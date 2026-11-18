"""Equal token budget for the gradient baselines: the ``budget:`` block, the
packed retained stream and the realised-exposure record.

A budgeted run sees the target ``target_passes`` times and every retained
document token exactly once (:data:`RETAIN_STREAM_RULE`); :class:`Exposure`
counts what the loss actually saw from the attention masks and
:func:`assert_budget` refuses a run that did not realise its budget. A config
without a ``budget:`` block is unbudgeted (:func:`resolve_budget` -> ``None``).
"""
from __future__ import annotations

from . import corpora as _C
from .corpora import _iter_local_jsonl_file, corpus_kind, load_corpus_texts, tokenize_batch

#: The keys of the top-level ``budget:`` block — all five, no others.
BUDGET_KEYS = ("target_passes", "retain_exposure_tokens", "steps", "forget_max_length",
               "retain_block")

#: How the retained set is consumed under a budget (recorded in every plan).
RETAIN_STREAM_RULE = (
    "packed, the whole set exactly once: per file of the retained corpus every document "
    "is tokenised without truncation (add_special_tokens=False), one separator token "
    "(eos) is prepended to each document, the file's documents are concatenated and "
    "cut into retain_block-token blocks (only the file's last block is short); the "
    "files' block lists are round-robin interleaved (file0 block0, file1 block0, file0 "
    "block1, ...); block i of N goes to optimiser step floor(i * steps / N), so every "
    "step holds floor(N / steps) or ceil(N / steps) blocks and the loss sees every "
    "document token exactly once")


def resolve_budget(cfg) -> dict | None:
    """The top-level ``budget:`` block, validated, or ``None`` when the config
    has none (the run is then unbudgeted).

    Exactly the keys :data:`BUDGET_KEYS`, every value a positive int, and
    ``target_passes`` a multiple of ``steps``; the result additionally carries
    ``passes_per_step = target_passes // steps``.
    """
    b = cfg.get("budget") if isinstance(cfg, dict) else None
    if b is None:
        return None
    if not isinstance(b, dict) or not b:
        raise ValueError(f"`budget:` must be a mapping with keys {list(BUDGET_KEYS)}, got {b!r}")
    unknown = sorted(set(b) - set(BUDGET_KEYS))
    if unknown:
        raise ValueError(f"unknown key(s) in `budget:`: {unknown}; known: {list(BUDGET_KEYS)}")
    missing = [k for k in BUDGET_KEYS if k not in b]
    if missing:
        raise ValueError(f"`budget:` is missing {missing}; it needs all of {list(BUDGET_KEYS)}")
    out = {}
    for k in BUDGET_KEYS:
        v = b[k]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"budget.{k} must be a positive int, got {v!r}")
        out[k] = int(v)
    if out["target_passes"] % out["steps"]:
        raise ValueError(f"budget.target_passes ({out['target_passes']}) must be a multiple "
                         f"of budget.steps ({out['steps']}): every step makes a whole number "
                         f"of passes over the target")
    out["passes_per_step"] = out["target_passes"] // out["steps"]
    return out


def separator_token_id(tokenizer) -> int:
    """The per-document separator of the packed stream: EOS (GPT-2's
    ``<|endoftext|>``), else BOS; a tokenizer with neither is refused."""
    for attr in ("eos_token_id", "bos_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            return int(v)
    raise ValueError("the tokenizer has neither eos_token_id nor bos_token_id; the packed "
                     "retained stream needs a document separator")


def _corpus_files(corpus: str) -> list:
    """``[(label, iterator-of-texts), ...]`` — one entry per file of a
    ``local_jsonl`` corpus, or one entry for a hub corpus (its whole split)."""
    if corpus not in _C.CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(_C.CORPORA)}")
    spec = _C.CORPORA[corpus]
    if corpus_kind(spec) == "local_jsonl":
        return [(rel, _iter_local_jsonl_file(spec, rel)) for rel in spec["path"]]
    if spec.get("stream"):
        raise ValueError(f"corpus {corpus!r} is streamed (multi-GB); the packed retained "
                         f"stream reads a corpus whole and needs a bounded one")
    return [(str(spec["path"]), iter(load_corpus_texts(corpus, min_len=0, max_rows=None)))]


def packed_retain_stream(tokenizer, corpus: str, *, block: int, sep_token_id: int,
                         min_len: int = 0) -> dict:
    """:data:`RETAIN_STREAM_RULE` as data: the retained ``corpus`` packed into
    ``block``-token 1-D ``LongTensor`` blocks, files round-robin interleaved.

    Per file every document (``len(text) > min_len``; the training loaders'
    filter, 0 keeps everything but the empty row) is tokenised with
    ``add_special_tokens=False`` and no truncation, one ``sep_token_id`` is
    prepended, the file is concatenated and cut into ``block``-token blocks —
    only the file's **last** block may be short — and the files' block lists are
    interleaved ``file0[0], file1[0], file0[1], ...`` (a shorter list simply
    runs out). No ``max_rows``, no padding, no per-document truncation.

    Returns ``{"blocks", "block_file", "block_n_sep", "n_docs", "n_doc_tokens",
    "n_sep_tokens", "per_file", "block", "sep_token_id", "corpus", "rule"}``:
    ``block_file[i]`` is the file index block ``i`` came from, ``block_n_sep[i]``
    the number of separator positions in it (exact, from the document
    boundaries, not from the token id), ``n_doc_tokens`` the number of document
    tokens over every file (what the loss must see exactly once) and
    ``n_sep_tokens == n_docs``.
    """
    import torch

    if int(block) < 1:
        raise ValueError(f"block must be >= 1, got {block!r}")
    block = int(block)
    sep = int(sep_token_id)
    per_file_blocks, per_file_nsep, per_file = [], [], []
    bs = 256
    for label, texts_iter in _corpus_files(corpus):
        docs = [t for t in texts_iter if isinstance(t, str) and len(t) > int(min_len)]
        ids, is_sep, n_doc_tokens = [], [], 0
        for i in range(0, len(docs), bs):
            enc = tokenizer(docs[i:i + bs], add_special_tokens=False, truncation=False)
            for doc_ids in enc["input_ids"]:
                ids.append(sep)
                is_sep.append(True)
                ids.extend(doc_ids)
                is_sep.extend([False] * len(doc_ids))
                n_doc_tokens += len(doc_ids)
        stream = torch.tensor(ids, dtype=torch.long)
        blocks = [stream[i:i + block] for i in range(0, len(ids), block)]
        nsep = [sum(is_sep[i:i + block]) for i in range(0, len(ids), block)]
        per_file_blocks.append(blocks)
        per_file_nsep.append(nsep)
        per_file.append({"path": label, "n_docs": len(docs), "n_doc_tokens": n_doc_tokens,
                         "n_sep_tokens": len(docs), "n_blocks": len(blocks),
                         "last_block_tokens": int(blocks[-1].shape[0]) if blocks else 0})
    if not any(per_file_blocks):
        raise RuntimeError(f"retained corpus {corpus!r} yielded no documents for the packed "
                           f"stream")
    blocks, block_file, block_n_sep = [], [], []
    for i in range(max(len(b) for b in per_file_blocks)):
        for f, fb in enumerate(per_file_blocks):
            if i < len(fb):
                blocks.append(fb[i])
                block_file.append(f)
                block_n_sep.append(int(per_file_nsep[f][i]))
    return {"blocks": blocks, "block_file": block_file, "block_n_sep": block_n_sep,
            "n_docs": sum(p["n_docs"] for p in per_file),
            "n_doc_tokens": sum(p["n_doc_tokens"] for p in per_file),
            "n_sep_tokens": sum(p["n_sep_tokens"] for p in per_file),
            "per_file": per_file, "block": block, "sep_token_id": sep, "corpus": corpus,
            "rule": RETAIN_STREAM_RULE}


def spread_ranges(n_items: int, n_slots: int) -> list:
    """Slot ``i`` of ``M`` gets items ``[i * n // M, (i + 1) * n // M)`` —
    floor or ceil of ``n / M`` each, every item exactly once, in order."""
    M = int(n_slots)
    if M < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots!r}")
    n = int(n_items)
    return [(i * n // M, (i + 1) * n // M) for i in range(M)]


def budget_retain_plan(tokenizer, corpus: str, *, budget: dict, min_len: int = 0) -> dict:
    """The retained side of a budgeted run: ``{"blocks", "block_n_sep", "ranges",
    "stream", "schedule"}`` — the packed stream of ``corpus`` at
    ``budget["retain_block"]``, ``ranges[step]`` the block slice optimiser step
    ``step`` consumes (:func:`spread_ranges` over ``budget["steps"]``), and
    ``schedule`` the record for ``metrics.json``. Every step holds at least one
    block (a set with fewer blocks than steps is refused)."""
    stream = packed_retain_stream(tokenizer, corpus, block=budget["retain_block"],
                                  sep_token_id=separator_token_id(tokenizer), min_len=min_len)
    n_blocks = len(stream["blocks"])
    steps = int(budget["steps"])
    if n_blocks < steps:
        raise ValueError(f"the retained corpus {corpus!r} packs into {n_blocks} blocks of "
                         f"{budget['retain_block']} tokens, fewer than budget.steps ({steps}); "
                         f"every step must see at least one block")
    ranges = spread_ranges(n_blocks, steps)
    per_step = [b - a for a, b in ranges]
    schedule = {"steps": steps, "n_blocks": n_blocks, "retain_block": int(budget["retain_block"]),
                "blocks_per_step_min": min(per_step), "blocks_per_step_max": max(per_step),
                "n_docs": stream["n_docs"], "n_doc_tokens": stream["n_doc_tokens"],
                "n_sep_tokens": stream["n_sep_tokens"], "per_file": stream["per_file"],
                "sep_token_id": stream["sep_token_id"], "corpus": corpus,
                "rule": RETAIN_STREAM_RULE}
    return {"blocks": stream["blocks"], "block_n_sep": stream["block_n_sep"],
            "ranges": ranges, "stream": stream, "schedule": schedule}


def windows_to_batch(windows, *, pad_token_id: int, padding_side: str = "right",
                     device=None, with_labels: bool = False) -> dict:
    """A tokenised batch (``input_ids`` / ``attention_mask`` [/ ``labels`` with
    pads at ``-100``]) from a list of 1-D id tensors, padded on ``padding_side``
    (RMU/KUDA force left; SimNPO uses the tokenizer's side)."""
    import torch

    if not windows:
        raise ValueError("windows_to_batch needs at least one window")
    L = max(int(w.shape[0]) for w in windows)
    n = len(windows)
    ids = torch.full((n, L), int(pad_token_id), dtype=torch.long)
    mask = torch.zeros((n, L), dtype=torch.long)
    for i, w in enumerate(windows):
        k = int(w.shape[0])
        if padding_side == "left":
            ids[i, L - k:] = w
            mask[i, L - k:] = 1
        else:
            ids[i, :k] = w
            mask[i, :k] = 1
    enc = {"input_ids": ids, "attention_mask": mask}
    if with_labels:
        labels = ids.clone()
        labels[mask == 0] = -100
        enc["labels"] = labels
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def target_tokens_unique(tokenizer, corpora, *, max_length: int, min_len: int = 0) -> tuple:
    """``(n_docs, n_tokens)`` of the target set as the training loss sees it:
    every document of every corpus in ``corpora`` (the ``min_len`` filter of the
    training loaders), tokenised exactly as :func:`tokenize_batch` does
    (truncated to ``max_length``), non-pad positions counted."""
    n_docs, n_tokens = 0, 0
    for c in corpora:
        texts = load_corpus_texts(c, min_len=min_len, max_rows=None)
        for i in range(0, len(texts), 64):
            enc = tokenize_batch(tokenizer, texts[i:i + 64], max_length=int(max_length))
            n_docs += int(enc["attention_mask"].shape[0])
            n_tokens += int(enc["attention_mask"].sum())
    return n_docs, n_tokens


class Exposure:
    """What the loss actually saw, counted from the attention masks (never from
    a nominal ``max_length``): target sequences and non-pad tokens, retained
    blocks, non-pad tokens and separator positions, the retained tokens a
    covariance pass read outside the loss (KUDA), and optimiser steps."""

    def __init__(self):
        self.target_sequences = 0
        self.target_tokens = 0
        self.retain_blocks = 0
        self.retain_tokens = 0
        self.retain_sep_tokens = 0
        self.retain_covariance_tokens = 0
        self.steps = 0

    def add_target(self, attention_mask):
        self.target_sequences += int(attention_mask.shape[0])
        self.target_tokens += int(attention_mask.sum())

    def add_retain(self, attention_mask, n_sep: int):
        self.retain_blocks += int(attention_mask.shape[0])
        self.retain_tokens += int(attention_mask.sum())
        self.retain_sep_tokens += int(n_sep)

    def add_retain_covariance(self, n_tokens: int):
        self.retain_covariance_tokens += int(n_tokens)

    def step(self, n: int = 1):
        self.steps += int(n)

    def record(self, *, budget, target_unique: int, retain_unique: int, retain_sep: int) -> dict:
        """The ``metrics.json`` ``budget`` record: the configured block and the
        realised exposure. ``retain_tokens_exposed`` counts every non-pad
        position the retained loss saw (separators included);
        ``retain_tokens_exposed_docs`` the document tokens only."""
        tu = int(target_unique)
        return {
            "configured": dict(budget) if budget else None,
            "target_tokens_unique": tu,
            "target_sequences": self.target_sequences,
            "target_tokens_exposed": self.target_tokens,
            "target_passes_realised": (self.target_tokens / tu) if tu else None,
            "retain_tokens_unique": int(retain_unique),
            "retain_separator_tokens_unique": int(retain_sep),
            "retain_blocks": self.retain_blocks,
            "retain_tokens_exposed": self.retain_tokens,
            "retain_tokens_exposed_docs": self.retain_tokens - self.retain_sep_tokens,
            "retain_separator_tokens": self.retain_sep_tokens,
            "retain_tokens_covariance": self.retain_covariance_tokens,
            "steps": self.steps,
        }

    @staticmethod
    def line(record: dict) -> str:
        p = record.get("target_passes_realised")
        return ("[budget] "
                f"steps={record['steps']} target_tokens_exposed={record['target_tokens_exposed']} "
                f"(unique {record['target_tokens_unique']}, passes "
                f"{'n/a' if p is None else f'{p:g}'}) "
                f"retain_tokens_exposed_docs={record['retain_tokens_exposed_docs']} "
                f"(unique {record['retain_tokens_unique']}; +{record['retain_separator_tokens']} "
                f"separators) retain_tokens_covariance={record['retain_tokens_covariance']}")

    @staticmethod
    def summary(record: dict, prefix: str = "budget/") -> dict:
        """The scalar fields of ``record`` as flat summary keys."""
        return {f"{prefix}{k}": v for k, v in record.items() if not isinstance(v, dict)}


def assert_budget(record: dict, budget: dict) -> None:
    """``RuntimeError`` unless the realised exposure IS the budget: ``steps``
    equal, the target seen ``target_passes`` times exactly, every retained
    document token seen exactly once, and the retained set within one block of
    ``retain_exposure_tokens``. Call this AFTER writing the run's metrics so the
    record is on disk whichever way it goes."""
    problems = []
    if record["steps"] != budget["steps"]:
        problems.append(f"steps {record['steps']} != budget.steps {budget['steps']}")
    want_t = budget["target_passes"] * record["target_tokens_unique"]
    if record["target_tokens_exposed"] != want_t:
        problems.append(f"target_tokens_exposed {record['target_tokens_exposed']} != "
                        f"target_passes {budget['target_passes']} x target_tokens_unique "
                        f"{record['target_tokens_unique']} = {want_t}")
    if record["retain_tokens_exposed_docs"] != record["retain_tokens_unique"]:
        problems.append(f"retain_tokens_exposed_docs {record['retain_tokens_exposed_docs']} != "
                        f"retain_tokens_unique {record['retain_tokens_unique']} (the whole "
                        f"retained set exactly once)")
    gap = abs(record["retain_tokens_unique"] - budget["retain_exposure_tokens"])
    if not gap < budget["retain_block"]:
        problems.append(f"|retain_tokens_unique {record['retain_tokens_unique']} - "
                        f"budget.retain_exposure_tokens {budget['retain_exposure_tokens']}| = "
                        f"{gap} >= retain_block {budget['retain_block']}")
    if problems:
        raise RuntimeError("the run did not realise its token budget: " + "; ".join(problems))
