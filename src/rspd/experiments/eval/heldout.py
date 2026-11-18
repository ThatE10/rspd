"""The held-out CE readout: ONE estimator shared by SRSPD and every baseline,
so a point of any method is scored on the same numbers.

A config's ``corpora:`` entries with ``topic: heldout`` plus a ``heldout:``
block (:func:`resolve_heldout_config`) define the readout; a
:class:`HeldoutReadout` encodes the rows once and scores the model before and
after an edit (:meth:`HeldoutReadout.score`), and :func:`heldout_delta` /
:func:`heldout_metrics` / :func:`heldout_rows_results` turn the two scores into
records. The estimator is :func:`rspd.experiments.eval.ce.block_ce`, pooled
over rows by token count.
"""
from __future__ import annotations

from ..data.corpora import encode, encode_all, load_corpus_texts, split_fit_heldout  # noqa: F401
from ..results import ResultRow
from .ce import block_ce

#: A held-out corpus with fewer rows than ``rows`` is read whole (the readout
#: is then over every row it has); the realised row count is recorded.
HELDOUT_DEFAULT_ROWS = 32
HELDOUT_DEFAULT_MAX_LENGTH = 1024


def pooled_ce(model, id_list):
    """Token-weighted mean next-token CE over several sequences (each scored on
    its own by :func:`rspd.experiments.eval.ce.block_ce`, i.e. teacher-forced,
    every position scored, in ``HELDOUT_WINDOW``-token windows)."""
    tot, n = 0.0, 0
    for ids in id_list:
        ce, k = block_ce(model, ids)
        if k:
            tot += ce * k
            n += k
    return (tot / n if n else float("nan")), n


def heldout_corpora(cfg) -> list[str]:
    """The names of the config's ``corpora:`` entries with ``topic: heldout``,
    in config order (``[]`` without a block)."""
    block = (cfg or {}).get("corpora") if isinstance(cfg, dict) else None
    if not isinstance(block, dict):
        return []
    return [n for n, s in block.items() if isinstance(s, dict) and s.get("topic") == "heldout"]


def resolve_heldout_config(cfg) -> dict | None:
    """The top-level ``heldout:`` block, validated, or ``None`` when absent (the
    readout is then skipped).

    Shape::

        heldout:
          rows: 32            # rows per corpus (default 32)
          max_length: 1024    # tokens per row (default 1024)
          tail: [heldout_eng, heldout_deu]   # optional: corpora read from the END

    **Row selection, and why ``tail`` exists.** The readout takes the FIRST
    ``rows`` rows of each ``topic: heldout`` corpus in file order -- except the
    corpora named in ``tail``, which contribute their LAST ``rows`` rows. That
    is the SRSPD target builder's convention
    (:func:`rspd.experiments.data.target.build_erasure_data`): its forget-side
    readout reads the first ``heldout_rows`` rows and its retain-side readout the
    last ``heldout_rows`` rows (the rows a hub corpus held out of the fit). The
    GPT-2 setup therefore sets ``tail: [heldout_eng, heldout_deu]`` so SRSPD and
    every baseline score the same rows of the same files.
    """
    block = (cfg or {}).get("heldout") if isinstance(cfg, dict) else None
    if block is None:
        return None
    if not isinstance(block, dict):
        raise ValueError(f"heldout: must be a mapping {{rows, max_length, tail}}, got {block!r}")
    unknown = set(block) - {"rows", "max_length", "tail", "min_len"}
    if unknown:
        raise ValueError(f"heldout: has unknown keys {sorted(unknown)}")
    rows = block.get("rows", HELDOUT_DEFAULT_ROWS)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"heldout.rows must be a positive int, got {rows!r}")
    max_length = block.get("max_length", HELDOUT_DEFAULT_MAX_LENGTH)
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError(f"heldout.max_length must be an int >= 2, got {max_length!r}")
    min_len = block.get("min_len", 0)
    if isinstance(min_len, bool) or not isinstance(min_len, int) or min_len < 0:
        raise ValueError(f"heldout.min_len must be a non-negative int, got {min_len!r}")
    corpora = heldout_corpora(cfg)
    if not corpora:
        raise ValueError("heldout: is set but no `corpora:` entry has `topic: heldout`")
    tail = block.get("tail") or []
    if isinstance(tail, str):
        tail = [tail]
    if not isinstance(tail, (list, tuple)) or not all(isinstance(t, str) for t in tail):
        raise ValueError(f"heldout.tail must be a list of corpus names, got {tail!r}")
    bad = [t for t in tail if t not in corpora]
    if bad:
        raise ValueError(f"heldout.tail names {bad}, which are not `topic: heldout` corpora "
                         f"{corpora}")
    return {"rows": rows, "max_length": max_length, "min_len": min_len,
            "corpora": corpora, "tail": [t for t in corpora if t in set(tail)]}


def heldout_rows(corpus: str, *, rows: int, tail: bool = False, min_len: int = 0) -> list[str]:
    """The readout rows of one corpus: the first ``rows`` rows in file order,
    or (``tail``) the last ``rows`` rows — :func:`split_fit_heldout`'s uncapped
    branch, which is what :func:`rspd.experiments.data.target.load_heldout_retain`
    applies."""
    if tail:
        texts = load_corpus_texts(corpus, min_len=min_len, max_rows=None)
        return split_fit_heldout(texts, None, rows)[1] if len(texts) > rows else texts
    return load_corpus_texts(corpus, min_len=min_len, max_rows=rows)


class HeldoutReadout:
    """The encoded readout rows of every ``topic: heldout`` corpus, scored on
    demand so a driver encodes once and scores before and after its edit.

    ``ids[name]`` is the list of ``[1, T]`` LongTensors (rows with < 2 tokens
    dropped); ``n_rows`` / ``n_tokens`` are the realised counts per corpus.
    """

    def __init__(self, tokenizer, corpora, *, rows, max_length, tail=(), min_len=0):
        self.rows, self.max_length = int(rows), int(max_length)
        self.tail = [c for c in corpora if c in set(tail)]
        self.ids, self.n_rows = {}, {}
        for c in corpora:
            texts = heldout_rows(c, rows=self.rows, tail=c in self.tail, min_len=min_len)
            self.ids[c] = encode_all(tokenizer, texts, self.max_length)
            self.n_rows[c] = len(texts)
        self.n_tokens = {c: sum(int(x.shape[-1]) for x in v) for c, v in self.ids.items()}

    @property
    def spec(self) -> dict:
        """What was scored, for ``metrics.json``."""
        return {"rows": self.rows, "max_length": self.max_length, "tail": list(self.tail),
                "n_rows": dict(self.n_rows), "n_tokens": dict(self.n_tokens),
                "estimator": "block_ce (teacher-forced mean next-token CE, "
                             "token-weighted over rows, HELDOUT_WINDOW windows)"}

    def score(self, model) -> dict:
        """``{corpus: ce}`` for the model as it is now; eval mode for the
        forward passes, the module's training flag restored afterwards."""
        import torch
        was_training = bool(getattr(model, "training", False))
        model.eval()
        try:
            with torch.no_grad():
                return {c: pooled_ce(model, v)[0] for c, v in self.ids.items()}
        finally:
            if was_training:
                model.train()


def load_heldout_readout(cfg, tokenizer) -> HeldoutReadout | None:
    """:class:`HeldoutReadout` from a config's ``heldout:`` block (its corpora
    must already be registered, :func:`register_config_corpora`), or ``None``
    when the block is absent."""
    H = resolve_heldout_config(cfg)
    if H is None:
        return None
    return HeldoutReadout(tokenizer, H["corpora"], rows=H["rows"], max_length=H["max_length"],
                          tail=H["tail"], min_len=H["min_len"])


def heldout_readout(model, tokenizer, corpora_by_name, *, rows, max_length, device=None,
                    tail=(), min_len=0) -> dict:
    """One-shot ``{name: ce}`` over ``corpora_by_name`` (a mapping of registered
    corpus names, or a list of them): the first ``rows`` rows of each in file
    order (the last ``rows`` for the names in ``tail``), each row's first
    ``max_length`` tokens, teacher-forced mean per-token CE pooled over the
    rows' tokens. ``device`` is accepted for symmetry with the loaders; the rows
    are moved to ``model.device`` by the estimator."""
    corpora = list(corpora_by_name)
    return HeldoutReadout(tokenizer, corpora, rows=rows, max_length=max_length, tail=tail,
                          min_len=min_len).score(model)


def heldout_delta(base: dict, edited: dict) -> dict:
    """``edited - base`` per corpus (``nan`` where either side is missing)."""
    return {c: (edited[c] - base[c]) if c in base and c in edited else float("nan")
            for c in edited}


def print_heldout(base: dict, edited: dict, *, tag: str = "heldout") -> None:
    """One ``[heldout] <name> base -> edited (delta)`` line per corpus."""
    for c in edited:
        b = base.get(c, float("nan"))
        print(f"[{tag}] {c} {b:.4f} -> {edited[c]:.4f} ({edited[c] - b:+.4f})", flush=True)


def heldout_metrics(readout: HeldoutReadout | None, base: dict | None, edited: dict | None) -> dict:
    """The three ``metrics.json`` dicts (``ce_heldout_base`` / ``ce_heldout`` /
    ``ce_heldout_delta``) plus ``heldout`` (the spec); every value ``None`` when
    the readout is off."""
    if readout is None or base is None or edited is None:
        return {"heldout": None, "ce_heldout_base": None, "ce_heldout": None,
                "ce_heldout_delta": None}
    return {"heldout": readout.spec, "ce_heldout_base": dict(base), "ce_heldout": dict(edited),
            "ce_heldout_delta": heldout_delta(base, edited)}


def heldout_summary(base: dict | None, edited: dict | None) -> dict:
    """Flat summary keys: ``heldout/<corpus>/{base,edited,delta}``."""
    if not base or not edited:
        return {}
    out = {}
    for c in edited:
        out[f"heldout/{c}/base"] = base.get(c)
        out[f"heldout/{c}/edited"] = edited[c]
        out[f"heldout/{c}/delta"] = edited[c] - base[c] if c in base else None
    return out




def heldout_rows_results(base: dict | None, edited: dict | None, *, model: str | None = None,
                         readout: HeldoutReadout | None = None) -> list:
    """The readout as :class:`rspd.experiments.results.ResultRow` rows: per
    corpus ``heldout_ce/<corpus>`` with metrics ``ce`` (edited), ``ce_base`` and
    ``ce_delta`` (nats; ``n`` = scored tokens when ``readout`` is given)."""
    if not base or not edited:
        return []
    out = []
    for c in edited:
        n = readout.n_tokens.get(c) if readout is not None else None
        b = base.get(c)
        out.append(ResultRow(f"heldout_ce/{c}", "ce", edited[c], model=model, n=n))
        out.append(ResultRow(f"heldout_ce/{c}", "ce_base", b, model=model, n=n))
        out.append(ResultRow(f"heldout_ce/{c}", "ce_delta",
                             None if b is None else edited[c] - b, model=model, n=n))
    return out

