"""The trilingual GPT-2 pretraining corpus and its BPE.

Two artifacts, built once and read by pretraining and by every retrain-to-recover
run (:mod:`rspd.experiments.eval.recovery`):

* ``<data root>/datasets/<name>/tokenizer/`` -- a 50,257-entry byte-level BPE
  fitted on an equal-**bytes** sample of each language (:func:`train_tokenizer`).
  The same size as GPT-2's vocabulary, so the model is still 124M parameters;
  fitted per language because GPT-2's English BPE fragments Dutch and German
  far more, which would make "1/3 of the tokens per language" carry unequal
  amounts of text. :func:`measure_fertility` records tokens per character per
  language as the evidence.
* ``<data root>/datasets/<name>/corpus/`` -- per language three spans
  ``train`` / ``val`` / ``recovery`` of ``uint16`` token shards plus
  ``corpus_manifest.json`` (:func:`build`).

Provenance:

* **No shuffling.** Documents are taken in stream order.
* **Every bucket names a dataset ``revision``**, and the manifest is hashed
  (``spec_sha256``). The published corpus spec (``specs/gpt2_trilingual_corpus.yaml``)
  names ``revision: main`` -- exactly what the published run streamed -- so a
  rebuild today may read different documents; the published shards are the
  reference.
* **Disjointness is structural.** The spans are filled sequentially from ONE
  pass over each language's stream (``train`` until its budget is met, then
  ``val``, then ``recovery``); every document has one destination, and the
  realised document boundaries are verified and recorded.
* **Token counts are exact.** ``n_tokens_by_language`` records what was written
  (``N_nld``, the permanence-ratio denominator of the recovery study), never
  the budget requested.

Entry point: :func:`prepare` (tokenizer, fertility, shards, ``results.md``).
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator, Mapping

import numpy as np

from .manifest import manifest_hash

#: Fields a language bucket may carry. Everything except ``skip_docs`` and
#: ``min_chars`` is required — ``revision`` above all, since it is the one
#: people leave out and the one that decides whether this is rebuildable.
BUCKET_FIELDS = ("lang", "dataset", "config", "split", "revision",
                 "text_field", "min_chars")

#: The three spans every language is cut into, in stream order. Order matters:
#: ``train`` first so that a shortened run still yields a usable model, and
#: ``recovery`` last because it is the largest and the most tolerant of a
#: stream that ends early.
SPANS = ("train", "val", "recovery")

#: uint16 holds a 50257-entry vocabulary with room to spare and halves the
#: on-disk size against int32. Asserted at write time rather than assumed.
DTYPE = np.uint16

#: Tokens per shard file. 100M uint16 = 200 MB, which is a comfortable unit to
#: memory-map and to re-download if one is corrupted.
SHARD_TOKENS = 100_000_000


def resolve_spec(cfg_block: Mapping[str, Any]) -> list[dict]:
    """Validate the ``corpus.buckets`` block: one bucket per language.

    Raises rather than defaulting on anything that affects reproducibility.
    """
    buckets = list(cfg_block or [])
    if not buckets:
        raise ValueError("corpus.buckets needs at least one language")
    out, seen = [], set()
    for i, raw in enumerate(buckets):
        unknown = set(raw) - set(BUCKET_FIELDS)
        if unknown:
            raise ValueError(f"corpus.buckets[{i}]: unknown field(s) "
                             f"{sorted(unknown)}; known: {sorted(BUCKET_FIELDS)}")
        for req in ("lang", "dataset", "split", "revision", "text_field"):
            if not raw.get(req):
                raise ValueError(
                    f"corpus.buckets[{i}]: '{req}' is required. "
                    f"'revision' especially — without it this corpus cannot be "
                    f"rebuilt once the dataset gains rows upstream.")
        lang = str(raw["lang"])
        if lang in seen:
            raise ValueError(f"corpus.buckets: duplicate language {lang!r}; "
                             f"one bucket per language")
        seen.add(lang)
        out.append({
            "lang": lang,
            "dataset": str(raw["dataset"]),
            "config": raw.get("config"),
            "split": str(raw["split"]),
            "revision": str(raw["revision"]),
            "text_field": str(raw["text_field"]),
            "min_chars": int(raw.get("min_chars") or 0),
        })
    return out


def resolve_spans(tokens_per_lang: int, val_tokens: int, recovery_tokens: int
                  ) -> dict[str, dict]:
    """Per-span **token budgets**, consumed in :data:`SPANS` order from one pass.

    There are no document-index ranges to reserve any more: a span is "the next
    N tokens of the stream", so the boundaries fall where they fall and are
    reported afterwards rather than predicted. Returns
    ``{span: {"tokens": int}}``.
    """
    if min(tokens_per_lang, val_tokens, recovery_tokens) <= 0:
        raise ValueError("every span needs a positive token budget")
    return {"train": {"tokens": int(tokens_per_lang)},
            "val": {"tokens": int(val_tokens)},
            "recovery": {"tokens": int(recovery_tokens)}}


def _stream(bucket: Mapping[str, Any]):
    from datasets import load_dataset
    kwargs = dict(path=bucket["dataset"], split=bucket["split"],
                  streaming=True, revision=bucket["revision"])
    if bucket.get("config"):
        kwargs["name"] = bucket["config"]
    return load_dataset(**kwargs)


def iter_docs(bucket: Mapping[str, Any]) -> Iterator[str]:
    """Every document of a bucket's stream, in order, filtered by ``min_chars``.

    One iterator serves all three spans — the caller stops taking when a span's
    budget is met and resumes for the next, which is what keeps this to a single
    pass. Documents below ``min_chars`` are skipped and consume no budget.
    """
    field, min_chars = bucket["text_field"], int(bucket.get("min_chars") or 0)
    ds = _stream(bucket)
    it = iter(ds)
    try:
        for doc in it:
            text = doc.get(field) or ""
            if len(text) >= min_chars:
                yield text
    finally:
        # A span almost always ends mid-shard, which abandons an in-flight
        # parquet download. `datasets` retries that on a background thread, and
        # if the interpreter finalizes while the retry is live the process dies
        # with "PyGILState_Release: auto-releasing thread-state" — SIGABRT,
        # after every byte has been written correctly. Close deterministically.
        try:
            it.close()
        except Exception:
            pass
        del it, ds


def write_span(tokenizer, docs: Iterator[str], span: Mapping[str, Any],
               out_dir: str, *, eot_id: int, encode_batch: int = 1000,
               doc_offset: int = 0, log=print) -> dict:
    """Fill one span from a **shared** document iterator, then stop.

    ``docs`` is not consumed past this span's budget, so the caller can hand the
    same iterator to the next span and the two cannot overlap — that is the
    whole disjointness argument, and it is a property of the control flow rather
    than of any index arithmetic.

    Returns the exact token count written and the document range consumed.
    """
    os.makedirs(out_dir, exist_ok=True)
    budget = int(span["tokens"])
    buf: list[int] = []
    written = shard_idx = n_docs = n_chars = 0
    shards: list[str] = []

    def flush(final: bool = False) -> None:
        nonlocal buf, written, shard_idx
        while len(buf) >= SHARD_TOKENS or (final and buf):
            take, buf = buf[:SHARD_TOKENS], buf[SHARD_TOKENS:]
            arr = np.array(take, dtype=DTYPE)
            path = os.path.join(out_dir, f"shard_{shard_idx:04d}.bin")
            arr.tofile(path)
            shards.append(os.path.basename(path))
            written += arr.size
            shard_idx += 1
            log(f"      shard {shard_idx:04d}: {arr.size:,} tokens "
                f"({written:,}/{budget:,})")
            if not final:
                break

    pending: list[str] = []

    def drain() -> bool:
        """Encode the buffered documents. True once the budget is met.

        Batched, not per document: one encode() per document is single-threaded
        Python into Rust and measured at ~143k tok/s, which put this corpus past
        its job timeout. The batch path releases the GIL.
        """
        nonlocal buf, n_docs, n_chars, pending
        if not pending:
            return False
        for ids in tokenizer(pending, add_special_tokens=False)["input_ids"]:
            buf.extend(ids)
            buf.append(eot_id)
        n_docs += len(pending)
        n_chars += sum(len(t) for t in pending)
        pending = []
        return written + len(buf) >= budget

    for text in docs:
        pending.append(text)
        if len(pending) >= encode_batch:
            if drain():
                break
            while len(buf) >= SHARD_TOKENS:
                flush()
    else:
        drain()

    # Truncate exactly at the budget so a span cannot borrow from the next.
    buf = buf[:max(0, budget - written)]
    flush(final=True)

    if written < budget:
        log(f"      WARNING: span ended {budget - written:,} tokens short of "
            f"its {budget:,} budget — the stream ran out. The manifest records "
            f"what was written, so the denominator stays honest.")

    return {"tokens": int(written), "docs": int(n_docs), "chars": int(n_chars),
            "shards": shards,
            "doc_lo": int(doc_offset), "doc_hi": int(doc_offset + n_docs),
            "fertility_tokens_per_char": (written / n_chars) if n_chars else None}


def build(cfg: Mapping[str, Any], tokenizer, out_root: str, *, log=print) -> dict:
    """Build every (language, span) in **one pass per language**.

    The manifest is the contract the trainer and the recovery loop both read:
    shard names, exact per-language token counts, and the realised document
    boundaries that make the spans disjoint.
    """
    corpus = cfg["corpus"]
    buckets = resolve_spec(corpus["buckets"])
    spans = resolve_spans(int(corpus["tokens_per_language"]),
                          int(corpus["val_tokens"]),
                          int(corpus["recovery_tokens"]))
    eot_id = tokenizer.eos_token_id
    if eot_id is None:
        raise ValueError("tokenizer has no eos_token_id; documents would run "
                         "into each other with no separator")

    # Only the target language is fine-tuned on in the Dutch-only recovery
    # protocol, so other languages' `recovery` spans would be tokens nothing reads.
    recovery_langs = set(corpus.get("recovery_languages")
                         or [b["lang"] for b in buckets])
    encode_batch = int(corpus.get("encode_batch", 1000))

    spec = {"buckets": buckets, "spans": spans, "eot_id": int(eot_id),
            "recovery_languages": sorted(recovery_langs),
            "vocab_size": int(tokenizer.vocab_size),
            "shard_tokens": SHARD_TOKENS, "layout": "sequential-single-pass"}
    digest = manifest_hash(spec)

    langs: dict[str, dict] = {}
    for bucket in buckets:
        lang = bucket["lang"]
        langs[lang] = {}
        docs = iter_docs(bucket)          # ONE iterator for all three spans
        cursor = 0
        for span_name in SPANS:
            if span_name == "recovery" and lang not in recovery_langs:
                log(f"  [{lang}/recovery] skipped — not a recovery language")
                langs[lang][span_name] = {
                    "tokens": 0, "docs": 0, "chars": 0, "shards": [],
                    "doc_lo": cursor, "doc_hi": cursor, "skipped": True,
                    "fertility_tokens_per_char": None}
                continue
            log(f"  [{lang}/{span_name}] -> {spans[span_name]['tokens']:,} tokens "
                f"(from document {cursor:,})")
            rec = write_span(tokenizer, docs, spans[span_name],
                             os.path.join(out_root, lang, span_name),
                             eot_id=int(eot_id), encode_batch=encode_batch,
                             doc_offset=cursor, log=log)
            cursor = rec["doc_hi"]
            langs[lang][span_name] = rec
            fert = rec["fertility_tokens_per_char"]
            log(f"    -> {rec['tokens']:,} tokens from {rec['docs']:,} docs"
                + (f", fertility {fert:.4f} tok/char" if fert else ""))
        del docs

    # Disjointness, verified on the realised boundaries rather than assumed.
    for lang, rec in langs.items():
        used = [(r["doc_lo"], r["doc_hi"]) for r in rec.values()
                if not r.get("skipped")]
        for (a_lo, a_hi), (b_lo, b_hi) in zip(sorted(used), sorted(used)[1:]):
            if a_hi > b_lo:
                raise ValueError(
                    f"{lang}: spans overlap at documents [{b_lo}, {a_hi}) — the "
                    f"held-out set is not held out; refusing to write a manifest")

    manifest = {
        "spec_sha256": digest, "spec": spec, "languages": langs,
        "n_tokens_by_language": {l: langs[l]["train"]["tokens"] for l in langs},
        "shard_dtype": np.dtype(DTYPE).name,
    }
    with open(os.path.join(out_root, "corpus_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    log(f"  [manifest] spec sha256:{digest[:12]}")
    return manifest


def load_manifest(out_root: str) -> dict:
    with open(os.path.join(out_root, "corpus_manifest.json")) as f:
        return json.load(f)


def span_shards(out_root: str, lang: str, span: str) -> list[str]:
    """Absolute paths of a span's shards, in order. The trainer's entry point."""
    man = load_manifest(out_root)
    rec = man["languages"][lang][span]
    return [os.path.join(out_root, lang, span, s) for s in rec["shards"]]


# --------------------------------------------------------------------------- #
# The trilingual BPE                                                           #
# --------------------------------------------------------------------------- #
#: The end-of-text token, matching GPT-2's so that anything expecting the
#: conventional name finds it.
EOT = "<|endoftext|>"

#: Same size as GPT-2's vocabulary — see the module docstring: this is what
#: holds the parameter count at 124M.
VOCAB_SIZE = 50257


def sample_texts(buckets, bytes_per_language: int, *, log=print) -> Iterator[str]:
    """Yield documents from the head of each language's stream, equal bytes each.

    The head is the ``train`` span (spans are filled sequentially, train first),
    which is the right place to fit from: the tokenizer is part of the model, so
    training text is correct and held-out text would be a leak.
    """
    for bucket in buckets:
        lang, used = bucket["lang"], 0
        for text in iter_docs(bucket):
            yield text
            used += len(text.encode("utf-8", errors="ignore"))
            if used >= bytes_per_language:
                break
        log(f"  [tokenizer] {lang}: sampled {used:,} bytes")


def train_tokenizer(buckets, out_dir: str, *, bytes_per_language: int,
          vocab_size: int = VOCAB_SIZE, log=print):
    """Fit a byte-level BPE and save it as a ``PreTrainedTokenizerFast``
    (needs the ``tokenizers`` package, a ``transformers`` dependency).

    Byte-level so there is no ``unk``: every byte sequence is representable,
    which matters across three languages where a rare accented form should cost
    more tokens rather than become an unknown.
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast

    os.makedirs(out_dir, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(
        sample_texts(buckets, bytes_per_language, log=log),
        vocab_size=vocab_size, min_frequency=2, special_tokens=[EOT],
    )
    tok.save(os.path.join(out_dir, "tokenizer.json"))

    fast = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(out_dir, "tokenizer.json"),
        eos_token=EOT, bos_token=EOT, unk_token=EOT, pad_token=EOT,
    )
    # model_max_length matches the context the trainer uses; leaving it unset
    # makes transformers warn on every call.
    fast.model_max_length = 1024
    fast.save_pretrained(out_dir)
    log(f"  [tokenizer] vocab {fast.vocab_size:,} -> {out_dir}")
    if fast.vocab_size > 65535:
        raise ValueError(f"vocab {fast.vocab_size} exceeds uint16; the shard "
                         f"dtype (DTYPE) would overflow")
    return fast


def measure_fertility(tokenizer, buckets, *, skip_bytes: int = 0,
                      chars_per_language: int = 2_000_000, log=print) -> dict:
    """Tokens per character, per language -- the evidence that the vocabulary
    treats the three languages alike.

    Measured **past** the bytes the vocabulary was fitted on (``skip_bytes``),
    so the number is not flattered by having seen the exact text. It is not
    measured on the ``val`` span: reaching that span means streaming the whole
    train span first, which costs tens of GB per language for a statistic that
    only needs a couple of million characters of held-out-from-the-fit text.
    """
    out = {}
    for bucket in buckets:
        lang, n_chars, n_tokens, seen = bucket["lang"], 0, 0, 0
        for text in iter_docs(bucket):
            if seen < skip_bytes:
                seen += len(text.encode("utf-8", errors="ignore"))
                continue
            n_tokens += len(tokenizer.encode(text))
            n_chars += len(text)
            if n_chars >= chars_per_language:
                break
        out[lang] = (n_tokens / n_chars) if n_chars else None
        log(f"  [fertility] {lang}: {out[lang]:.4f} tok/char "
            f"over {n_chars:,} chars")
    if all(v for v in out.values()):
        spread = max(out.values()) / min(out.values())
        log(f"  [fertility] spread {spread:.2f}x across languages "
            f"({'balanced' if spread <= 1.10 else 'NOT balanced (spread > 1.10)'})")
        out["_spread"] = spread
    return out


# --------------------------------------------------------------------------- #
# Prepare the corpus                                                           #
# --------------------------------------------------------------------------- #
def load_corpus_config(path: str) -> dict:
    """A corpus config (``name``, ``corpus``, ``tokenizer`` blocks; see
    ``specs/gpt2_trilingual_corpus.yaml``)."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("name", "corpus", "tokenizer"):
        if key not in cfg:
            raise ValueError(f"{path}: config needs '{key}'")
    return cfg


def corpus_root(name: str, *, root=None, create: bool = False) -> str:
    """``<data root>/datasets/<name>/corpus`` -- the shards and manifest."""
    from ..paths import datasets_dir
    return str(datasets_dir(name, root=root, create=create) / "corpus")


def tokenizer_dir(name: str, *, root=None) -> str:
    """``<data root>/datasets/<name>/tokenizer`` -- the trilingual BPE."""
    from ..paths import datasets_dir
    return str(datasets_dir(name, root=root, create=False) / "tokenizer")


def prepare(cfg: Mapping[str, Any], *, root=None, log_dir=None, log=print) -> dict:
    """Fit (or reuse) the BPE, measure fertility, write the shards and the
    manifest, and ``results.md`` under ``log_dir`` (default
    ``<data root>/logs/gpt2/<name>``). CPU-only and network-bound. Returns the
    corpus manifest."""
    import time

    from ..paths import datasets_dir, logs_dir

    out_dir = str(datasets_dir(cfg["name"], root=root))
    log_dir = str(log_dir or logs_dir("gpt2", cfg["name"], root=root))
    os.makedirs(log_dir, exist_ok=True)
    croot = os.path.join(out_dir, "corpus")
    tok_dir = os.path.join(out_dir, "tokenizer")
    os.makedirs(croot, exist_ok=True)
    buckets = resolve_spec(cfg["corpus"]["buckets"])
    spans = resolve_spans(int(cfg["corpus"]["tokens_per_language"]),
                          int(cfg["corpus"]["val_tokens"]),
                          int(cfg["corpus"]["recovery_tokens"]))
    log(f"[prepare] {cfg['name']}: languages {[b['lang'] for b in buckets]}")
    t0 = time.time()
    # Idempotent: a re-run after a transient stream failure must not refit the
    # vocabulary (which would silently invalidate every shard already written).
    if os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        log(f"[prepare] tokenizer: reusing {tok_dir} (vocab {tokenizer.vocab_size:,})")
    else:
        log("[prepare] fitting trilingual BPE ...")
        tokenizer = train_tokenizer(
            buckets, tok_dir, bytes_per_language=int(cfg["tokenizer"]["bytes_per_language"]),
            vocab_size=int(cfg["tokenizer"].get("vocab_size", VOCAB_SIZE)), log=log)
    t_tok = time.time() - t0
    # Fertility before the shards, so a bad vocabulary is caught early; measured
    # past the fitting sample, on text the vocabulary has not seen.
    fertility = measure_fertility(
        tokenizer, buckets, skip_bytes=int(cfg["tokenizer"]["bytes_per_language"]),
        chars_per_language=int(cfg["tokenizer"].get("fertility_chars", 2_000_000)), log=log)
    log("[prepare] writing shards ...")
    manifest = build(cfg, tokenizer, croot, log=log)
    manifest["fertility_tokens_per_char"] = fertility
    manifest["tokenizer_dir"] = os.path.relpath(tok_dir, out_dir)
    with open(os.path.join(croot, "corpus_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    elapsed = time.time() - t0
    counts = manifest["n_tokens_by_language"]
    total = sum(counts.values())
    lines = [f"# Corpus preparation -- `{cfg['name']}`", "",
             f"- tokenizer: vocab {tokenizer.vocab_size:,}, fitted in {t_tok/60:.1f} min",
             f"- total wall-clock: {elapsed/60:.1f} min",
             f"- spec sha256: `{manifest['spec_sha256'][:12]}`", "",
             "## Exact token counts (train span)", "",
             "| language | tokens | share | fertility (tok/char) |",
             "| --- | ---: | ---: | ---: |"]
    for lang, n in sorted(counts.items()):
        fert = fertility.get(lang)
        lines.append(f"| {lang} | {n:,} | {n/total:.3%} | {fert:.4f} |" if fert
                     else f"| {lang} | {n:,} | {n/total:.3%} | -- |")
    lines += ["", f"**Total: {total:,} tokens.**", "", "## Span sizes", "",
              "| language | train | val | recovery |", "| --- | ---: | ---: | ---: |"]
    for lang in sorted(manifest["languages"]):
        r = manifest["languages"][lang]
        lines.append(f"| {lang} | {r['train']['tokens']:,} | {r['val']['tokens']:,} | "
                     f"{r['recovery']['tokens']:,} |")
    with open(os.path.join(log_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    if fertility.get("_spread") and fertility["_spread"] > 1.10:
        log(f"[prepare] WARNING: fertility spread {fertility['_spread']:.2f}x exceeds 1.10")
    return manifest
