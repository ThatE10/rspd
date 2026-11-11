"""The retained corpus: documents streamed from the HF hub (or read from local
JSONL) to an exact token budget per bucket, packed into equal chunks.

This is ``Y`` for :func:`srspd.erase` -- the behaviour the edit must leave
alone -- and its provenance is the product: a retained set that cannot be
rebuilt makes a result irreproducible, so the builder is organised around a
content hash.

* **No shuffling.** Documents are taken in stream order from ``skip_docs``; the
  last document of a bucket is truncated at the token boundary. (A streaming
  shuffle is reproducible only if the shard layout is frozen, which a config
  cannot pin.)
* **Every hub bucket pins a dataset ``revision``.**
* **The manifest is hashed** (:func:`manifest_sha256`, recorded as
  ``retained_corpus_sha256``) and a pre-stream key (:func:`spec_key`) names the
  token cache, so a changed budget, revision, tokenizer or framing is a
  different cache entry by construction. :func:`load_or_build` reads the cache
  or builds and writes it atomically. The cache directory is a parameter of
  :func:`shared_cache_path` (default ``<data root>/datasets/retained_corpus_cache``,
  see :mod:`srspd_experiments.paths`).

Bucket fields are :data:`BUCKET_FIELDS`; ``kind`` is ``"hub"`` (default) or
``"local_jsonl"`` (``path`` names a JSONL file, data-root-relative or absolute;
its content sha256 joins :func:`spec_key`).

Framing: raw text by default (``chat_template=None``); ``"user"`` wraps every
document as one user turn through the tokenizer's chat template.

Rendered rows (``format``)
--------------------------
``mmlu_lm_eval`` renders an MMLU row (``question`` / ``subject`` / ``choices`` /
``answer``) as the lm-evaluation-harness ``mmlu`` prompt *with* its answer
letter; ``chat_messages`` renders a ``messages`` list through the tokenizer's
chat template (no generation prompt). The rendered string is what ``min_chars``
is measured on, what the document hashes cover and what is tokenised. An
``mmlu_lm_eval`` bucket may carry ``exclude_question_sha256``: sha256 digests
of whitespace-normalised questions (:func:`question_sha256`) whose rows are
skipped before rendering and counted as ``n_excluded`` -- how MMLU-validation
questions that duplicate scored test items are kept out of a retained set.
``format`` and ``exclude_question_sha256`` enter the resolved spec, the
manifest and both hashes **only when set**, so specs without them hash as they
always have (the tests pin two such hashes).

One BOS per document
--------------------
``add_bos`` prepends the tokenizer's BOS to every document **unless the
rendered document already tokenises to a leading BOS** (a chat template that
emits a literal ``<bos>``); such documents are counted as
``n_docs_bos_already_present`` (written only when non-zero, so corpora whose
renderings never start with BOS keep their hashes). :func:`load_or_build`
rebuilds any cached corpus in which two BOS ids are adjacent
(:func:`has_double_bos`).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator, Mapping, Sequence

import torch

#: Fields a bucket may carry. ``tokens`` is the budget; everything else is
#: provenance a result has to be able to quote. ``kind`` is ``"hub"`` (the
#: default: ``dataset``/``split``/``revision`` required) or ``"local_jsonl"``
#: (``path`` required: a data-root-relative or absolute JSONL file, one
#: document per line, read in order; ``dataset``/``split``/``revision`` are
#: then ``None``). A local file's content sha256 joins :func:`spec_key`, so a
#: rewritten file is a different cache entry.
BUCKET_FIELDS = ("name", "dataset", "config", "split", "revision",
                 "text_field", "tokens", "skip_docs", "min_chars", "kind", "path", "format",
                 "exclude_question_sha256")
BUCKET_KINDS = ("hub", "local_jsonl")

#: Row renderings a bucket may ask for (``format``). ``None`` is the default:
#: ``text_field`` is the document. The others render a structured row to text
#: before tokenisation (see :func:`render_document`). A ``local_jsonl`` bucket
#: may name a ``format`` too; its documents are then the parsed JSON rows.
BUCKET_FORMATS = (None, "mmlu_lm_eval", "chat_messages")

#: Framings ``build_retained_tokens`` understands.
CHAT_TEMPLATES = (None, "user")


def resolve_spec(cfg_block: Mapping[str, Any]) -> list[dict]:
    """Validate an ``erasure.retained_corpus.buckets`` block (the ``buckets`` list
    of a retained-corpus config).

    Every field is required except ``skip_docs`` (default 0), ``min_chars``
    (default 0) and ``format`` (default none: the raw ``text_field``) — including
    ``revision``, which is the one people leave out and the one that decides
    whether the corpus can be rebuilt a month from now. ``format`` and
    ``exclude_question_sha256`` (``mmlu_lm_eval`` only: 64-hex digests of
    whitespace-normalised questions to skip, stored sorted) are kept in the
    resolved bucket only when they are set, so a spec without them hashes
    exactly as it did before the keys existed.
    """
    buckets = list(cfg_block or [])
    if not buckets:
        raise ValueError("retained_corpus needs at least one bucket")
    out, seen = [], set()
    for i, raw in enumerate(buckets):
        unknown = set(raw) - set(BUCKET_FIELDS)
        if unknown:
            raise ValueError(f"bucket {i} has unknown fields {sorted(unknown)}; "
                             f"expected a subset of {BUCKET_FIELDS}")
        kind = str(raw.get("kind") or "hub")
        if kind not in BUCKET_KINDS:
            raise ValueError(f"bucket {i} ({raw.get('name', '?')!r}) has kind {kind!r}; "
                             f"expected one of {BUCKET_KINDS}")
        required = (("name", "path", "text_field", "tokens") if kind == "local_jsonl"
                    else ("name", "dataset", "split", "revision", "text_field", "tokens"))
        missing = [f for f in required if raw.get(f) in (None, "")]
        if missing:
            raise ValueError(f"bucket {i} ({raw.get('name', '?')!r}) is missing {missing}")
        name = str(raw["name"])
        if name in seen:
            raise ValueError(f"duplicate bucket name {name!r}")
        seen.add(name)
        tokens = int(raw["tokens"])
        if tokens < 1:
            raise ValueError(f"bucket {name!r} needs a positive token budget; got {tokens}")
        fmt = raw.get("format")
        fmt = None if fmt in (None, "") else str(fmt)
        if fmt not in BUCKET_FORMATS:
            raise ValueError(f"bucket {name!r} has unknown format {fmt!r}; "
                             f"expected one of {BUCKET_FORMATS}")
        excl = raw.get("exclude_question_sha256")
        if excl is None or excl == "" or (isinstance(excl, (list, tuple)) and not excl):
            excl = None
        else:
            if fmt != "mmlu_lm_eval":
                raise ValueError(f"bucket {name!r}: exclude_question_sha256 applies only to format "
                                 f"'mmlu_lm_eval' (it hashes the row's question); got format {fmt!r}")
            if isinstance(excl, str) or not isinstance(excl, (list, tuple)):
                raise ValueError(f"bucket {name!r}: exclude_question_sha256 must be a list of "
                                 f"64-hex sha256 digests, got {type(excl).__name__}")
            digests = []
            for d in excl:
                s = str(d).strip().lower()
                if len(s) != 64 or any(c not in "0123456789abcdef" for c in s):
                    raise ValueError(f"bucket {name!r}: {d!r} is not a 64-hex sha256 digest")
                digests.append(s)
            if len(set(digests)) != len(digests):
                raise ValueError(f"bucket {name!r}: exclude_question_sha256 has duplicate digests")
            excl = sorted(digests)
        if kind == "local_jsonl":
            if any(raw.get(f) not in (None, "") for f in ("dataset", "config", "split", "revision")):
                raise ValueError(f"bucket {name!r} is local_jsonl: dataset/config/split/"
                                 f"revision do not apply (its provenance is the file "
                                 f"and its sha256)")
            resolved = {
                "name": name, "kind": "local_jsonl", "path": str(raw["path"]),
                "dataset": None, "config": None, "split": None, "revision": None,
                "text_field": str(raw["text_field"]), "tokens": tokens,
                "skip_docs": int(raw.get("skip_docs", 0) or 0),
                "min_chars": int(raw.get("min_chars", 0) or 0),
            }
            if fmt is not None:      # absent, not None: keeps format-less hashes stable
                resolved["format"] = fmt
            if excl is not None:     # likewise: format-only specs keep their hashes
                resolved["exclude_question_sha256"] = excl
            out.append(resolved)
            continue
        # A hub bucket carries no `kind` key, so its resolved dict -- and with it
        # spec_key, i.e. the cache file name every existing run shares -- is
        # byte-identical to what it was before local_jsonl existed.
        resolved = {
            "name": name,
            "dataset": str(raw["dataset"]),
            "config": (None if raw.get("config") in (None, "") else str(raw["config"])),
            "split": str(raw["split"]),
            "revision": str(raw["revision"]),
            "text_field": str(raw["text_field"]),
            "tokens": tokens,
            "skip_docs": int(raw.get("skip_docs", 0) or 0),
            "min_chars": int(raw.get("min_chars", 0) or 0),
        }
        if fmt is not None:          # absent, not None: keeps format-less hashes stable
            resolved["format"] = fmt
        if excl is not None:         # likewise: format-only specs keep their hashes
            resolved["exclude_question_sha256"] = excl
        out.append(resolved)
    return out


def local_jsonl_path(bucket: Mapping[str, Any]) -> str:
    """Absolute path of a ``local_jsonl`` bucket's file (a relative ``path`` is
    under the data root, :func:`srspd_experiments.paths.data_dir`)."""
    from ..paths import resolve_data_ref

    return resolve_data_ref(str(bucket["path"]))


def local_jsonl_sha256(bucket: Mapping[str, Any]) -> str | None:
    """Content sha256 of a ``local_jsonl`` bucket's file, or ``None`` when the
    file is absent (the build then fails on its own, loudly)."""
    import os

    path = local_jsonl_path(bucket)
    if not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_local_jsonl(bucket: Mapping[str, Any]) -> Iterator[tuple[int, Any]]:
    """``(doc_index, document)`` of a ``local_jsonl`` bucket in file order from
    ``skip_docs``; the index counts every line, so ``skip_docs`` past a fit
    build's ``doc_index_end`` is document-disjoint exactly as for a hub stream.
    As for a hub bucket (:func:`iter_documents`), the document is the
    ``text_field`` string unless the bucket names a ``format``, in which case it
    is the whole parsed row (rows whose ``text_field`` is absent are skipped)
    for :func:`render_document`."""
    import os

    path = local_jsonl_path(bucket)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bucket {bucket['name']!r}: local_jsonl file "
                                f"{bucket['path']!r} -> {path} does not exist under the "
                                f"data root; build the corpus first")
    field, min_chars = bucket["text_field"], int(bucket.get("min_chars") or 0)
    formatted = bucket.get("format") is not None
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < int(bucket.get("skip_docs") or 0):
                continue
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if formatted:
                if row.get(field) is None:
                    continue
                yield i, row
                continue
            text = row.get(field)
            if not isinstance(text, str) or len(text) < min_chars:
                continue
            yield i, text


def question_sha256(question) -> str:
    """sha256 of the whitespace-normalised question (``" ".join(q.split())``) -- the key an
    ``exclude_question_sha256`` entry is matched on."""
    return hashlib.sha256(" ".join(str(question).split()).encode("utf-8")).hexdigest()


def iter_documents(bucket: Mapping[str, Any]) -> Iterator[tuple[int, Any]]:
    """Yield ``(doc_index, document)`` in stream order, starting at ``skip_docs``.

    For a bucket without ``format`` the document is the ``text_field`` string
    (rows whose field is not a string, or is shorter than ``min_chars``, are
    skipped). For a bucket with a ``format`` the document is the **whole row**
    (rows whose ``text_field`` is absent are skipped) and
    :func:`render_document` turns it into text — ``min_chars`` is then applied
    to the rendered text by :func:`build_retained_tokens`. A ``local_jsonl``
    bucket is read by :func:`iter_local_jsonl` under the same contract.

    ``datasets`` is imported here rather than at module scope so that importing
    this module stays cheap; the tests monkeypatch this function and never
    reach the import.
    """
    if str(bucket.get("kind") or "hub") == "local_jsonl":
        yield from iter_local_jsonl(bucket)
        return

    from datasets import load_dataset  # noqa: PLC0415  (see docstring)

    try:
        ds = load_dataset(bucket["dataset"], bucket["config"], split=bucket["split"],
                          revision=bucket["revision"], streaming=True)
    except RuntimeError as e:
        # Name the script-based-dataset failure rather than surfacing a bare
        # loader traceback.
        if "scripts are no longer supported" in str(e):
            raise RuntimeError(
                f"bucket {bucket['name']!r}: {bucket['dataset']} is a "
                f"script-based dataset, which this `datasets` refuses outright "
                f"({e}). A repo can carry BOTH a loading script and parquet "
                f"files and still be rejected -- the script's mere presence is "
                f"disqualifying. Pick a script-free repo (check "
                f"HfApi().dataset_info(repo).siblings for a top-level .py), or "
                f"load its parquet export via revision='refs/convert/parquet'."
            ) from e
        raise
    field, min_chars = bucket["text_field"], int(bucket["min_chars"])
    formatted = bucket.get("format") is not None
    for i, row in enumerate(ds):
        if i < int(bucket["skip_docs"]):
            continue
        if formatted:
            if row.get(field) is None:
                continue
            yield i, row
            continue
        text = row.get(field)
        if not isinstance(text, str) or len(text) < min_chars:
            continue
        yield i, text


def _render_mmlu_lm_eval(row: Mapping[str, Any]) -> str:
    """lm-evaluation-harness's ``mmlu`` prompt for one row, answer letter included.

    The harness's ``description`` names the subject with its underscores turned
    into spaces, ``doc_to_text`` lists the four choices and ends in ``Answer:``,
    and the target is the answer letter joined with a single space.
    """
    subject = " ".join(str(row.get("subject") or "").split("_"))
    choices = list(row["choices"])
    if len(choices) != 4:
        raise ValueError(f"mmlu_lm_eval expects 4 choices, got {len(choices)}")
    ans = row["answer"]
    if isinstance(ans, str) and ans.strip().upper() in ("A", "B", "C", "D"):
        idx = "ABCD".index(ans.strip().upper())
    else:
        idx = int(ans)
    if not 0 <= idx < 4:
        raise ValueError(f"mmlu_lm_eval answer index out of range: {ans!r}")
    return (f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            f"{str(row['question']).strip()}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            f"Answer: {'ABCD'[idx]}")


def render_document(tok, bucket: Mapping[str, Any], payload: Any) -> str:
    """The text of one document of ``bucket``: ``payload`` itself for a bucket
    without ``format``, else the row rendered per its ``format``.

    ``mmlu_lm_eval`` → :func:`_render_mmlu_lm_eval` (``text_field`` is
    ``question``; the row carries ``question`` / ``subject`` / ``choices`` /
    ``answer``). ``chat_messages`` → ``tok.apply_chat_template(messages,
    tokenize=False, add_generation_prompt=False)`` over the ``text_field`` list
    (the whole conversation, no dangling assistant turn).
    """
    fmt = bucket.get("format")
    if fmt is None:
        if not isinstance(payload, str):
            raise TypeError(f"bucket {bucket.get('name')!r} has no format; expected a text "
                            f"document, got {type(payload).__name__}")
        return payload
    if fmt == "mmlu_lm_eval":
        return _render_mmlu_lm_eval(payload)
    if fmt == "chat_messages":
        messages = payload[bucket["text_field"]]
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    raise ValueError(f"unknown format {fmt!r}; expected one of {BUCKET_FORMATS}")


def _wrap(tok, text: str, chat_template: str | None) -> str:
    """Apply the requested framing to one document."""
    if chat_template is None:
        return text
    if chat_template != "user":
        raise ValueError(f"unknown chat_template {chat_template!r}; "
                         f"expected one of {CHAT_TEMPLATES}")
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   add_generation_prompt=True, tokenize=False)


def build_retained_tokens(tok, spec: Sequence[Mapping[str, Any]], *,
                          chunk_tokens: int = 1024, seed: int = 0,
                          add_bos: bool = True,
                          chat_template: str | None = None,
                          on_document=None) -> tuple[torch.Tensor, dict]:
    """Stream each bucket to its budget and pack the result into chunks.

    Returns
    -------
    (ids, manifest) : (torch.Tensor, dict)
        ``ids`` is ``[n_chunks, chunk_tokens]`` int32 — the trailing partial
        chunk is dropped so every chunk is the same shape, which is what lets
        the projector's cost be predictable per chunk. ``manifest`` carries the
        per-bucket provenance and ``retained_corpus_sha256``.

    Notes
    -----
    ``seed`` does not select documents — nothing here is random, by design (see
    the module docstring). It is recorded in the manifest and consumed by the
    *sweep* for chunk-order shuffling, and it lives here so that one hash covers
    everything that determines the run.

    ``on_document`` (optional; not part of any hash) is called once per
    document that enters the corpus, in order, as ``on_document(bucket_name,
    doc_index, text, ids)``: ``text`` is the framed rendering that was
    tokenised and ``ids`` exactly the tokens taken from it (the BOS, when one
    was prepended, included; the last document of a bucket cut at the budget).
    It is how a consumer that cannot read token ids (a gradient baseline's
    JSONL loader) is handed the very documents this build selected.
    """
    if int(chunk_tokens) < 1:
        raise ValueError(f"chunk_tokens must be >= 1; got {chunk_tokens}")
    if chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"unknown chat_template {chat_template!r}; "
                         f"expected one of {CHAT_TEMPLATES}")
    if chat_template is not None:
        clash = [b["name"] for b in spec if b.get("format") == "chat_messages"]
        if clash:
            raise ValueError(f"buckets {clash} are already chat-templated by their format; "
                             f"a build-level chat_template would wrap them twice")

    bos = getattr(tok, "bos_token_id", None) if add_bos else None
    flat: list[int] = []
    bucket_reports = []

    for bucket in spec:
        budget = int(bucket["tokens"])
        got, n_docs, first, last = 0, 0, None, None
        doc_hashes = hashlib.sha256()
        formatted = bucket.get("format") is not None
        min_chars = int(bucket.get("min_chars", 0) or 0)
        excluded = frozenset(bucket.get("exclude_question_sha256") or ())
        n_excluded = 0
        n_bos_present = 0

        for doc_index, payload in iter_documents(bucket):
            if got >= budget:
                break
            if excluded and isinstance(payload, Mapping) \
                    and question_sha256(payload.get("question", "")) in excluded:
                n_excluded += 1               # a duplicate of a scored item: never rendered
                continue
            text = render_document(tok, bucket, payload)
            if formatted and len(text) < min_chars:   # raw buckets: filtered in the stream
                continue
            piece = _wrap(tok, text, chat_template)
            ids = tok(piece, add_special_tokens=False, return_tensors=None)["input_ids"]
            if bos is not None:
                if len(ids) and int(ids[0]) == int(bos):
                    # The rendering already starts with BOS (Gemma's chat
                    # template emits a literal ``<bos>``): prepending another
                    # would give every document two. Count it, don't double it.
                    n_bos_present += 1
                else:
                    ids = [int(bos)] + list(ids)
            if not ids:
                continue
            # Truncate the final document at the boundary rather than
            # overshooting: the budget is what a result quotes.
            take = min(len(ids), budget - got)
            flat.extend(int(t) for t in ids[:take])
            if on_document is not None:
                on_document(bucket["name"], doc_index, piece, [int(t) for t in ids[:take]])
            got += take
            n_docs += 1
            first = doc_index if first is None else first
            last = doc_index
            doc_hashes.update(hashlib.sha256(text.encode("utf-8")).digest())

        if got < budget:
            skip = int(bucket.get("skip_docs") or 0)
            if got == 0 and skip > 0:
                # The document-disjoint held-out twin (target.heldout_mix_spec)
                # starts one past the fit build's last document; a fit budget
                # that consumed the WHOLE split leaves it nothing to read.
                raise RuntimeError(
                    f"bucket {bucket['name']!r} has no documents from skip_docs={skip} on: "
                    f"the stream of {bucket['dataset']}@{bucket['revision']} "
                    f"{bucket['split']!r} ends before it. If this is a held-out twin "
                    f"(skip_docs = the fit build's doc_index_end + 1), the fit budget "
                    f"consumed every document of the split -- set heldout_mix_tokens: 0 "
                    f"or lower the fit bucket's tokens so some documents remain")
            raise RuntimeError(
                f"bucket {bucket['name']!r} exhausted its stream at {got} of "
                f"{budget} tokens; raise skip_docs, lower the budget, or check "
                f"that {bucket['dataset']}@{bucket['revision']} still has the "
                f"{bucket['split']!r} split")

        bucket_reports.append({
            **{k: bucket.get(k) for k in ("name", "dataset", "config", "split",
                                          "revision", "text_field", "skip_docs", "min_chars")},
            **({"kind": "local_jsonl", "path": bucket["path"],
                "path_sha256": local_jsonl_sha256(bucket)}
               if str(bucket.get("kind") or "hub") == "local_jsonl" else {}),
            **({"format": bucket["format"]} if formatted else {}),
            **({"exclude_question_sha256": sorted(excluded), "n_excluded": int(n_excluded)}
               if excluded else {}),
            # Only when it happened, so every corpus whose renderings never
            # start with BOS keeps its manifest (and its hash) byte for byte.
            **({"n_docs_bos_already_present": int(n_bos_present)} if n_bos_present else {}),
            "tokens_requested": budget,
            "tokens_taken": int(got),
            "n_docs": int(n_docs),
            "doc_index_start": (None if first is None else int(first)),
            "doc_index_end": (None if last is None else int(last)),
            "docs_sha256": doc_hashes.hexdigest(),
        })

    n_chunks = len(flat) // int(chunk_tokens)
    if n_chunks < 1:
        raise RuntimeError(f"corpus of {len(flat)} tokens is shorter than one "
                           f"chunk of {chunk_tokens}")
    kept = n_chunks * int(chunk_tokens)
    ids = torch.tensor(flat[:kept], dtype=torch.int32).reshape(n_chunks, int(chunk_tokens))

    manifest: dict[str, Any] = {
        "buckets": bucket_reports,
        "chunk_tokens": int(chunk_tokens),
        "n_chunks": int(n_chunks),
        "n_tokens": int(kept),
        "n_tokens_dropped": int(len(flat) - kept),
        "seed": int(seed),
        "add_bos": bool(add_bos),
        "bos_token_id": (None if bos is None else int(bos)),
        "chat_template": chat_template,
        "tokenizer": str(getattr(tok, "name_or_path", "unknown")),
    }
    manifest["retained_corpus_sha256"] = manifest_sha256(manifest)
    return ids, manifest


def has_double_bos(ids, bos_token_id) -> bool:
    """Whether a packed corpus holds two adjacent BOS tokens anywhere.

    Before :func:`build_retained_tokens` learned to skip the prepend when a
    rendering already starts with BOS, a chat bucket on a tokenizer whose
    template emits a literal ``<bos>`` (Gemma) was packed as ``BOS BOS ...``
    per document. The fix changes the tokens but not :func:`spec_key` (the key
    is computed before streaming and cannot see the template's output), so a
    cache written before it would be served unchanged. :func:`load_or_build`
    calls this on every cache hit and rebuilds a corpus that carries the
    defect; a corpus without a BOS id (``add_bos: false``) is never flagged.
    """
    if bos_token_id is None:
        return False
    flat = torch.as_tensor(ids).reshape(-1)
    if flat.numel() < 2:
        return False
    b = int(bos_token_id)
    return bool(((flat[:-1] == b) & (flat[1:] == b)).any())


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Content hash of everything that determines the corpus.

    Computed over a canonical JSON dump with ``retained_corpus_sha256`` itself
    excluded, so the hash of a manifest is stable whether or not it has already
    been stamped with one.
    """
    payload = {k: v for k, v in manifest.items() if k != "retained_corpus_sha256"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def spec_key(tok, spec: Sequence[Mapping[str, Any]], *, chunk_tokens: int = 1024,
             add_bos: bool = True, chat_template: str | None = None,
             **_ignored) -> str:
    """A short hash of everything that determines the corpus, computable
    *before* streaming it.

    ``manifest_sha256`` can only be taken after the corpus exists, which is no
    use for naming the file it should be cached in. This covers the same inputs
    — buckets, budgets, revisions, framing, tokenizer — and is what lets two
    different experiments share one corpus.
    """
    def _bucket_key(b):
        k = {f: b[f] for f in sorted(b)}
        if str(b.get("kind") or "hub") == "local_jsonl":
            k["path_sha256"] = local_jsonl_sha256(b)     # the file IS the provenance
        return k

    payload = {
        "buckets": [_bucket_key(b) for b in spec],
        "chunk_tokens": int(chunk_tokens),
        "add_bos": bool(add_bos),
        "chat_template": chat_template,
        "tokenizer": str(getattr(tok, "name_or_path", "unknown")),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def shared_cache_path(tok, spec: Sequence[Mapping[str, Any]], *, cache_dir=None, **kw) -> str:
    """Where a corpus lives so that every run using it shares one copy:
    ``<cache_dir>/retained_tokens_<spec_key>.pt``.

    Keyed on :func:`spec_key`, not on a run name: two runs that use the same
    spec must consume the *same* retained set, and content addressing makes
    sharing the default and divergence impossible. ``cache_dir`` defaults to
    ``<data root>/datasets/retained_corpus_cache`` (created on first call).
    """
    import os

    if cache_dir is None:
        from ..paths import datasets_dir
        cache_dir = datasets_dir("retained_corpus_cache")
    os.makedirs(str(cache_dir), exist_ok=True)
    return os.path.join(str(cache_dir), f"retained_tokens_{spec_key(tok, spec, **kw)}.pt")


def load_or_build(cache_path, tok, spec: Sequence[Mapping[str, Any]], **kw) -> tuple[torch.Tensor, dict]:
    """Return the cached corpus if its manifest still matches, else build it.

    The cache key is the manifest hash, so it cannot go stale silently: a
    changed budget, revision, tokenizer or framing produces a different hash and
    the corpus is rebuilt. A million tokens is 4 MB as int32, which is why this
    is worth doing at all -- every re-run then needs no network.
    """
    import os

    path = str(cache_path)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
        cached = blob.get("manifest", {})
        probe = dict(cached)
        if manifest_sha256(probe) == cached.get("retained_corpus_sha256"):
            expected = dict(cached)
            # Rebuild the would-be manifest header from the request and compare
            # only the fields the caller controls; the bucket reports are
            # outputs, not inputs, so they cannot be checked without streaming.
            same = (
                int(expected.get("chunk_tokens", -1)) == int(kw.get("chunk_tokens", 1024))
                and expected.get("chat_template") == kw.get("chat_template")
                and bool(expected.get("add_bos", True)) == bool(kw.get("add_bos", True))
                and [b["name"] for b in expected.get("buckets", [])] == [b["name"] for b in spec]
                and [b["revision"] for b in expected.get("buckets", [])] == [b["revision"] for b in spec]
                and [b.get("format") for b in expected.get("buckets", [])] == [b.get("format") for b in spec]
                and [b.get("exclude_question_sha256") for b in expected.get("buckets", [])]
                == [b.get("exclude_question_sha256") for b in spec]
                and [b["tokens_requested"] for b in expected.get("buckets", [])] == [int(b["tokens"]) for b in spec]
            )
            if same and not has_double_bos(blob["ids"], cached.get("bos_token_id")):
                return blob["ids"], cached

    ids, manifest = build_retained_tokens(tok, spec, **kw)
    tmp = path + ".tmp"
    torch.save({"ids": ids, "manifest": manifest}, tmp)
    os.replace(tmp, path)          # atomic: a killed run leaves no half cache
    return ids, manifest
