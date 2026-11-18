"""The SRSPD data path: the target ``X`` (rows or official windows of
registered corpora), the retained ``Y`` (a :mod:`retained_corpus` build), and
the held-out readouts, as token ids ready for :func:`rspd.api.erase`.

:func:`build_erasure_data` is the one entry point; it reproduces, step for
step, the data path every published SRSPD run used:

1. the target corpora are loaded (:func:`load_forget_corpora`): per row corpus
   the first ``max_rows`` rows (``len(text) > min_len``) are fit rows and the
   next ``heldout_rows`` the forget-side readout (or, with ``heldout_from``,
   the first ``heldout_rows`` rows of those corpora); a chunked corpus gives
   every official window but the last ``heldout_rows``;
2. the retain-side readout corpora give their last ``heldout_rows`` rows
   (:func:`load_heldout_retain`), encoded to ``retain_max_length`` tokens;
3. ``Y`` is the retained build, read from the content-addressed cache or built
   (:func:`retained_corpus.load_or_build`), and its document-disjoint held-out
   twin (:func:`heldout_mix_spec`) of ``heldout_mix_tokens`` per bucket gives
   the ``mix/<bucket>`` readout, whole chunks only (:func:`bucket_chunks`);
4. the fit rows are encoded to ``max_length`` tokens with the tokenizer's own
   special tokens (:func:`corpora.encode_all`, rows under 2 tokens dropped) and,
   with ``max_tokens``, cut to exactly that many tokens in order
   (:func:`cut_to_token_budget`).

Readout keys are ``forget/<corpus>``, ``retain/<corpus>`` and ``mix/<bucket>``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from . import corpora as C
from . import retained_corpus as rcorp
from .corpora import split_fit_heldout
from .spec import DataSpec, unique_in_order

#: Separator of :func:`texts_sha256` (ASCII record separator).
TEXT_SEP = "\x1e"


def split_windows(windows, heldout_rows):
    """``(fit, heldout)`` of a chunked corpus's official windows: the LAST
    ``heldout_rows`` windows are the held-out readout and every other window is
    the fit set (``X``). Raises when nothing would be left to fit."""
    windows = list(windows)
    if len(windows) <= heldout_rows:
        raise RuntimeError(f"a chunked corpus has {len(windows)} windows but heldout_rows is "
                           f"{heldout_rows}: nothing would be left to fit")
    if heldout_rows == 0:
        return windows, []
    return windows[:-heldout_rows], windows[-heldout_rows:]


def load_forget_corpora(names, *, min_len, max_rows, heldout_rows,
                        heldout_from=None, tokenizer=None, max_length=None) -> dict:
    """``{corpus: {"fit": [...], "heldout": [...], "windowed": bool}}`` via the
    registry loader :func:`corpora.load_corpus_texts` (sequential, unshuffled) --
    the same loader the gradient baselines use.

    ``max_rows`` ``None`` reads every row as fit rows (a fixed-size local target
    set under ``max_forget_tokens``). ``heldout_from`` -- a list of corpora --
    replaces "the rows after the fit rows" as the forget-side held-out readout:
    the first ``heldout_rows`` rows of each, keyed under their own names.

    A ``chunk: True`` corpus (MUSE) is read as its official windows instead
    (:func:`corpora.load_corpus_windows` -- token-id lists, BOS first, exactly
    ``max_length`` long, every row of the split, no ``min_len``): EVERY window
    but the last ``heldout_rows`` is a fit "text" (:func:`corpora.encode` passes
    the ids through untouched), the last ``heldout_rows`` are the held-out
    readout. ``tokenizer`` is required for it."""
    out = {}
    for c in names:
        if C.is_chunked_corpus(c):
            windows = C.load_corpus_windows(c, tokenizer, int(max_length))
            fit, held = split_windows(windows, heldout_rows)
            out[c] = {"fit": fit, "heldout": held, "windowed": True}
            print(f"[data] forget {c}: {len(windows)} official windows of {max_length} -> "
                  f"{len(fit)} fit windows (X), {len(held)} held-out windows", flush=True)
            continue
        if heldout_from is not None:
            texts = C.load_corpus_texts(c, min_len=min_len, max_rows=max_rows)
            fit, held = texts, []
        else:
            texts = C.load_corpus_texts(
                c, min_len=min_len,
                max_rows=(None if max_rows is None else max_rows + heldout_rows))
            fit, held = split_fit_heldout(texts, max_rows, heldout_rows)
        if not fit:
            raise RuntimeError(f"corpus {c!r} yielded no fit rows after the min_len filter")
        out[c] = {"fit": fit, "heldout": held, "windowed": False}
        print(f"[data] forget {c}: {len(fit)} fit rows, {len(held)} held-out rows", flush=True)
    for c in (heldout_from or []):
        held = C.load_corpus_texts(c, min_len=min_len, max_rows=heldout_rows)
        out[c] = {"fit": [], "heldout": held, "windowed": False}
        print(f"[data] held-out forget {c}: {len(held)} rows (readout only)", flush=True)
    return out


def cut_to_token_budget(id_list, budget):
    """The encoded texts in order, cut so the total is EXACTLY ``budget``
    tokens: the one that crosses the budget is truncated, the rest dropped. A
    truncated tail with a single token (no next-token target) is dropped too,
    and the total is then ``budget - 1``. Returns ``(ids, n_tokens)``; a corpus
    shorter than the budget comes back whole (the caller warns)."""
    out, got = [], 0
    for ids in id_list:
        if got >= budget:
            break
        take = min(int(ids.shape[-1]), budget - got)
        if take < 2:
            break
        out.append(ids[:, :take])
        got += take
    return out, got


def load_heldout_retain(names, *, min_len, heldout_rows, tokenizer=None,
                        max_length=None) -> dict:
    """``{corpus: [the last heldout_rows rows]}`` -- the retain-side readout
    (the rows a hub corpus holds out of a fit); nothing here enters ``X``. A
    ``chunk: True`` corpus (MUSE retain2) gives its last ``heldout_rows``
    official windows of ``max_length`` (``retain_max_length``) instead."""
    out = {}
    for c in names:
        if C.is_chunked_corpus(c):
            windows = C.load_corpus_windows(c, tokenizer, int(max_length))
            _fit, held = split_windows(windows, heldout_rows)
            out[c] = held
            print(f"[data] held-out retain {c}: last {len(held)} of {len(windows)} official "
                  f"windows of {max_length} (readout only)", flush=True)
            continue
        texts = C.load_corpus_texts(c, min_len=min_len, max_rows=None)
        _fit, held = split_fit_heldout(texts, None, heldout_rows)
        out[c] = held
        print(f"[data] held-out retain {c}: {len(held)} rows (readout only)", flush=True)
    return out


def bucket_chunks(manifest) -> dict:
    """``{bucket: [chunk indices lying ENTIRELY inside that bucket's tokens]}`` of a
    :func:`retained_corpus.build_retained_tokens` build. Buckets are packed
    back to back in spec order, so bucket ``i`` owns tokens
    ``[sum(tokens_taken[:i]), sum(tokens_taken[:i+1]))``; a chunk straddling a
    boundary belongs to neither (it is still in ``X``)."""
    ct = int(manifest["chunk_tokens"])
    n_chunks = int(manifest["n_chunks"])
    out, lo = {}, 0
    for b in manifest["buckets"]:
        hi = lo + int(b["tokens_taken"])
        first = -(-lo // ct)                     # ceil
        last = min(hi // ct, n_chunks)           # exclusive
        out[b["name"]] = list(range(first, max(first, last)))
        lo = hi
    return out


def round_robin_chunks(by_bucket, n_tokens, chunk_tokens) -> list:
    """Chunk indices for ``n_tokens`` tokens drawn evenly across buckets: whole
    chunks, one per bucket in turn (bucket order), until the budget is met or
    every bucket is used up."""
    want = -(-int(n_tokens) // int(chunk_tokens))
    lists = [list(v) for v in by_bucket.values()]
    out, j = [], 0
    while len(out) < want and any(j < len(v) for v in lists):
        for v in lists:
            if j < len(v) and len(out) < want:
                out.append(v[j])
        j += 1
    return out


def heldout_mix_spec(spec, manifest, tokens) -> list:
    """The held-out twin of a retained-mix spec: the same buckets (dataset,
    revision, split, field, ``min_chars``) with ``skip_docs`` one past the last
    document the fit build consumed (``doc_index_end`` in its manifest) and a
    budget of ``tokens`` each. The fit build truncates its last document at the
    budget and never reads further, so the two builds share no document and no
    token. ``tokens`` a multiple of ``chunk_tokens`` keeps every held-out chunk
    inside one bucket."""
    reports = {b["name"]: b for b in manifest["buckets"]}
    out = []
    for b in spec:
        end = reports[b["name"]]["doc_index_end"]
        skip = int(b.get("skip_docs", 0)) if end is None else int(end) + 1
        out.append(dict(b, tokens=int(tokens), skip_docs=skip))
    return out


def texts_sha256(texts) -> str:
    """sha256 of the fit set: texts joined by ``TEXT_SEP``; a pre-tokenised
    window (a list of ints) enters as its space-joined ids under an ``<ids>``
    prefix, so a windowed and a text target never collide."""
    parts = [t if isinstance(t, str) else "<ids>" + " ".join(str(int(i)) for i in t)
             for t in texts]
    return hashlib.sha256(TEXT_SEP.join(parts).encode("utf-8")).hexdigest()


@dataclass
class ErasureData:
    """The output of :func:`build_erasure_data`.

    ``target_ids`` -- list of ``[1, T]`` LongTensors (``X``, one sequence each);
    ``retained_ids`` -- ``[n_chunks, chunk_tokens]`` int32 (``Y``; pass
    ``list(retained_ids.long())`` or the tensor itself to :func:`rspd.api.erase`,
    which runs one forward per row); ``heldout_ids`` -- ``{readout key: [ids,
    ...]}`` (empty readouts dropped); ``manifest`` -- the provenance record
    (target row counts and sha256, token counts, the retained manifest and its
    ``retained_corpus_sha256``, the held-out twin's manifest)."""
    target_ids: list
    retained_ids: object
    heldout_ids: dict
    manifest: dict = field(default_factory=dict)

    @property
    def n_target_tokens(self) -> int:
        return sum(int(x.shape[-1]) for x in self.target_ids)

    @property
    def n_retained_tokens(self) -> int:
        return int(self.retained_ids.numel())


def build_erasure_data(tokenizer, spec: DataSpec, *, cache_dir=None, log=print) -> ErasureData:
    """Build ``X``, ``Y`` and the readouts of ``spec`` with ``tokenizer`` (see
    the module docstring). ``cache_dir`` is the retained-corpus cache directory
    (default ``<data root>/datasets/retained_corpus_cache``)."""
    import torch

    t, r = spec.target, spec.retained
    names = unique_in_order(t.corpora)
    forget = load_forget_corpora(names, min_len=t.min_len, max_rows=t.max_rows,
                                 heldout_rows=t.heldout_rows,
                                 heldout_from=(None if t.heldout_from is None
                                               else unique_in_order(t.heldout_from)),
                                 tokenizer=tokenizer, max_length=t.max_length)
    held_retain = load_heldout_retain(unique_in_order(spec.heldout_retain_corpora),
                                      min_len=t.min_len, heldout_rows=t.heldout_rows,
                                      tokenizer=tokenizer, max_length=spec.retain_max_length)
    forget_fit = [x for c in forget.values() for x in c["fit"]]

    rspec, build_kw = r.spec, r.build_kw
    ct = build_kw["chunk_tokens"]
    corpus_ids, manifest = rcorp.load_or_build(
        rcorp.shared_cache_path(tokenizer, rspec, cache_dir=cache_dir, **build_kw),
        tokenizer, rspec, **build_kw)
    held_mix, held_manifest = {}, None
    if spec.heldout_mix_tokens:
        hspec = heldout_mix_spec(rspec, manifest, spec.heldout_mix_tokens)
        hids, held_manifest = rcorp.load_or_build(
            rcorp.shared_cache_path(tokenizer, hspec, cache_dir=cache_dir, **build_kw),
            tokenizer, hspec, **build_kw)
        for bname, idx in bucket_chunks(held_manifest).items():
            held_mix[bname] = [hids[j:j + 1].to(torch.long) for j in idx]
        for b, hb in zip(manifest["buckets"], held_manifest["buckets"]):
            if not (b["doc_index_end"] is None or hb["doc_index_start"] > b["doc_index_end"]):
                raise RuntimeError(f"held-out bucket {b['name']} overlaps the fit documents")

    target_ids = C.encode_all(tokenizer, forget_fit, t.max_length)
    n_cut = None
    if t.max_tokens:
        target_ids, n_cut = cut_to_token_budget(target_ids, t.max_tokens)
        if n_cut < t.max_tokens:
            log(f"[data] WARNING: the target rows hold {n_cut} tokens, short of "
                f"max_tokens {t.max_tokens}; X is the whole set")
    heldout_ids = {f"forget/{c}": C.encode_all(tokenizer, v["heldout"], t.max_length)
                   for c, v in forget.items()}
    heldout_ids.update({f"retain/{c}": C.encode_all(tokenizer, v, spec.retain_max_length)
                        for c, v in held_retain.items()})
    heldout_ids.update({f"mix/{b}": v for b, v in held_mix.items()})
    heldout_ids = {k: v for k, v in heldout_ids.items() if v}
    n_x = sum(int(x.shape[-1]) for x in target_ids)
    rec = {
        "target": {"corpora": names, "windowed": t.windowed,
                   "n_rows": {c: len(v["fit"]) for c, v in forget.items()},
                   "n_heldout_rows": {c: len(v["heldout"]) for c, v in forget.items()},
                   "fit_sha256": texts_sha256(forget_fit), "n_sequences": len(target_ids),
                   "n_tokens": n_x, "max_tokens": t.max_tokens},
        "retained": {"n_chunks": int(corpus_ids.shape[0]), "n_tokens": int(corpus_ids.numel()),
                     "chunk_tokens": ct, "manifest": manifest},
        "heldout_mix_manifest": held_manifest,
        "heldout": {k: {"n_rows": len(v), "n_tokens": sum(int(x.shape[-1]) for x in v)}
                    for k, v in heldout_ids.items()},
        "spec": spec.as_dict(),
    }
    log(f"[data] X {n_x:,} tokens in {len(target_ids)} sequences; Y {int(corpus_ids.numel()):,} "
        f"tokens in {int(corpus_ids.shape[0])} chunks of {ct} "
        f"(sha256 {manifest['retained_corpus_sha256'][:12]}); readouts {sorted(heldout_ids)}")
    return ErasureData(target_ids=target_ids, retained_ids=corpus_ids, heldout_ids=heldout_ids,
                       manifest=rec)
