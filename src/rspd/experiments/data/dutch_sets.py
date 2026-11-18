"""Dutch-unlearning sets for a large instruct model: the target cut to exactly N
tokens of the model's tokenizer, and the retained set rendered to JSONL so the
gradient baselines read the very documents the SRSPD solve reads.

:func:`prepare` (config: ``specs/dutch_sets_1k.yaml``; the tokenizer is a
parameter) writes under ``<data root>/<out>/``:

* ``target_nld_1k.jsonl`` -- ONE row whose ``text`` tokenises, without special
  tokens, to exactly ``target.tokens`` (1,000) tokens: the GPT-2 sets' Dutch
  target article (``target_nld.jsonl``) re-tokenised and cut
  (:func:`cut_text_to_tokens`). The cut is the decode of the first N ids,
  re-encoded to check the round trip; if it does not round-trip (a multi-byte
  character split at the boundary), the longest character prefix that encodes
  to exactly N tokens is taken instead, and the manifest says which.
* ``retained/<bucket>.jsonl`` -- per bucket of the ``retained_corpus`` block
  (the same block the SRSPD data spec carries), the documents
  :func:`retained_corpus.build_retained_tokens` selected, in order, as text:
  exactly the framed rendering it tokenised (a ``chat_messages`` bucket already
  through the chat template) with the bucket's last document cut at the
  budget. Where the build prepended a BOS, the row's text starts with the
  tokenizer's literal BOS string, so the row tokenised with
  ``add_special_tokens=False`` (the baselines' packed retained stream,
  :func:`budget.packed_retain_stream`) yields the very ids the SRSPD solve
  reads; with special tokens on, :func:`corpora.strip_redundant_bos` keeps it
  at one BOS.
* ``retained_probe/<bucket>.jsonl`` -- the first ``probe_fraction`` of every
  bucket's JSONL tokens (whole documents, in order), for short probe runs that
  must carry the same retained load per step as a full run.
* ``manifest.json`` -- tokenizer, target cut, per bucket the SRSPD-side
  ``tokens_taken`` and the JSONL-side ``jsonl_doc_tokens`` (their sum is a
  baseline's ``budget.retain_exposure_tokens``), round-trip mismatches, BOS
  counts, the held-out files (pointed at: path, sha256, rows) and the retained
  corpus manifest (``retained_corpus_sha256``). ``results.md`` goes to
  ``<data root>/logs/data/<name>/``.

The retained token cache is also written to the content-addressed shared path
(:func:`retained_corpus.shared_cache_path`), so an SRSPD run with the same
block and tokenizer reads it instead of re-streaming.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

from ..paths import data_dir, data_ref, hf_token, logs_dir, resolve_data_ref
from . import retained_corpus as rcorp

CUT_METHODS = ("decode", "char_prefix")


# --------------------------------------------------------------------------- #
# Pure functions                                                               #
# --------------------------------------------------------------------------- #
def encode_plain(tok, text: str) -> list:
    """``text``'s ids without special tokens (no BOS)."""
    return [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]


def cut_text_to_tokens(tok, text: str, n_tokens: int) -> dict:
    """A prefix of ``text`` that tokenises, without special tokens, to EXACTLY
    ``n_tokens`` tokens.

    First the decode of the first ``n_tokens`` ids of ``text``; it is accepted
    when it re-encodes to those very ids (``round_trip: True``, method
    ``decode``). Otherwise (a byte-fallback character split at the boundary, or
    a merge across it) the longest character prefix of ``text`` whose encoding
    is exactly ``n_tokens`` long is taken (method ``char_prefix``;
    ``round_trip`` then says whether its ids equal the first ``n_tokens`` of
    the article's). Raises ``ValueError`` when the text is shorter than
    ``n_tokens`` tokens or no prefix encodes to exactly that many.
    """
    n = int(n_tokens)
    if n < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens!r}")
    ids = encode_plain(tok, text)
    if len(ids) < n:
        raise ValueError(f"the text tokenises to {len(ids)} tokens, fewer than {n}")
    head = ids[:n]
    dec = tok.decode(head, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if encode_plain(tok, dec) == head:
        return {"text": dec, "ids": head, "n_tokens": n, "method": "decode",
                "round_trip": True, "n_source_tokens": len(ids)}
    # Longest character prefix that encodes to exactly n tokens. Token count is
    # monotone non-decreasing in the prefix length for every tokenizer we use,
    # so a binary search for the last prefix with <= n tokens, then a check.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(encode_plain(tok, text[:mid])) <= n:
            lo = mid
        else:
            hi = mid - 1
    cand = text[:lo].rstrip() if len(encode_plain(tok, text[:lo].rstrip())) == n else text[:lo]
    cids = encode_plain(tok, cand)
    if len(cids) != n:
        raise ValueError(f"no character prefix of the text encodes to exactly {n} tokens "
                         f"(the longest with <= {n} has {len(cids)})")
    return {"text": cand, "ids": cids, "n_tokens": n, "method": "char_prefix",
            "round_trip": cids == head, "n_source_tokens": len(ids)}


def document_text(tok, rendering: str, ids: list, *, bos_token_id, bos_token,
                  full: bool) -> str:
    """The JSONL text of one retained document: the rendering the build
    tokenised (``full``) or the decode of the ids it took (the bucket's last,
    budget-cut document), with the tokenizer's literal BOS string in front when
    the build prepended a BOS the text does not already carry -- so that
    ``tok(text, add_special_tokens=False)`` reproduces ``ids``."""
    has_bos = bool(ids) and bos_token_id is not None and int(ids[0]) == int(bos_token_id)
    if full:
        text = rendering
    else:
        body = ids[1:] if has_bos and not rendering.startswith(bos_token or "\0") else ids
        text = tok.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if has_bos and bos_token and not text.startswith(bos_token):
        text = bos_token + text
    return text


def probe_subset(rows: list, n_tokens: list, fraction: float) -> int:
    """How many leading rows make up the first ``fraction`` of the tokens:
    whole rows, in order, the first one that reaches the share included (at
    least one row)."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError(f"probe_fraction must be in (0, 1], got {fraction!r}")
    want = float(fraction) * sum(n_tokens)
    got, k = 0, 0
    for k, n in enumerate(n_tokens, start=1):
        got += n
        if got >= want:
            break
    return max(1, k) if rows else 0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_jsonl(path: str, rows) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #
def resolve_config(cfg, *, tokenizer: str | None = None) -> dict:
    """Validate the config; returns it with defaults filled in. ``tokenizer``
    (a hub id or path) overrides / supplies the config's ``tokenizer:``."""
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    if tokenizer is not None:
        cfg = {**cfg, "tokenizer": tokenizer}
    for key in ("name", "tokenizer", "out", "target", "retained_corpus"):
        if key not in cfg:
            raise ValueError(f"config needs `{key}:`")
    t = cfg["target"]
    for key in ("source", "tokens"):
        if key not in t:
            raise ValueError(f"target needs `{key}:`")
    n = t["tokens"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"target.tokens must be a positive int, got {n!r}")
    rc = cfg["retained_corpus"]
    spec = rcorp.resolve_spec(rc.get("buckets") or [])
    build_kw = dict(chunk_tokens=int(rc.get("chunk_tokens", 1024)), seed=int(rc.get("seed", 0)),
                    add_bos=bool(rc.get("add_bos", True)),
                    chat_template=rc.get("chat_template"))
    frac = float(cfg.get("probe_fraction", 0.1))
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"probe_fraction must be in (0, 1], got {frac!r}")
    return {
        "name": str(cfg["name"]), "tokenizer": str(cfg["tokenizer"]), "out": str(cfg["out"]),
        "target": {"source": str(t["source"]), "text_field": str(t.get("text_field", "text")),
                   "tokens": int(n), "out_file": str(t.get("out_file", "target_nld_1k.jsonl"))},
        "heldout": [str(p) for p in (cfg.get("heldout") or [])],
        "spec": spec, "build_kw": build_kw, "probe_fraction": frac,
        "compare_tokenizers": [str(x) for x in (cfg.get("compare_tokenizers") or [])],
        "write_shared_cache": bool(cfg.get("write_shared_cache", True)),
    }


# --------------------------------------------------------------------------- #
# The stages                                                                   #
# --------------------------------------------------------------------------- #
def prepare_target(tok, R, out_dir: str) -> dict:
    src = resolve_data_ref(R["target"]["source"])
    rows = read_jsonl(src)
    field = R["target"]["text_field"]
    if not rows:
        raise ValueError(f"target source {src} has no rows")
    text = "\n\n".join(r[field] for r in rows)
    cut = cut_text_to_tokens(tok, text, R["target"]["tokens"])
    row = {k: v for k, v in rows[0].items() if k != field and len(rows) == 1}
    row.update({field: cut["text"], "n_tokens": cut["n_tokens"],
                "tokenizer": R["tokenizer"], "source": R["target"]["source"]})
    path = os.path.join(out_dir, R["target"]["out_file"])
    write_jsonl(path, [row])
    # re-verify from disk: what a driver will read
    back = read_jsonl(path)[0][field]
    n_back = len(encode_plain(tok, back))
    if n_back != R["target"]["tokens"]:
        raise RuntimeError(f"{path}: re-read target tokenises to {n_back} tokens, "
                           f"not {R['target']['tokens']}")
    n_special = len(tok(back)["input_ids"])
    print(f"[target] {R['target']['source']} ({len(rows)} row(s), {cut['n_source_tokens']} "
          f"tokens) -> {path}: {n_back} tokens without special tokens, {n_special} with "
          f"(method {cut['method']}, round trip {cut['round_trip']}, "
          f"{len(back)} chars of {len(text)})", flush=True)
    return {"path": data_ref(path), "sha256": sha256_file(path),
            "source": R["target"]["source"], "source_sha256": sha256_file(src),
            "source_rows": len(rows), "source_tokens": cut["n_source_tokens"],
            "tokens": n_back, "tokens_with_special": n_special, "method": cut["method"],
            "round_trip": cut["round_trip"], "chars": len(back), "source_chars": len(text)}


def prepare_retained(tok, R, out_dir: str, *, cache_dir=None) -> dict:
    docs = {b["name"]: [] for b in R["spec"]}

    def on_document(bucket, doc_index, piece, ids):
        docs[bucket].append((doc_index, piece, ids))

    t0 = time.time()
    ids, manifest = rcorp.build_retained_tokens(tok, R["spec"], on_document=on_document,
                                                **R["build_kw"])
    print(f"[retained] built {manifest['n_tokens']} tokens ({manifest['n_chunks']} chunks) in "
          f"{time.time() - t0:.0f}s; sha256 {manifest['retained_corpus_sha256'][:16]}",
          flush=True)
    cache = None
    if R["write_shared_cache"]:
        import torch
        cache = rcorp.shared_cache_path(tok, R["spec"], cache_dir=cache_dir, **R["build_kw"])
        tmp = cache + ".tmp"
        torch.save({"ids": ids, "manifest": manifest}, tmp)
        os.replace(tmp, cache)
        print(f"[retained] shared token cache -> {cache}", flush=True)

    bos_id = getattr(tok, "bos_token_id", None)
    bos = getattr(tok, "bos_token", None)
    buckets, files, probe_files = [], [], []
    for b, rep in zip(R["spec"], manifest["buckets"]):
        rows, counts, mismatch = [], [], 0
        bdocs = docs[b["name"]]
        for k, (doc_index, piece, take) in enumerate(bdocs):
            full_ids = None
            last = k == len(bdocs) - 1
            # a document is "full" unless it is the bucket's last and was cut
            full = True
            if last:
                full_ids = encode_plain(tok, piece)
                if (bos_id is not None and take and int(take[0]) == int(bos_id)
                        and not (full_ids and int(full_ids[0]) == int(bos_id))):
                    full_ids = [int(bos_id)] + full_ids
                full = len(full_ids) == len(take)
            text = document_text(tok, piece, take, bos_token_id=bos_id, bos_token=bos, full=full)
            got = encode_plain(tok, text)
            mismatch += int(got != list(take))
            rows.append({"text": text, "bucket": b["name"], "doc_index": int(doc_index),
                         "n_tokens": len(got), **({"truncated": True} if not full else {})})
            counts.append(len(got))
        path = os.path.join(out_dir, "retained", f"{b['name']}.jsonl")
        write_jsonl(path, rows)
        k = probe_subset(rows, counts, R["probe_fraction"])
        ppath = os.path.join(out_dir, "retained_probe", f"{b['name']}.jsonl")
        write_jsonl(ppath, rows[:k])
        starts_bos = [r["text"].startswith(bos) for r in rows] if bos else []
        rel = data_ref(path)
        prel = data_ref(ppath)
        files.append(rel)
        probe_files.append(prel)
        buckets.append({
            "name": b["name"], "format": b.get("format"), "path": rel, "sha256": sha256_file(path),
            "n_docs": len(rows), "tokens_taken": int(rep["tokens_taken"]),
            "jsonl_doc_tokens": int(sum(counts)), "n_docs_token_mismatch": int(mismatch),
            "n_docs_start_with_bos": int(sum(starts_bos)),
            "n_docs_bos_already_present": int(rep.get("n_docs_bos_already_present", 0)),
            "probe": {"path": prel, "sha256": sha256_file(ppath), "n_docs": k,
                      "jsonl_doc_tokens": int(sum(counts[:k]))},
        })
        print(f"[retained] {b['name']}: {len(rows)} docs, {sum(counts)} JSONL tokens "
              f"(build took {rep['tokens_taken']}), {mismatch} round-trip mismatch(es), "
              f"{sum(starts_bos)} start with BOS -> {path}; probe {k} docs, "
              f"{sum(counts[:k])} tokens", flush=True)
    total = sum(x["jsonl_doc_tokens"] for x in buckets)
    ptotal = sum(x["probe"]["jsonl_doc_tokens"] for x in buckets)
    return {"buckets": buckets, "files": files, "probe_files": probe_files,
            "jsonl_doc_tokens_total": int(total), "probe_jsonl_doc_tokens_total": int(ptotal),
            "solver_tokens": int(manifest["n_tokens"]),
            "solver_tokens_dropped": int(manifest["n_tokens_dropped"]),
            "shared_cache": cache, "corpus_manifest": manifest}


def describe_heldout(tok, R) -> list:
    out = []
    for rel in R["heldout"]:
        p = resolve_data_ref(rel)
        rows = read_jsonl(p)
        n = sum(len(encode_plain(tok, r.get("text", ""))) for r in rows)
        out.append({"path": rel, "sha256": sha256_file(p), "rows": len(rows), "tokens": n})
        print(f"[heldout] {rel}: {len(rows)} rows, {n} tokens (pointed at, not copied)",
              flush=True)
    return out


def compare_tokenizers(tok, names, target_text: str, retained_files: list) -> dict:
    """Whether each named tokenizer gives the same ids as ``tok`` on the target
    and on the first row of every retained file (a second model of the same
    family may reuse JSONL tokenised with the first's tokenizer)."""
    from transformers import AutoTokenizer
    res = {}
    samples = [target_text] + [read_jsonl(resolve_data_ref(f))[0]["text"]
                               for f in retained_files]
    for name in names:
        other = AutoTokenizer.from_pretrained(name, token=hf_token())
        same = [encode_plain(tok, s) == encode_plain(other, s) for s in samples]
        res[name] = {"identical_on_samples": all(same), "n_samples": len(same),
                     "bos_token_id": getattr(other, "bos_token_id", None)}
        print(f"[tokenizer] {name}: identical ids on {sum(same)}/{len(same)} samples", flush=True)
    return res


def results_md(m: dict) -> str:
    t, r = m["target"], m["retained"]
    lines = [f"# {m['name']}", "",
             f"Tokenizer `{m['tokenizer']}`.", "",
             f"Target `{t['path']}`: **{t['tokens']}** tokens without special tokens "
             f"({t['tokens_with_special']} with), method `{t['method']}`, round trip "
             f"{t['round_trip']}, {t['chars']} of {t['source_chars']} characters of "
             f"`{t['source']}` ({t['source_tokens']} tokens).", "",
             "| bucket | docs | build tokens | JSONL tokens | mismatches | start with BOS | "
             "probe docs | probe tokens |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for b in r["buckets"]:
        lines.append(f"| {b['name']} | {b['n_docs']} | {b['tokens_taken']} | "
                     f"{b['jsonl_doc_tokens']} | {b['n_docs_token_mismatch']} | "
                     f"{b['n_docs_start_with_bos']} | {b['probe']['n_docs']} | "
                     f"{b['probe']['jsonl_doc_tokens']} |")
    lines += ["", f"Retained JSONL total **{r['jsonl_doc_tokens_total']}** tokens "
              f"(`budget.retain_exposure_tokens` of a full baseline run); probe subset "
              f"**{r['probe_jsonl_doc_tokens_total']}** (a short probe run). The SRSPD "
              f"solve reads {r['solver_tokens']} ({r['solver_tokens_dropped']} dropped with "
              f"the trailing partial chunk); `retained_corpus_sha256` "
              f"`{r['corpus_manifest']['retained_corpus_sha256']}`.", ""]
    return "\n".join(lines)


def prepare(cfg: dict, *, tokenizer=None, tokenizer_id: str | None = None, cache_dir=None,
            log_dir=None) -> dict:
    """Run every stage of ``cfg`` and return the manifest. ``tokenizer`` (an
    object) or ``tokenizer_id`` (loaded with ``AutoTokenizer``) supplies the
    tokenizer when the config names none; ``cache_dir`` is the retained-corpus
    cache directory (default under the data root)."""
    R = resolve_config(cfg, tokenizer=tokenizer_id or (None if tokenizer is None else
                                                       str(getattr(tokenizer, "name_or_path",
                                                                   "tokenizer"))))
    out_dir = os.path.join(str(data_dir()), R["out"])
    os.makedirs(out_dir, exist_ok=True)
    log_dir = str(log_dir or logs_dir("data", R["name"]))
    os.makedirs(log_dir, exist_ok=True)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(R["tokenizer"], token=hf_token())
    tok = tokenizer
    print(f"[plan] {R['name']}: tokenizer {R['tokenizer']} (bos {tok.bos_token!r} = "
          f"{tok.bos_token_id}, eos {tok.eos_token!r} = {tok.eos_token_id}) -> {out_dir}",
          flush=True)
    target = prepare_target(tok, R, out_dir)
    retained = prepare_retained(tok, R, out_dir, cache_dir=cache_dir)
    heldout = describe_heldout(tok, R)
    cmp = {}
    if R["compare_tokenizers"]:
        cmp = compare_tokenizers(
            tok, R["compare_tokenizers"],
            read_jsonl(os.path.join(out_dir, R["target"]["out_file"]))[0]["text"],
            retained["files"])
    manifest = {"name": R["name"], "tokenizer": R["tokenizer"],
                "bos_token": tok.bos_token, "bos_token_id": tok.bos_token_id,
                "eos_token": tok.eos_token, "eos_token_id": tok.eos_token_id,
                "probe_fraction": R["probe_fraction"], "target": target, "retained": retained,
                "heldout": heldout, "compare_tokenizers": cmp}
    for d in (out_dir, log_dir):
        with open(os.path.join(d, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2, default=str)
    with open(os.path.join(log_dir, "results.md"), "w") as f:
        f.write(results_md(manifest))
    print(f"[done] manifest -> {os.path.join(out_dir, 'manifest.json')}; retained JSONL "
          f"{retained['jsonl_doc_tokens_total']} tokens, probe "
          f"{retained['probe_jsonl_doc_tokens_total']}", flush=True)
    return manifest
