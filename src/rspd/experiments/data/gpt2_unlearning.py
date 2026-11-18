"""The GPT-2 unlearning sets: sanitised Wikipedia text, in trilingual-BPE tokens.

The sets (budgets of the published 1M build, ``specs/gpt2_unlearning_1m.yaml``):

* ``target_nld`` -- **1,000 tokens** of Dutch: whole sanitised Dutch articles
  taken until the budget is met, the last one truncated at exactly 1,000;
* ``contrastive_eng`` / ``contrastive_deu`` -- **500,000 tokens** each (1M
  retained in total), which INCLUDE the English and German counterparts of
  every target article (found through the Wikipedia ``langlinks`` API and
  fetched as plain text with the ``extracts`` API), placed FIRST and flagged
  ``is_target_counterpart``, the rest filled with sampled sanitised articles
  from the pinned hub dump;
* ``heldout_{eng,deu,nld}`` -- **200,000 tokens** each of further articles,
  disjoint by page id from everything above.

Every budget is measured in the trilingual BPE's tokens
(:mod:`rspd.experiments.data.gpt2_corpus`), as the model sees them.

Getting the sets: download, don't rebuild
-----------------------------------------
The published JSONL files are the reference: :func:`ensure_unlearning_sets`
returns a local directory holding them, fetching the published copy when it is
missing (``source=`` a HF dataset repo id or a local directory), and
:func:`materialize_shards` writes the uint16 shards the recovery loop reads
from the JSONL (deterministic given the tokenizer). **Rebuilding** with
:func:`prepare_unlearning_sets` is supported but **time-dependent**: the
articles come from ``wikimedia/wikipedia`` at a pinned revision, subsampled
deterministically per page id (:func:`accept_page`), but the counterparts come
from the *live* Wikipedia API, so a rebuild today can select different
counterpart texts (and therefore different contrastive fills) than the
published build. The manifest lists every counterpart with its page id and the
build date; network calls retry with backoff and fail loudly.

Sanitisation (:func:`sanitize_wiki_text`)
-----------------------------------------
``<ref>`` blocks and HTML tags, ``{{templates}}``, ``[[links]]`` (keeping the
link text, dropping File/Category links), wiki bold / italic quote marks,
``[http://... text]`` external links; section headings are dropped, and
everything from the first reference-style heading (References / Referenties /
Einzelnachweise / Literatur / Weblinks / Externe links / See also / Zie ook /
Siehe auch / Bronnen) onward is cut; whitespace is collapsed; paragraphs under
``min_paragraph_chars`` (200) and documents under ``min_doc_chars`` (1,000)
after cleaning are dropped.

Layout (under ``<data root>/datasets/gpt2_unlearning/<name>/``)
--------------------------------------------------------------
``<role>_<lang>.jsonl`` (one document per line: id, title, lang, role, text,
is_target_counterpart, n_tokens), ``shards/<role>_<lang>/shard_0000.bin``
(uint16: the documents' tokens with one ``<|endoftext|>`` after each),
``manifest.json``. ``n_tokens`` counts the documents' own tokens (the budgets);
``n_shard_tokens`` adds the per-document separators.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Callable, Iterable, Iterator, Mapping

from . import gpt2_corpus as pc     # the pretraining corpus's shard writer
from .manifest import manifest_hash

#: Roles and their language, in the order the sets are built.
ROLES = ("target_nld", "contrastive_eng", "contrastive_deu",
         "heldout_eng", "heldout_deu", "heldout_nld")
LANG_OF_ROLE = {r: r.rsplit("_", 1)[1] for r in ROLES}
#: Headings (case-insensitive) from which on an article is cut.
STOP_HEADINGS = ("references", "referenties", "einzelnachweise", "literatur", "weblinks",
                 "externe links", "see also", "zie ook", "siehe auch", "bronnen")
DEFAULT_MIN_PARAGRAPH_CHARS = 200
DEFAULT_MIN_DOC_CHARS = 1000
#: The rules, verbatim, for the manifest.
SANITIZE_RULES = (
    "strip <ref>...</ref> and <ref .../> blocks, then every HTML tag",
    "remove {{...}} templates (innermost first, repeated)",
    "[[target|text]] / [[text]] -> text; File/Image/Category links removed",
    "[http://url text] -> text; '''bold''' / ''italic'' quotes removed",
    "section headings (== ... == lines, or a bare stop-heading line) dropped",
    "everything from the first of " + " / ".join(STOP_HEADINGS) + " onward cut",
    "whitespace collapsed within a paragraph",
    "paragraphs under min_paragraph_chars dropped",
    "documents under min_doc_chars after cleaning dropped",
)
#: The default User-Agent of the Wikipedia API calls (a generic client name).
USER_AGENT = "rspd[experiments]-unlearning-sets/1.0 (research corpus builder)"

_REF_RE = re.compile(r"<ref[^>/]*/\s*>|<ref[^>]*>.*?</ref\s*>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^<>]+>")
_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
_LINK_RE = re.compile(r"\[\[([^\[\]]*)\]\]")
_EXT_LINK_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?:\s+([^\]]*))?\]")
_HEADING_RE = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")
_FILE_PREFIXES = ("file:", "image:", "category:", "bestand:", "datei:", "kategorie:",
                  "categorie:", "afbeelding:", "media:")
_WS_RE = re.compile(r"\s+")


def _link_text(m: re.Match) -> str:
    inner = m.group(1)
    target = inner.split("|", 1)[0].strip().lower()
    if target.startswith(_FILE_PREFIXES):
        return ""
    return inner.rsplit("|", 1)[-1]


def _heading_name(line: str):
    """The heading text if ``line`` is a heading (``== X ==`` or a bare line that
    is exactly a stop heading), else ``None``."""
    m = _HEADING_RE.match(line)
    if m:
        return m.group(2)
    bare = line.strip().rstrip(":").strip().lower()
    if bare in STOP_HEADINGS:
        return bare
    return None


def sanitize_wiki_text(text: str, *, min_paragraph_chars: int = DEFAULT_MIN_PARAGRAPH_CHARS,
                       min_doc_chars: int = DEFAULT_MIN_DOC_CHARS) -> str:
    """Clean one article (hub-dump plain text or an ``extracts`` API text) per
    the module docstring. Returns the cleaned document, paragraphs joined by a
    blank line, or ``""`` when it ends under ``min_doc_chars``."""
    if not isinstance(text, str) or not text:
        return ""
    s = _REF_RE.sub(" ", text)
    s = _TAG_RE.sub(" ", s)
    prev = None
    while prev != s:                               # nested templates, innermost first
        prev, s = s, _TEMPLATE_RE.sub(" ", s)
    s = _LINK_RE.sub(_link_text, s)
    s = _EXT_LINK_RE.sub(lambda m: m.group(1) or " ", s)
    s = s.replace("'''", "").replace("''", "")
    paragraphs = []
    for line in s.split("\n"):
        name = _heading_name(line)
        if name is not None:
            if name.strip().rstrip(":").strip().lower() in STOP_HEADINGS:
                break
            continue                               # any other heading line is dropped
        para = _WS_RE.sub(" ", line).strip()
        if len(para) >= int(min_paragraph_chars):
            paragraphs.append(para)
    out = "\n\n".join(paragraphs)
    return out if len(out) >= int(min_doc_chars) else ""


# --------------------------------------------------------------------------- #
# Deterministic subsampling of a pinned stream                                 #
# --------------------------------------------------------------------------- #
def accept_page(page_id, seed: int, sample_rate: float) -> bool:
    """Whether a page is in the seeded subsample: a hash of ``(seed, id)``
    against ``sample_rate`` -- a function of the id alone, so the selection does
    not depend on stream position or on how far an earlier run got."""
    if sample_rate >= 1.0:
        return True
    h = hashlib.sha256(f"{int(seed)}:{page_id}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64 < float(sample_rate)


def iter_hub_articles(source: Mapping[str, Any], *, seed: int) -> Iterator[dict]:
    """``{"id", "title", "text"}`` of a pinned ``wikimedia/wikipedia`` config in
    dump order from ``skip_docs``, subsampled by :func:`accept_page`. Imports
    ``datasets`` lazily (runtime image only)."""
    from datasets import load_dataset

    ds = load_dataset(source["dataset"], source["config"], split=source.get("split", "train"),
                      revision=source["revision"], streaming=True)
    idf, tf, xf = (source.get("id_field", "id"), source.get("title_field", "title"),
                   source.get("text_field", "text"))
    rate = float(source.get("sample_rate", 1.0))
    it = iter(ds)
    try:
        for i, row in enumerate(it):
            if i < int(source.get("skip_docs", 0) or 0):
                continue
            pid = str(row.get(idf))
            if not accept_page(pid, seed, rate):
                continue
            yield {"id": pid, "title": str(row.get(tf) or ""), "text": row.get(xf) or "",
                   "stream_index": i}
    finally:
        try:
            it.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Token budgets                                                                #
# --------------------------------------------------------------------------- #
def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer([text], add_special_tokens=False)["input_ids"][0])


def truncate_to_tokens(tokenizer, text: str, budget: int) -> tuple[str, int]:
    """The longest prefix of ``text`` whose re-tokenisation is at most ``budget``
    tokens, with its exact count. Decoding a cut token sequence and encoding it
    again is not always length-preserving (a BPE merge can straddle the cut),
    so the cut is shrunk until it is -- on the trilingual BPE the first attempt
    almost always lands exactly on ``budget``."""
    ids = tokenizer([text], add_special_tokens=False)["input_ids"][0]
    if len(ids) <= budget:
        return text, len(ids)
    k = budget
    while k > 0:
        cut = tokenizer.decode(ids[:k]).rstrip()
        n = count_tokens(tokenizer, cut)
        if n <= budget:
            return cut, n
        k -= 1
    return "", 0


def fill_role(docs: Iterable[dict], *, budget: int, tokenizer, role: str, lang: str,
              exclude_ids: set | None = None, seen_ids: set | None = None,
              preset: list | None = None, log=print) -> dict:
    """Take whole documents from ``docs`` (already sanitised, in order) until
    ``budget`` tokens are met, truncating the last one at exactly the budget.

    ``preset`` documents (the counterparts) go FIRST, in the order given, and
    count toward the budget. ``exclude_ids`` are skipped; every id taken is added
    to ``seen_ids`` so the caller can keep later roles disjoint. ``docs`` is not
    consumed past the budget, so the same iterator can fill the next role.
    Returns ``{"docs": [...], "n_tokens": int, "complete": bool}``.
    """
    exclude_ids = exclude_ids or set()
    out, got = [], 0

    def _take(d, flag):
        nonlocal got
        text, n = d["text"], d.get("n_tokens")
        if n is None:
            n = count_tokens(tokenizer, text)
        truncated = False
        if got + n > budget:
            text, n = truncate_to_tokens(tokenizer, text, budget - got)
            truncated = True
            if n == 0:
                return True
        out.append({"id": str(d["id"]), "title": d.get("title", ""), "lang": lang,
                    "role": role, "text": text, "is_target_counterpart": bool(flag),
                    "source_lang": d.get("source_lang", lang), "n_tokens": int(n),
                    "truncated": truncated})
        got += n
        if seen_ids is not None:
            seen_ids.add(str(d["id"]))
        return got >= budget

    for d in (preset or []):
        if _take(d, True):
            break
    if got < budget:
        for d in docs:
            if str(d["id"]) in exclude_ids or (seen_ids is not None and str(d["id"]) in seen_ids):
                continue
            if _take(d, False):
                break
    complete = got >= budget
    if not complete:
        log(f"  WARNING: {role}: the stream ran out at {got:,} of {budget:,} tokens")
    log(f"  [{role}] {len(out)} documents, {got:,} tokens"
        + (f" ({sum(1 for d in out if d['is_target_counterpart'])} counterparts first)"
           if preset else ""))
    return {"docs": out, "n_tokens": int(got), "complete": complete}


def sanitised(docs: Iterable[dict], *, min_paragraph_chars: int, min_doc_chars: int,
              stats: dict | None = None) -> Iterator[dict]:
    """``docs`` with ``text`` sanitised; documents that clean to nothing are
    dropped (counted in ``stats['dropped']`` when given)."""
    for d in docs:
        if stats is not None:
            stats["seen"] = stats.get("seen", 0) + 1
        clean = sanitize_wiki_text(d.get("text", ""), min_paragraph_chars=min_paragraph_chars,
                                   min_doc_chars=min_doc_chars)
        if not clean:
            if stats is not None:
                stats["dropped"] = stats.get("dropped", 0) + 1
            continue
        yield dict(d, text=clean)


# --------------------------------------------------------------------------- #
# The Wikipedia API: langlinks + extracts, retried, loud                       #
# --------------------------------------------------------------------------- #
def get_json(url: str, params: Mapping[str, Any], *, user_agent: str = USER_AGENT,
             retries: int = 5, backoff: float = 1.0, timeout: float = 60.0,
             sleep=time.sleep) -> dict:
    """GET ``url`` with ``params`` and parse JSON, retrying ``retries`` times
    with exponential backoff; raises ``RuntimeError`` naming the URL after the
    last failure. No silent fallback: an unreachable API fails the build."""
    import urllib.error
    import urllib.parse
    import urllib.request

    full = url + "?" + urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
    last = None
    for attempt in range(int(retries)):
        try:
            req = urllib.request.Request(full, headers={"User-Agent": user_agent,
                                                        "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read().decode("utf-8"))
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(f"API error: {payload['error']}")
            return payload
        except Exception as exc:                   # noqa: BLE001 - retried, then re-raised
            last = exc
            if attempt + 1 < int(retries):
                sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Wikipedia API unreachable after {retries} attempts: {full} "
                       f"({type(last).__name__}: {last})")


def fetch_langlinks(titles: list, *, endpoint: str, langs: Mapping[str, str],
                    get=get_json, batch: int = 50) -> dict:
    """``{title: {lang_code: counterpart title}}`` through ``prop=langlinks`` on
    the source wiki (``endpoint``), following redirects. ``lllang`` takes ONE
    code per request, so it is one request per language per batch of
    ``batch`` titles. A title with no link in a language has no entry for it;
    a missing page has an empty dict."""
    codes = sorted(set(langs.values()))
    out = {t: {} for t in titles}
    for i in range(0, len(titles), batch):
        chunk = titles[i:i + batch]
        for code in codes:
            payload = get(endpoint, {"action": "query", "prop": "langlinks", "lllang": code,
                                     "lllimit": 500, "titles": "|".join(chunk), "redirects": 1,
                                     "format": "json", "formatversion": 2})
            query = payload.get("query", {})
            norm = {n["from"]: n["to"] for n in query.get("normalized", [])}
            redir = {r["from"]: r["to"] for r in query.get("redirects", [])}
            by_title = {p.get("title"): p for p in query.get("pages", [])}
            for t in chunk:
                resolved = redir.get(norm.get(t, t), norm.get(t, t))
                page = by_title.get(resolved) or {}
                if page.get("missing"):
                    continue
                for ll in page.get("langlinks", []) or []:
                    if ll.get("lang") == code:
                        out[t][code] = ll.get("title") or ll.get("*")
    return out


def fetch_extract(title: str, *, lang_code: str, endpoint_template: str, get=get_json) -> dict:
    """``{"pageid", "title", "text"}`` of one article as plain text through
    ``prop=extracts&explaintext=1`` on that language's wiki, following
    redirects. Raises ``LookupError`` for a missing page."""
    endpoint = endpoint_template.format(lang=lang_code)
    payload = get(endpoint, {"action": "query", "prop": "extracts", "explaintext": 1,
                             "exlimit": 1, "titles": title, "redirects": 1,
                             "format": "json", "formatversion": 2})
    pages = payload.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        raise LookupError(f"{lang_code}: no page {title!r}")
    p = pages[0]
    return {"pageid": str(p.get("pageid")), "title": p.get("title", title),
            "text": p.get("extract") or ""}


def fetch_counterparts(target_docs: list, *, langs: Mapping[str, str], langlinks_endpoint: str,
                       extracts_endpoint: str, get=get_json, min_paragraph_chars: int,
                       min_doc_chars: int, log=print) -> tuple[dict, list]:
    """The sanitised English / German counterparts of the target articles.

    Returns ``(by_lang, unmatched)``: ``by_lang[lang] = [doc, ...]`` in target
    order (each doc ``id`` = the counterpart's page id, ``source_lang`` the
    target's language, ``counterpart_of`` the target's id / title) and
    ``unmatched = [{target_id, title, lang, reason}, ...]`` for every
    (target, language) without a usable counterpart -- no langlink, a missing
    page, or a text that sanitises to nothing. Printed, never fatal.
    """
    titles = [d["title"] for d in target_docs]
    links = fetch_langlinks(titles, endpoint=langlinks_endpoint, langs=langs, get=get)
    by_lang = {lang: [] for lang in langs}
    unmatched = []
    for d in target_docs:
        for lang, code in langs.items():
            ctitle = links.get(d["title"], {}).get(code)
            if not ctitle:
                unmatched.append({"target_id": d["id"], "title": d["title"], "lang": lang,
                                  "reason": "no langlink"})
                log(f"  [counterpart] {lang}: no langlink for {d['title']!r}")
                continue
            try:
                page = fetch_extract(ctitle, lang_code=code, endpoint_template=extracts_endpoint,
                                     get=get)
            except LookupError as exc:
                unmatched.append({"target_id": d["id"], "title": d["title"], "lang": lang,
                                  "reason": f"missing page: {exc}"})
                log(f"  [counterpart] {lang}: {exc}")
                continue
            clean = sanitize_wiki_text(page["text"], min_paragraph_chars=min_paragraph_chars,
                                       min_doc_chars=min_doc_chars)
            if not clean:
                unmatched.append({"target_id": d["id"], "title": d["title"], "lang": lang,
                                  "reason": f"counterpart {page['title']!r} sanitises to under "
                                            f"{min_doc_chars} chars"})
                log(f"  [counterpart] {lang}: {page['title']!r} too short after cleaning")
                continue
            by_lang[lang].append({"id": page["pageid"], "title": page["title"], "text": clean,
                                  "source_lang": LANG_OF_ROLE["target_nld"],
                                  "counterpart_of": {"id": d["id"], "title": d["title"]}})
            log(f"  [counterpart] {lang}: {d['title']!r} -> {page['title']!r} (page {page['pageid']})")
    return by_lang, unmatched


# --------------------------------------------------------------------------- #
# The build                                                                    #
# --------------------------------------------------------------------------- #
def resolve_spec(cfg: Mapping[str, Any]) -> dict:
    """Validate the config and return the spec that is hashed into the manifest."""
    for key in ("name", "corpus_name", "sources", "budgets"):
        if key not in cfg:
            raise ValueError(f"config needs '{key}'")
    sources = {}
    for lang in ("nld", "eng", "deu"):
        src = (cfg["sources"] or {}).get(lang)
        if not isinstance(src, dict):
            raise ValueError(f"sources.{lang} is required")
        for req in ("dataset", "config", "revision"):
            if not src.get(req):
                raise ValueError(f"sources.{lang}.{req} is required (revision above all: "
                                 f"without it the sets cannot be rebuilt)")
        rate = float(src.get("sample_rate", 1.0))
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"sources.{lang}.sample_rate must be in (0, 1], got {rate}")
        sources[lang] = {"dataset": str(src["dataset"]), "config": str(src["config"]),
                         "split": str(src.get("split", "train")), "revision": str(src["revision"]),
                         "text_field": str(src.get("text_field", "text")),
                         "id_field": str(src.get("id_field", "id")),
                         "title_field": str(src.get("title_field", "title")),
                         "skip_docs": int(src.get("skip_docs", 0) or 0), "sample_rate": rate}
    budgets = {}
    for role in ROLES:
        b = (cfg["budgets"] or {}).get(role)
        if not isinstance(b, int) or isinstance(b, bool) or b < 1:
            raise ValueError(f"budgets.{role} must be a positive int (tokens), got {b!r}")
        budgets[role] = int(b)
    san = cfg.get("sanitize") or {}
    cp = cfg.get("counterparts") or {}
    langs = dict(cp.get("langs") or {"eng": "en", "deu": "de"})
    if set(langs) != {"eng", "deu"}:
        raise ValueError(f"counterparts.langs must map eng and deu to wiki codes, got {langs}")
    return {
        "name": str(cfg["name"]), "corpus_name": str(cfg["corpus_name"]),
        "seed": int(cfg.get("seed", 0)), "sources": sources, "budgets": budgets,
        "sanitize": {"min_paragraph_chars": int(san.get("min_paragraph_chars",
                                                         DEFAULT_MIN_PARAGRAPH_CHARS)),
                     "min_doc_chars": int(san.get("min_doc_chars", DEFAULT_MIN_DOC_CHARS)),
                     "rules": list(SANITIZE_RULES)},
        "counterparts": {"langs": langs,
                         "langlinks_endpoint": str(cp.get("langlinks_endpoint",
                                                          "https://nl.wikipedia.org/w/api.php")),
                         "extracts_endpoint": str(cp.get("extracts_endpoint",
                                                         "https://{lang}.wikipedia.org/w/api.php")),
                         "user_agent": str(cp.get("user_agent", USER_AGENT))},
        "shard_layout": "documents' tokens with one eos after each (pretrain_corpus.write_span)",
    }


def build_sets(spec: Mapping[str, Any], tokenizer, *, article_source: Callable[[str], Iterable[dict]],
               counterparts: Callable[[list], tuple[dict, list]], log=print) -> dict:
    """Select every role's documents. Pure selection: ``article_source(lang)``
    yields raw ``{"id", "title", "text"}`` articles of a language in stream order
    (the pinned hub dump; a fake in tests) and ``counterparts(target_docs)``
    returns :func:`fetch_counterparts`'s ``(by_lang, unmatched)``.

    One stream per language serves every role of that language in order
    (``target`` then ``heldout`` for Dutch; ``contrastive`` then ``heldout`` for
    English / German), so roles of one language are disjoint by construction;
    the counterparts' page ids are additionally excluded from the fills of
    their language. Returns ``{"roles": {role: fill}, "counterparts": {...},
    "sanitize_stats": {lang: {...}}}``.
    """
    san = spec["sanitize"]
    budgets = spec["budgets"]
    stats = {lang: {} for lang in ("nld", "eng", "deu")}
    streams = {lang: sanitised(article_source(lang), min_paragraph_chars=san["min_paragraph_chars"],
                               min_doc_chars=san["min_doc_chars"], stats=stats[lang])
               for lang in ("nld", "eng", "deu")}
    seen = {lang: set() for lang in streams}
    roles = {}

    log(f"[sets] target_nld: {budgets['target_nld']:,} tokens of Dutch")
    roles["target_nld"] = fill_role(streams["nld"], budget=budgets["target_nld"], tokenizer=tokenizer,
                                    role="target", lang="nld", seen_ids=seen["nld"], log=log)
    target_docs = roles["target_nld"]["docs"]
    log(f"[sets] counterparts of {len(target_docs)} target article(s) via the Wikipedia API")
    by_lang, unmatched = counterparts(target_docs)
    for lang in ("eng", "deu"):
        role = f"contrastive_{lang}"
        preset = by_lang.get(lang, [])
        excl = {str(d["id"]) for d in preset}
        log(f"[sets] {role}: {budgets[role]:,} tokens, {len(preset)} counterpart(s) first")
        roles[role] = fill_role(streams[lang], budget=budgets[role], tokenizer=tokenizer,
                                role="contrastive", lang=lang, exclude_ids=excl,
                                seen_ids=seen[lang], preset=preset, log=log)
    for lang in ("eng", "deu", "nld"):
        role = f"heldout_{lang}"
        log(f"[sets] {role}: {budgets[role]:,} tokens, disjoint from {len(seen[lang])} used page ids")
        roles[role] = fill_role(streams[lang], budget=budgets[role], tokenizer=tokenizer,
                                role="heldout", lang=lang, seen_ids=seen[lang], log=log)
    # disjointness, verified on the realised ids rather than assumed
    for lang in ("nld", "eng", "deu"):
        ids = [d["id"] for r in ROLES if LANG_OF_ROLE[r] == lang for d in roles[r]["docs"]]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{lang}: a page id appears in two roles; refusing to write")
    matched = [{"lang": lang, **{k: v for k, v in d.items() if k != "text"}}
               for lang, docs in by_lang.items() for d in docs]
    return {"roles": roles, "counterparts": {"matched": matched, "unmatched": unmatched},
            "sanitize_stats": stats}


def write_outputs(sets: Mapping[str, Any], spec: Mapping[str, Any], tokenizer, out_dir: str, *,
                  tokenizer_dir_ref: str, log=print) -> dict:
    """JSONL per role, uint16 shards per role through :func:`gpt2_corpus.write_span`,
    and ``manifest.json``; returns the manifest."""
    eot = tokenizer.eos_token_id
    if eot is None:
        raise ValueError("tokenizer has no eos_token_id; documents would run into each other")
    os.makedirs(out_dir, exist_ok=True)
    manifest_roles, n_tokens, n_shard_tokens, n_docs = {}, {}, {}, {}
    for role in ROLES:
        fill = sets["roles"][role]
        docs = fill["docs"]
        jsonl = os.path.join(out_dir, f"{role}.jsonl")
        with open(jsonl, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({k: d[k] for k in ("id", "title", "lang", "role", "text",
                                                      "is_target_counterpart", "n_tokens")},
                                   ensure_ascii=False) + "\n")
        total = sum(d["n_tokens"] for d in docs)
        shard_dir = os.path.join(out_dir, "shards", role)
        rec = pc.write_span(tokenizer, iter(d["text"] for d in docs), {"tokens": total + len(docs)},
                            shard_dir, eot_id=int(eot), log=log)
        if rec["tokens"] != total + len(docs):
            raise RuntimeError(f"{role}: shard holds {rec['tokens']} tokens, expected "
                               f"{total} + {len(docs)} separators")
        n_chars = sum(len(d["text"]) for d in docs)
        manifest_roles[role] = {
            "lang": LANG_OF_ROLE[role], "role": role.rsplit("_", 1)[0],
            "budget_tokens": spec["budgets"][role], "n_tokens": total, "complete": fill["complete"],
            "n_shard_tokens": rec["tokens"], "n_docs": len(docs), "n_chars": n_chars,
            "fertility_tokens_per_char": (total / n_chars) if n_chars else None,
            "jsonl": os.path.relpath(jsonl, out_dir),
            "shards": [os.path.join("shards", role, s) for s in rec["shards"]],
            "documents": [{k: d[k] for k in ("id", "title", "lang", "source_lang",
                                             "is_target_counterpart", "n_tokens", "truncated")}
                          for d in docs],
        }
        n_tokens[role], n_shard_tokens[role], n_docs[role] = total, rec["tokens"], len(docs)
    manifest = {
        "name": spec["name"], "corpus_name": spec["corpus_name"],
        "tokenizer_dir": tokenizer_dir_ref, "spec_sha256": manifest_hash(dict(spec)),
        "spec": dict(spec),
        "revisions": {lang: s["revision"] for lang, s in spec["sources"].items()},
        "n_tokens": n_tokens, "n_shard_tokens": n_shard_tokens, "n_docs": n_docs,
        "shard_dtype": "uint16", "shard_layout": spec["shard_layout"],
        "sanitisation_rules": list(spec["sanitize"]["rules"]),
        "sanitize_stats": sets["sanitize_stats"],
        "counterparts": sets["counterparts"],
        "roles": manifest_roles,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"  [manifest] {os.path.join(out_dir, 'manifest.json')} (spec sha256 "
        f"{manifest['spec_sha256'][:12]})")
    return manifest


def results_markdown(manifest: Mapping[str, Any], *, elapsed_s: float) -> str:
    lines = [f"# GPT-2 unlearning sets — `{manifest['name']}`", "",
             f"- tokenizer: `{manifest['tokenizer_dir']}` (the trilingual BPE of "
             f"`{manifest['corpus_name']}`)",
             f"- spec sha256: `{manifest['spec_sha256'][:12]}`",
             f"- wall-clock: {elapsed_s / 60:.1f} min", "",
             "## Sets", "",
             "| set | budget (tokens) | tokens | docs | shard tokens (incl. eos) | fertility (tok/char) | complete |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for role in ROLES:
        r = manifest["roles"][role]
        fert = r["fertility_tokens_per_char"]
        lines.append(f"| `{role}` | {r['budget_tokens']:,} | {r['n_tokens']:,} | {r['n_docs']} | "
                     f"{r['n_shard_tokens']:,} | {fert:.4f} | {'yes' if r['complete'] else 'NO'} |"
                     if fert else f"| `{role}` | {r['budget_tokens']:,} | {r['n_tokens']:,} | "
                                  f"{r['n_docs']} | {r['n_shard_tokens']:,} | — | "
                                  f"{'yes' if r['complete'] else 'NO'} |")
    lines += ["", "## Target articles", "", "| id | title | tokens | truncated |", "| --- | --- | ---: | --- |"]
    for d in manifest["roles"]["target_nld"]["documents"]:
        lines.append(f"| {d['id']} | {d['title']} | {d['n_tokens']} | {'yes' if d['truncated'] else 'no'} |")
    cp = manifest["counterparts"]
    lines += ["", f"## Counterparts ({len(cp['matched'])} matched, {len(cp['unmatched'])} unmatched)", "",
              "| lang | target | counterpart | page id |", "| --- | --- | --- | --- |"]
    for m in cp["matched"]:
        lines.append(f"| {m['lang']} | {m['counterpart_of']['title']} | {m['title']} | {m['id']} |")
    for u in cp["unmatched"]:
        lines.append(f"| {u['lang']} | {u['title']} | — ({u['reason']}) | — |")
    lines += ["", "## Sanitisation", ""] + [f"- {r}" for r in manifest["sanitisation_rules"]]
    lines += ["", "Documents seen / dropped per language: "
              + ", ".join(f"{lang} {s.get('seen', 0)}/{s.get('dropped', 0)}"
                          for lang, s in manifest["sanitize_stats"].items()), ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Download, shards, rebuild                                                    #
# --------------------------------------------------------------------------- #
#: Where the sets live under the data root.
UNLEARNING_DATASET = "gpt2_unlearning"
#: The roles an unlearning-set directory holds, as ``shards/<role>_<lang>/``.
UNLEARNING_ROLES = ("target", "contrastive", "heldout")
#: Published copies of the sets, ``{name: HF dataset repo id}``. A name absent
#: here must be given an explicit ``source=`` (or rebuilt).
PUBLISHED_SETS: dict = {}


def load_sets_config(path: str) -> dict:
    """A sets config (``name``, ``corpus_name``, ``sources``, ``budgets``, ...;
    see ``specs/gpt2_unlearning_1m.yaml``)."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("name", "corpus_name", "sources", "budgets"):
        if key not in cfg:
            raise ValueError(f"{path}: config needs '{key}'")
    return cfg


def sets_root(name: str, *, root=None) -> str:
    """``<data root>/datasets/gpt2_unlearning/<name>`` (not created)."""
    from ..paths import datasets_dir
    return str(datasets_dir(UNLEARNING_DATASET, root=root, create=False) / name)


def unlearning_shards(sets_dir: str, role: str, lang: str) -> list:
    """Absolute paths of ``shards/<role>_<lang>/shard_*.bin`` under ``sets_dir``,
    in order; ``[]`` when the role/language is absent."""
    from pathlib import Path

    d = Path(sets_dir) / "shards" / f"{role}_{lang}"
    if not d.is_dir():
        return []
    return [str(p) for p in sorted(d.glob("shard_*.bin"))]


def ensure_unlearning_sets(name: str, *, source=None, root=None, revision=None) -> str:
    """A local directory holding the ``<role>.jsonl`` files of set ``name``.

    Already present under the data root -> returned as is. Otherwise copied
    from ``source`` (a local directory) or downloaded from it (a HF dataset
    repo id; ``revision`` pins the dataset commit), defaulting to
    :data:`PUBLISHED_SETS` ``[name]``. Raises when no source is known --
    rebuilding instead (:func:`prepare_unlearning_sets`) is time-dependent, see
    the module docstring."""
    import shutil

    out = sets_root(name, root=root)
    if all(os.path.isfile(os.path.join(out, f"{r}.jsonl")) for r in ROLES):
        return out
    source = source or PUBLISHED_SETS.get(name)
    if source is None:
        raise FileNotFoundError(
            f"unlearning sets {name!r} are not under {out} and no published source is known; "
            f"pass source=<HF dataset repo id or directory>, or rebuild with "
            f"prepare_unlearning_sets (time-dependent: live Wikipedia API)")
    os.makedirs(out, exist_ok=True)
    if os.path.isdir(str(source)):
        for f in [f"{r}.jsonl" for r in ROLES] + ["manifest.json"]:
            src = os.path.join(str(source), f)
            if os.path.isfile(src):
                shutil.copyfile(src, os.path.join(out, f))
    else:
        from huggingface_hub import snapshot_download

        from ..paths import hf_token
        snapshot_download(repo_id=str(source), repo_type="dataset", revision=revision,
                          local_dir=out, token=hf_token(),
                          allow_patterns=["*.jsonl", "manifest.json"])
    missing = [r for r in ROLES if not os.path.isfile(os.path.join(out, f"{r}.jsonl"))]
    if missing:
        raise FileNotFoundError(f"{source}: no {missing} JSONL files")
    return out


def read_role(sets_dir: str, role: str) -> list:
    """The rows of ``<sets_dir>/<role>.jsonl`` in order."""
    with open(os.path.join(sets_dir, f"{role}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def materialize_shards(sets_dir: str, tokenizer, *, overwrite: bool = False, log=print) -> dict:
    """Write ``shards/<role>_<lang>/`` from the JSONL with
    :func:`gpt2_corpus.write_span` -- what the build wrote (the same texts, the
    same writer, one eos per document). Returns ``{role: n_shard_tokens}`` for
    the roles written; roles whose shard directory is non-empty are skipped
    unless ``overwrite``."""
    eot = tokenizer.eos_token_id
    if eot is None:
        raise ValueError("tokenizer has no eos_token_id; documents would run into each other")
    out = {}
    for role in ROLES:
        shard_dir = os.path.join(sets_dir, "shards", role)
        if os.path.isdir(shard_dir) and os.listdir(shard_dir) and not overwrite:
            continue
        docs = read_role(sets_dir, role)
        n = sum(count_tokens(tokenizer, d["text"]) for d in docs)
        rec = pc.write_span(tokenizer, iter(d["text"] for d in docs), {"tokens": n + len(docs)},
                            shard_dir, eot_id=int(eot), log=log)
        out[role] = int(rec["tokens"])
    return out


def prepare_unlearning_sets(cfg: Mapping[str, Any], *, tokenizer=None, root=None,
                            log_dir=None, article_source=None, counterparts=None,
                            log=print) -> dict:
    """Rebuild the sets of ``cfg`` from the pinned dumps and the live Wikipedia
    API (time-dependent, see the module docstring) and write them under
    :func:`sets_root`; ``results.md`` goes to ``log_dir`` (default
    ``<data root>/logs/gpt2/<name>``). ``tokenizer`` defaults to the trilingual
    BPE of ``cfg["corpus_name"]`` (:func:`gpt2_corpus.tokenizer_dir`).
    ``article_source`` / ``counterparts`` replace the hub stream and the API
    (tests). Returns the manifest."""
    import time

    from ..paths import data_ref, logs_dir
    from .gpt2_corpus import tokenizer_dir

    spec = resolve_spec(cfg)
    out_dir = sets_root(spec["name"], root=root)
    os.makedirs(out_dir, exist_ok=True)
    log_dir = str(log_dir or logs_dir("gpt2", spec["name"], root=root))
    os.makedirs(log_dir, exist_ok=True)
    tok_dir = tokenizer_dir(spec["corpus_name"], root=root)
    if tokenizer is None:
        if not os.path.exists(os.path.join(tok_dir, "tokenizer.json")):
            raise FileNotFoundError(f"no trilingual tokenizer at {tok_dir}; prepare the corpus "
                                    f"{spec['corpus_name']!r} first (gpt2_corpus.prepare)")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    t0 = time.time()
    cp, san = spec["counterparts"], spec["sanitize"]

    if article_source is None:
        def article_source(lang):
            return iter_hub_articles(spec["sources"][lang], seed=spec["seed"])

    if counterparts is None:
        def counterparts(target_docs):
            def get(url, params):
                return get_json(url, params, user_agent=cp["user_agent"])
            return fetch_counterparts(target_docs, langs=cp["langs"],
                                      langlinks_endpoint=cp["langlinks_endpoint"],
                                      extracts_endpoint=cp["extracts_endpoint"], get=get,
                                      min_paragraph_chars=san["min_paragraph_chars"],
                                      min_doc_chars=san["min_doc_chars"], log=log)

    sets = build_sets(spec, tokenizer, article_source=article_source,
                      counterparts=counterparts, log=log)
    manifest = write_outputs(sets, spec, tokenizer, out_dir,
                             tokenizer_dir_ref=data_ref(tok_dir, root=root), log=log)
    with open(os.path.join(log_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write(results_markdown(manifest, elapsed_s=time.time() - t0))
    return manifest
