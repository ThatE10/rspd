"""The named-corpus registry and its loaders: WMDP forget/retain corpora,
WikiText, MUSE News/Books raw splits, Dutch / Chinese Wikipedia, plus any
``local_jsonl`` corpora a config declares.

:data:`CORPORA` maps a corpus name to its hub coordinates (``path``,
``config``, ``split``, pinned ``revision`` where one is pinned, ``gated``,
``stream``, ``chunk``, ``topic`` / ``role``). The same registry serves the
SRSPD target builder and the gradient baselines, so every method reads a
corpus the same way.

* forget **bio** -- ``cais/wmdp-bio-forget-corpus`` (gated; needs ``HF_TOKEN``);
* forget **cyber** -- ``cais/wmdp-corpora`` config ``cyber-forget-corpus``;
* retain **wikitext** -- ``Salesforce/wikitext`` / ``wikitext-2-raw-v1`` test split;
* optional retain **bio/cyber** -- ``cais/wmdp-corpora`` ``{bio,cyber}-retain-corpus``;
* ``nl-wiki`` / ``zh-wiki`` -- ``wikimedia/wikipedia`` ``20231101.{nl,zh}`` at a
  pinned revision, streamed (multi-GB; every consumer caps rows);
* ``muse-{news,books}-{forget,retain1,retain2,holdout}`` -- the ``raw`` config
  of ``muse-bench/MUSE-{News,Books}`` at pinned revisions.

A config may add ``local_jsonl`` corpora (``corpora:`` at the top level, see
:func:`register_config_corpora`) -- plain-text JSONL files under the data root
(:mod:`rspd.experiments.paths`), such as the GPT-2 unlearning sets. They join
:data:`CORPORA` for the process and are read by the same loaders.

Two consumption shapes:

* :func:`load_text_batches` -- per corpus, a list of raw-string batches (RMU and
  KUDA tokenize inside their loop with a per-topic ``max_length``: bio 512,
  cyber/other 768);
* :func:`tokenize_batch` -- ``input_ids`` / ``attention_mask`` / ``labels`` with
  PAD positions masked to ``-100`` (SimNPO's per-token NLL must not count pads).

MUSE windows
------------
The MUSE corpora carry ``chunk: True``: the official unlearning loader joins
every row of a split with ``"\n\n"``, tokenizes the whole corpus once with
``add_special_tokens=False``, cuts it into windows of ``max_len - 1`` tokens
each prefixed with BOS, and rotate-pads the last window from the first. The
Books forget split is 1.13M tokens in 4 rows, so a per-row loader truncating to
a few hundred tokens would discard >99% of it. :func:`load_corpus_windows`
implements the official windows, :func:`load_text_batches` yields them for a
``chunk`` corpus, and :func:`tokenize_batch` / :func:`encode` accept the
resulting pre-tokenized windows without re-truncating them or adding a BOS.

``datasets`` / ``transformers`` / ``torch`` are imported inside the functions
that need them, so importing this module is cheap.
"""
from __future__ import annotations

import json
import os

from ..paths import hf_token, resolve_data_ref

#: Default minimum text length (characters) — the RMU/KUDA ``get_data`` filter.
DEFAULT_MIN_LEN = 50
#: Per-topic tokenisation length used by the RMU/KUDA repros.
FORGET_MAX_LENGTH = {"bio": 512, "cyber": 768}
RETAIN_MAX_LENGTH = 512

#: Hub coordinates for each corpus the loaders know. ``kind`` picks
#: the loader path; ``gated`` marks the ones that need a read token.
CORPORA = {
    # forget
    "bio-forget-corpus":   {"path": "cais/wmdp-bio-forget-corpus", "config": None,
                            "topic": "bio", "gated": True},
    "cyber-forget-corpus": {"path": "cais/wmdp-corpora", "config": "cyber-forget-corpus",
                            "topic": "cyber", "gated": False},
    # retain
    # `Salesforce/wikitext` is the canonical id for what upstream RMU loads as
    # bare `wikitext`; the bare alias still resolves on the hub, but the pinned
    # huggingface_hub rejects it (HfUriError: repo id must be namespace/name).
    # Same dataset, same config, same split, ungated — an id-form fix only.
    "wikitext":            {"path": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                            "split": "test", "topic": "retain", "gated": False},
    "bio-retain-corpus":   {"path": "cais/wmdp-corpora", "config": "bio-retain-corpus",
                            "topic": "retain", "gated": False},
    "cyber-retain-corpus": {"path": "cais/wmdp-corpora", "config": "cyber-retain-corpus",
                            "topic": "retain", "gated": False},
    # Dutch Wikipedia at a pinned revision. `stream` is set because the full
    # 20231101.nl config is multi-GB and every consumer caps rows -- the loader
    # iterates lazily and stops at `max_rows`.
    "nl-wiki":             {"path": "wikimedia/wikipedia", "config": "20231101.nl",
                            "split": "train", "topic": "nl", "gated": False,
                            "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                            "stream": True},
    # Chinese Wikipedia, held-out readout only: the same source and pinned
    # revision as nl-wiki, config 20231101.zh; streamed and row-capped the same
    # way. Never a fit set.
    "zh-wiki":             {"path": "wikimedia/wikipedia", "config": "20231101.zh",
                            "split": "train", "topic": "retain", "gated": False,
                            "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
                            "stream": True},
    # MUSE (muse-bench/MUSE-News, muse-bench/MUSE-Books; config `raw`). `chunk`
    # selects the official window loader (see the module docstring); `role`
    # says which side of an unlearning run a split may sit on — the topic is
    # the corpus family, not "retain", so the drivers' forget/retain validation
    # reads `role` through `is_retain_corpus`. `retain1` is the official
    # unlearning retain set; `retain2` and `holdout` are eval-side splits and
    # are registered so a config can name them, never as forget corpora. Every
    # split is pinned to one dataset-repo sha (the same revision a retained
    # bucket over retain1 pins), so X / the readouts and Y come from one snapshot.
    "muse-news-forget":    {"path": "muse-bench/MUSE-News", "config": "raw",
                            "split": "forget", "topic": "muse-news", "gated": False,
                            "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
                            "chunk": True, "role": "forget"},
    "muse-news-retain1":   {"path": "muse-bench/MUSE-News", "config": "raw",
                            "split": "retain1", "topic": "muse-news", "gated": False,
                            "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
                            "chunk": True, "role": "retain"},
    "muse-news-retain2":   {"path": "muse-bench/MUSE-News", "config": "raw",
                            "split": "retain2", "topic": "muse-news", "gated": False,
                            "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
                            "chunk": True, "role": "retain"},
    "muse-news-holdout":   {"path": "muse-bench/MUSE-News", "config": "raw",
                            "split": "holdout", "topic": "muse-news", "gated": False,
                            "revision": "506bd5b150b92814d45e4404a82f120ab2d748bf",
                            "chunk": True, "role": "retain"},
    "muse-books-forget":   {"path": "muse-bench/MUSE-Books", "config": "raw",
                            "split": "forget", "topic": "muse-books", "gated": False,
                            "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
                            "chunk": True, "role": "forget"},
    "muse-books-retain1":  {"path": "muse-bench/MUSE-Books", "config": "raw",
                            "split": "retain1", "topic": "muse-books", "gated": False,
                            "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
                            "chunk": True, "role": "retain"},
    "muse-books-retain2":  {"path": "muse-bench/MUSE-Books", "config": "raw",
                            "split": "retain2", "topic": "muse-books", "gated": False,
                            "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
                            "chunk": True, "role": "retain"},
    "muse-books-holdout":  {"path": "muse-bench/MUSE-Books", "config": "raw",
                            "split": "holdout", "topic": "muse-books", "gated": False,
                            "revision": "051ba90319e920d410d87cfdbd61f25843c1b892",
                            "chunk": True, "role": "retain"},
}


#: ``kind`` of a :data:`CORPORA` entry. Hub entries (the built-ins above) carry
#: no ``kind`` key and are ``"hub"``; a config's ``corpora:`` block may add
#: ``"local_jsonl"`` entries.
CORPUS_KINDS = ("hub", "local_jsonl")
#: Topics a config-level corpus may declare. The drivers' validation reads the
#: topic: SimNPO wants its retain side at ``"retain"`` and its forget side at
#: anything else; RMU/KUDA look the forget ``max_length`` up by topic (768 for
#: anything but ``bio``). ``target`` = forget, ``contrastive`` = retain,
#: ``heldout`` = a readout-only set.
CONFIG_CORPUS_TOPICS = ("target", "retain", "heldout", "bio", "cyber", "nl", "other")


def corpus_kind(spec: dict) -> str:
    return str(spec.get("kind") or "hub")


def register_config_corpora(cfg) -> list[str]:
    """Merge a config's top-level ``corpora:`` block into :data:`CORPORA` for
    this process and return the names registered (``[]`` when the config has
    none).

    Shape::

        corpora:
          target_nld:
            kind: local_jsonl
            path: datasets/gpt2_unlearning/gpt2_unlearning/target_nld.jsonl
            text_field: text          # default
            topic: target             # target = forget; retain; heldout
          contrastive:
            kind: local_jsonl
            path: [datasets/.../contrastive_eng.jsonl, datasets/.../contrastive_deu.jsonl]
            topic: retain

    ``path`` is data-root-relative (:func:`rspd.experiments.paths.data_dir`,
    ``$RSPD_DATA_DIR``) or absolute, and may be a LIST of paths that
    are read in order and concatenated -- which is how one named corpus can be
    "the English and the German contrastive set together" for a driver that
    takes a single retain corpus (SimNPO). A name that collides with a built-in
    hub corpus is refused; re-registering an identical entry is a no-op (a
    driver may call this more than once).
    """
    block = (cfg or {}).get("corpora") if isinstance(cfg, dict) else None
    if block is None:
        return []
    if not isinstance(block, dict) or not block:
        raise ValueError(f"corpora: must be a non-empty mapping name -> spec, got {block!r}")
    names = []
    for name, raw in block.items():
        where = f"corpora.{name}"
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"corpora: corpus names must be non-empty strings, got {name!r}")
        if not isinstance(raw, dict):
            raise ValueError(f"{where} must be a mapping, got {raw!r}")
        kind = str(raw.get("kind") or "")
        if kind != "local_jsonl":
            raise ValueError(f"{where}.kind must be 'local_jsonl' (the only config-level "
                             f"kind; hub corpora are the built-ins {sorted(CORPORA)}), "
                             f"got {raw.get('kind')!r}")
        path = raw.get("path")
        paths = [path] if isinstance(path, str) else path
        if (not isinstance(paths, (list, tuple)) or not paths
                or not all(isinstance(x, str) and x.strip() for x in paths)):
            raise ValueError(f"{where}.path must be a non-empty path or list of paths, "
                             f"got {path!r}")
        topic = raw.get("topic")
        if topic not in CONFIG_CORPUS_TOPICS:
            raise ValueError(f"{where}.topic must be one of {CONFIG_CORPUS_TOPICS} "
                             f"(target = forget, retain, heldout), got {topic!r}")
        unknown = set(raw) - {"kind", "path", "text_field", "topic"}
        if unknown:
            raise ValueError(f"{where} has unknown keys {sorted(unknown)}")
        spec = {"kind": "local_jsonl", "path": [str(x) for x in paths],
                "text_field": str(raw.get("text_field") or "text"),
                "topic": str(topic), "gated": False}
        prior = CORPORA.get(name)
        if prior is not None and prior != spec:
            raise ValueError(f"{where} collides with an existing corpus of that name "
                             f"({'a built-in hub corpus' if corpus_kind(prior) == 'hub' else 'a different local spec'})")
        CORPORA[name] = spec
        names.append(name)
    return names


def resolve_local_path(path: str) -> str:
    """A ``local_jsonl`` path: absolute as given, else under the data root."""
    return resolve_data_ref(path)


def _iter_local_jsonl_file(spec: dict, rel: str):
    """Every ``text_field`` value of ONE file of a ``local_jsonl`` corpus, line by
    line, in order. A missing file or a line without the field fails loudly
    (a silent empty corpus is a run that trains on nothing)."""
    field = spec.get("text_field", "text")
    path = resolve_local_path(rel)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"local_jsonl corpus file {rel!r} -> {path} does not "
                                f"exist under the data root; build the corpus first")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if field not in row:
                raise KeyError(f"{path}:{i + 1} has no {field!r} field (keys: {sorted(row)})")
            yield row[field]


def _iter_local_jsonl(spec: dict):
    """Every ``text_field`` value of a ``local_jsonl`` corpus, file by file, line
    by line, in order (:func:`_iter_local_jsonl_file` per file)."""
    for rel in spec["path"]:
        yield from _iter_local_jsonl_file(spec, rel)


def is_retain_corpus(corpus: str) -> bool:
    """Whether ``corpus`` may sit on the *retain* side of an unlearning run.

    The WMDP registry encodes the side in ``topic == "retain"``; the MUSE
    entries keep their corpus family as the topic and say the side in ``role``.
    Unknown names raise ``ValueError`` like the loaders do.
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    spec = CORPORA[corpus]
    return spec.get("topic") == "retain" or spec.get("role") == "retain"


def is_chunked_corpus(corpus: str) -> bool:
    """Whether ``corpus`` is consumed as official MUSE windows (``chunk: True``)
    rather than one text per row."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    return bool(CORPORA[corpus].get("chunk"))


# --------------------------------------------------------------------------- #
# Corpus loading                                                               #
# --------------------------------------------------------------------------- #
def load_corpus_texts(corpus: str, *, min_len: int = DEFAULT_MIN_LEN,
                      max_rows: int | None = None) -> list[str]:
    """The ``text`` column of one named corpus (see :data:`CORPORA`), filtered to
    ``len(text) > min_len`` characters and never shuffled (the RMU/KUDA loaders
    batch sequentially).

    Raises ``ValueError`` for an unknown name, and ``RuntimeError`` with a clear
    hint if a gated dataset is requested without a token.
    """
    from datasets import load_dataset

    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    spec = CORPORA[corpus]
    if corpus_kind(spec) == "local_jsonl":
        texts = []
        for t in _iter_local_jsonl(spec):
            if isinstance(t, str) and len(t) > min_len:
                texts.append(t)
                if max_rows is not None and len(texts) >= max_rows:
                    break
        return texts
    token = hf_token()
    if spec.get("gated") and not token:
        raise RuntimeError(
            f"corpus {corpus!r} ({spec['path']}) is gated and no HF_TOKEN is set; "
            "export a read token as HF_TOKEN and re-run")
    kwargs = {"split": spec.get("split", "train")}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if spec.get("stream"):
        # Multi-GB corpora (nl-wiki) are consumed lazily: every caller caps rows,
        # so streaming avoids downloading the whole config for a 600-row read.
        kwargs["streaming"] = True
    if token:
        kwargs["token"] = token
    ds = load_dataset(spec["path"], **kwargs)
    texts = []
    for row in ds:
        t = row.get("text")
        if isinstance(t, str) and len(t) > min_len:
            texts.append(t)
            if max_rows is not None and len(texts) >= max_rows:
                break
    return texts


def load_corpus_texts_all(corpus: str, *, max_rows: int | None = None) -> list[str]:
    """EVERY ``text`` row of a corpus, verbatim — including empty strings — with
    **no length filter**.

    This mirrors what the upstream KUDA null-space projector actually sees: ROME's
    ``layer_stats`` wraps the raw ``wikitext-2-raw-v1`` *test* split in a
    ``TokenizedDataset`` that tokenises **every** row (an empty row becomes a lone
    BOS token), where the training-data loader (:func:`load_corpus_texts`) drops
    anything at or below ``min_len``. The two paths are deliberately different
    upstream, and the covariance one keeps the empties; excluding them shifts the
    second-moment spectrum (BOS carries real mass) and therefore the null subspace,
    which measurably changes a KUDA run.
    """
    from datasets import load_dataset

    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; known: {sorted(CORPORA)}")
    spec = CORPORA[corpus]
    if corpus_kind(spec) == "local_jsonl":
        texts = []
        for t in _iter_local_jsonl(spec):
            if isinstance(t, str):
                texts.append(t)
                if max_rows is not None and len(texts) >= max_rows:
                    break
        return texts
    token = hf_token()
    if spec.get("gated") and not token:
        raise RuntimeError(
            f"corpus {corpus!r} ({spec['path']}) is gated and no HF_TOKEN is set; "
            "export a read token as HF_TOKEN and re-run")
    if spec.get("stream") and max_rows is None:
        raise ValueError(f"corpus {corpus!r} is streamed (multi-GB) and this is the "
                         f"unfiltered all-rows loader: max_rows is required, or the "
                         f"iteration never terminates in bounded time")
    kwargs = {"split": spec.get("split", "train")}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    if spec.get("revision"):
        kwargs["revision"] = spec["revision"]
    if spec.get("stream"):
        kwargs["streaming"] = True
    if token:
        kwargs["token"] = token
    ds = load_dataset(spec["path"], **kwargs)
    texts = []
    for row in ds:
        t = row.get("text")
        if isinstance(t, str):
            texts.append(t)
            if max_rows is not None and len(texts) >= max_rows:
                break
    return texts


# --------------------------------------------------------------------------- #
# MUSE windows — muse_bench/baselines/dataset.py::DefaultDataset (.txt branch)  #
# --------------------------------------------------------------------------- #
#: The separator ``muse_bench/load_data.py`` writes between rows of a ``raw``
#: split before the official loader tokenizes the file as one string.
MUSE_JOIN = "\n\n"


def chunk_token_ids(tokens, max_len: int, bos_id: int) -> list[list[int]]:
    """The official MUSE windows of one tokenized corpus — a pure function.

    ``tokens`` is the corpus tokenized **without** special tokens. Windows are
    ``tokens[i : i + max_len - 1]`` for ``i`` in ``range(0, len, max_len - 1)``,
    each prefixed with ``bos_id`` (``DefaultDataset.__init__``, ``add_bos_token``
    branch). If the last window is short it is extended with the **first**
    window's tokens (BOS included, exactly as upstream's ``torch.concat([last,
    first])[:max_len]``), so every window is exactly ``max_len`` long. An empty
    corpus yields no windows.
    """
    if max_len < 2:
        raise ValueError(f"max_len must be >= 2 (one BOS + >= 1 token), got {max_len}")
    tokens = [int(t) for t in tokens]
    step = max_len - 1
    windows = [[int(bos_id)] + tokens[i:i + step] for i in range(0, len(tokens), step)]
    if windows and len(windows[-1]) < max_len:
        windows[-1] = (windows[-1] + windows[0])[:max_len]
    return windows


_WINDOW_CACHE: dict = {}


def load_corpus_windows(corpus: str, tokenizer, max_len: int, *,
                        max_rows: int | None = None) -> list[list[int]]:
    """The official MUSE windows (lists of token ids, each exactly ``max_len``
    long, BOS first) of a ``chunk: True`` corpus.

    Mirrors ``load_data.py`` + ``DefaultDataset``: **every** row of the split,
    verbatim and unfiltered (``min_len`` does not apply — the official file is
    the rows joined with ``"\n\n"``), tokenized once with
    ``add_special_tokens=False``, then :func:`chunk_token_ids`. ``max_rows``
    caps the number of *windows* (a smoke-run knob), not the rows. Memoised per
    ``(corpus, tokenizer, max_len)`` within a process, since a KUDA run asks for
    the same retain windows twice (training and the covariance pass).
    """
    if not is_chunked_corpus(corpus):
        raise ValueError(f"corpus {corpus!r} is not a chunked (MUSE) corpus")
    if tokenizer is None:
        raise ValueError(f"corpus {corpus!r} is chunked and needs a tokenizer to window")
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        raise ValueError("the tokenizer has no bos_token_id; MUSE windows are BOS-prefixed")
    key = (corpus, str(getattr(tokenizer, "name_or_path", id(tokenizer))), int(max_len))
    windows = _WINDOW_CACHE.get(key)
    if windows is None:
        rows = load_corpus_texts_all(corpus)
        joined = MUSE_JOIN.join(rows)
        tokens = tokenizer(joined, add_special_tokens=False)["input_ids"]
        windows = chunk_token_ids(tokens, int(max_len), int(bos_id))
        _WINDOW_CACHE[key] = windows
        print(f"[data] {corpus}: {len(rows)} rows -> {len(tokens)} tokens -> "
              f"{len(windows)} windows of {max_len} (official MUSE chunking)", flush=True)
    if max_rows is not None:
        windows = windows[:max_rows]
    return list(windows)


def load_text_batches(corpora, *, batch_size: int, min_len: int = DEFAULT_MIN_LEN,
                      max_num_batches: int | None = None,
                      max_rows: int | None = None, tokenizer=None,
                      chunk_max_len: int | None = None) -> list[list[list]]:
    """One list of batches per corpus, sequentially batched (RMU/KUDA shape —
    the caller tokenizes inside its loop with the per-topic ``max_length``).

    ``corpora`` is a list of corpus names (e.g. ``["bio-forget-corpus",
    "cyber-forget-corpus"]``). Returns ``[[batch0, batch1, ...], ...]`` aligned
    with ``corpora``; each ``batchN`` is a list of ``<= batch_size`` items.
    ``max_num_batches`` caps each topic's batch list (the ``num_batches =
    min(max_num_batches, min over topics)`` convention lives in the driver).

    For a WMDP corpus the items are *raw strings*. For a ``chunk: True`` (MUSE)
    corpus they are the official pre-tokenized windows
    (:func:`load_corpus_windows`), which needs ``tokenizer`` and
    ``chunk_max_len`` (the window length, i.e. the ``max_length`` the caller
    will tokenize at); ``min_len`` is not applied to them. :func:`tokenize_batch`
    consumes either kind.
    """
    out = []
    for corpus in corpora:
        if is_chunked_corpus(corpus):
            if chunk_max_len is None:
                raise ValueError(f"corpus {corpus!r} is chunked: pass chunk_max_len (the "
                                 f"window length) to load_text_batches")
            texts = load_corpus_windows(corpus, tokenizer, chunk_max_len, max_rows=max_rows)
        else:
            texts = load_corpus_texts(corpus, min_len=min_len, max_rows=max_rows)
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        if max_num_batches is not None:
            batches = batches[:max_num_batches]
        out.append(batches)
    return out


def adds_bos(tokenizer) -> bool:
    """Whether ``tokenizer(text)`` (special tokens on, the default) prepends its
    BOS itself -- Gemma / Llama / Mistral do, GPT-2 does not. Probed on a
    one-word string rather than read off a flag, because the flag's name and
    existence differ between tokenizer classes."""
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is None:
        return False
    try:
        ids = tokenizer("a")["input_ids"]
    except Exception:
        return False
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return bool(ids) and int(ids[0]) == int(bos_id)


def strip_redundant_bos(tokenizer, texts) -> list:
    """``texts`` with a leading literal BOS string removed from each one, when
    the tokenizer is about to add its own BOS (:func:`adds_bos`).

    A chat-templated document rendered to text (the Dutch retained JSONL of
    :mod:`rspd.experiments.data.dutch_sets`) can start with a literal
    ``<bos>``; tokenised with special tokens on, it would otherwise begin with
    two BOS ids. Exactly one leading occurrence is stripped -- nothing inside a
    text is touched -- and a tokenizer that adds no BOS (GPT-2), or a text that
    does not start with one, is returned unchanged. The packed retained stream
    (:func:`packed_retain_stream`) tokenises with ``add_special_tokens=False``
    and needs no guard: there the literal is the document's only BOS.
    """
    texts = list(texts)
    bos = getattr(tokenizer, "bos_token", None)
    if not isinstance(bos, str) or not bos or not any(
            isinstance(t, str) and t.startswith(bos) for t in texts):
        return texts
    if not adds_bos(tokenizer):
        return texts
    return [t[len(bos):] if isinstance(t, str) and t.startswith(bos) else t for t in texts]


def tokenize_batch(tokenizer, texts, *, max_length: int, device=None):
    """Tokenise a batch of strings for full-sequence LM training (SimNPO shape):
    ``input_ids`` / ``attention_mask`` / ``labels`` with ``labels`` a copy of
    ``input_ids`` and every PAD position set to ``-100``.

    Masking pad→-100 (rather than the upstream WMDP track's eos-as-pad, which
    counts pads) keeps SimNPO's per-token average ``NLL / |y|`` faithful. A
    text that already starts with a literal BOS (a chat-rendered document) is
    not given a second one (:func:`strip_redundant_bos`).

    ``texts`` may also be **pre-tokenized** sequences (lists of token ids — the
    MUSE windows of :func:`load_corpus_windows`). Those are stacked as they are:
    no special tokens are added (the window already starts with BOS) and they
    are **never truncated** — a window longer than ``max_length`` is a config
    mismatch and raises. Shorter ones are padded on the tokenizer's own
    ``padding_side`` with ``pad_token_id`` (equal-length windows need none).
    """
    import torch  # noqa: F401  (kept local; enc tensors carry the dtype)

    texts = list(texts)
    if texts and _is_pretokenized(texts[0]):
        enc = _stack_pretokenized(tokenizer, texts, max_length=max_length)
    else:
        enc = tokenizer(strip_redundant_bos(tokenizer, texts), return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    enc["labels"] = labels
    if device is not None:
        enc = {k: v.to(device) for k, v in enc.items()}
    return enc


def add_position_ids(enc):
    """Insert ``position_ids`` derived from ``attention_mask`` (``cumsum - 1``,
    clamped at 0 on the pads) into a tokenized batch, in place, and return it.

    For a model with ABSOLUTE position embeddings (GPT-2) a left-padded batch
    is otherwise mis-positioned: ``GPT2Model.forward`` numbers positions
    ``arange(L)`` when none are given, so every real token of a left-padded row
    is shifted by that row's pad count. RMU and KUDA force ``padding_side =
    "left"`` (upstream fidelity), so their GPT-2 configs opt into this
    (``position_ids_from_mask: true``). It is opt-in and off by default so
    rotary-position models keep their upstream numerics (upstream passes no
    position ids either). A batch that already carries ``position_ids`` is
    returned untouched; a batch without an ``attention_mask`` too.
    """
    mask = enc.get("attention_mask") if hasattr(enc, "get") else None
    if mask is None or "position_ids" in enc:
        return enc
    pos = mask.long().cumsum(-1) - 1
    pos.masked_fill_(mask == 0, 0)
    enc["position_ids"] = pos.to(mask.device)
    return enc


def _is_pretokenized(item) -> bool:
    """A list/tuple of ints (or an int tensor) is a token-id sequence; a str is
    text."""
    if isinstance(item, str):
        return False
    if isinstance(item, (list, tuple)):
        return len(item) > 0 and all(isinstance(t, int) and not isinstance(t, bool)
                                     for t in item)
    return hasattr(item, "dim") and hasattr(item, "tolist")      # a torch tensor


def _stack_pretokenized(tokenizer, seqs, *, max_length: int):
    """``{"input_ids", "attention_mask"}`` tensors from token-id sequences: no
    specials added, no truncation (raises), padding on ``tokenizer.padding_side``."""
    import torch

    rows = [list(map(int, s.tolist() if hasattr(s, "tolist") else s)) for s in seqs]
    longest = max(len(r) for r in rows)
    if longest > max_length:
        raise ValueError(f"a pre-tokenized window has {longest} tokens but max_length is "
                         f"{max_length}; windows are never re-truncated — make the "
                         f"window length and the tokenization length agree")
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0) or 0
    side = getattr(tokenizer, "padding_side", "right")
    ids = torch.full((len(rows), longest), int(pad_id), dtype=torch.long)
    mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for i, r in enumerate(rows):
        if side == "left":
            ids[i, longest - len(r):] = torch.tensor(r, dtype=torch.long)
            mask[i, longest - len(r):] = 1
        else:
            ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
            mask[i, :len(r)] = 1
    return {"input_ids": ids, "attention_mask": mask}



# --------------------------------------------------------------------------- #
# Tokenizer                                                                    #
# --------------------------------------------------------------------------- #
def load_tokenizer(tokenizer_id: str, *, padding_side: str | None = None,
                   truncation_side: str | None = None):
    """The tokenizer alone -- e.g. to window a ``chunk: True`` corpus before a
    model load. A Gemma tokenizer loads fast and keeps its own pad; any other
    loads slow (fast when it is a local directory with only ``tokenizer.json``)
    with ``pad = eos`` when it has no pad token; padding / truncation side are
    opt-in (``None`` keeps the tokenizer's own)."""
    from transformers import AutoTokenizer

    for attr, value in (("padding_side", padding_side),
                        ("truncation_side", truncation_side)):
        if value is not None and value not in ("left", "right"):
            raise ValueError(f"{attr} must be 'left', 'right' or None "
                             f"(the tokenizer's own default), got {value!r}")
    is_gemma = "gemma" in tokenizer_id.lower()
    # A local checkpoint that ships only tokenizer.json (the trilingual GPT-2
    # BPE of rspd.experiments.data.gpt2_corpus is a PreTrainedTokenizerFast
    # with no vocab.json/merges.txt) has no slow tokenizer to fall back to;
    # every other non-Gemma model keeps the slow one.
    use_fast = bool(is_gemma or (os.path.isdir(tokenizer_id)
                                 and os.path.exists(os.path.join(tokenizer_id, "tokenizer.json"))))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, use_fast=use_fast, token=hf_token())
    if not is_gemma and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side
    return tokenizer


# --------------------------------------------------------------------------- #
# Fit / held-out split and encoding                                            #
# --------------------------------------------------------------------------- #
def split_fit_heldout(texts, max_rows, heldout_rows):
    """``(fit, heldout)`` from one corpus's rows as loaded: with a row cap, the
    fit rows are the first ``max_rows`` and the held-out rows the (up to)
    ``heldout_rows`` after them; uncapped, the held-out rows are the last
    ``heldout_rows`` and the fit rows everything before them."""
    texts = list(texts)
    if max_rows is not None:
        return texts[:max_rows], texts[max_rows:max_rows + heldout_rows]
    if heldout_rows and len(texts) > heldout_rows:
        return texts[:-heldout_rows], texts[-heldout_rows:]
    return texts, []


def encode(tok, text, max_length):
    """The text's first ``max_length`` tokens (the tokenizer's own special
    tokens, i.e. BOS where it adds one), as a ``[1, T]`` LongTensor. Cut on our
    side, so the kept end is the head whatever the tokenizer's
    ``truncation_side`` (some tokenizers truncate on the left).

    A pre-tokenised window (a list of ints / an int tensor, e.g. an official
    MUSE window from :func:`load_corpus_windows`, BOS already first) is passed
    through as it is -- never re-tokenised, never given a second BOS and never
    truncated: a window longer than ``max_length`` raises, since silently
    cutting it would change the target set."""
    import torch
    if _is_pretokenized(text):
        ids = [int(t) for t in (text.tolist() if hasattr(text, "tolist") else text)]
        if len(ids) > max_length:
            raise ValueError(f"a pre-tokenized window has {len(ids)} tokens but max_length is "
                             f"{max_length}; windows are never re-truncated -- make the "
                             f"window length and max_length agree")
        return torch.tensor([ids], dtype=torch.long)
    ids = tok(strip_redundant_bos(tok, [text])[0])["input_ids"][:max_length]
    return torch.tensor([ids], dtype=torch.long)


def encode_all(tok, texts, max_length):
    """Encoded texts with at least two tokens (a single token has no target)."""
    return [x for x in (encode(tok, t, max_length) for t in texts) if x.shape[-1] >= 2]
