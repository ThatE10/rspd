"""The corpus registry and its loaders (``srspd_experiments.data.corpora``):
registry hygiene (every entry a namespaced hub id with the keys the loader
reads), the official MUSE windows, ``tokenize_batch`` on windows and strings,
pinned MUSE revisions, the config-level ``local_jsonl`` corpora, and
left-padding position ids. CPU, no network (``load_dataset`` is monkeypatched)."""
from __future__ import annotations

import json

import pytest
import torch

from gpt2_tiny_helpers import corpora_block, tiny_gpt2, tiny_tokenizer, write_sets
from srspd_experiments.data import corpora as common


@pytest.fixture
def fresh_corpora(monkeypatch):
    """Registration mutates the module-level registry; give each test its own copy."""
    monkeypatch.setattr(common, "CORPORA", dict(common.CORPORA))
    return common.CORPORA


@pytest.mark.parametrize("name", sorted(common.CORPORA))
def test_every_corpus_path_is_a_namespaced_repo_id(name):
    """``namespace/name`` — bare ids resolve in the browser but not in the client.

    huggingface_hub raises ``HfUriError: Repository id must be 'namespace/name'``
    for a bare alias like ``wikitext``, so the canonical owner must be spelled
    out even when the hub would redirect.
    """
    path = common.CORPORA[name]["path"]
    owner, sep, repo = path.partition("/")
    assert sep and owner and repo, (
        f"corpus {name!r} has path {path!r}; hub ids must be 'namespace/name'"
    )
    assert "/" not in repo, f"corpus {name!r} path {path!r} has too many segments"


@pytest.mark.parametrize("name", sorted(common.CORPORA))
def test_every_corpus_declares_the_keys_the_loader_reads(name):
    spec = common.CORPORA[name]
    assert spec.keys() >= {"path", "topic", "gated"}, (
        f"corpus {name!r} is missing a key the loader reads: {sorted(spec)}"
    )
    assert spec["topic"] in {"bio", "cyber", "retain", "nl", "muse-news", "muse-books"}
    assert isinstance(spec["gated"], bool)


def test_only_muse_corpora_are_chunked_and_they_say_their_side():
    """``chunk`` selects the official MUSE window loader and ``role`` the side of
    the run a split may sit on; a WMDP corpus has neither and keeps encoding its
    side in ``topic == "retain"``."""
    chunked = {n for n, s in common.CORPORA.items() if s.get("chunk")}
    assert chunked == {f"muse-{c}-{s}" for c in ("news", "books")
                       for s in ("forget", "retain1", "retain2", "holdout")}
    for n in chunked:
        spec = common.CORPORA[n]
        assert spec["config"] == "raw" and spec["split"] == n.split("-")[-1]
        assert spec["role"] == ("forget" if n.endswith("forget") else "retain")
        assert common.is_chunked_corpus(n)
    assert common.is_retain_corpus("muse-news-retain1")
    assert common.is_retain_corpus("muse-books-holdout")
    assert not common.is_retain_corpus("muse-books-forget")
    # unchanged for the WMDP registry
    assert common.is_retain_corpus("wikitext") and not common.is_retain_corpus("bio-forget-corpus")
    assert not common.is_chunked_corpus("wikitext")
    with pytest.raises(ValueError, match="unknown corpus"):
        common.is_retain_corpus("no-such-corpus")


# --------------------------------------------------------------------------- #
# MUSE windows — muse_bench/baselines/dataset.py::DefaultDataset, .txt branch  #
# --------------------------------------------------------------------------- #
def test_chunk_token_ids_is_the_official_bos_prefixed_window_cut():
    """Windows of ``max_len - 1`` tokens, each prefixed with BOS; the last one,
    when short, is extended with the FIRST window (BOS included) and cut to
    ``max_len`` — ``torch.concat([last, first])[:max_len]`` upstream."""
    BOS = 99
    assert common.chunk_token_ids(list(range(1, 10)), 4, BOS) == [
        [BOS, 1, 2, 3], [BOS, 4, 5, 6], [BOS, 7, 8, 9]]
    assert common.chunk_token_ids(list(range(1, 11)), 4, BOS) == [
        [BOS, 1, 2, 3], [BOS, 4, 5, 6], [BOS, 7, 8, 9], [BOS, 10, BOS, 1]]
    # one short window rotates onto itself
    assert common.chunk_token_ids([5, 6], 4, BOS) == [[BOS, 5, 6, BOS]]
    assert common.chunk_token_ids([], 4, BOS) == []
    for w in common.chunk_token_ids(list(range(1000)), 17, BOS):
        assert len(w) == 17 and w[0] == BOS
    with pytest.raises(ValueError):
        common.chunk_token_ids([1, 2], 1, BOS)


class _ToyTokenizer:
    """Whitespace tokenizer: id = 1 + int(word); BOS 0, PAD 7, right padding."""
    bos_token_id = 0
    pad_token_id = 7
    eos_token_id = 8
    padding_side = "right"
    name_or_path = "toy"

    def __call__(self, text, add_special_tokens=True, **kw):
        assert add_special_tokens is False, "MUSE windows tokenize WITHOUT specials"
        return {"input_ids": [1 + int(w) for w in text.split()]}


def test_load_corpus_windows_joins_every_row_and_ignores_min_len(monkeypatch):
    """The official file is ``"\n\n".join(rows)`` — every row, no length
    filter — tokenized once; the join itself contributes no tokens here, so the
    windows are a plain cut of the concatenated rows."""
    rows = ["0 1 2", "", "3 4 5 6"]              # the empty row is kept (upstream does)
    seen = {}

    def fake_all(corpus, *, max_rows=None):
        seen["corpus"] = corpus
        return rows
    monkeypatch.setattr(common, "load_corpus_texts_all", fake_all)
    common._WINDOW_CACHE.clear()
    w = common.load_corpus_windows("muse-books-forget", _ToyTokenizer(), 4)
    assert seen["corpus"] == "muse-books-forget"
    assert w == [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 0, 1]]
    # max_rows caps WINDOWS, not rows
    assert common.load_corpus_windows("muse-books-forget", _ToyTokenizer(), 4,
                                      max_rows=2) == w[:2]
    with pytest.raises(ValueError, match="not a chunked"):
        common.load_corpus_windows("wikitext", _ToyTokenizer(), 4)
    with pytest.raises(ValueError, match="needs a tokenizer"):
        common.load_corpus_windows("muse-books-forget", None, 4)
    common._WINDOW_CACHE.clear()


def test_load_text_batches_yields_windows_for_a_chunked_corpus_and_strings_otherwise(
        monkeypatch):
    monkeypatch.setattr(common, "load_corpus_texts_all",
                        lambda corpus, *, max_rows=None: ["0 1 2 3 4 5"])
    monkeypatch.setattr(common, "load_corpus_texts",
                        lambda corpus, *, min_len, max_rows=None: ["a", "b", "c"])
    common._WINDOW_CACHE.clear()
    out = common.load_text_batches(["muse-news-forget", "wikitext"], batch_size=2,
                                   tokenizer=_ToyTokenizer(), chunk_max_len=4)
    assert out[0] == [[[0, 1, 2, 3], [0, 4, 5, 6]]]
    assert out[1] == [["a", "b"], ["c"]]
    with pytest.raises(ValueError, match="chunk_max_len"):
        common.load_text_batches(["muse-news-forget"], batch_size=2,
                                 tokenizer=_ToyTokenizer())
    common._WINDOW_CACHE.clear()


def test_only_the_bio_forget_corpus_is_gated():
    """The gated flag drives token plumbing; drift here fails only in-container."""
    gated = {n for n, s in common.CORPORA.items() if s["gated"]}
    assert gated == {"bio-forget-corpus"}


# --------------------------------------------------------------------------- #
# tokenize_batch on pre-tokenized windows                                      #
# --------------------------------------------------------------------------- #
def test_tokenize_batch_stacks_windows_without_specials_or_truncation():
    """Windows go in as they are: no second BOS, equal-length windows need no
    pad, a window longer than ``max_length`` is a config mismatch (raises)."""
    tok = _ToyTokenizer()
    enc = common.tokenize_batch(tok, [[0, 1, 2, 3], [0, 4, 5, 6]], max_length=4)
    assert enc["input_ids"].tolist() == [[0, 1, 2, 3], [0, 4, 5, 6]]
    assert enc["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1]]
    assert enc["labels"].tolist() == enc["input_ids"].tolist()
    with pytest.raises(ValueError, match="never re-truncated"):
        common.tokenize_batch(tok, [[0, 1, 2, 3, 4]], max_length=4)


def test_tokenize_batch_pads_short_windows_on_the_tokenizer_side_and_masks_labels():
    tok = _ToyTokenizer()
    enc = common.tokenize_batch(tok, [[0, 1, 2], [0, 4]], max_length=4)
    assert enc["input_ids"].tolist() == [[0, 1, 2], [0, 4, 7]]
    assert enc["labels"].tolist() == [[0, 1, 2], [0, 4, -100]]
    tok.padding_side = "left"
    enc = common.tokenize_batch(tok, [[0, 1, 2], [0, 4]], max_length=4)
    assert enc["input_ids"].tolist() == [[0, 1, 2], [7, 0, 4]]
    assert enc["attention_mask"].tolist() == [[1, 1, 1], [0, 1, 1]]
    tok.padding_side = "right"


def test_tokenize_batch_still_calls_the_tokenizer_for_strings():
    """The WMDP path is untouched: strings go through the tokenizer with
    padding + truncation at ``max_length``."""
    calls = {}

    class _Tok:
        def __call__(self, texts, **kw):
            calls.update(kw, texts=texts)
            return {"input_ids": torch.tensor([[1, 2, 3]]),
                    "attention_mask": torch.tensor([[1, 1, 0]])}
    enc = common.tokenize_batch(_Tok(), ["hello"], max_length=3)
    assert calls["texts"] == ["hello"] and calls["truncation"] is True
    assert calls["padding"] is True and calls["max_length"] == 3
    assert enc["labels"].tolist() == [[1, 2, -100]]


def test_muse_corpora_pin_a_revision_and_the_loaders_forward_it(monkeypatch):
    """X (the forget windows) and the retain2 readout must come from the same
    dataset snapshot as the pinned SRSPD retain1 bucket."""
    revs = {"muse-bench/MUSE-News": "506bd5b150b92814d45e4404a82f120ab2d748bf",
            "muse-bench/MUSE-Books": "051ba90319e920d410d87cfdbd61f25843c1b892"}
    muse = {n: s for n, s in common.CORPORA.items() if n.startswith("muse-")}
    assert len(muse) == 8
    for n, spec in muse.items():
        assert spec["revision"] == revs[spec["path"]], n
    seen = {}

    def fake_load_dataset(path, config=None, **kw):
        seen.update(path=path, config=config, **kw)
        return [{"text": "a b"}]

    import datasets as _ds
    monkeypatch.setattr(_ds, "load_dataset", fake_load_dataset)
    common.load_corpus_texts_all("muse-books-forget")
    assert seen["revision"] == revs["muse-bench/MUSE-Books"] and seen["split"] == "forget"


def test_register_config_corpora_adds_local_jsonl_entries(fresh_corpora):
    names = common.register_config_corpora({"corpora": corpora_block()})
    assert names == ["target_nld", "contrastive", "heldout_nld", "heldout_eng", "heldout_deu"]
    spec = common.CORPORA["contrastive"]
    assert spec["kind"] == "local_jsonl" and spec["topic"] == "retain" and spec["gated"] is False
    assert spec["path"] == ["datasets/gpt2_unlearning/gpt2_unlearning_1m/contrastive_eng.jsonl",
                            "datasets/gpt2_unlearning/gpt2_unlearning_1m/contrastive_deu.jsonl"]
    assert spec["text_field"] == "text"
    assert common.CORPORA["target_nld"]["path"] == [
        "datasets/gpt2_unlearning/gpt2_unlearning_1m/target_nld.jsonl"]
    # idempotent; no block is a no-op
    assert common.register_config_corpora({"corpora": corpora_block()}) == names
    assert common.register_config_corpora({"name": "x"}) == []
    assert common.register_config_corpora(None) == []


@pytest.mark.parametrize("mutate,match", [
    (lambda b: b["target_nld"].update(kind="hub"), "kind"),
    (lambda b: b["target_nld"].pop("path"), "path"),
    (lambda b: b["target_nld"].update(path=[]), "path"),
    (lambda b: b["target_nld"].update(topic="forget"), "topic"),
    (lambda b: b["target_nld"].update(revision="abc"), "unknown keys"),
    (lambda b: b.update(wikitext=dict(b["target_nld"])), "collides"),
])
def test_register_config_corpora_rejects_bad_specs(fresh_corpora, mutate, match):
    block = corpora_block()
    mutate(block)
    with pytest.raises(ValueError, match=match):
        common.register_config_corpora({"corpora": block})


def test_a_different_spec_under_the_same_name_is_refused(fresh_corpora):
    common.register_config_corpora({"corpora": corpora_block()})
    other = corpora_block()
    other["target_nld"]["topic"] = "heldout"
    with pytest.raises(ValueError, match="collides"):
        common.register_config_corpora({"corpora": other})


def test_local_jsonl_loaders_read_in_order_filter_and_cap(fresh_corpora, tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    write_sets(tmp_path, n_target=3, n_contrastive=4, n_heldout=2, words=12)
    common.register_config_corpora({"corpora": corpora_block()})
    texts = common.load_corpus_texts("target_nld", min_len=0)
    assert len(texts) == 3 and all(isinstance(t, str) for t in texts)
    # the list path concatenates eng then deu, in file order
    both = common.load_corpus_texts("contrastive", min_len=0)
    eng = common.load_corpus_texts_all("contrastive", max_rows=4)
    assert len(both) == 8 and both[:4] == eng
    assert common.load_corpus_texts("contrastive", min_len=0, max_rows=5)[:4] == eng
    # min_len filters by characters exactly as for the hub corpora
    assert common.load_corpus_texts("target_nld", min_len=10 ** 6) == []
    assert common.load_corpus_texts_all("target_nld") == texts
    # a missing file fails loudly, naming the path
    missing = {"corpora": {"nope": {"kind": "local_jsonl", "path": "datasets/x/nope.jsonl",
                                    "topic": "target"}}}
    common.register_config_corpora(missing)
    with pytest.raises(FileNotFoundError, match="nope.jsonl"):
        common.load_corpus_texts("nope")
    # a row without the text field too
    bad = tmp_path / "datasets" / "gpt2_unlearning" / "gpt2_unlearning_1m" / "bad.jsonl"
    bad.write_text(json.dumps({"id": 1}) + "\n")
    common.register_config_corpora({"corpora": {"bad": {
        "kind": "local_jsonl", "path": "datasets/gpt2_unlearning/gpt2_unlearning_1m/bad.jsonl",
        "topic": "target"}}})
    with pytest.raises(KeyError, match="text"):
        common.load_corpus_texts("bad")
    # load_text_batches goes through the same loader
    batches = common.load_text_batches(["contrastive"], batch_size=3, min_len=0)[0]
    assert [len(b) for b in batches] == [3, 3, 2]


def test_text_field_is_configurable(fresh_corpora, tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    p = tmp_path / "datasets" / "c.jsonl"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"content": "hello world"}) + "\n\n" + json.dumps({"content": "x"}) + "\n")
    common.register_config_corpora({"corpora": {"c": {"kind": "local_jsonl",
                                                       "path": "datasets/c.jsonl",
                                                       "text_field": "content",
                                                       "topic": "retain"}}})
    assert common.load_corpus_texts("c", min_len=0) == ["hello world", "x"]
    assert common.load_corpus_texts("c", min_len=5) == ["hello world"]
    assert common.resolve_local_path("/abs/x.jsonl") == "/abs/x.jsonl"


# --------------------------------------------------------------------------- #
# position ids under left padding                                              #
# --------------------------------------------------------------------------- #
def test_add_position_ids_counts_real_tokens_from_zero():
    mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    enc = {"input_ids": torch.zeros_like(mask), "attention_mask": mask}
    out = common.add_position_ids(enc)
    assert out is enc
    assert enc["position_ids"].tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]
    # idempotent, and a batch without a mask is left alone
    common.add_position_ids(enc)
    assert enc["position_ids"].tolist() == [[0, 0, 0, 1, 2], [0, 1, 2, 3, 4]]
    assert "position_ids" not in common.add_position_ids({"input_ids": mask})


def test_position_ids_make_a_left_padded_gpt2_forward_match_the_unpadded_one():
    """The reason the GPT-2 configs opt in: absolute embeddings mis-position a
    left-padded row unless the positions are supplied."""
    m, tok = tiny_gpt2(), tiny_tokenizer()
    tok.padding_side = "left"
    enc = tok(["w1 w2 w3 w4 w5", "w6 w7"], return_tensors="pt", padding=True)
    short = tok(["w6 w7"], return_tensors="pt")
    with torch.no_grad():
        ref = m(**short).logits[0]
        naive = m(**enc).logits[1, -2:]
        fixed = m(**common.add_position_ids(dict(enc))).logits[1, -2:]
    assert not torch.allclose(naive, ref, atol=1e-5)
    assert torch.allclose(fixed, ref, atol=1e-5)
