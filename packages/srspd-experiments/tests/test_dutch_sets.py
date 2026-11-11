"""``srspd_experiments.data.dutch_sets``: the target cut to exactly
N tokens (decode round trip, character-prefix fallback), the retained JSONL that
reproduces the build's ids under ``add_special_tokens=False`` with one BOS per
document, the probe subset, and an end-to-end run on a fake stream. CPU, no
network."""
from __future__ import annotations

import json

import pytest
import yaml

from gemma_tiny_helpers import BOS, gemma_like_tokenizer
from srspd_experiments.data import budget as B
from srspd_experiments.data import corpora as C
from srspd_experiments.data import dutch_sets as P
from srspd_experiments.data import retained_corpus as rcorp
from srspd_experiments.data import spec_path


def _words(n, start=0):
    return " ".join(f"w{(start + i) % 300}" for i in range(n))


class TestCut:
    def test_decode_cut_is_exact_and_round_trips(self):
        tok = gemma_like_tokenizer()
        cut = P.cut_text_to_tokens(tok, _words(50), 20)
        assert cut["method"] == "decode" and cut["round_trip"]
        assert P.encode_plain(tok, cut["text"]) == P.encode_plain(tok, _words(50))[:20]
        assert len(P.encode_plain(tok, cut["text"])) == 20

    def test_char_prefix_fallback_when_decode_does_not_round_trip(self):
        tok = gemma_like_tokenizer()
        real_decode = tok.decode

        class Lossy:                 # decode appends junk that encodes to an extra token
            def __getattr__(self, k):
                return getattr(tok, k)

            def __call__(self, *a, **kw):
                return tok(*a, **kw)

            def decode(self, ids, **kw):
                return real_decode(ids, **kw) + " w7"
        cut = P.cut_text_to_tokens(Lossy(), _words(50), 20)
        assert cut["method"] == "char_prefix" and cut["round_trip"]
        assert len(P.encode_plain(tok, cut["text"])) == 20
        assert _words(50).startswith(cut["text"])

    def test_too_short_is_an_error(self):
        with pytest.raises(ValueError, match="fewer than"):
            P.cut_text_to_tokens(gemma_like_tokenizer(), _words(5), 20)


def test_probe_subset_takes_whole_leading_rows():
    assert P.probe_subset(list("abcd"), [10, 10, 10, 10], 0.25) == 1
    assert P.probe_subset(list("abcd"), [10, 10, 10, 10], 0.3) == 2
    assert P.probe_subset(list("abcd"), [10, 10, 10, 10], 1.0) == 4
    with pytest.raises(ValueError):
        P.probe_subset(list("ab"), [1, 1], 0.0)


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("SRSPD_DATA_DIR", str(root))
    d = root / "datasets" / "src"
    d.mkdir(parents=True)
    (d / "target.jsonl").write_text(json.dumps({"text": _words(60), "title": "Atlantische Oceaan"})
                                    + "\n")
    (d / "held.jsonl").write_text(json.dumps({"text": _words(9)}) + "\n")
    tok = gemma_like_tokenizer()

    def fake_iter(bucket):
        if bucket.get("format") == "chat_messages":
            for i in range(50):
                yield i, {"messages": [{"role": "user", "content": _words(4, i)},
                                       {"role": "assistant", "content": _words(5, i + 1)}]}
        else:
            for i in range(50):
                yield i, _words(13, i)
    monkeypatch.setattr(rcorp, "iter_documents", fake_iter)
    monkeypatch.setattr(C, "CORPORA", dict(C.CORPORA))
    cfg = {"name": "prep_test", "tokenizer": "fake/gemma-like", "out": "datasets/g4",
           "probe_fraction": 0.2,
           "target": {"source": "datasets/src/target.jsonl", "tokens": 25},
           "heldout": ["datasets/src/held.jsonl"],
           "retained_corpus": {"seed": 1, "chunk_tokens": 16, "add_bos": True,
                               "chat_template": None, "buckets": [
               {"name": "plain", "dataset": "fake/a", "config": None, "split": "train",
                "revision": "r1", "text_field": "text", "tokens": 100},
               {"name": "chat", "dataset": "fake/b", "config": None, "split": "test",
                "revision": "r2", "text_field": "messages", "format": "chat_messages",
                "tokens": 90}]}}
    return root, tok, cfg


def test_end_to_end(tmp_path, monkeypatch):
    root, tok, cfg = _setup(tmp_path, monkeypatch)
    m = P.prepare(cfg, tokenizer=tok)
    out = root / "datasets" / "g4"
    # target: exactly 25 tokens without special tokens, one BOS with them
    trow = json.loads((out / "target_nld_1k.jsonl").read_text())
    assert len(P.encode_plain(tok, trow["text"])) == 25 and trow["title"] == "Atlantische Oceaan"
    assert m["target"]["tokens"] == 25 and m["target"]["tokens_with_special"] == 26
    # retained: the JSONL reproduces the build's ids exactly, bucket by bucket
    ids, man = rcorp.build_retained_tokens(tok, rcorp.resolve_spec(
        cfg["retained_corpus"]["buckets"]), chunk_tokens=16, seed=1)
    flat = []
    for b in ("plain", "chat"):
        rows = [json.loads(l) for l in (out / "retained" / f"{b}.jsonl").read_text().splitlines()]
        for r in rows:
            ids_r = P.encode_plain(tok, r["text"])
            assert ids_r[0] == BOS and ids_r[1] != BOS          # one BOS per document
            flat.extend(ids_r)
        rep = next(x for x in m["retained"]["buckets"] if x["name"] == b)
        assert rep["n_docs_token_mismatch"] == 0
        assert rep["jsonl_doc_tokens"] == rep["tokens_taken"]
        assert rep["n_docs_start_with_bos"] == rep["n_docs"]
    assert m["retained"]["buckets"][1]["n_docs_bos_already_present"] > 0
    assert flat[:ids.numel()] == ids.reshape(-1).tolist()
    assert m["retained"]["jsonl_doc_tokens_total"] == 190
    # the baselines' packed stream counts exactly that many document tokens
    C.register_config_corpora({"corpora": {"g4_retained": {
        "kind": "local_jsonl", "path": m["retained"]["files"], "text_field": "text",
        "topic": "retain"}}})
    s = B.packed_retain_stream(tok, "g4_retained", block=16, sep_token_id=tok.eos_token_id)
    assert s["n_doc_tokens"] == m["retained"]["jsonl_doc_tokens_total"]
    # probe: a leading subset of whole rows
    for b in m["retained"]["buckets"]:
        full = (root / b["path"]).read_text().splitlines()
        probe = (root / b["probe"]["path"]).read_text().splitlines()
        assert probe == full[:len(probe)] and 0 < len(probe) < len(full)
    # the shared token cache is the SRSPD build
    import torch
    blob = torch.load(m["retained"]["shared_cache"])
    assert torch.equal(blob["ids"], ids)
    assert m["heldout"][0]["rows"] == 1 and m["heldout"][0]["tokens"] == 9
    assert (root / "logs" / "data" / "prep_test" / "results.md").exists()
    assert json.loads((out / "manifest.json").read_text())["retained"]["corpus_manifest"][
        "retained_corpus_sha256"] == man["retained_corpus_sha256"]


def test_shipped_config_resolves_with_the_tokenizer_as_a_parameter():
    with open(spec_path("dutch_sets_1k")) as f:
        cfg = yaml.safe_load(f)
    assert "tokenizer" not in cfg
    with pytest.raises(ValueError, match="tokenizer"):
        P.resolve_config(cfg)
    R = P.resolve_config(cfg, tokenizer="org/some-tokenizer")
    assert R["target"]["tokens"] == 1000 and R["tokenizer"] == "org/some-tokenizer"
    assert R["probe_fraction"] == 0.1 and R["build_kw"]["seed"] == 20260812
    names = [b["name"] for b in R["spec"]]
    assert names == ["en_prose", "de_web", "code", "de_wiki", "chat"]
    assert [b["tokens"] for b in R["spec"]] == [200000, 400000, 200000, 200000, 150000]
    chat = R["spec"][-1]
    assert chat["format"] == "chat_messages" and chat["tokens"] == 150000
    assert chat["revision"] == "8049631c405ae6576f93f445c6b8166f76f5505a"
