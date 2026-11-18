"""The declarative data spec and the SRSPD data path (``data.spec`` +
``data.target.build_erasure_data``) on the tiny local unlearning sets: spec
validation, fit / held-out row selection, the exact token budget, the retained
build and its document-disjoint held-out twin, the readout keys, and the ids
going straight into ``rspd.erase``. CPU, no network."""
from __future__ import annotations

import pytest
import torch

from gpt2_tiny_helpers import corpora_block, tiny_gpt2, tiny_tokenizer, write_sets
from rspd.experiments.data import DataSpec, build_erasure_data
from rspd.experiments.data import corpora as C
from rspd.experiments.data import target as T

SETS = "datasets/gpt2_unlearning/gpt2_unlearning_1m"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(C, "CORPORA", {k: v for k, v in C.CORPORA.items()
                                       if C.corpus_kind(v) != "local_jsonl"})
    write_sets(tmp_path, n_target=4, n_contrastive=30, n_heldout=12, words=40)
    return tmp_path


def _spec(**over):
    d = {"corpora": corpora_block(),
         "target": {"corpora": ["target_nld"], "max_rows": 3, "min_len": 0, "max_length": 32,
                    "heldout_rows": 1},
         "retained": {"chunk_tokens": 16, "seed": 3, "add_bos": True, "buckets": [
             {"name": "eng", "kind": "local_jsonl", "path": f"{SETS}/contrastive_eng.jsonl",
              "text_field": "text", "tokens": 160},
             {"name": "deu", "kind": "local_jsonl", "path": f"{SETS}/contrastive_deu.jsonl",
              "text_field": "text", "tokens": 160}]},
         "heldout_retain_corpora": ["heldout_eng"], "retain_max_length": 24,
         "heldout_mix_tokens": 32}
    d.update(over)
    return d


def test_defaults_are_the_published_ones(env):
    d = _spec(target={"corpora": ["target_nld"], "max_rows": 3})
    d.pop("retain_max_length")
    d.pop("heldout_mix_tokens")
    d["retained"] = dict(d["retained"], chunk_tokens=1024)
    s = DataSpec.from_dict(d)
    assert (s.target.min_len, s.target.max_length, s.target.heldout_rows) == (50, 768, 8)
    assert (s.retain_max_length, s.heldout_mix_tokens) == (512, 2048)
    s = DataSpec.from_dict(_spec())
    assert s.retained.build_kw == {"chunk_tokens": 16, "seed": 3, "add_bos": True,
                                   "chat_template": None}
    assert DataSpec.from_dict({k: v for k, v in _spec().items()
                               if k != "heldout_retain_corpora"}).heldout_retain_corpora == \
        ("wikitext",)
    d = s.as_dict()
    assert d["retained"]["buckets"][0]["kind"] == "local_jsonl"


@pytest.mark.parametrize("over, match", [
    ({"target": {"corpora": ["target_nld"]}}, "max_rows"),
    ({"target": {"corpora": ["nope"], "max_rows": 1}}, "unknown corpus"),
    ({"target": {"corpora": ["target_nld"], "max_rows": 0}}, "max_rows"),
    ({"target": {"corpora": ["target_nld"], "max_rows": 1, "window": 3}}, "unknown keys"),
    ({"target": {"corpora": ["nl-wiki"], "max_tokens": 100}}, "streamed"),
    ({"target": {"corpora": ["muse-news-forget"], "max_rows": 3}}, "chunked"),
    ({"target": {"corpora": ["muse-news-forget", "wikitext"]}}, "mixes"),
    ({"heldout_retain_corpora": ["target_nld"]}, "both a target"),
    ({"heldout_mix_tokens": 20}, "multiple of"),
    ({"retained": {"buckets": []}}, "at least one bucket"),
    ({"retained": {"buckets": [{"name": "x"}], "chunk": 3}}, "unknown keys"),
    ({"mu_per_token": 1.0}, "unknown keys"),
])
def test_bad_specs_are_refused(env, over, match):
    with pytest.raises(ValueError, match=match):
        DataSpec.from_dict(_spec(**over))


def test_build_erasure_data_reproduces_the_srspd_data_path(env):
    tok = tiny_tokenizer()
    spec = DataSpec.from_dict(_spec())
    data = build_erasure_data(tok, spec, log=lambda *_: None)
    rows = C.load_corpus_texts("target_nld", min_len=0)
    # X: the first max_rows rows, each its first max_length tokens
    assert len(data.target_ids) == 3
    assert [x.tolist()[0] for x in data.target_ids] == \
        [tok(r)["input_ids"][:32] for r in rows[:3]]
    assert data.n_target_tokens == 3 * 32
    # Y: the retained build, exactly what load_or_build returns for the spec
    from rspd.experiments.data import retained_corpus as rc
    ids, man = rc.load_or_build(rc.shared_cache_path(tok, spec.retained.spec,
                                                     **spec.retained.build_kw),
                                tok, spec.retained.spec, **spec.retained.build_kw)
    assert torch.equal(data.retained_ids, ids) and data.n_retained_tokens == 320
    assert data.manifest["retained"]["manifest"]["retained_corpus_sha256"] == \
        man["retained_corpus_sha256"]
    # readouts: the row after the fit rows, the LAST heldout_rows retain rows, the mix twin
    assert set(data.heldout_ids) == {"forget/target_nld", "retain/heldout_eng", "mix/eng",
                                     "mix/deu"}
    assert data.heldout_ids["forget/target_nld"][0].tolist()[0] == tok(rows[3])["input_ids"][:32]
    last_eng = C.load_corpus_texts("heldout_eng", min_len=0)[-1]
    assert data.heldout_ids["retain/heldout_eng"][0].tolist()[0] == \
        tok(last_eng)["input_ids"][:24]
    assert [x.shape for x in data.heldout_ids["mix/eng"]] == [torch.Size([1, 16])] * 2
    twin = data.manifest["heldout_mix_manifest"]["buckets"]
    for b, hb in zip(man["buckets"], twin):
        assert hb["doc_index_start"] > b["doc_index_end"]
    assert data.manifest["target"]["n_rows"] == {"target_nld": 3}
    assert len(data.manifest["target"]["fit_sha256"]) == 64


def test_max_tokens_cuts_x_to_exactly_the_budget(env):
    tok = tiny_tokenizer()
    s = _spec(target={"corpora": ["target_nld"], "max_rows": 3, "max_tokens": 50,
                      "min_len": 0, "max_length": 32, "heldout_rows": 1})
    data = build_erasure_data(tok, DataSpec.from_dict(s), log=lambda *_: None)
    assert data.n_target_tokens == 50 and [x.shape[-1] for x in data.target_ids] == [32, 18]


def test_heldout_from_names_the_forget_side_readout(env):
    tok = tiny_tokenizer()
    s = _spec(target={"corpora": ["target_nld"], "max_rows": 4, "min_len": 0,
                      "max_length": 32, "heldout_rows": 2, "heldout_from": ["heldout_nld"]})
    data = build_erasure_data(tok, DataSpec.from_dict(s), log=lambda *_: None)
    assert len(data.target_ids) == 4 and "forget/target_nld" not in data.heldout_ids
    assert len(data.heldout_ids["forget/heldout_nld"]) == 2


def test_windowed_target_takes_every_window_but_the_last(env, monkeypatch):
    calls = []

    def fake_windows(corpus, tokenizer, max_len, *, max_rows=None):
        calls.append((corpus, max_len))
        return [[0] + [5 + i] * (max_len - 1) for i in range(5)]
    monkeypatch.setattr(C, "load_corpus_windows", fake_windows)
    s = _spec(target={"corpora": ["muse-news-forget"], "max_length": 8, "heldout_rows": 2},
              heldout_retain_corpora=["heldout_eng"])
    data = build_erasure_data(tiny_tokenizer(), DataSpec.from_dict(s), log=lambda *_: None)
    assert calls[0] == ("muse-news-forget", 8)
    assert [x.tolist()[0][1] for x in data.target_ids] == [5, 6, 7]
    assert len(data.heldout_ids["forget/muse-news-forget"]) == 2
    assert data.manifest["target"]["windowed"] is True


def test_cut_and_chunk_helpers():
    ids = [torch.ones(1, 5, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)]
    assert T.cut_to_token_budget(ids, 6)[1] == 5         # a 1-token tail has no target
    assert T.cut_to_token_budget(ids, 7)[1] == 7
    man = {"chunk_tokens": 4, "n_chunks": 5,
           "buckets": [{"name": "a", "tokens_taken": 10}, {"name": "b", "tokens_taken": 10}]}
    assert T.bucket_chunks(man) == {"a": [0, 1], "b": [3, 4]}
    assert T.round_robin_chunks({"a": [0, 1], "b": [3, 4]}, 12, 4) == [0, 3, 1]
    with pytest.raises(RuntimeError, match="nothing would be left"):
        T.split_windows([[1], [2]], 2)
    assert T.texts_sha256(["a"]) != T.texts_sha256([[1]])


def test_the_ids_go_straight_into_srspd_erase(env):
    import rspd
    tok = tiny_tokenizer()
    data = build_erasure_data(tok, DataSpec.from_dict(_spec()), log=lambda *_: None)
    model = tiny_gpt2()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    rep = rspd.erase(model, target=data.target_ids, retained=data.retained_ids,
                      mu_per_token=[1.0], layers="gpt2")
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert changed and rep.matrices
    rspd.restore(model)
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())
