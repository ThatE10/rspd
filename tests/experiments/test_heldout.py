"""The shared held-out CE readout (``rspd.experiments.eval.heldout``): the
``heldout:`` block, head / tail row selection, the block_ce estimator pooled by
tokens, and the delta / summary / result-row helpers. CPU, tiny GPT-2."""
from __future__ import annotations

import math

import pytest

from gpt2_tiny_helpers import corpora_block, tiny_gpt2, tiny_tokenizer, write_sets
from rspd.experiments.data import corpora as CO
from rspd.experiments.eval import heldout as C
from rspd.experiments.eval.ce import block_ce


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(CO, "CORPORA", {k: v for k, v in CO.CORPORA.items()
                                        if CO.corpus_kind(v) != "local_jsonl"})
    write_sets(tmp_path, n_target=3, n_contrastive=6, n_heldout=5, words=20)
    CO.register_config_corpora({"corpora": corpora_block()})
    return tmp_path


# --------------------------------------------------------------------------- #
# the config block                                                             #
# --------------------------------------------------------------------------- #
def test_absent_block_means_no_readout():
    assert C.resolve_heldout_config({}) is None
    assert C.resolve_heldout_config({"corpora": corpora_block()}) is None
    assert C.load_heldout_readout({"corpora": corpora_block()}, tiny_tokenizer()) is None


def test_block_resolves_with_defaults_and_tail_in_corpus_order():
    H = C.resolve_heldout_config({"corpora": corpora_block(),
                                  "heldout": {"tail": ["heldout_deu", "heldout_eng"]}})
    assert H == {"rows": 32, "max_length": 1024, "min_len": 0,
                 "corpora": ["heldout_nld", "heldout_eng", "heldout_deu"],
                 "tail": ["heldout_eng", "heldout_deu"]}


@pytest.mark.parametrize("block, match", [
    ({"rows": 0}, "heldout.rows"),
    ({"rows": True}, "heldout.rows"),
    ({"max_length": 1}, "heldout.max_length"),
    ({"tail": ["contrastive"]}, "not `topic: heldout`"),
    ({"window": 3}, "unknown keys"),
    ("yes", "must be a mapping"),
])
def test_bad_blocks_are_refused(block, match):
    with pytest.raises(ValueError, match=match):
        C.resolve_heldout_config({"corpora": corpora_block(), "heldout": block})


def test_block_without_heldout_corpora_is_refused():
    block = {k: v for k, v in corpora_block().items() if v["topic"] != "heldout"}
    with pytest.raises(ValueError, match="no `corpora:` entry has `topic: heldout`"):
        C.resolve_heldout_config({"corpora": block, "heldout": {"rows": 4}})


# --------------------------------------------------------------------------- #
# row selection and the estimator                                              #
# --------------------------------------------------------------------------- #
def test_head_and_tail_rows_follow_the_target_builder(env):
    all_rows = CO.load_corpus_texts("heldout_nld", min_len=0)
    assert len(all_rows) == 5
    assert C.heldout_rows("heldout_nld", rows=2) == all_rows[:2]
    assert C.heldout_rows("heldout_nld", rows=2, tail=True) == all_rows[-2:]
    assert C.heldout_rows("heldout_nld", rows=2, tail=True) == \
        C.split_fit_heldout(all_rows, None, 2)[1]                # target.load_heldout_retain
    # a corpus shorter than `rows` is read whole, either way
    assert C.heldout_rows("heldout_nld", rows=9) == all_rows
    assert C.heldout_rows("heldout_nld", rows=9, tail=True) == all_rows


def test_readout_is_block_ce_pooled_deterministic_and_restores_train_mode(env):
    tok, model = tiny_tokenizer(), tiny_gpt2()
    ro = C.HeldoutReadout(tok, ["heldout_nld", "heldout_eng"], rows=2, max_length=16,
                          tail=["heldout_eng"])
    assert ro.n_rows == {"heldout_nld": 2, "heldout_eng": 2}
    assert all(x.shape[-1] == 16 for v in ro.ids.values() for x in v)
    model.train()
    a = ro.score(model)
    assert model.training                                   # restored
    b = ro.score(model)
    assert a == b and set(a) == {"heldout_nld", "heldout_eng"}
    assert all(math.isfinite(v) and v > 0 for v in a.values())
    # == token-weighted mean of block_ce over the rows
    model.eval()
    for c, ids in ro.ids.items():
        tot = n = 0
        for x in ids:
            ce, k = block_ce(model, x)
            tot += ce * k
            n += k
        assert a[c] == pytest.approx(tot / n, rel=1e-6)
    # the one-shot form and the config-driven form give the same numbers
    one = C.heldout_readout(model, tok, {"heldout_nld": {}, "heldout_eng": {}}, rows=2,
                            max_length=16, tail=["heldout_eng"])
    assert one == a
    cfg = {"corpora": corpora_block(), "heldout": {"rows": 2, "max_length": 16,
                                                   "tail": ["heldout_eng", "heldout_deu"]}}
    full = C.load_heldout_readout(cfg, tok).score(model)
    assert {k: full[k] for k in a} == a
    assert ro.spec["n_tokens"] == {"heldout_nld": 32, "heldout_eng": 32}
    # head vs tail rows differ, so the choice is load-bearing
    head = C.HeldoutReadout(tok, ["heldout_eng"], rows=2, max_length=16).score(model)
    assert head["heldout_eng"] != a["heldout_eng"]


def test_delta_summary_and_metrics_helpers():
    base, edited = {"a": 1.0, "b": 2.0}, {"a": 1.5, "b": 1.0}
    assert C.heldout_delta(base, edited) == {"a": 0.5, "b": -1.0}
    s = C.heldout_summary(base, edited)
    assert s["heldout/a/base"] == 1.0 and s["heldout/a/edited"] == 1.5 and s["heldout/b/delta"] == -1.0
    assert C.heldout_summary(None, edited) == {}
    off = C.heldout_metrics(None, None, None)
    assert off == {"heldout": None, "ce_heldout_base": None, "ce_heldout": None,
                   "ce_heldout_delta": None}


def test_result_rows_carry_ce_base_and_delta(env):
    tok = tiny_tokenizer()
    ro = C.HeldoutReadout(tok, ["heldout_nld"], rows=2, max_length=16)
    rows = C.heldout_rows_results({"heldout_nld": 3.0}, {"heldout_nld": 3.5}, model="m",
                                  readout=ro)
    got = {(r.task, r.metric): r for r in rows}
    assert got[("heldout_ce/heldout_nld", "ce")].value == 3.5
    assert got[("heldout_ce/heldout_nld", "ce_base")].value == 3.0
    assert got[("heldout_ce/heldout_nld", "ce_delta")].value == pytest.approx(0.5)
    assert got[("heldout_ce/heldout_nld", "ce")].n == 32 and rows[0].tp is None
    assert C.heldout_rows_results(None, {"a": 1.0}) == []
