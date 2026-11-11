"""The shared held-out CE readout (:mod:`srspd_experiments.eval.heldout`:
``resolve_heldout_config`` / ``HeldoutReadout`` / ``heldout_readout``) and its
wiring into the four GPT-2 runners: the row selection follows the SRSPD runner's
convention (first ``rows`` rows per corpus, the LAST ``rows`` rows for the
``tail`` corpora = its retain-side readout), the estimator is
:func:`srspd_experiments.eval.ce.block_ce`, the readout is deterministic, and the
same tiny checkpoint over the same local sets scores identically through the
SimNPO, RMU, KUDA and SRSPD runners. CPU, no network."""
from __future__ import annotations

import importlib

import json
import math

import pytest

from conftest import isolate_corpora  # noqa: E402
import torch
import yaml

from gpt2_tiny_helpers import corpora_block, save_checkpoint, tiny_gpt2, tiny_tokenizer, write_sets

from srspd_experiments.methods import _common as C  # noqa: E402
from recipe_configs import legacy  # noqa: E402

#: the GPT-2 configs the readout is wired through (recipe runs, see recipe_configs)
STEMS = {"gpt2_unlearn_simnpo": "gpt2_simnpo", "gpt2_unlearn_rmu": "gpt2_rmu",
         "gpt2_unlearn_kuda": "gpt2_kuda"}
HELDOUT = {"rows": 2, "max_length": 16, "tail": ["heldout_eng", "heldout_deu"]}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    # hub corpora only: a local_jsonl entry another test registered (the 1M or the
    # 2M sets under the same names) would collide with this test's config
    isolate_corpora(monkeypatch, drop_local=True)
    save_checkpoint(tmp_path)
    write_sets(tmp_path, n_target=3, n_contrastive=6, n_heldout=5, words=20)
    C.register_config_corpora({"corpora": corpora_block()})
    return tmp_path


def _cfg(stem, name, **top):
    cfg = legacy(STEMS[stem])
    cfg.update(name=name, device_map="cpu", wandb={"enabled": False}, heldout=dict(HELDOUT))
    cfg["corpora"] = corpora_block()
    cfg.update(top)
    return cfg


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
def test_head_and_tail_rows_follow_hf_rspd(env):
    all_rows = C.load_corpus_texts("heldout_nld", min_len=0)
    assert len(all_rows) == 5
    assert C.heldout_rows("heldout_nld", rows=2) == all_rows[:2]
    assert C.heldout_rows("heldout_nld", rows=2, tail=True) == all_rows[-2:]
    assert C.heldout_rows("heldout_nld", rows=2, tail=True) == \
        C.split_fit_heldout(all_rows, None, 2)[1]                # hf_rspd.load_heldout_retain
    # a corpus shorter than `rows` is read whole, either way
    assert C.heldout_rows("heldout_nld", rows=9) == all_rows
    assert C.heldout_rows("heldout_nld", rows=9, tail=True) == all_rows


def test_readout_is_block_ce_pooled_deterministic_and_restores_train_mode(env):
    from srspd_experiments.eval.ce import block_ce
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


# --------------------------------------------------------------------------- #
# the four drivers score the same checkpoint the same way                     #
# --------------------------------------------------------------------------- #
def _reference(env):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ck = env / "models" / "recipes" / "gpt2_trilingual_base"
    tok = AutoTokenizer.from_pretrained(str(ck))
    model = AutoModelForCausalLM.from_pretrained(str(ck)).eval()
    return C.heldout_readout(model, tok, ["heldout_nld", "heldout_eng", "heldout_deu"],
                             rows=2, max_length=16, tail=["heldout_eng", "heldout_deu"])


def _metrics(env, name):
    return json.loads((env / "logs" / "methods" / name / "metrics.json").read_text())


def _check(m, ref):
    for k in ("ce_heldout_base", "ce_heldout", "ce_heldout_delta"):
        assert set(m[k]) == {"heldout_nld", "heldout_eng", "heldout_deu"}, k
    assert m["ce_heldout_base"] == pytest.approx(ref, rel=1e-6)
    for c in ref:
        assert m["ce_heldout_delta"][c] == pytest.approx(m["ce_heldout"][c] - m["ce_heldout_base"][c])
    assert m["heldout"]["rows"] == 2 and m["heldout"]["max_length"] == 16
    assert m["heldout"]["tail"] == ["heldout_eng", "heldout_deu"]
    assert m["heldout"]["n_rows"] == {"heldout_nld": 2, "heldout_eng": 2, "heldout_deu": 2}


def test_simnpo_rmu_and_kuda_report_the_shared_readout(env, tmp_path):
    ref = _reference(env)

    simnpo = importlib.import_module("srspd_experiments.methods.simnpo")
    cfg = _cfg("gpt2_unlearn_simnpo", "ho_simnpo")
    cfg["train"].update(max_steps=2, batch_size=2, grad_accum=1, log_every=1)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m = simnpo.run(str(tmp_path / "s.yaml"), no_push=True)
    _check(m, ref)
    _check(_metrics(env, "ho_simnpo"), ref)
    assert any(m["ce_heldout_delta"][c] != 0.0 for c in ref)      # the edit moved the model

    rmu = importlib.import_module("srspd_experiments.methods.rmu")
    cfg = _cfg("gpt2_unlearn_rmu", "ho_rmu")
    cfg["rmu"].update(layer_id=2, layer_ids=[1, 2], max_num_batches=2, batch_size=2, epochs=1,
                      steering_coeffs=[4.0])
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m = rmu.run(str(tmp_path / "r.yaml"), push_to_hub=False)
    _check(m, ref)
    _check(_metrics(env, "ho_rmu"), ref)

    kuda = importlib.import_module("srspd_experiments.methods.kuda")
    cfg = _cfg("gpt2_unlearn_kuda", "ho_kuda")
    cfg["kuda"].update(layer_id=2, layer_ids=[1, 2], max_num_batches=2, epoch=1, batch_size=2,
                       tau=1e-2)
    cfg["kuda"]["nullspace"].update(max_length=16, batch_size=4, cov_dtype="float64")
    (tmp_path / "k.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    m = kuda.run(str(tmp_path / "k.yaml"), no_push=True)
    _check(m, ref)
    _check(_metrics(env, "ho_kuda"), ref)


def test_srspd_baseline_readout_matches_the_shared_one(env, tmp_path):
    from recipe_configs import config
    ref = _reference(env)
    srspd_runner = importlib.import_module("srspd_experiments.methods.srspd")
    cfg = config("gpt2_trilingual_recovery", "srspd_sweep")
    cfg.update(name="ho_srspd", device_map="cpu")
    d = cfg["data"]
    d["corpora"] = corpora_block()
    d["target"].update(max_tokens=30, max_length=16, heldout_rows=2)
    d["retain_max_length"] = 16
    d["retained"]["chunk_tokens"] = 64
    d["retained"]["buckets"] = [dict(b, tokens=64) for b in d["retained"]["buckets"]]
    cfg["srspd"].update(mu_per_token=[0.5], gram_budget_gib=2 ** 20 / 2 ** 30, export=False)
    m = srspd_runner.run(cfg)
    got = m["baseline_ce_heldout"]
    assert got["forget/heldout_nld"] == pytest.approx(ref["heldout_nld"], rel=1e-6)
    assert got["retain/heldout_eng"] == pytest.approx(ref["heldout_eng"], rel=1e-6)
    assert got["retain/heldout_deu"] == pytest.approx(ref["heldout_deu"], rel=1e-6)
    (ent,) = m["exports"].values()
    assert ent["label"] == "srspd_mpt0.5" and set(ent["ce_heldout_delta"]) == set(got)


# --------------------------------------------------------------------------- #
# the KUDA covariance cache keys local corpora by content                      #
# --------------------------------------------------------------------------- #
def test_kuda_cov_key_moves_with_the_local_retain_files(env):
    kuda = importlib.import_module("srspd_experiments.methods.kuda")
    cfg = _cfg("gpt2_unlearn_kuda", "ho_kuda_key")
    R = kuda.resolve_kuda_config(cfg)
    k1 = kuda._cov_key(R, 3)
    assert k1 == kuda._cov_key(kuda.resolve_kuda_config(cfg), 3)          # stable
    # same corpus NAME, different file content (a rebuilt contrastive set)
    p = env / "datasets" / "gpt2_unlearning" / "gpt2_unlearning_1m" / "contrastive_eng.jsonl"
    p.write_text(p.read_text() + json.dumps({"text": "w3 w4 w5 w6"}) + "\n")
    assert kuda._cov_key(R, 3) != k1
    # hub corpora carry no file fingerprint (the Gemma / Zephyr keys are untouched)
    assert kuda._local_corpora_fingerprint(["wikitext"]) == {}
