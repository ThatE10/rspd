"""``rspd.api.erase`` / ``rspd.api.restore`` / ``EditReport`` behaviour on tiny models:
dose selection, re-dosing without a second pass, rank truncation, scratch
spills, refusals, and the pipeline helpers."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import rspd  # noqa: E402
import tiny_models as tm  # noqa: E402
from rspd import pipeline as pl  # noqa: E402
from rspd.api import as_sequences  # noqa: E402


def _params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _same(m, before):
    return all(torch.equal(before[n], p) for n, p in m.named_parameters())


def test_input_forms():
    a = as_sequences(torch.arange(5))
    b = as_sequences(torch.arange(6).reshape(2, 3))
    c = as_sequences([[1, 2], torch.tensor([[3, 4, 5]])])
    d = as_sequences([7, 8, 9])
    assert [s.shape for s in a] == [(1, 5)] and [s.shape for s in b] == [(1, 3), (1, 3)]
    assert [s.tolist() for s in c] == [[[1, 2]], [[3, 4, 5]]] and d[0].tolist() == [[7, 8, 9]]
    tok = tm.word_tokenizer()
    e = as_sequences(["w1 w2 w3", "w4"], tok)
    assert [s.shape[-1] for s in e] == [3, 1]
    with pytest.raises(TypeError, match="tokenizer"):
        as_sequences("w1 w2")
    with pytest.raises(ValueError, match="empty"):
        as_sequences([[]])
    with pytest.raises(TypeError, match="integers"):
        as_sequences(torch.randn(4))


def test_dose_grid_apply_dose_and_report_round_trip(tmp_path):
    m = tm.mistral()
    before = _params(m)
    tgt, ret = tm.token_sets()
    with pytest.raises(ValueError, match="apply_mu_per_token"):
        rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=[0.1, 10.0])
    assert _same(m, before)
    events = []
    rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=[0.1, 10.0], apply=False,
                      progress=lambda e, i: events.append(e))
    assert not rep.applied and rep.dose is None and _same(m, before)
    assert {"plan", "pass_start", "solve", "pass_done"} <= set(events)
    lx = {k: v["leaks_by_type"]["q_proj"]["L_X_rel"]["median"] for k, v in rep.by_dose.items()}
    assert lx[repr(10.0)] < lx[repr(0.1)]              # a larger dose erases more
    rep.apply_dose(10.0)
    w10 = _params(m)
    assert rep.applied and rep.dose["mu_per_token"] == 10.0 and not _same(m, before)
    rep.apply_dose(0.1)                                  # restores, then re-applies
    rep.apply_dose(10.0)
    assert _same(m, w10)
    with pytest.raises(ValueError, match="was not solved"):
        rep.apply_dose(3.0)
    with pytest.raises(RuntimeError, match="restore"):
        rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0)
    p = tmp_path / "r.json"
    rep.to_json(p)
    d = json.loads(p.read_text())
    assert d["dose"]["mu_per_token"] == 10.0 and "_session" not in d
    assert set(d["dose"]) == {"mu_per_token", "mu_raw", "mu", "n_x", "n_y", "formula"}
    assert rspd.api.restore(m) and _same(m, before) and not rep.applied
    assert rspd.api.restore(m) == []
    rep.close()
    with pytest.raises(RuntimeError, match="no live solve"):
        rep.apply_dose(10.0)


def test_rank_rule_applies_the_truncated_component():
    m = tm.mistral()
    before = _params(m)
    tgt, ret = tm.token_sets()
    full = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0)
    w_full = _params(m)
    rspd.api.restore(m)
    for rule in ("erank", 0.5):
        rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0, rank_rule=rule)
        assert rep.rank_rule == rule
        for n, r in rep.matrices.items():
            assert 1 <= r["rank"] <= r["rank_bound"], n
            assert r["L_X_rel"] >= full.matrices[n]["L_X_rel"] - 1e-9, n   # truncation erases less
        trunc = rep.by_dose[repr(1.0)]["truncated"][str(rule)]
        assert set(trunc["leaks"]) == set(rep.matrices)
        assert not _same(m, w_full) or all(r["rank"] == r["rank_bound"]
                                           for r in rep.matrices.values())
        rspd.api.restore(m)
        assert _same(m, before)
    with pytest.raises(ValueError, match="readout"):
        rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0, rank_rule=0.5, solver="grams")
    with pytest.raises(ValueError, match="lam_rel"):
        rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0, rank_rule=1.5)


@pytest.mark.parametrize("solver", ["readout", "grams"])
def test_scratch_dir_spills_to_disk_and_matches_ram(tmp_path, solver):
    tgt, ret = tm.token_sets()
    a, b = tm.mistral(), tm.mistral()
    ra = rspd.api.erase(a, target=tgt, retained=ret, mu_per_token=2.0, solver=solver)
    rb = rspd.api.erase(b, target=tgt, retained=ret, mu_per_token=2.0, solver=solver,
                     scratch_dir=tmp_path / "scratch")
    sd = rb.stats["scratch_dir"]
    assert sd and any((tmp_path / "scratch").iterdir())
    assert _same(a, dict(b.named_parameters()))          # bitwise: the spill is lossless
    rb.close()
    assert not (tmp_path / "scratch" / sd.split("/")[-1]).exists()
    ra.close()


def test_multipass_plan_equals_single_pass():
    tgt, ret = tm.token_sets()
    a, b = tm.llama(), tm.llama()
    ra = rspd.api.erase(a, target=tgt, retained=ret, mu_per_token=1.0)
    rb = rspd.api.erase(b, target=tgt, retained=ret, mu_per_token=1.0,
                     gram_budget_gib=8 * 32 * 32 / 2 ** 30)
    assert len(ra.pass_plan) == 1 and len(rb.pass_plan) > 2 and rb.stats["resume_from_cache"]
    assert _same(a, dict(b.named_parameters()))


def test_kv_shared_model_disables_resume_and_warns():
    tgt, ret = tm.token_sets()
    m = tm.gemma4_text()
    rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0,
                      gram_budget_gib=8 * 32 * 32 / 2 ** 30)
    assert len(rep.pass_plan) > 1 and rep.stats["resume_from_cache"] is False
    assert any("kv_shared_layers" in w for w in rep.warnings)
    rep.close()


def test_selector_restricts_the_edit():
    m = tm.qwen3()
    before = _params(m)
    tgt, ret = tm.token_sets()
    rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0,
                      layers={"types": ["down_proj"], "blocks": [1]})
    assert list(rep.matrices) == ["L01.mlp.down_proj"]
    changed = [n for n, p in m.named_parameters() if not torch.equal(before[n], p)]
    assert changed == ["model.layers.1.mlp.down_proj.weight"]
    rspd.api.restore(m)


def test_pipeline_helpers():
    assert pl.solve_key_order(["a", "b", "c"], {"a": 30, "b": 10, "c": 10}) == ["b", "c", "a"]
    D, G = torch.randn(7, 5, dtype=torch.float64), torch.randn(5, 5, dtype=torch.float64)
    G = G @ G.T
    assert pl.quad_trace(D, G, block=2) == pytest.approx(float(torch.trace(D @ G @ D.T)))
    s = pl.summarize_by_type({"L00.mlp.up_proj": {"x": 1.0}, "L01.mlp.up_proj": {"x": 3.0},
                              "L00.self_attn.q_proj": {"x": 2.0}}, ("x",))
    assert s["up_proj"] == {"n": 2, "x": {"median": 2.0, "min": 1.0, "max": 3.0}}
    assert pl.check_scratch_space("/", 1, free=10) == 10
    with pytest.raises(RuntimeError, match="scratch_dir"):
        pl.check_scratch_space("/", 100, free=10)
    assert pl.scratch_bytes_needed({"a": 10}, n_lam=1, elem_bytes=2, margin=1.0) == 40
    assert pl.rank_rule_spec("erank") == "erank" and pl.rank_rule_spec(0.25) == 0.25


def test_grams_solver_matches_the_readout_solver_on_bf16_weights():
    """The two solvers compute the same ``M`` in float64 by different arithmetic;
    after the fp32 edit is cast to bf16 a few entries may land one bf16 ulp
    apart, none further."""
    tgt, ret = tm.token_sets()
    a, b = tm.mistral(dtype=torch.bfloat16), tm.mistral(dtype=torch.bfloat16)
    ra = rspd.api.erase(a, target=tgt, retained=ret, mu_per_token=[0.5, 5.0],
                     apply_mu_per_token=5.0)
    rb = rspd.api.erase(b, target=tgt, retained=ret, mu_per_token=[0.5, 5.0],
                     apply_mu_per_token=5.0, solver="grams")
    assert rb.solver == "grams" and rb.input_keys["L00.down_in"]["gram_bytes"] == 16 * 40 * 40
    assert ra.edit["delta_fro_sum"] == pytest.approx(rb.edit["delta_fro_sum"], rel=1e-3)
    pb = dict(b.named_parameters())
    for n, p in a.named_parameters():
        x, y = p.detach().float(), pb[n].detach().float()
        assert (x - y).abs().max() <= 2 ** -7 * x.abs().max() + 1e-6, n
        assert float((x != y).float().mean()) < 0.05, n
    for key in rb.by_dose:
        for mat, lk in rb.by_dose[key]["leaks"].items():
            for f in ("L_X_rel", "L_Y_rel"):
                assert lk[f] == pytest.approx(ra.by_dose[key]["leaks"][mat][f], rel=1e-3,
                                              abs=1e-6), (key, mat, f)
