"""Usability fixes of issue #13: the model-owned edit lock, the full ``gamma``
spectrum on a report, the CLI's solve-only path, the descriptive leak-energy
names, the ``export_checkpoint`` re-export, the block order of
``report.matrices`` and the single non-PD summary warning."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import rspd  # noqa: E402
import tiny_models as tm  # noqa: E402
from rspd.models.harness import key_layer  # noqa: E402
from rspd.pipeline import NON_PD_MESSAGE  # noqa: E402

LAYERS = {"types": ["mlp.c_fc", "mlp.c_proj"], "blocks": [1, 2]}


def _params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _bitwise(m, before):
    return all(torch.equal(before[n], p) for n, p in m.named_parameters())


def _sets():
    return tm.token_sets()


def test_older_open_report_cannot_stack_a_second_edit():
    """The scenario of the issue: an older still-open report applies past the
    refusal, a fresh erase starts from edited weights, restore returns []."""
    model = tm.gpt2()
    target, retained = _sets()
    pristine = _params(model)
    old = rspd.api.erase(model, target=target, retained=retained, layers=LAYERS,
                      mu_per_token=[0.3, 1.0], apply=False)        # still open, unapplied
    new = rspd.api.erase(model, target=target, retained=retained, layers=LAYERS,
                      mu_per_token=1.0)                              # holds the model
    assert new.applied
    edited = _params(model)
    with pytest.raises(RuntimeError, match=r"held by EditReport #\d+ .*applied mu_per_token "
                                           r"1\.0.*not by this report"):
        old.apply_dose(1.0)
    assert _bitwise(model, edited) and not old.applied             # nothing was stacked
    with pytest.raises(RuntimeError, match=r"carries an applied erasure held by EditReport"):
        rspd.api.erase(model, target=target, retained=retained, layers=LAYERS, mu_per_token=1.0)
    names = rspd.api.restore(model)                                     # the holder's edit
    assert names == sorted(new.matrices) and not new.applied
    assert _bitwise(model, pristine)
    old.apply_dose(0.3)                                              # lock released: allowed
    assert old.applied
    with pytest.raises(RuntimeError, match="held by EditReport"):
        new.apply_dose(1.0)
    assert sorted(rspd.api.restore(model)) == sorted(old.matrices) and not old.applied
    assert _bitwise(model, pristine) and rspd.api.restore(model) == []
    fresh = rspd.api.erase(model, target=target, retained=retained, layers=LAYERS,
                        mu_per_token=1.0)                            # from pristine weights
    assert _bitwise(model, edited)
    fresh.close()
    old.close()
    new.close()
    assert _bitwise(model, pristine) and rspd.api.restore(model) == []


def test_full_spectrum_feeds_suggest_mu_per_token():
    model = tm.gpt2()
    target, retained = _sets()
    top = rspd.api.erase(model, target=target, retained=retained, layers=LAYERS,
                      mu_per_token=1.0, apply=False)
    key = next(iter(top.specificity))
    assert "spectrum" not in top.specificity[key]["specificity"]        # default unchanged
    with pytest.raises(ValueError, match="spectrum='full'"):
        top.spectrum(key)
    top.close()
    rep = rspd.api.erase(model, target=target, retained=retained, layers=LAYERS,
                      mu_per_token=1.0, apply=False, spectrum="full")
    spec = rep.specificity[key]["specificity"]
    assert len(spec["spectrum"]) == spec["n"] and spec["spectrum"][:8] == spec["top"]
    grid = rspd.core.suggest_mu_per_token(spec, fractions=(0.5,), quantiles=(0.5,))
    name = next(n for n, r in rep.matrices.items() if r["input_key"] == key)
    assert grid == rspd.core.suggest_mu_per_token(rep.spectrum(name), fractions=(0.5,),
                                              quantiles=(0.5,))
    back = json.loads(rep.to_json())["specificity"][key]["specificity"]
    assert rspd.core.suggest_mu_per_token(back, fractions=(0.5,), quantiles=(0.5,)) == grid
    rep.close()
    with pytest.raises(ValueError, match="readout"):
        rspd.api.erase(model, target=target, retained=retained, mu_per_token=1.0,
                    solver="grams", spectrum="full")
    with pytest.raises(ValueError, match="spectrum must be"):
        rspd.api.erase(model, target=target, retained=retained, mu_per_token=1.0,
                    spectrum="all")


def test_cli_no_apply_solves_every_dose(capsys, tmp_path):
    pytest.importorskip("safetensors")
    from rspd.cli import main
    ckpt = tmp_path / "ckpt"
    tm.mistral().save_pretrained(ckpt)
    tm.word_tokenizer().save_pretrained(ckpt)
    tgt, ret = tmp_path / "t.txt", tmp_path / "r.txt"
    tgt.write_text("\n".join(" ".join(f"w{(i * 5 + j) % 60}" for j in range(20))
                             for i in range(3)))
    ret.write_text("\n".join(" ".join(f"w{60 + (i * 3 + j) % 60}" for j in range(30))
                             for i in range(6)))
    base = ["erase", str(ckpt), "--target", str(tgt), "--retained", str(ret),
            "--mu-per-token", "0.3", "1", "--dtype", "float32"]

    def run(argv):
        code = main(argv)
        return code, json.loads(capsys.readouterr().out)
    code, out = run(base)
    assert code == 2 and "--apply-mu-per-token" in out["error"] and "--no-apply" in out["error"]
    assert "apply=False" not in out["error"]
    code, out = run(base + ["--no-apply", "--out", str(tmp_path / "x")])
    assert code == 2 and "--out" in out["error"]
    code, rep = run(base + ["--no-apply"])
    assert code == 0 and not rep["applied"] and rep["dose"] is None
    assert set(rep["by_dose"]) == {"0.3", "1.0"}


def test_descriptive_energy_names_alias_the_deprecated_keys():
    model = tm.gpt2()
    target, retained = _sets()
    rep = rspd.api.erase(model, target=target, retained=retained, layers=LAYERS,
                      mu_per_token=1.0, rank_rule=0.1, apply_rank_rule=None)
    for rows in (rep.matrices, *[rep._session.state.lam_info[0]]):
        for r in rows.values():
            assert r["target_surviving_rel"] == r["erased_rel"]          # same value, bitwise
            assert r["retained_disturbed_rel"] == r["retained_rel"]
            assert r["target_surviving_rel"] == pytest.approx(r["L_X_rel"] ** 2)
    rep.close()
    W = torch.randn(5, 4, dtype=torch.float64)
    X, Y = torch.randn(12, 4, dtype=torch.float64), torch.randn(30, 4, dtype=torch.float64)
    G_X, G_Y = X.T @ X, Y.T @ Y
    comp = rspd.core.erased_component(W, X, Y, mu=1.0)
    for lk in (rspd.core.leaks(W, comp, X, Y), rspd.core.leaks_gram(W, comp, X, G_Y),
               rspd.core.leaks_grams(W, rspd.core.erased_component_grams(W, G_X, G_Y, mu=1.0),
                                 G_X, G_Y)):
        assert lk["target_surviving_rel"] == lk["erased_rel"]
        assert lk["retained_disturbed_rel"] == lk["retained_rel"]


def test_export_checkpoint_is_re_exported():
    import rspd.models
    from rspd.models.export import export_checkpoint
    assert rspd.export_checkpoint is export_checkpoint
    assert rspd.models.export_checkpoint is export_checkpoint
    assert "export_checkpoint" in rspd.__all__ and "export_checkpoint" in rspd.models.__all__


def test_matrices_are_in_block_then_forward_order():
    model = tm.gpt2()
    target, retained = _sets()
    rep = rspd.api.erase(model, target=target, retained=retained, mu_per_token=1.0)
    names = list(rep.matrices)
    order = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]
    assert names == [f"L{b:02d}.{t}" for b in range(3) for t in order]
    assert [key_layer(n) for n in names] == sorted(key_layer(n) for n in names)
    for ent in rep.by_dose.values():
        assert list(ent["leaks"]) == names
    rep.close()


def test_non_pd_solves_give_one_summary_warning():
    model = tm.gpt2()                    # LayerNorm-fed c_attn / c_fc: G_mu rank-deficient
    target, retained = _sets()
    events = []
    rep = rspd.api.erase(model, target=target, retained=retained, mu_per_token=[0.3, 1.0],
                      apply_mu_per_token=1.0, progress=lambda e, i: events.append((e, i)))
    msgs = [w for w in rep.warnings if NON_PD_MESSAGE in w]
    assert len(msgs) == 1 and len(rep.non_pd) > 1
    assert sum(e == "warning" and NON_PD_MESSAGE in i["message"] for e, i in events) == 1
    paths = {}
    for r in rep.non_pd:
        paths[r["solve_path"]] = paths.get(r["solve_path"], 0) + 1
        assert set(r) == {"matrix", "input_key", "mu_per_token", "solve_path", "ridge_delta",
                          "requested_ridge_delta"}
    for p, c in paths.items():
        assert f"{p}: {c}" in msgs[0]
    hit = [r for r in rep.non_pd if r["mu_per_token"] == 1.0]
    for r in hit:
        assert rep.matrices[r["matrix"]]["solve_path"] == r["solve_path"]
    assert json.loads(rep.to_json())["non_pd"] == rep.non_pd
    rep.close()
