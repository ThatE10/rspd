"""The recipe catalogue, the legacy dose translation, ``reproduce`` plans,
``compare`` and the ``rspd-exp`` command line. No network, no model."""
from __future__ import annotations

import csv
import json
import os
import socket

import pytest

from rspd.experiments import cli
from rspd.experiments.recipes import catalog as C
from rspd.experiments.recipes import legacy as L
from rspd.experiments.recipes.compare import compare
from rspd.experiments.results import ResultRow, write_results

RESULTS = ["gemma4_31b_dutch_gmmlu", "gemma4_e4b_dutch_gmmlu", "gpt2_trilingual_recovery",
           "muse_books", "muse_news", "zephyr_wmdp"]


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access during a dry run")
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)


def test_the_shipped_results_are_exactly_the_published_ones():
    assert C.list_results() == RESULTS


@pytest.mark.parametrize("result", RESULTS)
def test_every_recipe_loads_and_is_internally_consistent(result):
    R = C.load_recipe(result)
    assert R["model"] and R["tensor_parallel_size"] >= 1 and R["eval"]["kind"] in C.EVAL_KINDS
    ids = [e["id"] for _r, e in C.iter_exports(R)]
    assert len(ids) == len(set(ids))
    for run, e in C.iter_exports(R):
        assert e["published"], f"{e['id']} has no published number to compare against"
        if run is None:
            continue
        if run["method"] == "srspd":
            assert e["mu_per_token"] > 0
            assert e["checkpoint_subdir"].startswith(f"srspd_mpt{e['mu_per_token']:g}")
            s = run["config"]["srspd"]
            assert e["mu_per_token"] in s["mu_per_token"]
            assert L.LEGACY_DOSE_KEY not in s and L.LEGACY_DOSE_KEY not in run["config"]
        if run["method"] in C.METHOD_BLOCK:
            assert C.METHOD_BLOCK[run["method"]] in run["config"]
            for k in ("push_to_hub", "wandb", "hub_repo"):
                assert k not in run["config"]


@pytest.mark.parametrize("result", RESULTS)
def test_the_eval_tp_is_the_published_rows_tp(result):
    R = C.load_recipe(result)
    if R["eval"]["kind"] == "lm_eval":
        from rspd.experiments.eval.lm_eval import tensor_parallel_size
        assert tensor_parallel_size(R["eval"]["config"]) == R["tensor_parallel_size"]
    assert R["tensor_parallel_size"] == (2 if result == "gemma4_31b_dutch_gmmlu" else 1)


@pytest.mark.parametrize("result", RESULTS)
@pytest.mark.parametrize("mode", C.MODES)
def test_dry_run_resolves_every_recipe_without_network(result, mode, no_network, capsys):
    assert cli.main(["reproduce", result, "--mode", mode, "--dry-run"]) == 0
    p = json.loads(capsys.readouterr().out)
    R = C.load_recipe(result)
    n_exports = sum(len(r["exports"]) for r in R["runs"])
    evals = [s for s in p["steps"] if s["step"] == "eval"]
    per_export = 1 + len(R["eval"].get("extra") or [])
    if mode == "full":
        assert len([s for s in p["steps"] if s["step"] == "train"]) == len(R["runs"])
        assert len(evals) == (n_exports + 1) * per_export
    else:
        skips = [s for s in p["steps"] if s["step"] == "skip"]
        assert sorted(s["export"] for s in skips) == sorted(C.weight_gaps(R))
        assert len(evals) == (n_exports - len(skips) + 1) * per_export
    for s in evals:
        key = "checkpoint" if s["kind"] == "recovery" else "model_path"
        assert s["config"][key]


def test_legacy_dose_translation_is_the_one_formula():
    assert L.mu_per_token_from_legacy(10, 1000, 999424) == 10 * 1000 / 999424
    with pytest.raises(ValueError):
        L.mu_per_token_from_legacy(0, 1, 1)
    with pytest.raises(ValueError):
        L.mu_per_token_from_legacy(1, 0, 1)
    R = C.load_recipe("gpt2_trilingual_recovery")
    _run, e = C.find_export(R, "srspd_mu10")
    assert e["mu_per_token"] == pytest.approx(0.0100058, rel=1e-5)
    assert e["checkpoint_subdir"] == "srspd_mpt0.0100058"


def test_legacy_config_translation_keeps_the_data_path():
    legacy = {"name": "x", "model": "m", "dtype": "float32",
              "corpora": {"t": {"kind": "local_jsonl", "path": "datasets/t.jsonl",
                                "topic": "target"}},
              "rs" + "pd": {"forget_corpora": ["t"], "max_forget_tokens": 100, "min_len": 0,
                            "max_length": 64, "heldout_rows": 3, "heldout_mix_tokens": 0,
                            "retained_corpus": {"chunk_tokens": 32, "seed": 1, "buckets": [
                                {"name": "b", "kind": "local_jsonl", "path": "datasets/t.jsonl",
                                 "tokens": 64}]},
                            L.LEGACY_DOSE_KEY: [2.0], "solver": "grams", "ridge_rel": 1e-10,
                            "gram": {"budget_gib_per_device": 4, "resume_from_cache": False},
                            "skip_" + "circ" + "uits": True, "rho_retain_tokens": 7}}
    out = L.srspd_config_from_legacy(legacy, n_x=100, n_y=400)
    assert out["srspd"]["mu_per_token"] == [0.5] and out["srspd"]["solver"] == "grams"
    assert out["srspd"]["gram_budget_gib"] == 4.0 and out["srspd"]["resume_from_cache"] is False
    t = out["data"]["target"]
    assert (t["corpora"], t["max_tokens"], t["min_len"], t["max_length"], t["heldout_rows"]) == \
        (["t"], 100, 0, 64, 3)
    assert out["data"]["retained"]["chunk_tokens"] == 32 and out["data"]["heldout_mix_tokens"] == 0
    with pytest.raises(ValueError, match="n_x and n_y"):
        L.srspd_config_from_legacy(legacy)
    b = L.baseline_config_from_legacy({"name": "a", "model": "m", "push_to_hub": True,
                                       "wandb": {"enabled": True}, "kuda": {}}, name="b")
    assert b == {"name": "b", "model": "m", "kuda": {}}


def test_a_recipe_dose_given_twice_is_refused(tmp_path):
    import yaml
    R = yaml.safe_load(open(C.recipe_path("gpt2_trilingual_recovery")))
    e = [r for r in R["runs"] if r["method"] == "srspd"][0]["exports"][0]
    e["mu_per_token"] = 0.01
    p = tmp_path / "gpt2_trilingual_recovery.yaml"
    p.write_text(yaml.safe_dump(R))
    with pytest.raises(C.RecipeError, match="both"):
        C.load_recipe("gpt2_trilingual_recovery", path=str(p))


def test_a_tp_mismatch_in_a_recipe_is_refused(tmp_path):
    import yaml
    R = yaml.safe_load(open(C.recipe_path("gemma4_31b_dutch_gmmlu")))
    R["eval"]["config"]["model_args"]["tensor_parallel_size"] = 1
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(R))
    with pytest.raises(C.RecipeError, match="TP must match"):
        C.load_recipe("gemma4_31b_dutch_gmmlu", path=str(p))


def test_hf_weights_catalogue_matches_the_recipes():
    path = os.path.join(C.RECIPES_DIR, "hf_weights.csv")
    rows = list(csv.DictReader(l for l in open(path) if not l.startswith("#")))
    got = {(r["result"], r["export"]): (r["hf_repo"], r["subfolder"] or None) for r in rows}
    want = {}
    for res in RESULTS:
        R = C.load_recipe(res)
        for run, e in C.iter_exports(R):
            if run is not None and e.get("weights"):
                want[(res, e["id"])] = (e["weights"]["repo"], e["weights"]["subfolder"])
    assert got == want and len(got) == 43


def _rows_for(R, export_id, shift=0.0, tp=None):
    _run, e = C.find_export(R, export_id)
    return [ResultRow(p["task"], p["metric"], p["value"] + shift, model=export_id,
                      tp=tp if tp is not None else R["tensor_parallel_size"])
            for p in e["published"]]


def test_compare_passes_fails_and_reports_missing():
    R = C.load_recipe("zephyr_wmdp")
    ok = compare(R, _rows_for(R, "rmu"))
    assert ok["ok"] and ok["n_pass"] == 4 and ok["n_fail"] == ok["n_missing"] == 0
    bad = compare(R, _rows_for(R, "rmu", shift=0.05))
    assert not bad["ok"] and bad["n_fail"] >= 3
    part = compare(R, _rows_for(R, "rmu")[:2])
    assert not part["ok"] and part["n_missing"] == 2
    tp = compare(R, _rows_for(R, "rmu", tp=2))
    assert not tp["ok"] and tp["tp_mismatch"]
    rel = C.load_recipe("gpt2_trilingual_recovery")
    rows = _rows_for(rel, "srspd_mu10")
    rows = [ResultRow(r.task, r.metric, r.value * 1.4, model=r.model, tp=1) for r in rows]
    assert compare(rel, rows)["ok"]                       # within the 50% relative tolerance


def test_compare_cli_exit_code_and_export_override(tmp_path, capsys):
    R = C.load_recipe("muse_news")
    rows = [ResultRow(r.task, r.metric, r.value, model="/some/checkpoint/path", tp=1)
            for r in _rows_for(R, "srspd_retain1_mpt0.01")]
    p = write_results(str(tmp_path / "r.json"), rows)
    assert cli.main(["compare", "muse_news", p, "--export", "srspd_retain1_mpt0.01"]) == 0
    assert "PASS" in capsys.readouterr().out
    assert cli.main(["compare", "muse_news", p]) == 1      # no row names an export


def test_list_and_show(capsys):
    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert all(r in out for r in RESULTS)
    assert cli.main(["show", "gemma4_e4b_dutch_gmmlu"]) == 0
    out = capsys.readouterr().out
    assert "trunc_srspd_mu10_lam0.01" in out and "rank_rule 0.01" in out
    assert "weights: not published" in out


def test_execute_runs_a_skip_and_collects_relabelled_rows(tmp_path, monkeypatch):
    from rspd.experiments.recipes import reproduce as X
    R = C.load_recipe("zephyr_wmdp")
    p = C.plan(R, "cheap", exports=["kuda_port", "rmu"])
    kinds = [s["step"] for s in p["steps"]]
    assert kinds == ["download", "eval", "eval", "skip"]
    seen = []
    monkeypatch.setattr(X, "_download", lambda st: seen.append(st["repo"]))
    monkeypatch.setattr(X, "_eval", lambda st, out: [ResultRow("wmdp_bio", "acc", 0.33,
                                                                model=st["config"]["model_path"],
                                                                tp=1)])
    path = X.execute(p, log=lambda *_: None)
    out = json.load(open(path))
    assert seen == ["thate10/rspd-wmdp-rmu-zephyr-7b"]
    assert {r["model"] for r in out["rows"]} == {"rmu"}
    assert out["rows"][0]["meta"]["checkpoint"].endswith(os.path.join("rmu"))
    assert out["steps"][-1] == {"step": "skip", "export": "kuda_port",
                               "reason": "edited weights not published; use --mode full"}
