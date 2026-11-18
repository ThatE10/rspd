"""The SRSPD method runner: data spec -> :func:`rspd.erase` at every dose ->
one exported checkpoint and one held-out readout per (dose, rank rule).

    from rspd.experiments.methods import srspd as srspd_runner
    metrics = srspd_runner.run("my_srspd.yaml")       # or a mapping

Config (YAML or mapping)::

    name: my_run                     # output directory name
    model: <hub id | checkpoint dir | models/... under the data root>
    tokenizer: null                  # load the tokenizer from another repo
    dtype: bfloat16                  # load AND export dtype
    device_map: auto
    seed: 42
    data: {...}                      # rspd.experiments.data.DataSpec (target X, retained Y)
    srspd:
      mu_per_token: [0.003, 0.01]    # the dose(s); one solve serves every dose
      solver: readout                # readout | grams (target streamed into G_X)
      rank_rule: []                  # truncated exports besides full rank: lam_rel
                                     # floats in (0, 1) and/or "erank" (readout only)
      ridge_rel: 0.0
      gram_budget_gib: 12.0          # float64 Gram budget per device
      resume_from_cache: true
      eig_extremes: true
      lam_svd_dtype: float32
      layers: null                   # rspd layer selector (null = auto preset)
      scratch_dir: null              # spill pristine weights / readouts here
      export: true                   # write a checkpoint per (dose, rule)

What runs, in order (the data path and the edit are those of every published
SRSPD run):

1. :func:`rspd.experiments.data.build_erasure_data` builds ``X``, ``Y`` and the
   held-out readouts (``forget/<corpus>``, ``retain/<corpus>``, ``mix/<bucket>``)
   with the run's tokenizer;
2. the model is loaded (:func:`rspd.experiments.models.load_causal_lm`) and the
   baseline held-out CE scored;
3. ONE :func:`rspd.erase` call (``apply=False``) runs one set of Gram passes,
   solves every ``mu_per_token`` and truncates every rank rule of each dose
   from one shared SVD of ``M X^T`` per matrix;
4. every ``(dose, rule)`` setting is applied in turn
   (:meth:`rspd.EditReport.apply_dose`), exported and scored: the full-rank
   edit ``W - M`` to ``<data root>/models/methods/<name>/srspd_mpt<mpt>/``,
   the truncated edits ``W - M_r`` to ``srspd_mpt<mpt>_lam<lam>`` /
   ``srspd_mpt<mpt>_erank`` beside it.

Every edit is restored bit for bit before the next. ``metrics.json`` (under
``<data root>/logs/methods/<name>/``) records the dose arithmetic (``mu`` and
the per-token weights derived from ``mu_per_token``, ``n_x``, ``n_y``), the leaks
per matrix and per type, the held-out CE and its delta per export, the data
manifest and each export record; ``results.json`` holds the held-out CE as
:class:`rspd.experiments.results.ResultRow` rows.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import time

from ..data.spec import DataSpec
from ..paths import logs_dir, models_dir

METHOD = "srspd"
SOLVERS = ("readout", "grams")
ERANK = "erank"
DEFAULT_GRAM_BUDGET_GIB = 12.0
SRSPD_KEYS = ("mu_per_token", "solver", "rank_rule", "ridge_rel", "gram_budget_gib",
              "resume_from_cache", "eig_extremes", "lam_svd_dtype", "layers", "scratch_dir",
              "export")
TOP_KEYS = ("name", "model", "tokenizer", "dtype", "device_map", "seed", "data", "srspd",
            "tracking")


def _pos_float(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v) > 0.0:
        raise ValueError(f"{where} must be a number > 0, got {v!r}")
    return float(v)


def resolve_config(cfg) -> dict:
    """Validate a run config (module docstring); returns a plain dict with the
    :class:`DataSpec` under ``"data"``. Every failure is a ``ValueError``."""
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    unknown = set(cfg) - set(TOP_KEYS)
    if unknown:
        raise ValueError(f"config has unknown keys {sorted(unknown)}; known: {TOP_KEYS}")
    for k in ("name", "model"):
        if not isinstance(cfg.get(k), str) or not cfg[k].strip():
            raise ValueError(f"config has no `{k}:`")
    dtype = cfg.get("dtype", "bfloat16")
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be one of bfloat16/float16/float32, got {dtype!r}")
    s = cfg.get("srspd")
    if not isinstance(s, dict):
        raise ValueError("config has no `srspd:` block")
    unknown = set(s) - set(SRSPD_KEYS)
    if unknown:
        raise ValueError(f"srspd has unknown keys {sorted(unknown)}; known: {SRSPD_KEYS}")
    mpt = s.get("mu_per_token")
    if isinstance(mpt, (int, float)) and not isinstance(mpt, bool):
        mpt = [mpt]
    if not isinstance(mpt, (list, tuple)) or not mpt:
        raise ValueError("rspd.mu_per_token must be a number > 0 or a non-empty list of them")
    mpt = sorted({_pos_float(v, f"rspd.mu_per_token[{i}]") for i, v in enumerate(mpt)})
    solver = s.get("solver", "readout")
    if solver not in SOLVERS:
        raise ValueError(f"rspd.solver must be one of {SOLVERS}, got {solver!r}")
    rules = s.get("rank_rule") or []
    if not isinstance(rules, (list, tuple)):
        rules = [rules]
    out_rules, seen = [], set()
    for i, r in enumerate(rules):
        if r == ERANK:
            spec = ERANK
        else:
            if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0.0 < float(r) < 1.0:
                raise ValueError(f"rspd.rank_rule[{i}] must be a lam_rel in (0, 1) or "
                                 f"{ERANK!r}, got {r!r}")
            spec = float(r)
        if spec not in seen:
            seen.add(spec)
            out_rules.append(spec)
    out_rules = sorted([r for r in out_rules if r != ERANK]) + ([ERANK] if ERANK in seen else [])
    if out_rules and solver == "grams":
        raise ValueError("rspd.rank_rule needs solver: readout (truncation acts on the target "
                         "response M X^T, which the grams solver never forms)")
    ridge = s.get("ridge_rel", 0.0)
    if isinstance(ridge, bool) or not isinstance(ridge, (int, float)) or not 0.0 <= float(ridge) < 1e-4:
        raise ValueError(f"rspd.ridge_rel must be in [0, 1e-4), got {ridge!r}")
    lsd = s.get("lam_svd_dtype", "float32")
    if lsd not in ("float32", "float64"):
        raise ValueError(f"rspd.lam_svd_dtype must be float32 or float64, got {lsd!r}")
    if not isinstance(cfg.get("data"), dict):
        raise ValueError("config has no `data:` block (a DataSpec)")
    data = DataSpec.from_dict(cfg["data"])
    seed = cfg.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    return {"name": cfg["name"], "model": cfg["model"], "tokenizer": cfg.get("tokenizer"),
            "dtype": dtype, "device_map": cfg.get("device_map", "auto"), "seed": seed,
            "data": data, "mu_per_token": mpt, "solver": solver, "rank_rule": out_rules,
            "ridge_rel": float(ridge),
            "gram_budget_gib": _pos_float(s.get("gram_budget_gib", DEFAULT_GRAM_BUDGET_GIB),
                                          "rspd.gram_budget_gib"),
            "resume_from_cache": bool(s.get("resume_from_cache", True)),
            "eig_extremes": bool(s.get("eig_extremes", True)), "lam_svd_dtype": lsd,
            "layers": s.get("layers"), "scratch_dir": s.get("scratch_dir"),
            "export": bool(s.get("export", True)), "tracking": cfg.get("tracking")}


def export_label(mu_per_token, rule=None) -> str:
    """``srspd_mpt0.01`` (full rank), ``srspd_mpt0.01_lam0.001``, ``srspd_mpt0.01_erank``."""
    base = f"srspd_mpt{float(mu_per_token):g}"
    if rule is None:
        return base
    return f"{base}_{ERANK}" if rule == ERANK else f"{base}_lam{float(rule):g}"


def _load(config):
    if isinstance(config, (str, os.PathLike)):
        import yaml
        with open(config, "rb") as f:
            raw = f.read()
        return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest(), os.fspath(config)
    raw = json.dumps(config, sort_keys=True, default=str).encode()
    return dict(config), hashlib.sha256(raw).hexdigest(), None


def score_heldout(model, heldout_ids) -> dict:
    """``{readout key: pooled mean next-token CE}`` (the shared estimator)."""
    import torch

    from ..eval.heldout import pooled_ce
    with torch.no_grad():
        return {k: pooled_ce(model, v)[0] for k, v in heldout_ids.items()}


def _leak_summary(rep_dict) -> dict:
    return {"leaks_by_type": rep_dict["leaks_by_type"],
            "leaks_per_matrix": {n: {"L_X_rel": m["L_X_rel"], "L_Y_rel": m["L_Y_rel"]}
                                 for n, m in rep_dict["matrices"].items()}}


def run(config, *, model=None, tokenizer=None, out_root=None, log_root=None,
        progress=None) -> dict:
    """Run the config (module docstring). ``model`` / ``tokenizer`` skip the
    load (the model must match ``config["model"]``; it is edited in place and
    restored). ``out_root`` / ``log_root`` override the checkpoint / log
    directories. Returns the metrics dict (also written to ``metrics.json``)."""
    import numpy as np
    import torch

    import rspd

    from ..data.target import build_erasure_data
    from ..eval.tracking import Tracker
    from ..models import export_checkpoint, load_causal_lm, write_metrics
    from ..results import ResultRow, write_results

    cfg, config_sha, config_path = _load(config)
    R = resolve_config(cfg)
    name = R["name"]
    log_dir = str(log_root or logs_dir("methods", name))
    model_root = str(out_root or models_dir("methods", name))
    os.makedirs(log_dir, exist_ok=True)
    t0 = time.time()
    random.seed(R["seed"])
    np.random.seed(R["seed"])
    torch.manual_seed(R["seed"])
    tracker = Tracker(R["tracking"], out_dir=log_dir, run_name=name,
                      config={k: v for k, v in R.items() if k != "data"})
    print(f"[rspd] {R['model']} · solver {R['solver']} · mu_per_token {R['mu_per_token']}"
          + (f" · rank_rule {R['rank_rule']}" if R["rank_rule"] else "")
          + f" · ridge_rel {R['ridge_rel']:g} -> logs {log_dir}, models {model_root}",
          flush=True)

    if model is None:
        model, tok = load_causal_lm(R["model"], dtype=R["dtype"], device_map=R["device_map"],
                                    tokenizer_id=R["tokenizer"])
    else:
        tok = tokenizer
    if tok is None:
        raise ValueError("a tokenizer is required with a pre-loaded model")
    model.eval()
    data = build_erasure_data(tok, R["data"])
    retained = list(data.retained_ids.to(torch.long))

    ce0 = score_heldout(model, data.heldout_ids)
    print("[rspd] baseline held-out CE " + " ".join(f"{k} {v:.4f}" for k, v in ce0.items()),
          flush=True)

    exports, rows = {}, []
    with torch.no_grad():
        rep = rspd.erase(model, target=data.target_ids, retained=retained,
                          mu_per_token=R["mu_per_token"], apply=False,
                          layers=R["layers"], solver=R["solver"],
                          rank_rule=list(R["rank_rule"]) or None,
                          ridge_rel=R["ridge_rel"], gram_budget_gib=R["gram_budget_gib"],
                          scratch_dir=R["scratch_dir"],
                          resume_from_cache=R["resume_from_cache"],
                          eig_extremes=R["eig_extremes"], lam_svd_dtype=R["lam_svd_dtype"],
                          progress=progress)
        try:
            for m in R["mu_per_token"]:
                for rule in [None] + list(R["rank_rule"]):
                    rep.apply_dose(m, rule)
                    lab = export_label(m, rule)
                    d = rep.to_dict()
                    ent = {"label": lab, "rank_rule": rule, "dose": d["dose"],
                           "edit": d["edit"], **_leak_summary(d)}
                    if R["export"]:
                        ent["checkpoint"] = export_checkpoint(
                            model, tok, os.path.join(model_root, lab), source=R["model"])
                    ce = score_heldout(model, data.heldout_ids)
                    ent["ce_heldout"] = ce
                    ent["ce_heldout_delta"] = {k: ce[k] - ce0[k] for k in ce}
                    exports[lab] = ent
                    for k, v in ce.items():
                        rows.append(ResultRow(task=f"heldout/{k}", metric="ce", value=v,
                                              model=lab, n=None,
                                              meta={"mu_per_token": m, "rank_rule": rule}))
                    tracker.log({f"{lab}/ce_heldout/{k}": v for k, v in ce.items()})
                    print(f"[rspd] {lab}: held-out ΔCE " + " ".join(
                        f"{k} {ce[k] - ce0[k]:+.4f}" for k in ce)
                        + f" · survival {d['edit']['survival']:.4f}", flush=True)
            solve = {k: v for k, v in rep.to_dict().items()
                     if k in ("convention", "solver", "doses", "rank_rules", "settings",
                              "ridge_rel", "n_x", "n_y", "by_dose", "input_keys", "quirks",
                              "selector", "pass_plan", "stats", "warnings")}
        finally:
            rep.close()
    metrics = {
        "name": name, "method": METHOD, "config_path": config_path,
        "config_sha256": config_sha, "model": R["model"],
        "tokenizer": R["tokenizer"] or R["model"], "dtype": R["dtype"],
        "convention": "W_new = W - M; X = target (erased), Y = retained (kept)",
        "mu_per_token": R["mu_per_token"], "solver": R["solver"], "rank_rule": R["rank_rule"],
        "ridge_rel": R["ridge_rel"], "n_x": data.n_target_tokens, "n_y": data.n_retained_tokens,
        "data_manifest": data.manifest, "baseline_ce_heldout": ce0, "exports": exports,
        "solve": solve, "model_root": model_root, "log_dir": log_dir,
        "tracking": tracker.resolved, "torch": torch.__version__,
        "wall_seconds": round(time.time() - t0, 1),
    }
    write_metrics(log_dir, metrics)
    write_results(os.path.join(log_dir, "results.json"),
                  [ResultRow(task=f"heldout/{k}", metric="ce", value=v, model=R["model"])
                   for k, v in ce0.items()] + rows, run=name, method=METHOD)
    tracker.finish()
    return metrics
