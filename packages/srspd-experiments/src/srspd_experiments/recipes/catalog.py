"""The recipe catalogue: load, validate and resolve the shipped recipes.

A recipe (``recipes/<result>.yaml``) describes one published result set::

    result: <id>                     # the file stem
    title: ...
    model: <hub id of the unmodified model>
    tensor_parallel_size: 1          # the TP of every evaluation of this result
    hardware: {method: ..., eval: ...}
    eval: {kind: lm_eval | muse | recovery, config: {...}, extra: [...]}
    baseline: {id, label, published: [...]}   # the unmodified model's row
    references: [...]                # numbers printed by other papers (not checked)
    runs:
      - id: <run id>
        method: srspd | rmu | kuda | simnpo | kuda_authors_code | ...
        label: ...
        hardware: ...                # overrides the recipe's
        config: {...}                # the runner config (dose as mu_per_token)
        legacy_counts: {n_x, n_y}    # only when an export gives a legacy dose
        exports:
          - id: <export id>          # unique within the recipe
            mu_per_token: 0.01       # SRSPD only (or `legacy: {<legacy dose>: v}`)
            rank_rule: null          # SRSPD only: a lam_rel or "erank"
            checkpoint_subdir: ...   # derived for SRSPD; null for the baselines
            eval_overrides: {...}    # merged into the eval config (e.g. recovery)
            weights: {repo, subfolder, private} | null
            published: [{task, metric, value, tol | rel_tol}, ...]

Loading converts every legacy dose to ``mu_per_token`` (through
:func:`srspd_experiments.recipes.legacy.mu_per_token_from_legacy`, the one
translation point) and fills the SRSPD runner's dose list, so ``mu_per_token``
is the only dose any later stage sees. :func:`plan` resolves a recipe into
concrete steps for ``mode="full"`` (train with the method runner, then
evaluate the export) or ``mode="cheap"`` (download the published edited
weights, then evaluate) without touching the network.
"""
from __future__ import annotations

import copy
import os

from . import legacy as L

RECIPES_DIR = os.path.dirname(os.path.abspath(__file__))
MODES = ("cheap", "full")
EVAL_KINDS = ("lm_eval", "muse", "recovery")
#: What the runner key of each method is called inside its config.
METHOD_BLOCK = {"srspd": "srspd", "rmu": "rmu", "kuda": "kuda", "simnpo": "simnpo"}
RECIPE_KEYS = ("result", "title", "description", "model", "base_checkpoint",
               "tensor_parallel_size", "hardware", "eval", "baseline", "references", "runs",
               "notes", "data_prep")
RUN_KEYS = ("id", "method", "label", "hardware", "config", "legacy_counts", "exports", "notes")
EXPORT_KEYS = ("id", "label", "mu_per_token", "legacy", "rank_rule", "checkpoint_subdir",
               "eval_overrides", "weights", "published", "notes")
PUBLISHED_KEYS = ("task", "metric", "value", "tol", "rel_tol", "note")


class RecipeError(ValueError):
    """A recipe that does not validate."""


def list_results() -> list:
    """The ids of the shipped recipes (``recipes/*.yaml`` stems), sorted."""
    return sorted(f[:-5] for f in os.listdir(RECIPES_DIR) if f.endswith(".yaml"))


def recipe_path(result: str) -> str:
    p = os.path.join(RECIPES_DIR, f"{result}.yaml")
    if not os.path.isfile(p):
        raise KeyError(f"no recipe {result!r}; have {list_results()}")
    return p


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise RecipeError(f"{where} must be a number, got {v!r}")
    return float(v)


def _check_published(items, where) -> list:
    out = []
    for i, p in enumerate(items or []):
        w = f"{where}.published[{i}]"
        if not isinstance(p, dict):
            raise RecipeError(f"{w} must be a mapping")
        unknown = set(p) - set(PUBLISHED_KEYS)
        if unknown:
            raise RecipeError(f"{w} has unknown keys {sorted(unknown)}")
        for k in ("task", "metric"):
            if not isinstance(p.get(k), str) or not p[k]:
                raise RecipeError(f"{w}.{k} must be a non-empty string")
        q = dict(p)
        q["value"] = _num(p.get("value"), f"{w}.value")
        if ("tol" in p) == ("rel_tol" in p):
            raise RecipeError(f"{w} needs exactly one of tol / rel_tol")
        if "tol" in p:
            q["tol"] = _num(p["tol"], f"{w}.tol")
        else:
            q["rel_tol"] = _num(p["rel_tol"], f"{w}.rel_tol")
        out.append(q)
    return out


def export_subdir(mu_per_token, rank_rule=None) -> str:
    """The SRSPD runner's export directory name (``srspd_mpt<mpt>[_lam<l>|_erank]``)."""
    from ..methods.srspd import export_label
    return export_label(mu_per_token, rank_rule)


def _resolve_run(run, recipe_id, seen_exports) -> dict:
    w = f"{recipe_id}.runs[{run.get('id')!r}]"
    unknown = set(run) - set(RUN_KEYS)
    if unknown:
        raise RecipeError(f"{w} has unknown keys {sorted(unknown)}")
    from ..methods import METHODS
    method = run.get("method")
    if method not in METHODS:
        raise RecipeError(f"{w}.method must be one of {sorted(METHODS)}, got {method!r}")
    cfg = copy.deepcopy(run.get("config") or {})
    exports = []
    counts = run.get("legacy_counts")
    for e in run.get("exports") or []:
        we = f"{w}.exports[{e.get('id')!r}]"
        unknown = set(e) - set(EXPORT_KEYS)
        if unknown:
            raise RecipeError(f"{we} has unknown keys {sorted(unknown)}")
        if not isinstance(e.get("id"), str) or not e["id"]:
            raise RecipeError(f"{we} needs an id")
        if e["id"] in seen_exports:
            raise RecipeError(f"{we}: export id {e['id']!r} is not unique in the recipe")
        seen_exports.add(e["id"])
        x = copy.deepcopy(e)
        if method == "srspd":
            if "legacy" in e:
                if "mu_per_token" in e:
                    raise RecipeError(f"{we} gives both mu_per_token and a legacy dose")
                leg = e["legacy"] or {}
                if set(leg) != {L.LEGACY_DOSE_KEY}:
                    raise RecipeError(f"{we}.legacy must hold exactly the legacy dose")
                if not isinstance(counts, dict):
                    raise RecipeError(f"{w} needs legacy_counts {{n_x, n_y}} to convert "
                                      f"a legacy dose")
                x["mu_per_token"] = L.mu_per_token_from_legacy(
                    leg[L.LEGACY_DOSE_KEY], counts.get("n_x"), counts.get("n_y"))
                x["legacy"] = dict(leg)
            if "mu_per_token" not in x:
                raise RecipeError(f"{we} needs mu_per_token (or a legacy dose)")
            x["mu_per_token"] = _num(x["mu_per_token"], f"{we}.mu_per_token")
            x.setdefault("rank_rule", None)
            x["checkpoint_subdir"] = export_subdir(x["mu_per_token"], x["rank_rule"])
        else:
            for k in ("mu_per_token", "legacy", "rank_rule"):
                if k in e:
                    raise RecipeError(f"{we}.{k} applies to SRSPD exports only")
            x.setdefault("checkpoint_subdir", None)
        wts = x.get("weights")
        if wts is not None:
            if not isinstance(wts, dict) or not isinstance(wts.get("repo"), str):
                raise RecipeError(f"{we}.weights must be null or {{repo, subfolder}}")
            x["weights"] = {"repo": wts["repo"], "subfolder": wts.get("subfolder"),
                            "private": bool(wts.get("private", True))}
        x["published"] = _check_published(e.get("published"), we)
        x.setdefault("eval_overrides", {})
        exports.append(x)
    if not exports:
        raise RecipeError(f"{w} has no exports")
    if method == "srspd":
        s = cfg.setdefault("srspd", {})
        if any(k == L.LEGACY_DOSE_KEY for k in s):
            raise RecipeError(f"{w}.config.srspd carries a legacy dose; give it per export")
        s["mu_per_token"] = sorted({x["mu_per_token"] for x in exports})
        rules = sorted({x["rank_rule"] for x in exports if x["rank_rule"] not in (None, "erank")})
        s["rank_rule"] = rules + (["erank"] if any(x["rank_rule"] == "erank" for x in exports)
                                  else [])
    block = METHOD_BLOCK.get(method)
    if block and not isinstance(cfg.get(block), dict):
        raise RecipeError(f"{w}.config needs a `{block}:` block")
    if method in METHOD_BLOCK and cfg.get("name") is None:
        cfg["name"] = run["id"]
    return {"id": run["id"], "method": method, "label": run.get("label") or run["id"],
            "hardware": run.get("hardware"), "config": cfg, "legacy_counts": counts,
            "exports": exports, "notes": run.get("notes")}


def load_recipe(result: str, *, path: str | None = None) -> dict:
    """Load and validate one recipe; every dose resolved to ``mu_per_token``."""
    import yaml

    with open(path or recipe_path(result)) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise RecipeError(f"{result}: a recipe must be a mapping")
    unknown = set(raw) - set(RECIPE_KEYS)
    if unknown:
        raise RecipeError(f"{result} has unknown keys {sorted(unknown)}")
    if raw.get("result") != result:
        raise RecipeError(f"{result}: `result:` is {raw.get('result')!r}, not the file stem")
    tp = raw.get("tensor_parallel_size")
    if isinstance(tp, bool) or not isinstance(tp, int) or tp < 1:
        raise RecipeError(f"{result}.tensor_parallel_size must be a positive int, got {tp!r}")
    ev = raw.get("eval") or {}
    if ev.get("kind") not in EVAL_KINDS:
        raise RecipeError(f"{result}.eval.kind must be one of {EVAL_KINDS}")
    if ev["kind"] == "lm_eval":
        mtp = int(((ev.get("config") or {}).get("model_args") or {}).get("tensor_parallel_size", 1))
        if mtp != tp:
            raise RecipeError(f"{result}: eval model_args.tensor_parallel_size {mtp} != "
                              f"tensor_parallel_size {tp} (TP must match the published rows)")
    seen = set()
    base = dict(raw.get("baseline") or {})
    if base:
        base["published"] = _check_published(base.get("published"), f"{result}.baseline")
        seen.add(base.setdefault("id", "baseline"))
    runs = [_resolve_run(r, result, seen) for r in raw.get("runs") or []]
    ids = [r["id"] for r in runs]
    if len(set(ids)) != len(ids):
        raise RecipeError(f"{result}: run ids are not unique")
    out = dict(raw)
    out.update(baseline=base or None, runs=runs, references=list(raw.get("references") or []))
    return out


def iter_exports(recipe):
    """``(run, export)`` pairs in recipe order (the baseline as run ``None``)."""
    if recipe.get("baseline"):
        yield None, recipe["baseline"]
    for r in recipe["runs"]:
        for e in r["exports"]:
            yield r, e


def find_export(recipe, export_id):
    for r, e in iter_exports(recipe):
        if e["id"] == export_id:
            return r, e
    raise KeyError(f"{recipe['result']}: no export {export_id!r}")


def weight_gaps(recipe) -> list:
    """Export ids whose edited weights are not published (cheap mode cannot run them)."""
    return [e["id"] for r, e in iter_exports(recipe) if r is not None and e.get("weights") is None]


# --------------------------------------------------------------------------- #
# Plans                                                                        #
# --------------------------------------------------------------------------- #
def _eval_config(recipe, export, model_path, base_path=None, *, ev=None, suffix="") -> dict:
    ev = copy.deepcopy(ev or recipe["eval"])
    cfg = copy.deepcopy(ev.get("config") or {})
    over = {} if suffix else copy.deepcopy(export.get("eval_overrides") or {})
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    name = f"{recipe['result']}__{export['id']}" + (f"__{suffix}" if suffix else "")
    kind = ev.get("kind", recipe["eval"]["kind"])
    if kind == "recovery":
        cfg.update(name=name, checkpoint=model_path)
        cfg.setdefault("condition", export["id"])
        if base_path is not None:
            cfg["base_checkpoint"] = base_path
    else:
        cfg.update(name=name, model_path=model_path)
    return {"kind": kind, "config": cfg, "eval_id": suffix or "main"}


def _eval_steps(recipe, export, run_id, model_path, base_path) -> list:
    """One ``eval`` step for the recipe's main evaluation and one per ``eval.extra``."""
    out = [{"step": "eval", "export": export["id"], "run": run_id,
            **_eval_config(recipe, export, model_path, base_path)}]
    for x in recipe["eval"].get("extra") or []:
        out.append({"step": "eval", "export": export["id"], "run": run_id,
                    **_eval_config(recipe, export, model_path, base_path,
                                   ev={"kind": x.get("kind", recipe["eval"]["kind"]),
                                       "config": x["config"]}, suffix=x["id"])})
    return out


def plan(recipe, mode: str, *, root=None, exports=None) -> dict:
    """The resolved steps of ``reproduce`` -- no network, no model load.

    ``mode="full"``: per run one ``train`` step (method, runner config with the
    output directories it will write) and per export one ``eval`` step on the
    exported checkpoint. ``mode="cheap"``: per export a ``download`` step
    (``repo`` / ``subfolder`` / ``local_dir``) and an ``eval`` step, or a
    ``skip`` step naming the gap when the weights are not published. The
    baseline is always evaluated from the unmodified model. ``exports``
    restricts the plan to those export ids (their runs, in full mode)."""
    from ..paths import data_dir, models_dir

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    want = None if exports is None else set(exports)
    rid = recipe["result"]
    steps = []
    dl_root = os.path.join(str(data_dir(root, create=False)), "models", "recipes", rid)
    base_model = recipe.get("base_checkpoint") or recipe["model"]
    for i, d in enumerate(recipe.get("data_prep") or []):
        steps.append({"step": "prepare", "index": i, **copy.deepcopy(d)})
    if recipe.get("baseline") and (want is None or recipe["baseline"]["id"] in want):
        steps += _eval_steps(recipe, recipe["baseline"], None, base_model, base_model)
    for run in recipe["runs"]:
        sel = [e for e in run["exports"] if want is None or e["id"] in want]
        if not sel:
            continue
        if mode == "full":
            out_root = (os.path.join(str(models_dir(create=False, root=root)), "methods",
                                     run["config"].get("name", run["id"]))
                        if run["method"] in METHOD_BLOCK else os.path.join(dl_root, run["id"]))
            steps.append({"step": "train", "run": run["id"], "method": run["method"],
                          "config": copy.deepcopy(run["config"]), "out_root": out_root,
                          "hardware": run.get("hardware") or (recipe.get("hardware") or {})
                          .get("method")})
            for e in sel:
                sub = e.get("checkpoint_subdir")
                mp = os.path.join(out_root, sub) if sub else out_root
                steps += _eval_steps(recipe, e, run["id"], mp, base_model)
        else:
            for e in sel:
                w = e.get("weights")
                if w is None:
                    steps.append({"step": "skip", "export": e["id"], "run": run["id"],
                                  "reason": "edited weights not published; use --mode full"})
                    continue
                local = os.path.join(dl_root, e["id"])
                mp = os.path.join(local, w["subfolder"]) if w.get("subfolder") else local
                steps.append({"step": "download", "export": e["id"], "run": run["id"],
                              "repo": w["repo"], "subfolder": w.get("subfolder"),
                              "private": w.get("private", True), "local_dir": local})
                steps += _eval_steps(recipe, e, run["id"], mp, base_model)
    return {"result": rid, "mode": mode, "model": base_model,
            "tensor_parallel_size": recipe["tensor_parallel_size"],
            "hardware": recipe.get("hardware"), "steps": steps}
