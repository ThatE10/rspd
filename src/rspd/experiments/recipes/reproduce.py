"""Execute a recipe plan (:func:`rspd.experiments.recipes.catalog.plan`).

Steps run in order: ``train`` (the method runner; the authors'-code runners
with ``out_dir`` = the step's ``out_root``), ``download`` (``huggingface_hub``
``snapshot_download`` of the published edited weights -- a private repo needs
``HF_TOKEN``), ``eval`` (:mod:`rspd.experiments.eval.lm_eval`,
:mod:`~rspd.experiments.eval.muse` or :mod:`~rspd.experiments.eval.recovery`)
and ``skip`` (recorded). Every evaluation's rows are re-labelled with the
export id as ``model`` (the checkpoint path goes to ``meta.checkpoint``) and
collected into ``<data root>/logs/recipes/<result>/results.json``, which
:func:`rspd.experiments.recipes.compare.compare` reads.
"""
from __future__ import annotations

import os

from ..paths import hf_token, logs_dir


def _train(step, *, python_authors_stack=None):
    from ..methods import get
    m = step["method"]
    mod = get(m)
    if m in ("srspd",):
        return mod.run(step["config"], out_root=step["out_root"])
    if m in ("rmu", "kuda", "simnpo"):
        return mod.run(step["config"])
    if m == "kuda_authors_code_muse":
        if not python_authors_stack:
            raise RuntimeError("the authors' MUSE code needs their stack: pass "
                               "--authors-python <interpreter with methods.upstream_kuda_muse.STACK>")
        return mod.run(step["config"]["corpus"], python=python_authors_stack,
                       out_root=step["out_root"], name=None)
    cfg = dict(step["config"] or {})
    return mod.run(out_dir=step["out_root"], **({"model_id": cfg["model"]} if "model" in cfg else {}))


def _download(step):
    from huggingface_hub import snapshot_download
    allow = None if not step.get("subfolder") else [f"{step['subfolder']}/*"]
    return snapshot_download(step["repo"], local_dir=step["local_dir"], allow_patterns=allow,
                             token=hf_token())


def _prepare(step, root=None):
    """A ``data_prep`` entry: ``{download: {repo, local_dir, repo_type?}}`` (the
    local dir data-root relative; skipped when present) or ``{call:
    "module:function", spec: <shipped data spec>?, args: [...], kwargs: {...}}``
    (``spec`` loads :func:`rspd.experiments.data.spec_path` as the first argument)."""
    import importlib

    import yaml

    from ..paths import resolve_data_ref
    if "download" in step:
        d = step["download"]
        local = resolve_data_ref(d["local_dir"], root=root)
        if os.path.isdir(local) and os.listdir(local):
            return local
        from huggingface_hub import snapshot_download
        return snapshot_download(d["repo"], repo_type=d.get("repo_type", "model"),
                                 local_dir=local, token=hf_token())
    mod, _, fn = step["call"].partition(":")
    f = getattr(importlib.import_module(mod), fn)
    args = list(step.get("args") or [])
    if step.get("spec"):
        from ..data import spec_path
        with open(spec_path(step["spec"])) as fh:
            args.insert(0, yaml.safe_load(fh))
    return f(*args, **dict(step.get("kwargs") or {}))


def _eval(step, out_dir):
    kind, cfg = step["kind"], step["config"]
    if kind == "lm_eval":
        from ..eval import lm_eval
        return lm_eval.run(cfg, out_dir=out_dir)
    from ..results import read_results
    if kind == "muse":
        from ..eval import muse
        muse.run(cfg, out_dir=out_dir)
        return read_results(os.path.join(out_dir, "results_rows.json"))
    from ..eval import recovery
    recovery.run(cfg, out_dir=out_dir)
    return read_results(os.path.join(out_dir, "results.json"))


def execute(p, *, root=None, python_authors_stack=None, log=print) -> str:
    """Run every step of plan ``p``; returns the collected results path."""
    from ..results import ResultRow, write_results

    out_root = str(logs_dir("recipes", p["result"], root=root))
    rows, record = [], []
    for st in p["steps"]:
        log(f"[reproduce] {st['step']} {st.get('export') or st.get('run') or st.get('call') or st.get('download')}")
        if st["step"] == "skip":
            record.append({"step": "skip", "export": st["export"], "reason": st["reason"]})
            continue
        if st["step"] == "prepare":
            _prepare(st, root=root)
        elif st["step"] == "train":
            _train(st, python_authors_stack=python_authors_stack)
        elif st["step"] == "download":
            _download(st)
        elif st["step"] == "eval":
            got = _eval(st, os.path.join(out_root, st["export"]))
            ckpt = st["config"].get("model_path") or st["config"].get("checkpoint")
            for r in got:
                rows.append(ResultRow(r.task, r.metric, r.value, model=st["export"], tp=r.tp,
                                      n=r.n, meta={**r.meta, "checkpoint": ckpt}))
        record.append({k: v for k, v in st.items() if k != "config"})
    return write_results(os.path.join(out_root, "results.json"), rows, result=p["result"],
                         mode=p["mode"], steps=record)
