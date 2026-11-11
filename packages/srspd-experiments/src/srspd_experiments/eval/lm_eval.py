"""lm-evaluation-harness over a (local, edited) checkpoint.

    from srspd_experiments.eval import lm_eval as E
    cfg = E.resolve_config({"name": "nl_eval", "model_path": "models/x/y",
                            "tasks": ["global_mmlu_nl_local"],
                            "model_args": {"dtype": "bfloat16", "tensor_parallel_size": 1}})
    rows = E.run(cfg)                       # -> list[ResultRow], also written to disk

Runs ``python -m lm_eval`` in a subprocess of the current interpreter (the
``[eval]`` extra: ``lm_eval==0.4.12``, ``vllm==0.26.0``). The **vLLM offline
backend** is the default (log-likelihood multiple choice and fast generation);
``backend: hf`` is the slow fallback. The local tasks shipped in ``tasks/``
(Global-MMLU ``{en,de,nl,zh}`` five-shot, ``global_mmlu_<lang>_local``) are
passed with ``--include_path`` so they resolve beside the harness's own.

Config keys: ``name`` (output directory), ``model_path`` (hub id, or a
data-root-relative checkpoint ``models/...``), ``tasks``; optional ``backend``
(``vllm``), ``model_args`` (passed as ``k=v,...``; for vLLM e.g. ``dtype``,
``tensor_parallel_size``, ``gpu_memory_utilization``, ``max_model_len``,
``tokenizer``), ``batch_size`` (``auto``), ``seed`` (1234), ``num_fewshot``,
``apply_chat_template`` / ``fewshot_as_multiturn`` (default true when the
template is on), ``limit``, ``log_samples``.

**Tensor parallelism changes results.** TP changes the reduction order of the
sharded matmuls and so perturbs logprobs; every result row records ``tp``
(``model_args.tensor_parallel_size``, 1 when unset; ``None`` for the HF
backend), and evaluations put side by side must share it.

Outputs (``out_dir``, default ``<data root>/logs/eval/<name>``): ``lm_eval/``
(the harness's own ``results_*.json``), ``results.json`` (the common
:mod:`srspd_experiments.results` schema), ``results.md`` and ``config.yaml``.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

from ..paths import logs_dir, resolve_model_path
from ..results import ResultRow, write_results

TASKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tasks")
BACKENDS = {"vllm": "vllm", "hf": "hf"}
#: The shipped local tasks.
LOCAL_TASKS = tuple(sorted(os.path.splitext(f)[0] for f in os.listdir(TASKS_DIR)
                           if f.endswith(".yaml"))) if os.path.isdir(TASKS_DIR) else ()
CONFIG_KEYS = ("name", "model_path", "tasks", "backend", "model_args", "batch_size", "seed",
               "num_fewshot", "apply_chat_template", "fewshot_as_multiturn", "limit",
               "log_samples")


def resolve_config(cfg: dict) -> dict:
    """Validate an eval config; returns a copy with defaults filled in."""
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    for key in ("name", "model_path", "tasks"):
        if key not in cfg:
            raise ValueError(f"eval config needs '{key}'")
    unknown = set(cfg) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"eval config has unknown keys {sorted(unknown)}; known: {CONFIG_KEYS}")
    tasks = cfg["tasks"]
    if isinstance(tasks, str):
        tasks = [t for t in tasks.split(",") if t]
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    backend = cfg.get("backend", "vllm")
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}, got {backend!r}")
    margs = cfg.get("model_args") or {}
    if not isinstance(margs, dict):
        raise ValueError(f"model_args must be a mapping, got {margs!r}")
    out = {"backend": backend, "batch_size": "auto", "seed": 1234, "limit": None,
           "apply_chat_template": False, "fewshot_as_multiturn": True, "log_samples": False,
           "num_fewshot": None, **cfg}
    out["tasks"] = list(tasks)
    out["model_args"] = dict(margs)
    return out


def tensor_parallel_size(cfg: dict):
    """The TP degree an evaluation runs at: ``model_args.tensor_parallel_size``
    (default 1) on vLLM, ``None`` on the HF backend (no sharded matmuls)."""
    if cfg.get("backend", "vllm") != "vllm":
        return None
    return int((cfg.get("model_args") or {}).get("tensor_parallel_size", 1))


def build_command(cfg: dict, out_dir: str, *, root=None, python: str | None = None) -> list:
    """The ``python -m lm_eval`` argv for a resolved config."""
    cfg = resolve_config(cfg)
    margs = dict(cfg["model_args"])
    margs["pretrained"] = resolve_model_path(cfg["model_path"], root=root)
    model_args = ",".join(f"{k}={v}" for k, v in margs.items())
    cmd = [python or sys.executable, "-m", "lm_eval",
           "--model", BACKENDS[cfg["backend"]],
           "--model_args", model_args,
           "--tasks", ",".join(cfg["tasks"]),
           "--batch_size", str(cfg["batch_size"]),
           "--output_path", os.path.join(out_dir, "lm_eval"),
           "--seed", str(cfg["seed"])]
    if os.path.isdir(TASKS_DIR):
        cmd += ["--include_path", TASKS_DIR]
    if cfg.get("num_fewshot") is not None:
        cmd += ["--num_fewshot", str(int(cfg["num_fewshot"]))]
    if cfg.get("apply_chat_template"):
        cmd += ["--apply_chat_template"]
        if cfg.get("fewshot_as_multiturn", True):
            cmd += ["--fewshot_as_multiturn"]
    if cfg.get("limit") is not None:
        cmd += ["--limit", str(cfg["limit"])]
    if cfg.get("log_samples"):
        cmd += ["--log_samples"]
    return cmd


def newest_results_json(out_dir: str):
    """The newest ``lm_eval/**/results_*.json`` under ``out_dir``, or ``None``."""
    paths = glob.glob(os.path.join(out_dir, "lm_eval", "**", "results_*.json"), recursive=True)
    return max(paths, key=os.path.getmtime) if paths else None


def _stderr_key(key: str) -> str:
    """``acc,none`` -> ``acc_stderr,none``; ``acc`` -> ``acc_stderr``."""
    if "," in key:
        m, f = key.split(",", 1)
        return f"{m}_stderr,{f}"
    return f"{key}_stderr"


def parse_results(payload: dict, *, model: str | None = None, tp=None) -> list:
    """Result rows from an lm-eval ``results_*.json`` payload: one row per
    ``(task, metric)`` with a numeric value (``alias`` and ``*_stderr`` keys are
    folded into ``meta``). ``metric`` is the harness key without its filter
    suffix (``acc,none`` -> ``acc``; the filter is in ``meta["filter"]``); ``n``
    comes from ``n-samples`` when the payload has it."""
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    n_samples = payload.get("n-samples", {}) if isinstance(payload, dict) else {}
    nshot = payload.get("n-shot", {}) if isinstance(payload, dict) else {}
    rows = []
    for task, metrics in sorted(results.items()):
        if not isinstance(metrics, dict):
            continue
        ns = n_samples.get(task) or {}
        n = ns.get("effective", ns.get("original")) if isinstance(ns, dict) else None
        for key, value in sorted(metrics.items()):
            if key == "alias" or "_stderr" in key:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            name, _, filt = key.partition(",")
            stderr = metrics.get(_stderr_key(key))
            meta = {"filter": filt or None, "harness_key": key,
                    "stderr": stderr if isinstance(stderr, (int, float)) else None,
                    "num_fewshot": nshot.get(task)}
            rows.append(ResultRow(task, name, float(value), model=model, tp=tp, n=n, meta=meta))
    return rows


def results_markdown(cfg: dict, cmd: list, rows: list, results_path: str) -> str:
    lines = [f"# Benchmark results -- {cfg['name']}", "",
             f"- model: `{cfg['model_path']}`",
             f"- backend: `{cfg.get('backend', 'vllm')}` (tp {tensor_parallel_size(cfg)})",
             f"- limit: {cfg.get('limit') if cfg.get('limit') is not None else 'full run'}",
             f"- command: `{' '.join(cmd)}`",
             f"- raw: `{results_path}`", "",
             "| task | metric | value | stderr |", "| --- | --- | --- | --- |"]
    for r in rows:
        se = r.meta.get("stderr")
        lines.append(f"| {r.task} | {r.meta.get('harness_key', r.metric)} | {r.value:.4f} | "
                     f"{'--' if se is None else f'{se:.4f}'} |")
    return "\n".join(lines) + "\n"


def run(cfg: dict, *, out_dir=None, root=None, tasks_override=None, runner=None) -> list:
    """Run lm-eval for ``cfg`` and return the result rows (also written to
    ``results.json``). ``tasks_override`` replaces the task list (per-task
    numbers are independent of what else is in the set, so a subset rerun stays
    comparable at the same TP). ``runner(cmd) -> returncode`` replaces the
    subprocess (tests)."""
    import yaml

    cfg = resolve_config(cfg)
    if tasks_override:
        cfg = {**cfg, "tasks": list(tasks_override)}
    out_dir = str(out_dir or logs_dir("eval", cfg["name"], root=root))
    os.makedirs(out_dir, exist_ok=True)
    cmd = build_command(cfg, out_dir, root=root)
    print(f"[{cfg['name']}] running: {' '.join(cmd)}", flush=True)
    rc = (runner or (lambda c: subprocess.run(c).returncode))(cmd)
    if rc != 0:
        raise RuntimeError(f"lm_eval exited with {rc}")
    results_path = newest_results_json(out_dir)
    if results_path is None:
        raise RuntimeError(f"no results_*.json found under {out_dir}/lm_eval")
    with open(results_path) as f:
        payload = json.load(f)
    tp = tensor_parallel_size(cfg)
    rows = parse_results(payload, model=cfg["model_path"], tp=tp)
    write_results(os.path.join(out_dir, "results.json"), rows, name=cfg["name"],
                  backend=cfg["backend"], tp=tp, command=cmd,
                  lm_eval_results=os.path.relpath(results_path, out_dir))
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(results_markdown(cfg, cmd, rows, os.path.relpath(results_path, out_dir)))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[{cfg['name']}] SUMMARY " + " ".join(f"{r.task}:{r.meta['harness_key']}={r.value:.4f}"
                                                for r in rows), flush=True)
    return rows
