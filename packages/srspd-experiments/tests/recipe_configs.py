"""Method configs for the ported baseline tests, read from the shipped recipes.

:func:`config` returns a run's runner config from a recipe; :data:`LEGACY`
names a few further settings the original test suite pinned, rebuilt as a
shipped run plus the keys that differ (each listed explicitly)."""
from __future__ import annotations

import copy

from srspd_experiments.recipes import catalog


def config(result: str, run_id: str) -> dict:
    """The runner config of run ``run_id`` of recipe ``result`` (a fresh copy)."""
    R = catalog.load_recipe(result)
    (run,) = [r for r in R["runs"] if r["id"] == run_id]
    return copy.deepcopy(run["config"])


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur[k]
    if value is _DROP:
        cur.pop(keys[-1], None)
    else:
        cur[keys[-1]] = value


_DROP = object()

#: name -> (result, run, {dotted key: value}) -- shipped run + the differing keys.
LEGACY = {
    "gpt2_simnpo": ("gpt2_trilingual_recovery", "simnpo_lr1e4_b01", {"train.lr": 3.0e-5}),
    "gpt2_rmu": ("gpt2_trilingual_recovery", "rmu_s100_a10_lr1e4",
                 {"rmu.steering_coeffs": [20.0], "rmu.alpha": [1200.0], "rmu.lr": 5.0e-5}),
    "gpt2_kuda": ("gpt2_trilingual_recovery", "kuda_tau1e2_lr3e4",
                  {"kuda.tau": 1.0e-3, "kuda.lr": 1.0e-4, "kuda.epoch": 40}),
    "gpt2_eb_simnpo": ("gpt2_trilingual_recovery", "equalbudget_simnpo_lr1e4_b01", {}),
    "gpt2_eb_rmu": ("gpt2_trilingual_recovery", "equalbudget_rmu_s100_a10_lr1e4", {}),
    "gpt2_eb_kuda": ("gpt2_trilingual_recovery", "equalbudget_kuda_tau1e2_lr3e4", {}),
    "zephyr_rmu": ("zephyr_wmdp", "rmu", {}),
    "zephyr_simnpo": ("zephyr_wmdp", "simnpo_port", {}),
    "zephyr_kuda": ("zephyr_wmdp", "kuda_port",
                    {"kuda.init_jitter": 1.0e-3, "kuda.min_len": _DROP,
                     "kuda.nullspace.max_length": 512}),
    "zephyr_kuda_faithful": ("zephyr_wmdp", "kuda_port", {}),
}


def legacy(name: str, **top) -> dict:
    """:data:`LEGACY` ``name`` as a config dict (top-level keys from ``top`` applied)."""
    result, run, diff = LEGACY[name]
    cfg = config(result, run)
    for k, v in diff.items():
        _set(cfg, k, v)
    cfg.update(top)
    return cfg
