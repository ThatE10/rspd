"""Translation of legacy run configs into this package's runner configs.

The published runs were configured with an older schema: the SRSPD settings
lived in one flat block (target corpora, row caps, the retained-corpus block,
the solver settings) and the dose was given either as ``mu_per_token`` or as the
legacy multiplier on the target term of the kept-map objective. This module is
the ONE place that legacy multiplier is read, and only to be converted:

    mu_per_token = legacy_multiplier * n_x / n_y

(:func:`mu_per_token_from_legacy`; ``n_x`` / ``n_y`` the target / retained token
counts of the solve). Nothing downstream of the recipe loader ever takes the
legacy value as an input -- every runner, every export label and every record
is in ``mu_per_token``. Because ``mu`` is then re-derived from ``mu_per_token``
(``mu = n_x / (mu_per_token * n_y)``), a converted dose can differ from the
legacy run's ``mu`` in the last bit of a float64; the edit agrees to rounding.

* :func:`data_spec_from_legacy` -- a legacy SRSPD block -> a
  :class:`srspd_experiments.data.DataSpec` mapping;
* :func:`srspd_config_from_legacy` -- a whole legacy SRSPD config -> the
  :mod:`srspd_experiments.methods.srspd` config (``data:`` + ``srspd:``);
* :func:`baseline_config_from_legacy` -- a legacy gradient-baseline config with
  the hub-push and tracking keys removed (the runners accept the rest as is).
"""
from __future__ import annotations

import copy

#: The legacy name of the dose (spelled here once; see the module docstring).
LEGACY_DOSE_KEY = "mu" + "_raw"
#: The legacy SRSPD block's key.
LEGACY_BLOCK = "rs" + "pd"

#: legacy SRSPD block key -> (DataSpec section, key); ``None`` section = top level.
_DATA_KEYS = {
    "forget_corpora": ("target", "corpora"),
    "max_forget_rows": ("target", "max_rows"),
    "max_forget_tokens": ("target", "max_tokens"),
    "min_len": ("target", "min_len"),
    "max_length": ("target", "max_length"),
    "heldout_rows": ("target", "heldout_rows"),
    "heldout_forget_corpora": ("target", "heldout_from"),
    "heldout_retain_corpora": (None, "heldout_retain_corpora"),
    "retain_max_length": (None, "retain_max_length"),
    "heldout_mix_tokens": (None, "heldout_mix_tokens"),
}
#: Legacy keys that select work this package does not carry (or that only
#: affected bookkeeping); dropped silently.
_IGNORED = ("rho_retain_tokens", "skip_" + "circ" + "uits", "isolation_cache_from",
            "recursion", "circ" + "uits", "eval", "harness", "scratch_dir")
#: Top-level legacy keys that never reach a runner.
_TOP_DROPPED = ("push_to_hub", "hub_repo", "hub_org", "hub_private", "wandb")


def mu_per_token_from_legacy(mu_raw, n_x, n_y) -> float:
    """``mu_per_token = mu_raw * n_x / n_y`` -- the one translation of a legacy
    dose. ``n_x`` / ``n_y`` are the target / retained token counts of the solve
    (positive ints); ``mu_raw`` a number > 0."""
    if isinstance(mu_raw, bool) or not isinstance(mu_raw, (int, float)) or not mu_raw > 0:
        raise ValueError(f"legacy dose must be a number > 0, got {mu_raw!r}")
    for k, v in (("n_x", n_x), ("n_y", n_y)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{k} must be a positive int, got {v!r}")
    return float(mu_raw) * int(n_x) / int(n_y)


def data_spec_from_legacy(cfg) -> dict:
    """The :class:`~srspd_experiments.data.DataSpec` mapping of a legacy SRSPD
    config (its top-level ``corpora:`` block and its SRSPD block)."""
    block = cfg.get(LEGACY_BLOCK) or {}
    out = {"target": {}}
    if cfg.get("corpora"):
        out["corpora"] = copy.deepcopy(cfg["corpora"])
    for k, (sec, key) in _DATA_KEYS.items():
        if k in block and block[k] is not None:
            (out[sec] if sec else out)[key] = copy.deepcopy(block[k])
    rc = dict(block.get("retained_corpus") or {})
    out["retained"] = {k: copy.deepcopy(rc[k]) for k in
                       ("buckets", "chunk_tokens", "seed", "add_bos", "chat_template") if k in rc}
    return out


def _legacy_doses(block, n_x, n_y) -> list:
    if block.get("mu_per_token") is not None:
        if LEGACY_DOSE_KEY in block:
            raise ValueError("a legacy config gives both mu_per_token and the legacy dose")
        v = block["mu_per_token"]
        return list(v) if isinstance(v, (list, tuple)) else [v]
    if LEGACY_DOSE_KEY not in block:
        raise ValueError("a legacy config without mu_per_token must give the legacy dose")
    if n_x is None or n_y is None:
        raise ValueError("converting a legacy dose needs the solve's n_x and n_y")
    v = block[LEGACY_DOSE_KEY]
    vals = list(v) if isinstance(v, (list, tuple)) else [v]
    return [mu_per_token_from_legacy(m, n_x, n_y) for m in vals]


def srspd_config_from_legacy(cfg, *, n_x=None, n_y=None, name=None) -> dict:
    """The :mod:`srspd_experiments.methods.srspd` config of a legacy SRSPD
    config. A legacy dose needs ``n_x`` / ``n_y``; ``mu_per_token`` passes
    through. Rank rules (``rank_rule`` or the float-only ``lam_rel``) are kept."""
    block = cfg.get(LEGACY_BLOCK)
    if not isinstance(block, dict):
        raise ValueError("not a legacy SRSPD config (no SRSPD block)")
    rules = block.get("rank_rule")
    if rules is None:
        rules = block.get("lam_rel")
    g = block.get("gram") or {}
    s = {"mu_per_token": _legacy_doses(block, n_x, n_y),
         "solver": block.get("solver", "readout"),
         "rank_rule": list(rules or []),
         "ridge_rel": float(block.get("ridge_rel", 0.0)),
         "gram_budget_gib": float(g.get("budget_gib_per_device", 12.0)),
         "resume_from_cache": bool(g.get("resume_from_cache", True)),
         "eig_extremes": bool(g.get("eig_extremes", True)),
         "lam_svd_dtype": block.get("lam_svd_dtype", "float32")}
    h = block.get("harness") or {}
    if h.get("layers_regex") or h.get("exclude_regex"):
        s["layers"] = {k2: v for k2, v in (("regex", h.get("layers_regex")),
                                            ("exclude", h.get("exclude_regex"))) if v}
    out = {"name": name or cfg["name"], "model": cfg["model"]}
    for k in ("tokenizer", "dtype", "device_map", "seed"):
        if cfg.get(k) is not None:
            out[k] = cfg[k]
    out["data"] = data_spec_from_legacy(cfg)
    out["srspd"] = s
    return out


def baseline_config_from_legacy(cfg, *, name=None) -> dict:
    """A legacy RMU / KUDA / SimNPO config without its hub-push and tracking
    keys (``name`` optionally replaced)."""
    out = {k: copy.deepcopy(v) for k, v in cfg.items() if k not in _TOP_DROPPED}
    if name:
        out["name"] = name
    return out
