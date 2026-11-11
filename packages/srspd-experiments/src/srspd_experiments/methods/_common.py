"""The shared scaffold every gradient baseline in :mod:`srspd_experiments.methods`
reuses, gathered into one namespace (``from . import _common as C``) so the
method modules differ only in their loss and update rule, never in how they load
a corpus, resolve a decoder's blocks, track a run or save a checkpoint.

Everything here is re-exported from :mod:`srspd_experiments.data.corpora`,
:mod:`srspd_experiments.data.budget`, :mod:`srspd_experiments.eval.heldout`,
:mod:`srspd_experiments.models` and :mod:`srspd_experiments.eval.tracking`; a
test that wants to replace a loader for every caller patches the defining
module, not this one.

Two behaviours differ from the scaffold the published runs used, neither of
which can change a trained weight:

* tracking is OFF unless the config's ``tracking:`` (or legacy ``wandb:``) block
  says ``enabled: true`` (:func:`init_tracker`);
* nothing is pushed to a model hub: :func:`push_model` records why and returns.
"""
from __future__ import annotations

from ..data.budget import (BUDGET_KEYS, RETAIN_STREAM_RULE, Exposure, assert_budget,  # noqa: F401
                           budget_retain_plan, packed_retain_stream, resolve_budget,
                           separator_token_id, spread_ranges, target_tokens_unique,
                           windows_to_batch)
from ..data.corpora import (CORPORA, CORPUS_KINDS, DEFAULT_MIN_LEN, FORGET_MAX_LENGTH,  # noqa: F401
                            MUSE_JOIN, RETAIN_MAX_LENGTH, add_position_ids, adds_bos,
                            chunk_token_ids, corpus_kind, encode, encode_all,
                            is_chunked_corpus, is_retain_corpus, load_corpus_texts,
                            load_corpus_texts_all, load_corpus_windows, load_text_batches,
                            load_tokenizer, register_config_corpora, resolve_local_path,
                            split_fit_heldout, strip_redundant_bos, tokenize_batch)
from ..eval.heldout import (HELDOUT_DEFAULT_MAX_LENGTH, HELDOUT_DEFAULT_ROWS,  # noqa: F401
                            HeldoutReadout, heldout_corpora, heldout_delta, heldout_metrics,
                            heldout_readout, heldout_rows, heldout_summary,
                            load_heldout_readout, pooled_ce, print_heldout,
                            resolve_heldout_config)
from ..eval.tracking import Tracker
from ..models import (DEFAULT_BLOCK_PARAM_NAMES, DEFAULT_MLP_OUT_MODULE,  # noqa: F401
                      MLP_OUT_MODULE_NAMES, get_block_module, get_decoder_layers,
                      get_down_proj_module, get_mlp_out_module, get_named_block_params,
                      get_named_down_proj, load_causal_lm, save_model, save_processor,
                      weight_layout, write_metrics)
from ..paths import hf_token  # noqa: F401

#: Why :func:`push_model` never pushes.
NO_PUSH_REASON = "srspd-experiments never pushes checkpoints; upload the saved directory yourself"


def init_tracker(cfg, *, name, out_dir, config=None) -> Tracker:
    """The run's tracker: the config's ``tracking:`` block, or a legacy
    ``wandb:`` block (``enabled`` / ``project`` / ``entity`` / ``tags`` /
    ``notes``; ``mode`` online only when it says so). Off unless enabled."""
    cfg = cfg if isinstance(cfg, dict) else {}
    block = cfg.get("tracking")
    if block is None:
        legacy = cfg.get("wandb")
        block = None if not isinstance(legacy, dict) else {
            "enabled": bool(legacy.get("enabled", False)), "mode": legacy.get("mode", "offline"),
            "project": legacy.get("project"), "entity": legacy.get("entity"),
            "tags": legacy.get("tags"), "notes": legacy.get("notes")}
    return Tracker(block, out_dir=out_dir, run_name=name, config=config)


#: The name the method modules call (kept so their bodies read as before).
init_wandb = init_tracker


def push_model(model, tokenizer, repo_name, *, private=True, org=None, enabled=False) -> dict:
    """No-op: returns ``{"pushed": False, "repo_id", "commit": None, "reason"}``."""
    repo_id = f"{org}/{repo_name}" if org else repo_name
    print(f"[push] skipped -- {NO_PUSH_REASON}", flush=True)
    return {"pushed": False, "repo_id": repo_id, "commit": None, "reason": NO_PUSH_REASON,
            "requested": bool(enabled)}


def config_file(config, *, method: str) -> str:
    """A run config as a YAML path: a path is returned as is; a mapping is
    written (sorted keys off, ``safe_dump``) to
    ``<data root>/logs/methods/<name>/config.input.yaml`` and that path returned,
    so the runner reads, hashes and copies the same bytes either way."""
    import os

    import yaml

    from ..paths import logs_dir

    if isinstance(config, (str, os.PathLike)):
        return os.fspath(config)
    if not isinstance(config, dict):
        raise TypeError(f"{method}: config must be a YAML path or a mapping, "
                        f"got {type(config).__name__}")
    name = config.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("config has no `name:` key (it is the output directory)")
    path = os.path.join(str(logs_dir("methods", name)), "config.input.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(dict(config), f, sort_keys=False)
    return path


def git_short_sha() -> str | None:
    """The short commit of the installed source tree (``-dirty`` when modified),
    or ``None`` outside a git checkout."""
    import os
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if out.returncode != 0 or not out.stdout.strip():
            return None
        dirty = subprocess.call(["git", "-C", here, "diff", "--quiet"],
                                stderr=subprocess.DEVNULL) != 0
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return None
