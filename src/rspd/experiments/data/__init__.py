"""Data builders: what the target ``X`` and the retained ``Y`` of an erasure
are, and the sets the evaluations score.

Modules:

* :mod:`.spec` -- :class:`DataSpec` / :class:`TargetSpec` / :class:`RetainedSpec`
  dataclasses, loadable from YAML; :func:`build_erasure_data` (in :mod:`.target`)
  turns a spec into token ids for :func:`rspd.erase`;
* :mod:`.corpora` -- the named-corpus registry (WMDP, WikiText, MUSE, Wikipedia,
  config-level ``local_jsonl``) and its loaders, MUSE windows, tokenisation helpers;
* :mod:`.retained_corpus` -- the hashed, cached retained-corpus builder;
* :mod:`.budget` -- the equal-token-budget plumbing of the gradient baselines;
* :mod:`.gpt2_corpus` / :mod:`.gpt2_unlearning` -- the trilingual GPT-2
  pretraining corpus and BPE, and the GPT-2 unlearning sets;
* :mod:`.dutch_sets` -- the Dutch-unlearning sets for a large instruct model;
* ``specs/*.yaml`` -- the published data specs (:func:`spec_path`).

Every function that touches disk takes a root / directory argument and falls
back to :mod:`rspd.experiments.paths` (``$RSPD_DATA_DIR`` or ``~/.cache/rspd``).
"""
from __future__ import annotations

import os

from .spec import DataSpec, HeldoutSpec, RetainedSpec, TargetSpec
from .target import ErasureData, build_erasure_data

SPECS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "specs")


def spec_path(name: str) -> str:
    """Absolute path of a shipped data spec (``specs/<name>.yaml``)."""
    p = os.path.join(SPECS_DIR, name if name.endswith(".yaml") else f"{name}.yaml")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no shipped spec {name!r}; have "
                                f"{sorted(f[:-5] for f in os.listdir(SPECS_DIR))}")
    return p


__all__ = ["DataSpec", "TargetSpec", "RetainedSpec", "HeldoutSpec", "ErasureData",
           "build_erasure_data", "spec_path", "SPECS_DIR"]
