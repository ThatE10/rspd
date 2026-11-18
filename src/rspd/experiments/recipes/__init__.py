"""Reproduction recipes for the published results.

* :mod:`.catalog` -- load / validate the shipped ``<result>.yaml`` recipes, list
  exports, resolve a recipe into a ``reproduce`` plan (no network);
* :mod:`.reproduce` -- execute a plan (train or download, then evaluate);
* :mod:`.compare` -- check result rows against a recipe's published numbers;
* :mod:`.legacy` -- the one translation of legacy configs and doses
  (:func:`~.legacy.mu_per_token_from_legacy`);
* ``hf_weights.csv`` -- the catalogue of published edited weights.

Command line: ``rspd-exp list | show | reproduce | compare``.
"""
from __future__ import annotations

from .catalog import (RECIPES_DIR, RecipeError, find_export, iter_exports, list_results,  # noqa
                      load_recipe, plan, weight_gaps)
from .legacy import mu_per_token_from_legacy  # noqa: F401
