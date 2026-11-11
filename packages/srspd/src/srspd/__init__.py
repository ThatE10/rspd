"""srspd: closed-form selective erasure of behaviour from linear layers, by subtraction.

Convention (fixed): ``X`` is the TARGET set (erased), ``Y`` the RETAINED set
(kept), both batch-first ``[n, d_in]``; ``W`` the pristine weight
``[d_out, d_in]``. The erased component is

    M = W X^T X (X^T X + mu Y^T Y + delta I)^+

and the edit is ALWAYS ``W_new = W - M`` (:func:`apply_erasure`). Rank
truncation acts on ``M`` by the COALA projector over ``X`` (Parkina & Rakhuba,
arXiv:2507.07580). Leaks ``L_X_rel = ||(W - M) X^T|| / ||W X^T||`` and
``L_Y_rel = ||M Y^T|| / ||W Y^T||`` are measured per edited matrix. The dose is
``mu_per_token`` (:mod:`srspd.dose`); ``mu`` is derived from it.

Modules: :mod:`srspd.solve` (solve, truncation, edit, leaks, specificity),
:mod:`srspd.coala` (the COALA projector and compression), :mod:`srspd.dose`,
:mod:`srspd.erank`, :mod:`srspd.report` (edit-size report).

On a resident HF model: :func:`erase` (harvest, Gram passes, solve at every
requested ``mu_per_token``, apply ``W - M`` to every selected matrix, leaks per
matrix -> :class:`EditReport`), :func:`restore` (bit-for-bit), and
:func:`inspect_model` (what would be edited, input keys, Gram bytes, quirks).
Subpackages :mod:`srspd.models` (selector, quirks, harness, export) and
:mod:`srspd.capture` (activation sources, Gram sinks). Only ``torch`` is
needed to import the package; ``transformers`` / ``safetensors`` (the ``[hf]``
extra) are imported where used. Command line: ``srspd inspect | erase | report``.
"""
from __future__ import annotations

from .coala import compress_weight, fit_coala
from .dose import Dose, dose_from_mu_per_token, erased_fraction, suggest_mu_per_token
from .erank import effective_rank
from .api import EditReport, erase, restore
from .models.inspect import inspect_model
from .solve import (CORE_TABLE, ErasedComponent, ErasureConventionError, KeptMap,
                    apply_erasure, erased_component, erased_component_gram,
                    erased_component_grams, factor_grams, leaks, leaks_gram, leaks_grams,
                    readout_gram, specificity, truncate_component)

__version__ = "0.0.1"

__all__ = [
    "erased_component", "erased_component_gram", "erased_component_grams", "readout_gram",
    "factor_grams", "truncate_component", "apply_erasure", "leaks", "leaks_gram",
    "leaks_grams", "specificity", "ErasedComponent", "KeptMap", "ErasureConventionError",
    "CORE_TABLE", "fit_coala", "compress_weight", "Dose", "dose_from_mu_per_token",
    "suggest_mu_per_token", "erased_fraction", "effective_rank", "erase", "restore",
    "inspect_model", "EditReport", "__version__",
]
