"""``rspd.core``: the tensor-level mathematics. Plain tensors in, plain tensors
out -- no model classes, no ``transformers``.

* :mod:`rspd.core.types` -- :class:`Component`, :class:`Components`: what every fit returns;
* :mod:`rspd.core.solve` -- the closed-form component, truncation, the edit ``W - M``,
  leaks, specificity;
* :mod:`rspd.core.coala` -- the COALA projector and compression;
* :mod:`rspd.core.dose` -- ``mu_per_token`` and what it derives;
* :mod:`rspd.core.erank` -- effective rank.
"""
from __future__ import annotations

from .coala import compress_weight, fit_coala
from .dose import Dose, dose_from_mu_per_token, erased_fraction, suggest_mu_per_token
from .erank import effective_rank
from .solve import (CORE_TABLE, ErasedComponent, ErasureConventionError, KeptMap,
                    apply_erasure, erased_component, erased_component_gram,
                    erased_component_grams, factor_grams, leaks, leaks_gram, leaks_grams,
                    readout_gram, specificity, truncate_component)
from .types import Component, Components

__all__ = [
    "Component", "Components",
    "erased_component", "erased_component_gram", "erased_component_grams", "readout_gram",
    "factor_grams", "truncate_component", "apply_erasure", "leaks", "leaks_gram",
    "leaks_grams", "specificity", "ErasedComponent", "KeptMap", "ErasureConventionError",
    "CORE_TABLE", "fit_coala", "compress_weight", "Dose", "dose_from_mu_per_token",
    "suggest_mu_per_token", "erased_fraction", "effective_rank",
]
