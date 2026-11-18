"""Method runners: SRSPD and the unlearning baselines it is compared against.

* :mod:`.srspd` -- data spec -> :func:`rspd.erase` (both solvers, rank rules,
  a list of ``mu_per_token``) -> one export and held-out readout per dose;
* :mod:`.rmu` -- RMU (Li et al., arXiv:2403.03218; port of
  github.com/centerforaisafety/wmdp, MIT);
* :mod:`.kuda` -- KUDA (Fang et al., arXiv:2602.19275; port of
  github.com/AceNagi/KUDA, MIT);
* :mod:`.simnpo` -- SimNPO (Fan et al., arXiv:2410.07163; port of
  github.com/OPTML-Group/Unlearn-Simple, MIT);
* :mod:`.upstream_kuda`, :mod:`.upstream_kuda_muse`, :mod:`.upstream_simnpo` --
  the authors' own code, cloned at pinned commits and run verbatim (plus
  documented, result-neutral environment shims).

Every port is behaviour-identical to the runs it reproduces: the same
hyperparameters, schedules, seeds and equal-budget accounting
(:mod:`rspd.experiments.data.budget`). The runners take a YAML path or a
mapping and write under :mod:`rspd.experiments.paths` (``logs/methods/<name>``,
``models/methods/<name>``). They never push to a model hub and track only when
the config enables it.

:data:`METHODS` maps a recipe's ``method`` to the importable module path;
:func:`get` imports it (torch / transformers load only then).
"""
from __future__ import annotations

import importlib

METHODS = {
    "srspd": "rspd.experiments.methods.srspd",
    "rmu": "rspd.experiments.methods.rmu",
    "kuda": "rspd.experiments.methods.kuda",
    "simnpo": "rspd.experiments.methods.simnpo",
    "kuda_authors_code": "rspd.experiments.methods.upstream_kuda",
    "kuda_authors_code_muse": "rspd.experiments.methods.upstream_kuda_muse",
    "simnpo_authors_code": "rspd.experiments.methods.upstream_simnpo",
}


def get(method: str):
    """The runner module of ``method`` (a key of :data:`METHODS`)."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; known: {sorted(METHODS)}")
    return importlib.import_module(METHODS[method])


__all__ = ["METHODS", "get"]
