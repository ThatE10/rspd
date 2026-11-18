"""rspd[experiments]: data builders, evaluation harness, baselines and
reproduction recipes for :mod:`rspd`.

* :mod:`rspd.experiments.data` -- target / retained specs and builders, the
  corpus registry, the GPT-2 corpora and unlearning sets, the Dutch sets;
* :mod:`rspd.experiments.eval` -- lm-evaluation-harness wrapper (Global-MMLU
  local tasks, WMDP, MMLU, WikiText), KRD, the MUSE evaluator, held-out CE
  readouts, the GPT-2 retrain-to-recover loop;
* :mod:`rspd.experiments.results` -- the common result-row schema;
* :mod:`rspd.experiments.paths` -- the data root (``$RSPD_DATA_DIR``).

Convention (as in :mod:`rspd`): ``X`` is the TARGET set (erased), ``Y`` the
RETAINED set (kept); the edit is ``W - M``; the dose is ``mu_per_token``.
"""
from __future__ import annotations

__version__ = "0.0.1"
