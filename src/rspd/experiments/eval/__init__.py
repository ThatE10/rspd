"""Evaluation harness.

* :mod:`.lm_eval` -- lm-evaluation-harness wrapper (vLLM backend by default,
  HF fallback), the local Global-MMLU tasks (``tasks/``), result parsing into
  :mod:`rspd.experiments.results` rows with the tensor-parallel degree recorded;
* :mod:`.krd` -- Knowledge Removal Degree (single-domain and the two-domain
  harmonic mean) from lm-eval results;
* :mod:`.muse` / :mod:`.rouge_l` -- the MUSE evaluator (VerbMem, KnowMem, PrivLeak);
* :mod:`.ce` / :mod:`.heldout` -- teacher-forced CE and the shared held-out readout;
* :mod:`.trainer` / :mod:`.recovery` -- the GPT-2 training loop and the
  retrain-to-recover study;
* :mod:`.tracking` -- optional Weights & Biases logging (off by default).

Heavy dependencies (``torch``, ``transformers``, ``lm_eval``, ``rouge_score``,
``sklearn``) are imported inside the functions that need them.
"""
