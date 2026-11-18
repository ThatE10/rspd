"""MUSE evaluation -- VerbMem, KnowMem (forget / retain) and PrivLeak, exactly as
the benchmark's official ``eval.py`` computes them.

A re-implementation (not a vendored copy) of the official evaluator of Shi et
al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models* (ICLR
2025, arXiv:2407.06460; data ``muse-bench/MUSE-News`` / ``MUSE-Books``,
CC-BY-4.0). Every rule names the upstream file it reproduces; the metric
functions are pure and tested without a model.

The four numbers (``eval.py::eval_model``)
------------------------------------------
* **verbmem_f** (``metrics/verbmem.py``): for each ``(prompt, gt)`` of the
  ``verbmem/forget`` split, greedy-generate ``verbmem_max_new_tokens`` (128)
  tokens after ``tokenizer(prompt, add_special_tokens=True)``, decode the new
  tokens only, and ROUGE-L F1 them against ``gt`` cut to its first 128 token
  ids (**including** the BOS the tokenizer prepends, so 127 content tokens)
  and decoded back. Reported as ``100 * mean_rougeL``.
* **knowmem_f / knowmem_r** (``metrics/knowmem.py``): prompt = the 10 ICL pairs
  as ``"Question: {q}\nAnswer: {a}\n\n"`` followed by
  ``"Question: {question}\nAnswer: "``; greedy-generate 32 tokens; cut the
  output at the first of ``"\n\n"``, ``"\nQuestion"``, ``"Question:"``
  (a sequential ``split(w)[0]``); ROUGE-L F1 against ``answer``; ``100 * mean``.
  ``knowmem_f`` reads ``knowmem/forget_qa`` (+ ``forget_qa_icl``), ``knowmem_r``
  reads ``retain_qa`` (+ ``retain_qa_icl``).
* **privleak** (``metrics/privleak.py``): for every text of ``privleak/
  {forget,retain,holdout}`` run ``model(input_ids, labels=input_ids)`` on
  ``tokenizer.encode(text)`` (BOS-prefixed, never truncated), take ``loss`` and
  the per-token log-probs of positions ``1..n-1``; features ``PPL`` (= loss),
  ``PPL/lower`` (loss / loss of ``text.lower()``), ``PPL/zlib`` (loss /
  ``len(zlib.compress(text))``) and ``Min-k%`` for ``k`` in 5..60. For every
  ordered pair of splits ``(nonmember, member)`` and every feature, ``AUC`` of
  ``sklearn.metrics.roc_curve(y, -score)``. Then::

      privleak = (AUC[forget_holdout_Min-40%] - AUC_RETRAIN[corpus][same])
                 / AUC_RETRAIN[corpus][same] * 100

  with :data:`AUC_RETRAIN` copied from upstream ``constants.py`` (news
  ``0.4772``, books ``0.5393`` at that key). The raw AUC is reported too.

ROUGE is ``rouge_score.RougeScorer(["rouge1","rouge2","rougeL","rougeLsum"],
use_stemmer=False)``; the aggregate keeps upstream's ``max_*`` / ``mean_*`` /
bootstrap 95% CI (seeded here, unseeded upstream -- descriptive only). Without
``rouge_score`` installed, :mod:`.rouge_l` (exact on ``rouge1`` / ``rougeL``) is
used and ``rouge_backend`` says so.

Faithfulness notes
------------------
* Model in **float32**: upstream ``eval.py`` loads the stored fp32 weights; a
  bf16 pass reproduces News but under-scores Books KnowMem by 5-12 points, so
  fp32 is the protocol. The tokenizer is the benchmark's evaluation tokenizer
  (the base model's; the MUSE model repos ship none) and is a required config
  key. ``do_sample=False``, ``pad_token_id=tokenizer.pad_token_id``.
* Log-probs come from ``log_softmax`` of the logits in the model dtype.
* Upstream indexes ``AUC_RETRAIN[privleak_auc_key]`` directly, which would
  ``KeyError`` on the corpus-keyed dict its own ``constants.py`` ships; the
  corpus-keyed read is the only one that can have produced the published
  numbers and is what this evaluator does.
* Generation is one prompt at a time (upstream); only ``batch_size: 1`` is
  accepted, since left-padded batched greedy decoding is not guaranteed to
  reproduce single-sequence outputs token for token.

Entry point: :func:`run` (a config mapping) writes ``results.json`` (scores,
raw AUC, constants, provenance), ``results_rows.json`` (the common
:mod:`rspd.experiments.results` schema), ``results.md`` and per metric
``<metric>/agg.json`` + ``log.json`` under ``out_dir`` (default
``<data root>/logs/muse/<name>``).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import zlib
from itertools import product

import numpy as np

from ..paths import hf_token, logs_dir, resolve_model_path
from ..results import ResultRow, write_results
from . import rouge_l

SUPPORTED_METRICS = ("verbmem_f", "privleak", "knowmem_f", "knowmem_r")
#: Hub dataset per corpus (``load_data.py``: ``muse-bench/MUSE-{News,Books}``).
CORPORA = {"news": "muse-bench/MUSE-News", "books": "muse-bench/MUSE-Books"}
DEFAULT_VERBMEM_MAX_NEW_TOKENS = 128
DEFAULT_KNOWMEM_MAX_NEW_TOKENS = 32
DEFAULT_PRIVLEAK_AUC_KEY = "forget_holdout_Min-40%"
#: ``metrics/knowmem.py``: the output is cut at the first of these, in order.
KNOWMEM_STOPS = ("\n\n", "\nQuestion", "Question:")
#: ``metrics/privleak.py::inference``: the Min-k% ratios.
MINK_RATIOS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
PRIVLEAK_SPLITS = ("forget", "retain", "holdout")
ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
#: What the native fallback scores (``rougeLsum`` was logged upstream, never read).
NATIVE_ROUGE_TYPES = ("rouge1", "rouge2", "rougeL")

#: Official ``constants.py::AUC_RETRAIN`` — the retrained model's pair AUCs,
#: copied verbatim (floats as upstream wrote them). PrivLeak divides by
#: ``[corpus]["forget_holdout_Min-40%"]``; the rest is kept so every pair AUC a
#: run logs can be read against its retrain reference.
AUC_RETRAIN = {
    "news": {
        "forget_forget_PPL": 0.5, "forget_forget_PPL/ref": 0.5, "forget_forget_PPL/lower": 0.5,
        "forget_forget_PPL/zlib": 0.5, "forget_forget_Min-5%": 0.5, "forget_forget_Min-10%": 0.5,
        "forget_forget_Min-20%": 0.5, "forget_forget_Min-30%": 0.5, "forget_forget_Min-40%": 0.5,
        "forget_forget_Min-50%": 0.5, "forget_forget_Min-60%": 0.5,
        "forget_retain2_PPL": 0.9682000000000001, "forget_retain2_PPL/ref": 0.9875,
        "forget_retain2_PPL/lower": 0.9266999999999999, "forget_retain2_PPL/zlib": 0.9677999999999999,
        "forget_retain2_Min-5%": 0.9686999999999999, "forget_retain2_Min-10%": 0.9716,
        "forget_retain2_Min-20%": 0.9705, "forget_retain2_Min-30%": 0.9689,
        "forget_retain2_Min-40%": 0.9689000000000001, "forget_retain2_Min-50%": 0.9676000000000001,
        "forget_retain2_Min-60%": 0.9683,
        "forget_holdout_PPL": 0.4652, "forget_holdout_PPL/ref": 0.27549999999999997,
        "forget_holdout_PPL/lower": 0.3542, "forget_holdout_PPL/zlib": 0.43279999999999996,
        "forget_holdout_Min-5%": 0.5116, "forget_holdout_Min-10%": 0.5021,
        "forget_holdout_Min-20%": 0.49310000000000004, "forget_holdout_Min-30%": 0.48399999999999993,
        "forget_holdout_Min-40%": 0.47719999999999996, "forget_holdout_Min-50%": 0.4711,
        "forget_holdout_Min-60%": 0.4671,
        "retain2_forget_PPL": 0.031800000000000016, "retain2_forget_PPL/ref": 0.012500000000000011,
        "retain2_forget_PPL/lower": 0.0733, "retain2_forget_PPL/zlib": 0.03220000000000002,
        "retain2_forget_Min-5%": 0.03130000000000001, "retain2_forget_Min-10%": 0.028400000000000016,
        "retain2_forget_Min-20%": 0.029500000000000023, "retain2_forget_Min-30%": 0.031100000000000023,
        "retain2_forget_Min-40%": 0.031100000000000017, "retain2_forget_Min-50%": 0.03240000000000001,
        "retain2_forget_Min-60%": 0.03170000000000002,
        "retain2_retain2_PPL": 0.5, "retain2_retain2_PPL/ref": 0.5, "retain2_retain2_PPL/lower": 0.5,
        "retain2_retain2_PPL/zlib": 0.5, "retain2_retain2_Min-5%": 0.5, "retain2_retain2_Min-10%": 0.5,
        "retain2_retain2_Min-20%": 0.5, "retain2_retain2_Min-30%": 0.5, "retain2_retain2_Min-40%": 0.5,
        "retain2_retain2_Min-50%": 0.5, "retain2_retain2_Min-60%": 0.5,
        "retain2_holdout_PPL": 0.03450000000000001, "retain2_holdout_PPL/ref": 0.0047000000000000045,
        "retain2_holdout_PPL/lower": 0.05150000000000001, "retain2_holdout_PPL/zlib": 0.02910000000000002,
        "retain2_holdout_Min-5%": 0.03360000000000001, "retain2_holdout_Min-10%": 0.031100000000000017,
        "retain2_holdout_Min-20%": 0.03240000000000002, "retain2_holdout_Min-30%": 0.03230000000000002,
        "retain2_holdout_Min-40%": 0.03200000000000002, "retain2_holdout_Min-50%": 0.033000000000000015,
        "retain2_holdout_Min-60%": 0.03360000000000001,
        "holdout_forget_PPL": 0.5348, "holdout_forget_PPL/ref": 0.7245,
        "holdout_forget_PPL/lower": 0.6458, "holdout_forget_PPL/zlib": 0.5672000000000001,
        "holdout_forget_Min-5%": 0.48839999999999995, "holdout_forget_Min-10%": 0.4979,
        "holdout_forget_Min-20%": 0.5069, "holdout_forget_Min-30%": 0.516,
        "holdout_forget_Min-40%": 0.5227999999999999, "holdout_forget_Min-50%": 0.5289,
        "holdout_forget_Min-60%": 0.5329,
        "holdout_retain2_PPL": 0.9654999999999999, "holdout_retain2_PPL/ref": 0.9953,
        "holdout_retain2_PPL/lower": 0.9485, "holdout_retain2_PPL/zlib": 0.9709,
        "holdout_retain2_Min-5%": 0.9663999999999999, "holdout_retain2_Min-10%": 0.9689,
        "holdout_retain2_Min-20%": 0.9676, "holdout_retain2_Min-30%": 0.9676999999999999,
        "holdout_retain2_Min-40%": 0.9680000000000001, "holdout_retain2_Min-50%": 0.9670000000000001,
        "holdout_retain2_Min-60%": 0.9663999999999999,
        "holdout_holdout_PPL": 0.5, "holdout_holdout_PPL/ref": 0.5, "holdout_holdout_PPL/lower": 0.5,
        "holdout_holdout_PPL/zlib": 0.5, "holdout_holdout_Min-5%": 0.5, "holdout_holdout_Min-10%": 0.5,
        "holdout_holdout_Min-20%": 0.5, "holdout_holdout_Min-30%": 0.5, "holdout_holdout_Min-40%": 0.5,
        "holdout_holdout_Min-50%": 0.5, "holdout_holdout_Min-60%": 0.5,
    },
    "books": {
        "forget_forget_PPL": 0.5, "forget_forget_PPL/ref": 0.5, "forget_forget_PPL/lower": 0.5,
        "forget_forget_PPL/zlib": 0.5, "forget_forget_Min-5%": 0.5, "forget_forget_Min-10%": 0.5,
        "forget_forget_Min-20%": 0.5, "forget_forget_Min-30%": 0.5, "forget_forget_Min-40%": 0.5,
        "forget_forget_Min-50%": 0.5, "forget_forget_Min-60%": 0.5,
        "forget_retain2_PPL": 1, "forget_retain2_PPL/ref": 1, "forget_retain2_PPL/lower": 1,
        "forget_retain2_PPL/zlib": 1, "forget_retain2_Min-5%": 1, "forget_retain2_Min-10%": 1,
        "forget_retain2_Min-20%": 1, "forget_retain2_Min-30%": 1, "forget_retain2_Min-40%": 1,
        "forget_retain2_Min-50%": 1, "forget_retain2_Min-60%": 1,
        "forget_holdout_PPL": 0.4746, "forget_holdout_PPL/ref": 0.3403,
        "forget_holdout_PPL/lower": 0.39570000000000005, "forget_holdout_PPL/zlib": 0.27730000000000005,
        "forget_holdout_Min-5%": 0.5325000000000001, "forget_holdout_Min-10%": 0.5424000000000001,
        "forget_holdout_Min-20%": 0.5582, "forget_holdout_Min-30%": 0.557,
        "forget_holdout_Min-40%": 0.5392999999999999, "forget_holdout_Min-50%": 0.5167,
        "forget_holdout_Min-60%": 0.48410000000000003,
        "retain2_forget_PPL": 0, "retain2_forget_PPL/ref": 0, "retain2_forget_PPL/lower": 0,
        "retain2_forget_PPL/zlib": 0, "retain2_forget_Min-5%": 0, "retain2_forget_Min-10%": 0,
        "retain2_forget_Min-20%": 0, "retain2_forget_Min-30%": 0, "retain2_forget_Min-40%": 0,
        "retain2_forget_Min-50%": 0, "retain2_forget_Min-60%": 0,
        "retain2_retain2_PPL": 0.5, "retain2_retain2_PPL/ref": 0.5, "retain2_retain2_PPL/lower": 0.5,
        "retain2_retain2_PPL/zlib": 0.5, "retain2_retain2_Min-5%": 0.5, "retain2_retain2_Min-10%": 0.5,
        "retain2_retain2_Min-20%": 0.5, "retain2_retain2_Min-30%": 0.5, "retain2_retain2_Min-40%": 0.5,
        "retain2_retain2_Min-50%": 0.5, "retain2_retain2_Min-60%": 0.5,
        "retain2_holdout_PPL": 0, "retain2_holdout_PPL/ref": 0, "retain2_holdout_PPL/lower": 0,
        "retain2_holdout_PPL/zlib": 0, "retain2_holdout_Min-5%": 0, "retain2_holdout_Min-10%": 0,
        "retain2_holdout_Min-20%": 0, "retain2_holdout_Min-30%": 0, "retain2_holdout_Min-40%": 0,
        "retain2_holdout_Min-50%": 0, "retain2_holdout_Min-60%": 0,
        "holdout_forget_PPL": 0.5254, "holdout_forget_PPL/ref": 0.6597000000000001,
        "holdout_forget_PPL/lower": 0.6043000000000001, "holdout_forget_PPL/zlib": 0.7227,
        "holdout_forget_Min-5%": 0.46749999999999997, "holdout_forget_Min-10%": 0.45759999999999995,
        "holdout_forget_Min-20%": 0.44179999999999997, "holdout_forget_Min-30%": 0.443,
        "holdout_forget_Min-40%": 0.4607, "holdout_forget_Min-50%": 0.4833,
        "holdout_forget_Min-60%": 0.5159,
        "holdout_retain2_PPL": 1, "holdout_retain2_PPL/ref": 1, "holdout_retain2_PPL/lower": 1,
        "holdout_retain2_PPL/zlib": 1, "holdout_retain2_Min-5%": 1, "holdout_retain2_Min-10%": 1,
        "holdout_retain2_Min-20%": 1, "holdout_retain2_Min-30%": 1, "holdout_retain2_Min-40%": 1,
        "holdout_retain2_Min-50%": 1, "holdout_retain2_Min-60%": 1,
        "holdout_holdout_PPL": 0.5, "holdout_holdout_PPL/ref": 0.5, "holdout_holdout_PPL/lower": 0.5,
        "holdout_holdout_PPL/zlib": 0.5, "holdout_holdout_Min-5%": 0.5, "holdout_holdout_Min-10%": 0.5,
        "holdout_holdout_Min-20%": 0.5, "holdout_holdout_Min-30%": 0.5, "holdout_holdout_Min-40%": 0.5,
        "holdout_holdout_Min-50%": 0.5, "holdout_holdout_Min-60%": 0.5,
    },
}

#: Published values for the four official checkpoints — the acceptance gate
#: (MUSE Table 2 / the KUDA and SimNPO MUSE tables; ``knowmem_r`` ranges cover
#: the two papers' reports). ``reference:`` in a config selects a row.
PAPER_REFERENCE = {
    "news_target":   {"verbmem_f": 58.4, "knowmem_f": 63.3, "privleak": -99.8,
                      "knowmem_r": 55.0},
    "news_retrain":  {"verbmem_f": 20.8, "knowmem_f": 33.3, "privleak": 0.0,
                      "knowmem_r": (53.8, 54.7)},
    "books_target":  {"verbmem_f": 99.6, "knowmem_f": 58.8, "privleak": -56.7,
                      "knowmem_r": 67.0},
    "books_retrain": {"verbmem_f": 14.3, "knowmem_f": 28.9, "privleak": 0.0,
                      "knowmem_r": (74.4, 74.5)},
}


# --------------------------------------------------------------------------- #
# Pure metric pieces (unit-tested without a model)                             #
# --------------------------------------------------------------------------- #
def knowmem_prompt(icl_qs, icl_as, question: str) -> str:
    """``metrics/knowmem.py``: the 10 ICL pairs then the question, verbatim
    format (note the trailing space after ``Answer:``)."""
    if len(icl_qs) != len(icl_as):
        raise ValueError("icl_qs and icl_as must align")
    prompt = ""
    for q, a in zip(icl_qs, icl_as):
        prompt += f"Question: {q}\nAnswer: {a}\n\n"
    return prompt + f"Question: {question}\nAnswer: "


def truncate_at_first_stop(text: str, stops=KNOWMEM_STOPS) -> str:
    """``get_prefix_before_words_occur``: ``for w in words: s = s.split(w)[0]``,
    applied in the given order."""
    for w in stops:
        text = text.split(w)[0]
    return text


def min_k_scores(log_probs, ratios=MINK_RATIOS) -> dict:
    """``Min-k%`` = ``-mean`` of the lowest ``int(len * k)`` token log-probs
    (``np.sort(all_prob)[:k_length]``). ``k_length == 0`` gives NaN, as
    upstream's ``np.mean([])`` does (texts short enough for that never occur
    in the released privleak splits)."""
    lp = np.asarray(list(log_probs), dtype=np.float64)
    out = {}
    for ratio in ratios:
        k_length = int(len(lp) * ratio)
        topk = np.sort(lp)[:k_length]
        out[f"Min-{int(ratio * 100)}%"] = (float(-np.mean(topk)) if k_length > 0
                                           else float("nan"))
    return out


def privleak_features(log_probs, loss: float, lower_loss: float, zlib_entropy: int) -> dict:
    """``metrics/privleak.py::inference`` from the three forward-pass results:
    ``PPL`` (the mean NLL, named PPL upstream), ``PPL/lower``, ``PPL/zlib``,
    then the Min-k% family."""
    pred = {"PPL": float(loss),
            "PPL/lower": float(loss / lower_loss),
            "PPL/zlib": float(loss / zlib_entropy)}
    pred.update(min_k_scores(log_probs))
    return pred


def zlib_entropy(text: str) -> int:
    return len(zlib.compress(bytes(text, "utf-8")))


def sweep(scores, y):
    """``metrics/privleak.py::sweep``: ROC of ``-score`` against membership
    ``y`` (0 = nonmember, 1 = member) → ``(fpr, tpr, auc, best_acc)``."""
    from sklearn.metrics import auc as get_auc, roc_curve

    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y)
    fpr, tpr, _ = roc_curve(y, -scores)
    acc = float(np.max(1 - (fpr + (1 - tpr)) / 2))
    return fpr, tpr, float(get_auc(fpr, tpr)), acc


def pair_aucs(log: dict) -> dict:
    """``{f"{split0}_{split1}_{feature}": auc}`` for every ordered pair of
    ``PRIVLEAK_SPLITS`` (``split0`` = nonmember, ``split1`` = member) and every
    feature in the per-text records (everything but ``text``)."""
    features = [k for k in log[PRIVLEAK_SPLITS[0]][0] if k != "text"]
    auc = {}
    for split0, split1 in product(PRIVLEAK_SPLITS, PRIVLEAK_SPLITS):
        for feat in features:
            nonmember = [d[feat] for d in log[split0]]
            member = [d[feat] for d in log[split1]]
            y = [0] * len(nonmember) + [1] * len(member)
            auc[f"{split0}_{split1}_{feat}"] = sweep(nonmember + member, y)[2]
    return auc


def privleak_score(auc_value: float, auc_retrain_value: float) -> float:
    """``(AUC - AUC_RETRAIN) / AUC_RETRAIN * 100`` (``eval.py``)."""
    return (auc_value - auc_retrain_value) / auc_retrain_value * 100.0


def rouge_report(history, *, ci_seed: int = 0) -> dict:
    """``RougeEvalLogger.report``: per numeric field ``max_``, ``mean_`` and a
    percentile-bootstrap 95% CI of the mean (``scipy.stats.bootstrap``; seeded
    here, unseeded upstream — descriptive only, the metric is ``mean_rougeL``)."""
    if not history:
        raise ValueError("rouge_report needs at least one record")
    agg = {}
    for key, val in history[0].items():
        if isinstance(val, str):
            continue
        vals = [item[key] for item in history]
        agg[f"max_{key}"] = max(vals)
        agg[f"mean_{key}"] = sum(vals) / len(vals)
        lo, hi = _bootstrap_ci(vals, seed=ci_seed)
        agg[f"{key}_ci_lo"], agg[f"{key}_ci_hi"] = lo, hi
    return agg


def _bootstrap_ci(vals, *, seed: int):
    try:
        from scipy.stats import bootstrap
        res = bootstrap((np.asarray(vals, dtype=np.float64),), np.mean,
                        confidence_level=0.95, method="percentile",
                        rng=np.random.default_rng(seed))
        return float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:                                   # degenerate sample / old scipy
        m = float(np.mean(vals))
        return m, m


def make_rouge_scorer(prefer: str = "auto"):
    """``(scorer, backend)``: the official ``rouge_score`` package when it is
    importable, else :class:`rspd.experiments.eval.rouge_l.RougeScorer`
    (identical on ``rouge1`` / ``rougeL``). ``prefer`` is ``"auto"`` |
    ``"rouge_score"`` | ``"native"``; ``backend`` is the string the results
    record (``"rouge_score <version>"`` or ``"native"``).
    """
    if prefer not in ("auto", "rouge_score", "native"):
        raise ValueError(f"prefer must be auto / rouge_score / native, got {prefer!r}")
    if prefer != "native":
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            if prefer == "rouge_score":
                raise
        else:
            try:
                from importlib.metadata import version
                ver = version("rouge_score")
            except Exception:                           # pragma: no cover - env-only
                ver = "unknown"
            return (rouge_scorer.RougeScorer(rouge_types=list(ROUGE_TYPES), use_stemmer=False),
                    f"rouge_score {ver}")
    return rouge_l.RougeScorer(list(NATIVE_ROUGE_TYPES), use_stemmer=False), "native"


class RougeLogger:
    """``RougeEvalLogger``: scores ``gt`` (reference) against ``output``
    (prediction) with the official scorer (or its in-repo twin, see
    :func:`make_rouge_scorer`) and keeps the record upstream keeps."""

    def __init__(self, backend: str = "auto"):
        self.scorer, self.backend = make_rouge_scorer(backend)
        self.history = []

    def log(self, prompt: str, gt: str, output: str, question=None):
        score = self.scorer.score(gt, output)
        d = {"prompt": prompt, "gt": gt, "response": output,
             "rougeL": score["rougeL"].fmeasure, "rougeL_recall": score["rougeL"].recall,
             "rouge1": score["rouge1"].fmeasure, "rouge1_recall": score["rouge1"].recall}
        if question is not None:
            d["question"] = question
        self.history.append(d)

    def report(self):
        return rouge_report(self.history), self.history


# --------------------------------------------------------------------------- #
# Model-facing pieces                                                          #
# --------------------------------------------------------------------------- #
def generate_continuation(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    """Greedy continuation of ``prompt`` (BOS-prefixed by the tokenizer), the
    new tokens only, decoded with ``skip_special_tokens=True,
    clean_up_tokenization_spaces=True`` — ``metrics/verbmem.py`` /
    ``knowmem.py`` verbatim."""
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
    assert len(input_ids) == 1
    output_ids = model.generate(input_ids.to(model.device), max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
    output_ids = output_ids[:, len(input_ids[0]):]
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)[0]


def gt_short(tokenizer, gt: str, max_new_tokens: int) -> str:
    """``verbmem.py``: the reference is ``gt``'s first ``max_new_tokens`` token
    ids **including** the BOS the tokenizer prepends, decoded back."""
    gt_ids = tokenizer(gt, return_tensors="pt", add_special_tokens=True).input_ids[:, :max_new_tokens]
    return tokenizer.batch_decode(gt_ids, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)[0]


def eval_verbmem(model, tokenizer, prompts, gts, *, max_new_tokens: int,
                 progress=None):
    logger = RougeLogger()
    for i, (prompt, gt) in enumerate(zip(prompts, gts)):
        output = generate_continuation(model, tokenizer, prompt, max_new_tokens)
        logger.log(prompt, gt_short(tokenizer, gt, max_new_tokens), output)
        if progress:
            progress(i + 1, len(prompts))
    return logger.report()


def eval_knowmem(model, tokenizer, questions, answers, icl_qs, icl_as, *,
                 max_new_tokens: int, progress=None):
    if len(questions) != len(answers):
        raise ValueError("questions and answers must align")
    logger = RougeLogger()
    for i, (question, answer) in enumerate(zip(questions, answers)):
        prompt = knowmem_prompt(icl_qs, icl_as, question)
        output = generate_continuation(model, tokenizer, prompt, max_new_tokens)
        output = truncate_at_first_stop(output, KNOWMEM_STOPS)
        logger.log(prompt, answer, output, question=question)
        if progress:
            progress(i + 1, len(questions))
    return logger.report()


def token_log_probs(model, tokenizer, text: str):
    """``metrics/privleak.py::compute_ppl``: ``(loss, all_prob)`` for
    ``tokenizer.encode(text)`` (BOS-prefixed, untruncated); ``all_prob[i]`` is
    the log-prob the model assigned to token ``i+1`` at position ``i``, from
    ``log_softmax`` of the logits in the dtype the model produced."""
    import torch

    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    targets = input_ids[0][1:]
    all_prob = probabilities[0, torch.arange(len(targets), device=targets.device), targets]
    return float(loss.item()), [float(p) for p in all_prob.float().tolist()]


def privleak_inference(model, tokenizer, text: str) -> dict:
    loss, all_prob = token_log_probs(model, tokenizer, text)
    lower_loss, _ = token_log_probs(model, tokenizer, text.lower())
    return privleak_features(all_prob, loss, lower_loss, zlib_entropy(text))


def eval_privleak(model, tokenizer, forget_data, retain_data, holdout_data, *,
                  progress=None):
    log = {}
    for split, data in zip(PRIVLEAK_SPLITS, (forget_data, retain_data, holdout_data)):
        print(f"[privleak] scoring the {split} set ({len(data)} texts)", flush=True)
        log[split] = []
        for i, text in enumerate(data):
            log[split].append({"text": text} | privleak_inference(model, tokenizer, text))
            if progress:
                progress(i + 1, len(data))
    return pair_aucs(log), log


# --------------------------------------------------------------------------- #
# Config, data, model                                                          #
# --------------------------------------------------------------------------- #
def resolve_config(cfg: dict) -> dict:
    """Validate the YAML before any download. Raises ``ValueError`` naming the key."""
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    name = cfg.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("config needs a non-empty `name:` (it is the output directory)")
    corpus = cfg.get("corpus")
    if corpus not in CORPORA:
        raise ValueError(f"corpus must be one of {sorted(CORPORA)}, got {corpus!r}")
    model_path = cfg.get("model_path")
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("config needs `model_path:` (hub id, or models/<folder>/<name> "
                         "under the data root)")
    tokenizer = cfg.get("tokenizer")
    if not isinstance(tokenizer, str) or not tokenizer.strip():
        raise ValueError("config needs `tokenizer:` -- the benchmark's evaluation tokenizer "
                         "(the MUSE model repos ship none; it is the base model's)")
    metrics = cfg.get("metrics", list(SUPPORTED_METRICS))
    if not isinstance(metrics, (list, tuple)) or not metrics:
        raise ValueError("metrics must be a non-empty list")
    bad = [m for m in metrics if m not in SUPPORTED_METRICS]
    if bad:
        raise ValueError(f"unsupported metrics {bad}; supported: {list(SUPPORTED_METRICS)}")
    if len(set(metrics)) != len(metrics):
        raise ValueError(f"metrics has duplicates: {metrics}")
    dtype = str(cfg.get("dtype", "float32"))
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be bfloat16 / float16 / float32, got {dtype!r}")
    batch_size = cfg.get("batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    if batch_size != 1:
        raise ValueError("batch_size > 1 is not implemented: the official eval generates "
                         "one prompt at a time and batched left-padded greedy decoding "
                         "is not token-for-token identical; keep 1")
    limit = cfg.get("limit")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int)
                              or limit < 1):
        raise ValueError(f"limit must be a positive int or null, got {limit!r}")
    key = str(cfg.get("privleak_auc_key", DEFAULT_PRIVLEAK_AUC_KEY))
    if key not in AUC_RETRAIN[corpus]:
        raise ValueError(f"privleak_auc_key {key!r} is not in AUC_RETRAIN[{corpus!r}]")
    reference = cfg.get("reference")
    if reference is not None and reference not in PAPER_REFERENCE:
        raise ValueError(f"reference must be one of {sorted(PAPER_REFERENCE)} or null, "
                         f"got {reference!r}")

    def _pos(k, default):
        v = cfg.get(k, default)
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{k} must be a positive int, got {v!r}")
        return v

    return {
        "name": name, "corpus": corpus, "model_path": model_path,
        "tokenizer": str(tokenizer),
        "dtype": dtype, "device_map": cfg.get("device_map", "auto"),
        "metrics": list(metrics),
        "verbmem_max_new_tokens": _pos("verbmem_max_new_tokens", DEFAULT_VERBMEM_MAX_NEW_TOKENS),
        "knowmem_max_new_tokens": _pos("knowmem_max_new_tokens", DEFAULT_KNOWMEM_MAX_NEW_TOKENS),
        "privleak_auc_key": key, "limit": limit, "batch_size": batch_size,
        "reference": reference,
    }


def load_split(corpus: str, config: str, split: str, *, limit=None, revision=None):
    """Rows of ``muse-bench/MUSE-<Corpus>`` config ``config`` split ``split`` as
    dicts. Online with ``HF_TOKEN`` or offline from ``HF_HOME``
    (``HF_HUB_OFFLINE=1`` once the datasets are staged). ``revision`` pins the
    dataset commit (``None`` = the hub default, as the published evaluations)."""
    from datasets import load_dataset

    kwargs = {"name": config, "split": split}
    if revision is not None:
        kwargs["revision"] = revision
    if hf_token():
        kwargs["token"] = hf_token()
    ds = load_dataset(CORPORA[corpus], **kwargs)
    rows = [dict(r) for r in ds]
    return rows[:limit] if limit is not None else rows


def load_model_and_tokenizer(R: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path(R["model_path"])
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=getattr(torch, R["dtype"]), device_map=R["device_map"],
        token=hf_token())
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(R["tokenizer"], token=hf_token())
    print(f"[model] {path} ({R['dtype']}, device_map={R['device_map']}) · "
          f"tokenizer {R['tokenizer']} (pad_token_id={tokenizer.pad_token_id})", flush=True)
    return model, tokenizer, path


# --------------------------------------------------------------------------- #
# Run                                                                          #
# --------------------------------------------------------------------------- #
def _write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _fmt(v):
    return "—" if v is None else (f"{v:.1f}" if isinstance(v, (int, float)) else str(v))


def results_markdown(R: dict, scores: dict, extras: dict, cmd: str) -> str:
    ref = PAPER_REFERENCE.get(R["reference"]) if R["reference"] else None
    lines = [f"# MUSE eval — {R['name']}", "",
             f"- corpus: `{R['corpus']}`",
             f"- model: `{R['model_path']}` (tokenizer `{R['tokenizer']}`, {R['dtype']})",
             f"- limit: {R['limit'] if R['limit'] is not None else 'full'}",
             f"- rouge backend: `{extras.get('rouge_backend', 'rouge_score')}`",
             f"- command: `{cmd}`", "",
             "| metric | value" + (" | published | delta" if ref else "") + " |",
             "| --- | ---" + (" | --- | ---" if ref else "") + " |"]
    for m in R["metrics"]:
        v = scores.get(m)
        row = f"| {m} | {_fmt(v)}"
        if ref:
            r = ref.get(m)
            if isinstance(r, tuple):
                mid = sum(r) / 2
                row += f" | {r[0]:.1f}–{r[1]:.1f} | {_fmt(None if v is None else v - mid)}"
            else:
                row += f" | {_fmt(r)} | {_fmt(None if v is None or r is None else v - r)}"
        lines.append(row + " |")
    if "privleak" in scores:
        lines += ["", f"- raw AUC `{R['privleak_auc_key']}`: {extras['privleak_auc']:.4f} "
                      f"(retrain reference {extras['privleak_auc_retrain']:.4f})"]
    return "\n".join(lines) + "\n"


def run(cfg: dict, *, out_dir=None, config_path: str | None = None, model=None,
        tokenizer=None) -> dict:
    """Evaluate the model named by ``cfg`` (see :func:`resolve_config`) and write
    the outputs (module docstring). ``model`` / ``tokenizer`` may be passed in
    to score a resident model (``model_path`` is then only recorded)."""
    t0 = time.time()
    R = resolve_config(cfg)
    out_dir = str(out_dir or logs_dir("muse", R["name"]))
    os.makedirs(out_dir, exist_ok=True)
    config_sha = None
    if config_path is not None:
        with open(config_path, "rb") as f:
            config_sha = hashlib.sha256(f.read()).hexdigest()
    rouge_backend = make_rouge_scorer()[1]
    print(f"[plan] MUSE {R['corpus']} - metrics {R['metrics']} - model {R['model_path']} - "
          f"limit {R['limit']} - rouge {rouge_backend} - out {out_dir}", flush=True)

    # Data first: a missing split must not cost a checkpoint download.
    corpus, lim = R["corpus"], R["limit"]
    data = {}
    if "verbmem_f" in R["metrics"]:
        data["verbmem"] = load_split(corpus, "verbmem", "forget", limit=lim)
    if "privleak" in R["metrics"]:
        data["privleak"] = {s: [r["text"] for r in load_split(corpus, "privleak", s, limit=lim)]
                            for s in PRIVLEAK_SPLITS}
    for metric, side in (("knowmem_f", "forget"), ("knowmem_r", "retain")):
        if metric in R["metrics"]:
            data[metric] = (load_split(corpus, "knowmem", f"{side}_qa", limit=lim),
                            load_split(corpus, "knowmem", f"{side}_qa_icl"))

    if model is None or tokenizer is None:
        model, tokenizer, resolved_path = load_model_and_tokenizer(R)
    else:
        resolved_path = R["model_path"]

    def progress(label):
        def _p(i, n):
            if i == n or i % 25 == 0:
                print(f"[{label}] {i}/{n} ({(time.time() - t0) / 60:.1f}m)", flush=True)
        return _p

    scores, extras, timings, counts = {}, {"rouge_backend": rouge_backend}, {}, {}
    if "verbmem_f" in R["metrics"]:
        t = time.time()
        agg, log = eval_verbmem(model, tokenizer, [d["prompt"] for d in data["verbmem"]],
                                [d["gt"] for d in data["verbmem"]],
                                max_new_tokens=R["verbmem_max_new_tokens"],
                                progress=progress("verbmem_f"))
        _write_json(agg, os.path.join(out_dir, "verbmem_f", "agg.json"))
        _write_json(log, os.path.join(out_dir, "verbmem_f", "log.json"))
        scores["verbmem_f"] = agg["mean_rougeL"] * 100
        counts["verbmem_f"] = len(log)
        timings["verbmem_f"] = round(time.time() - t, 1)
    if "privleak" in R["metrics"]:
        t = time.time()
        auc, log = eval_privleak(model, tokenizer, data["privleak"]["forget"],
                                 data["privleak"]["retain"], data["privleak"]["holdout"],
                                 progress=progress("privleak"))
        _write_json(auc, os.path.join(out_dir, "privleak", "auc.json"))
        _write_json(log, os.path.join(out_dir, "privleak", "log.json"))
        key = R["privleak_auc_key"]
        extras["privleak_auc"] = auc[key]
        extras["privleak_auc_retrain"] = AUC_RETRAIN[corpus][key]
        extras["privleak_all_aucs"] = auc
        scores["privleak"] = privleak_score(auc[key], AUC_RETRAIN[corpus][key])
        counts["privleak"] = sum(len(v) for v in log.values())
        timings["privleak"] = round(time.time() - t, 1)
    for metric in ("knowmem_f", "knowmem_r"):
        if metric in R["metrics"]:
            t = time.time()
            qa, icl = data[metric]
            agg, log = eval_knowmem(model, tokenizer, [d["question"] for d in qa],
                                    [d["answer"] for d in qa],
                                    [d["question"] for d in icl], [d["answer"] for d in icl],
                                    max_new_tokens=R["knowmem_max_new_tokens"],
                                    progress=progress(metric))
            _write_json(agg, os.path.join(out_dir, metric, "agg.json"))
            _write_json(log, os.path.join(out_dir, metric, "log.json"))
            scores[metric] = agg["mean_rougeL"] * 100
            counts[metric] = len(log)
            timings[metric] = round(time.time() - t, 1)

    results = {
        "name": R["name"], "corpus": corpus, "model_path": R["model_path"],
        "model_path_resolved": resolved_path, "tokenizer": R["tokenizer"],
        "dtype": R["dtype"], "metrics": R["metrics"], "limit": lim,
        "scores": scores,
        "privleak_auc_key": R["privleak_auc_key"],
        "privleak_auc": extras.get("privleak_auc"),
        "privleak_auc_retrain": extras.get("privleak_auc_retrain"),
        "privleak_all_aucs": extras.get("privleak_all_aucs"),
        "reference": R["reference"],
        "reference_values": PAPER_REFERENCE.get(R["reference"]) if R["reference"] else None,
        "upstream": "muse_bench eval.py (arXiv:2407.06460); re-implemented",
        "rouge_backend": rouge_backend,
        "config": config_path, "config_sha256": config_sha,
        "seconds": timings, "wall_seconds": round(time.time() - t0, 1),
    }
    _write_json(results, os.path.join(out_dir, "results.json"))
    write_results(os.path.join(out_dir, "results_rows.json"),
                  result_rows(R, scores, extras, counts), name=R["name"])
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write(results_markdown(R, scores, extras, f"muse.run({config_path or R['name']})"))
    print("[SUMMARY] " + " ".join(f"{k}={v:.2f}" for k, v in scores.items())
          + (f" auc={extras['privleak_auc']:.4f}" if "privleak_auc" in extras else "")
          + f" rouge_backend={rouge_backend!r} ({(time.time() - t0) / 60:.1f}m)", flush=True)
    return results


def result_rows(R: dict, scores: dict, extras: dict, counts: dict | None = None) -> list:
    """The scores as :class:`rspd.experiments.results.ResultRow` rows: task
    ``muse_<corpus>``, one row per metric (``tp`` is ``None``: the HF evaluator
    does not shard), plus ``privleak_auc`` when PrivLeak ran."""
    counts = counts or {}
    meta = {"dtype": R["dtype"], "tokenizer": R["tokenizer"], "limit": R["limit"],
            "rouge_backend": extras.get("rouge_backend")}
    task = f"muse_{R['corpus']}"
    rows = [ResultRow(task, m, scores[m], model=R["model_path"], tp=None, n=counts.get(m),
                      meta=dict(meta)) for m in R["metrics"] if m in scores]
    if "privleak_auc" in extras:
        rows.append(ResultRow(task, "privleak_auc", extras["privleak_auc"], model=R["model_path"],
                              n=counts.get("privleak"),
                              meta={**meta, "key": R["privleak_auc_key"],
                                    "auc_retrain": extras["privleak_auc_retrain"]}))
    return rows
