"""A dependency-free re-implementation of ``rouge_score`` for the MUSE metric.

The MUSE evaluator (:mod:`rspd.experiments.eval.muse`) scores with
``rouge_score.rouge_scorer.RougeScorer([...], use_stemmer=False)`` and reads
only ``rouge1`` / ``rougeL`` precision, recall and F1. ``rouge_score`` is the
``[muse]`` extra; when it is not installed this module is used instead. It
reproduces the package **exactly** for what the metric uses:

* **Tokenizer** (``rouge_score/tokenize.py``, no stemmer): ``str.lower()``,
  every run of characters outside ``[a-z0-9]`` replaced by one space, split on
  whitespace, empty tokens dropped. (Upstream also stems tokens longer than 3
  characters when a stemmer is given — never the case here.)
* **rougeN** (``rouge_scorer._score_ngrams``): ``Counter`` intersection of the
  n-gram multisets; ``P = hits / max(len(pred_ngrams), 1)``,
  ``R = hits / max(len(ref_ngrams), 1)``.
* **rougeL** (``rouge_scorer._score_lcs``): LCS length over the token sequences;
  ``P = lcs / len(pred)``, ``R = lcs / len(ref)``; all three ``0`` when either
  side has no tokens.
* **F** (``scoring.fmeasure``): ``2PR / (P + R)`` if ``P + R > 0`` else ``0.0``.

``rougeLsum`` (newline sentence splitting + union-LCS) is not reproduced — it was
only ever logged upstream, never read. The tests compare this module against
the real package on randomised and edge-case pairs and assert equality to
1e-12 wherever ``rouge_score`` is importable.

Interface mirrors the package: ``RougeScorer(rouge_types, use_stemmer=False)``
with ``.rouge_types`` and ``.score(target, prediction) -> {type: Score}``.
"""
from __future__ import annotations

import collections
import re

Score = collections.namedtuple("Score", ["precision", "recall", "fmeasure"])

_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")
_SPACES_RE = re.compile(r"\s+")
_VALID_TOKEN_RE = re.compile(r"^[a-z0-9]+$")
_ROUGE_N_RE = re.compile(r"^rouge([0-9]+)$")

SUPPORTED_TYPES = ("rouge1", "rouge2", "rougeL")


def tokenize(text: str) -> list:
    """``rouge_score.tokenize.tokenize(text, stemmer=None)``."""
    text = _NON_ALPHANUM_RE.sub(" ", str(text).lower())
    return [t for t in _SPACES_RE.split(text) if _VALID_TOKEN_RE.match(t)]


def fmeasure(precision: float, recall: float) -> float:
    """``rouge_score.scoring.fmeasure``."""
    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    return 0.0


def _ngrams(tokens, n: int):
    return collections.Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def score_ngrams(target_tokens, prediction_tokens, n: int) -> Score:
    """``rouge_scorer._score_ngrams`` on the n-gram ``Counter`` of each side."""
    target, prediction = _ngrams(target_tokens, n), _ngrams(prediction_tokens, n)
    hits = 0
    for ngram in target:
        hits += min(target[ngram], prediction[ngram])
    precision = hits / max(sum(prediction.values()), 1)
    recall = hits / max(sum(target.values()), 1)
    return Score(precision, recall, fmeasure(precision, recall))


def lcs_length(ref, can) -> int:
    """Length of the longest common subsequence (``rouge_scorer._lcs_table``,
    kept to one rolling row)."""
    if not ref or not can:
        return 0
    prev = [0] * (len(can) + 1)
    for r in ref:
        cur = [0] * (len(can) + 1)
        for j, c in enumerate(can, start=1):
            cur[j] = prev[j - 1] + 1 if r == c else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def score_lcs(target_tokens, prediction_tokens) -> Score:
    """``rouge_scorer._score_lcs``."""
    if not target_tokens or not prediction_tokens:
        return Score(0, 0, 0)
    lcs = lcs_length(target_tokens, prediction_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(target_tokens)
    return Score(precision, recall, fmeasure(precision, recall))


class RougeScorer:
    """Drop-in for ``rouge_score.rouge_scorer.RougeScorer`` restricted to
    ``rouge1`` / ``rouge2`` / ``rougeL`` (any ``rougeN``) without stemming."""

    def __init__(self, rouge_types, use_stemmer: bool = False, split_summaries: bool = False):
        if use_stemmer:
            raise ValueError("the native scorer has no stemmer; MUSE uses use_stemmer=False")
        if split_summaries:
            raise ValueError("the native scorer does not implement split_summaries")
        rouge_types = list(rouge_types)
        for t in rouge_types:
            if t != "rougeL" and not _ROUGE_N_RE.match(t):
                raise ValueError(f"unsupported rouge type {t!r} (native scorer: rougeN, rougeL)")
        self.rouge_types = rouge_types

    def score(self, target: str, prediction: str) -> dict:
        target_tokens, prediction_tokens = tokenize(target), tokenize(prediction)
        out = {}
        for t in self.rouge_types:
            if t == "rougeL":
                out[t] = score_lcs(target_tokens, prediction_tokens)
            else:
                out[t] = score_ngrams(target_tokens, prediction_tokens,
                                      int(_ROUGE_N_RE.match(t).group(1)))
        return out
