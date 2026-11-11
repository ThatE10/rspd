"""``srspd_experiments.eval.rouge_l`` must reproduce ``rouge_score`` exactly on what
the MUSE metric reads: ``rouge1`` / ``rougeL`` precision, recall and F1 with
``use_stemmer=False``.

The equality tests skip where the package is absent (the point of the native
scorer is an image without it); the hand-computed cases and the backend
plumbing in ``eval.muse`` run everywhere.
"""
from __future__ import annotations

import importlib.util
import random

import pytest

from srspd_experiments.eval import muse as M
from srspd_experiments.eval import rouge_l as RL

TYPES = ("rouge1", "rouge2", "rougeL")

EDGE_PAIRS = [
    ("", ""), ("", "a b c"), ("a b c", ""), ("   ", "\n\t"),
    ("the cat sat on the mat", "the cat sat on the mat"),
    ("the cat sat on the mat", "a cat sat on a mat today"),
    ("a a a a", "a a"), ("a a", "a a a a"), ("a b a b a", "b a b a b"),
    ("Hello, World!", "hello world"), ("Hello,World!", "hello world"),
    ("naïve café — déjà vu", "naive cafe deja vu"),
    ("İstanbul", "istanbul"), ("ÀÉÎÕÜ", "aeiou"), ("straße", "strasse"),
    ("x1 2y 3", "3 2y x1"), ("2024-07-04", "2024 07 04"), ("1,000,000", "1000000"),
    ("it's", "its"), ("don't stop", "dont stop"), ("e-mail", "email"),
    ("under_score", "under score"), ("tab\tsep\nnew", "tab sep new"),
    ("日本語 テキスト", "text"), ("emoji 😀 here", "emoji here"),
    ("a" * 5000, "a" * 5000 + " b"), (" ".join(["w"] * 600), " ".join(["w", "v"] * 300)),
    ("Question: what?\nAnswer: this.", "Answer: this.\n\nQuestion: that?"),
    ("THE the The tHe", "the"), ("A.B.C.", "abc"), ("a.b.c", "a b c"),
]


def _rand_text(rng, alphabet, max_tokens):
    n = rng.randint(0, max_tokens)
    return "".join(rng.choice(alphabet) for _ in range(n))


@pytest.fixture(scope="module")
def official():
    rs = pytest.importorskip("rouge_score")
    from rouge_score import rouge_scorer
    assert rs is not None
    return rouge_scorer.RougeScorer(list(TYPES), use_stemmer=False)


def _assert_same(official, native, ref, pred):
    a, b = official.score(ref, pred), native.score(ref, pred)
    for t in TYPES:
        for field in ("precision", "recall", "fmeasure"):
            x, y = getattr(a[t], field), getattr(b[t], field)
            assert abs(x - y) <= 1e-12, (t, field, x, y, ref[:80], pred[:80])


def test_tokenizer_matches_rouge_score():
    pytest.importorskip("rouge_score")
    from rouge_score import tokenize as official_tok
    for ref, pred in EDGE_PAIRS:
        for s in (ref, pred):
            assert RL.tokenize(s) == official_tok.tokenize(s, None), s[:80]


def test_edge_pairs_match_rouge_score(official):
    native = RL.RougeScorer(list(TYPES), use_stemmer=False)
    for ref, pred in EDGE_PAIRS:
        _assert_same(official, native, ref, pred)
        _assert_same(official, native, pred, ref)


def test_random_pairs_match_rouge_score(official):
    native = RL.RougeScorer(list(TYPES), use_stemmer=False)
    rng = random.Random(1234)
    alphabets = [
        "ab ",                                            # tiny vocab, many repeats
        "abc  ,.\n",                                      # punctuation + whitespace
        "abcdefghijklmnopqrstuvwxyz0123456789 ,.!?;:-'\"()[]\n\t",
        "AaBbCc1 ÀàÉéİı ß 日本 😀 —–_",                     # case / unicode
    ]
    for i in range(400):
        alphabet = alphabets[i % len(alphabets)]
        ref = _rand_text(rng, alphabet, 300)
        pred = _rand_text(rng, alphabet, 300)
        _assert_same(official, native, ref, pred)


def test_hand_computed_values_without_the_package():
    sc = RL.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    s = sc.score("w1 w2", "w1 w2 w3")                   # P 2/3, R 1 -> F 0.8
    assert s["rougeL"].precision == pytest.approx(2 / 3)
    assert s["rougeL"].recall == 1.0 and s["rougeL"].fmeasure == pytest.approx(0.8)
    assert s["rouge1"].fmeasure == pytest.approx(0.8)
    s = sc.score("a b c d", "d c b a")                   # unigrams all hit, LCS = 1
    assert s["rouge1"].fmeasure == 1.0 and s["rougeL"].fmeasure == pytest.approx(0.25)
    s = sc.score("", "a b")
    assert s["rouge1"] == (0, 0, 0.0) and s["rougeL"] == (0, 0, 0)
    assert RL.tokenize("Hello, World! 42") == ["hello", "world", "42"]
    assert RL.fmeasure(0.0, 0.0) == 0.0
    with pytest.raises(ValueError):
        RL.RougeScorer(["rougeLsum"])
    with pytest.raises(ValueError):
        RL.RougeScorer(["rougeL"], use_stemmer=True)


def test_eval_muse_backend_selection_and_record_fields():
    scorer, backend = M.make_rouge_scorer("native")
    assert backend == "native" and sorted(scorer.rouge_types) == sorted(M.NATIVE_ROUGE_TYPES)
    auto_backend = M.make_rouge_scorer()[1]
    if importlib.util.find_spec("rouge_score"):
        assert auto_backend.startswith("rouge_score ")
    else:
        assert auto_backend == "native"
    with pytest.raises(ValueError):
        M.make_rouge_scorer("nltk")

    logger = M.RougeLogger(backend="native")
    assert logger.backend == "native"
    logger.log("p", "w1 w2", "w1 w2 w3")
    logger.log("p", "zz", "w1 w2 w3", question="q")
    assert set(logger.history[0]) == {"prompt", "gt", "response", "rougeL", "rougeL_recall",
                                      "rouge1", "rouge1_recall"}
    assert logger.history[1]["question"] == "q"
    assert logger.history[0]["rougeL"] == pytest.approx(0.8)
    assert logger.history[0]["rougeL_recall"] == 1.0 and logger.history[1]["rougeL"] == 0.0
    agg, hist = logger.report()
    assert agg["mean_rougeL"] == pytest.approx(0.4) and hist is logger.history
