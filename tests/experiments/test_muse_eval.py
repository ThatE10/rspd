"""Unit tests for ``rspd.experiments.eval.muse`` -- the official MUSE metrics,
pinned without a model.

The pure pieces (prompt format, the knowmem truncation rule, Min-k% scoring,
the AUC pairing, the PrivLeak formula with the official constants, the ROUGE
aggregate, config validation) are tested directly; the three evaluation loops
are exercised against a *fake* model/tokenizer pair that records what it was
asked, so the slicing rules upstream relies on (drop the prompt tokens, cut
``gt`` to 128 ids *including* BOS, BOS-prefixed untruncated privleak encoding,
log-prob of token ``i+1`` at position ``i``) are checked byte for byte.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from rspd.experiments.eval import muse as M
from rspd.experiments.paths import resolve_model_path


# --------------------------------------------------------------------------- #
# Prompt construction and the knowmem truncation rule                          #
# --------------------------------------------------------------------------- #
def test_knowmem_prompt_is_the_official_icl_format():
    p = M.knowmem_prompt(["q1", "q2"], ["a1", "a2"], "Q?")
    assert p == "Question: q1\nAnswer: a1\n\nQuestion: q2\nAnswer: a2\n\nQuestion: Q?\nAnswer: "
    assert M.knowmem_prompt([], [], "Q?") == "Question: Q?\nAnswer: "
    with pytest.raises(ValueError):
        M.knowmem_prompt(["q"], [], "Q?")


@pytest.mark.parametrize("raw, expected", [
    ("Paris.\n\nQuestion: next", "Paris."),
    ("Paris.\nQuestion: next", "Paris."),
    ("Paris. Question: next", "Paris. "),
    ("Paris.\nAnswer: more\nQuestion", "Paris.\nAnswer: more"),
    ("no stop here", "no stop here"),
    ("", ""),
])
def test_truncate_at_first_stop_applies_the_splits_in_order(raw, expected):
    assert M.truncate_at_first_stop(raw) == expected


def test_truncation_order_matters_like_upstream():
    """``split("\\n\\n")[0]`` runs first, then ``"\\nQuestion"``, then
    ``"Question:"`` — a sequential ``split(w)[0]``, not a min over positions.
    The two readings agree on every string (each split keeps a prefix), so pin
    the sequential form on a case where the stops nest."""
    s = "a\nQuestion: b\n\nc"
    assert M.truncate_at_first_stop(s) == "a"


# --------------------------------------------------------------------------- #
# PrivLeak pieces                                                               #
# --------------------------------------------------------------------------- #
def test_min_k_scores_take_the_lowest_int_fraction_of_log_probs():
    lp = [-1.0, -5.0, -2.0, -0.5, -4.0, -3.0, -0.1, -6.0, -2.5, -1.5]   # n = 10
    out = M.min_k_scores(lp)
    assert set(out) == {"Min-5%", "Min-10%", "Min-20%", "Min-30%", "Min-40%",
                        "Min-50%", "Min-60%"}
    # int(10 * 0.05) = 0 -> NaN (np.mean([]) upstream)
    assert math.isnan(out["Min-5%"])
    assert out["Min-10%"] == pytest.approx(6.0)             # lowest 1: -6
    assert out["Min-20%"] == pytest.approx((6.0 + 5.0) / 2)
    assert out["Min-40%"] == pytest.approx((6 + 5 + 4 + 3) / 4)
    assert out["Min-60%"] == pytest.approx((6 + 5 + 4 + 3 + 2.5 + 2) / 6)


def test_privleak_features_are_the_three_ratios_plus_min_k():
    feats = M.privleak_features([-1.0, -2.0, -3.0, -4.0], loss=2.5, lower_loss=2.0,
                                zlib_entropy=50)
    assert feats["PPL"] == 2.5
    assert feats["PPL/lower"] == pytest.approx(1.25)
    assert feats["PPL/zlib"] == pytest.approx(0.05)
    assert feats["Min-50%"] == pytest.approx(3.5)
    assert "text" not in feats


def test_zlib_entropy_is_the_compressed_byte_length():
    import zlib
    assert M.zlib_entropy("hello " * 50) == len(zlib.compress(b"hello " * 50))


def test_sweep_is_roc_of_minus_score_with_member_one():
    """A lower score (higher likelihood) means "member": a perfectly separated
    set where members score lower gives AUC 1, the reverse gives 0."""
    pytest.importorskip("sklearn")
    scores = [5.0, 4.0, 3.0, 1.0, 0.5, 0.2]
    y = [0, 0, 0, 1, 1, 1]
    _, _, auc, acc = M.sweep(scores, y)
    assert auc == pytest.approx(1.0) and acc == pytest.approx(1.0)
    _, _, auc_rev, _ = M.sweep(scores[::-1], y)
    assert auc_rev == pytest.approx(0.0)


def test_pair_aucs_covers_every_ordered_pair_and_feature():
    pytest.importorskip("sklearn")
    log = {
        "forget":  [{"text": "f1", "PPL": 1.0, "Min-40%": 1.0}, {"text": "f2", "PPL": 1.5, "Min-40%": 1.2}],
        "retain":  [{"text": "r1", "PPL": 3.0, "Min-40%": 2.0}, {"text": "r2", "PPL": 3.5, "Min-40%": 2.5}],
        "holdout": [{"text": "h1", "PPL": 5.0, "Min-40%": 4.0}, {"text": "h2", "PPL": 6.0, "Min-40%": 5.0}],
    }
    auc = M.pair_aucs(log)
    assert len(auc) == 3 * 3 * 2
    # nonmember = forget (low scores), member = holdout (high scores): members
    # score HIGHER, so -score ranks them lower -> AUC 0
    assert auc["forget_holdout_PPL"] == pytest.approx(0.0)
    assert auc["holdout_forget_PPL"] == pytest.approx(1.0)
    assert auc["forget_forget_Min-40%"] == pytest.approx(0.5)


def test_privleak_formula_and_the_official_constants():
    assert M.AUC_RETRAIN["news"]["forget_holdout_Min-40%"] == pytest.approx(0.4772)
    assert M.AUC_RETRAIN["books"]["forget_holdout_Min-40%"] == pytest.approx(0.5393)
    assert len(M.AUC_RETRAIN["news"]) == len(M.AUC_RETRAIN["books"]) == 9 * 11
    # the retrain model itself scores 0 by construction
    assert M.privleak_score(0.4772, 0.4772) == 0.0
    # a target model that leaks (AUC ~ 0) is ~ -100
    assert M.privleak_score(0.001, 0.4772) == pytest.approx(-99.79, abs=0.01)
    assert M.privleak_score(0.5393 * 1.5, 0.5393) == pytest.approx(50.0)


def test_paper_reference_rows_match_the_readme_gate():
    r = M.PAPER_REFERENCE
    assert r["news_target"] == {"verbmem_f": 58.4, "knowmem_f": 63.3, "privleak": -99.8,
                                "knowmem_r": 55.0}
    assert r["books_target"]["verbmem_f"] == 99.6 and r["books_retrain"]["knowmem_f"] == 28.9
    assert r["news_retrain"]["privleak"] == 0.0 and r["books_retrain"]["privleak"] == 0.0


# --------------------------------------------------------------------------- #
# ROUGE aggregate                                                               #
# --------------------------------------------------------------------------- #
def test_rouge_report_keeps_upstreams_max_mean_and_ci_fields():
    hist = [{"prompt": "p", "gt": "g", "response": "r", "rougeL": 0.2, "rougeL_recall": 0.1,
             "rouge1": 0.3, "rouge1_recall": 0.2},
            {"prompt": "p", "gt": "g", "response": "r", "rougeL": 0.6, "rougeL_recall": 0.5,
             "rouge1": 0.7, "rouge1_recall": 0.6}]
    agg = M.rouge_report(hist)
    assert agg["mean_rougeL"] == pytest.approx(0.4) and agg["max_rougeL"] == 0.6
    assert agg["mean_rouge1"] == pytest.approx(0.5)
    assert set(agg) == {f"{p}_{k}" for p in ("max", "mean")
                        for k in ("rougeL", "rougeL_recall", "rouge1", "rouge1_recall")} | \
        {f"{k}_ci_{s}" for k in ("rougeL", "rougeL_recall", "rouge1", "rouge1_recall")
         for s in ("lo", "hi")}
    assert agg["rougeL_ci_lo"] <= agg["mean_rougeL"] <= agg["rougeL_ci_hi"]
    with pytest.raises(ValueError):
        M.rouge_report([])


def test_rouge_logger_uses_the_official_scorer_settings():
    rs = pytest.importorskip("rouge_score")
    logger = M.RougeLogger()
    assert sorted(logger.scorer.rouge_types) == sorted(M.ROUGE_TYPES)
    # use_stemmer=False upstream: the scorer's tokenizer carries no stemmer
    assert getattr(logger.scorer._tokenizer, "_stemmer", None) is None
    logger.log("prompt", "the cat sat on the mat", "the cat sat on the mat")
    logger.log("prompt", "the cat sat on the mat", "a dog", question="q")
    agg, hist = logger.report()
    assert hist[0]["rougeL"] == pytest.approx(1.0) and hist[1]["rougeL"] == 0.0
    assert hist[1]["question"] == "q" and "question" not in hist[0]
    assert agg["mean_rougeL"] == pytest.approx(0.5)
    del rs


# --------------------------------------------------------------------------- #
# The evaluation loops against a fake model/tokenizer                           #
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


class _FakeTok:
    """Whitespace tokenizer. ``__call__``/``encode`` prepend BOS (id 1) when
    ``add_special_tokens`` (the default); ids are ``10 + word index`` in a
    fixed vocabulary; decoding skips BOS/EOS and joins with spaces."""
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    vocab = ["<pad>", "<s>", "</s>"]

    def _id(self, w):
        if w not in self.vocab:
            self.vocab.append(w)
        return self.vocab.index(w)

    def _ids(self, text, add_special_tokens):
        ids = [self._id(w) for w in text.split()]
        return ([self.bos_token_id] if add_special_tokens else []) + ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._ids(text, add_special_tokens)

        class _Enc:
            input_ids = torch.tensor([ids])
        return _Enc()

    def encode(self, text, add_special_tokens=True):
        return self._ids(text, add_special_tokens)

    def batch_decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        out = []
        for row in ids.tolist():
            words = [self.vocab[i] for i in row if not (skip_special_tokens and i in (1, 2))]
            out.append(" ".join(words))
        return out


class _FakeModel:
    """``generate`` echoes a fixed continuation after the prompt; ``__call__``
    returns a loss and logits that put all mass on a known target so the
    log-prob slicing is checkable."""
    device = torch.device("cpu")

    def __init__(self, tok, continuation):
        self.tok = tok
        self.continuation = continuation
        self.calls = []

    def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id):
        assert do_sample is False and pad_token_id is None
        self.calls.append(("generate", input_ids.tolist(), max_new_tokens))
        cont = [self.tok._id(w) for w in self.continuation.split()][:max_new_tokens]
        return torch.cat([input_ids, torch.tensor([cont])], dim=1)

    def __call__(self, input_ids, labels=None):
        self.calls.append(("forward", input_ids.tolist()))
        n = input_ids.shape[1]
        V = len(self.tok.vocab) + 1
        logits = torch.full((1, n, V), -10.0)
        # position i predicts token i+1 with log-prob ~ -(i+1)/10 (bf16 not used here)
        for i in range(n - 1):
            logits[0, i, input_ids[0, i + 1]] = 10.0
        loss = torch.tensor(float(n))             # anything; it is passed through
        return loss, logits


def test_generate_continuation_returns_only_the_new_tokens():
    tok = _FakeTok()
    model = _FakeModel(tok, "x y z")
    out = M.generate_continuation(model, tok, "a b", 2)
    assert out == "x y"
    kind, ids, mnt = model.calls[-1]
    assert ids[0][0] == tok.bos_token_id and len(ids[0]) == 3 and mnt == 2


def test_gt_short_counts_the_bos_among_the_first_max_new_tokens_ids():
    """Upstream cuts ``gt_ids[:, :128]`` AFTER the tokenizer prepended BOS, so
    the reference holds ``max_new_tokens - 1`` content tokens."""
    tok = _FakeTok()
    assert M.gt_short(tok, "w1 w2 w3 w4 w5", 3) == "w1 w2"


def test_eval_verbmem_scores_decoded_output_against_the_truncated_gt():
    pytest.importorskip("rouge_score")
    tok = _FakeTok()
    model = _FakeModel(tok, "w1 w2 w3 w4")
    agg, log = M.eval_verbmem(model, tok, ["p1", "p2"], ["w1 w2 w3 w4 w5", "zz"],
                              max_new_tokens=3)
    # output = first 3 generated words; gt cut to 2 content words (3 ids incl. BOS)
    assert log[0]["response"] == "w1 w2 w3" and log[0]["gt"] == "w1 w2"
    assert log[0]["rougeL"] == pytest.approx(0.8)            # P 2/3, R 1 -> F 0.8
    assert log[1]["rougeL"] == 0.0
    assert agg["mean_rougeL"] == pytest.approx(0.4)


def test_eval_knowmem_builds_the_icl_prompt_and_truncates_the_answer():
    pytest.importorskip("rouge_score")
    tok = _FakeTok()
    model = _FakeModel(tok, "Paris \n\n Question: next")
    agg, log = M.eval_knowmem(model, tok, ["Capital?"], ["Paris"], ["q0"], ["a0"],
                              max_new_tokens=32)
    assert log[0]["prompt"] == "Question: q0\nAnswer: a0\n\nQuestion: Capital?\nAnswer: "
    assert log[0]["question"] == "Capital?"
    # the fake tokenizer splits on whitespace, so "\n\n" survives decoding as its
    # own token and the stop rule cuts there
    assert log[0]["response"] == "Paris "
    assert log[0]["rougeL"] == pytest.approx(1.0) and agg["mean_rougeL"] == pytest.approx(1.0)


def test_token_log_probs_follow_compute_ppl_slicing():
    """BOS-prefixed encoding, no truncation, ``all_prob[i]`` = log-prob of
    token ``i+1`` at position ``i`` — ``n - 1`` values for ``n`` ids."""
    tok = _FakeTok()
    model = _FakeModel(tok, "")
    loss, lp = M.token_log_probs(model, tok, "a b c d")
    kind, ids = model.calls[-1]
    assert ids[0][0] == tok.bos_token_id and len(ids[0]) == 5
    assert loss == 5.0 and len(lp) == 4
    # the fake puts +10 on the right next token and -10 elsewhere -> ~0 log-prob
    assert all(v > -1e-3 for v in lp)


def test_eval_privleak_logs_each_split_and_pairs_the_aucs():
    pytest.importorskip("sklearn")
    tok = _FakeTok()
    model = _FakeModel(tok, "")
    # >= 20 tokens each so every int(n * k) >= 1 and no Min-k% is NaN (as the
    # released privleak splits are; sklearn refuses NaN scores)
    f = "f " + " ".join(f"w{i}" for i in range(24))
    r = "r " + " ".join(f"w{i}" for i in range(24))
    h = "h " + " ".join(f"w{i}" for i in range(30))
    auc, log = M.eval_privleak(model, tok, [f], [r], [h])
    assert set(log) == {"forget", "retain", "holdout"}
    assert log["forget"][0]["text"] == f
    assert set(log["forget"][0]) == {"text", "PPL", "PPL/lower", "PPL/zlib"} | \
        {f"Min-{k}%" for k in (5, 10, 20, 30, 40, 50, 60)}
    assert "forget_holdout_Min-40%" in auc and len(auc) == 9 * 10
    # both texts of a pair get the same (fake) loss -> AUC 0.5 everywhere PPL is defined
    assert auc["forget_retain_PPL"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Config validation and the shipped configs                                     #
# --------------------------------------------------------------------------- #
def _cfg(**over):
    cfg = {"name": "unit", "corpus": "news", "model_path": "muse-bench/MUSE-news_target",
           "tokenizer": "org/base-tokenizer"}
    cfg.update(over)
    return cfg


def test_resolve_config_defaults_match_the_official_eval():
    R = M.resolve_config(_cfg())
    assert R["tokenizer"] == "org/base-tokenizer"
    assert R["metrics"] == ["verbmem_f", "privleak", "knowmem_f", "knowmem_r"]
    assert (R["verbmem_max_new_tokens"], R["knowmem_max_new_tokens"]) == (128, 32)
    assert R["privleak_auc_key"] == "forget_holdout_Min-40%"
    assert R["dtype"] == "float32" and R["batch_size"] == 1 and R["limit"] is None


@pytest.mark.parametrize("over, match", [
    ({"corpus": "wiki"}, "corpus"),
    ({"metrics": ["verbmem_f", "nope"]}, "unsupported"),
    ({"metrics": []}, "non-empty"),
    ({"metrics": ["privleak", "privleak"]}, "duplicates"),
    ({"batch_size": 2}, "not implemented"),
    ({"batch_size": 0}, "batch_size"),
    ({"limit": 0}, "limit"),
    ({"dtype": "int8"}, "dtype"),
    ({"privleak_auc_key": "forget_holdout_Min-45%"}, "privleak_auc_key"),
    ({"reference": "news_kuda"}, "reference"),
    ({"model_path": ""}, "model_path"),
    ({"name": ""}, "name"),
    ({"tokenizer": None}, "tokenizer"),
    ({"verbmem_max_new_tokens": -1}, "verbmem_max_new_tokens"),
])
def test_resolve_config_rejects(over, match):
    with pytest.raises(ValueError, match=match):
        M.resolve_config(_cfg(**over))


def test_resolve_model_path_passes_hub_ids_and_resolves_local_checkpoints(tmp_path,
                                                                           monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    assert resolve_model_path("muse-bench/MUSE-news_target") == "muse-bench/MUSE-news_target"
    (tmp_path / "models" / "x" / "ckpt").mkdir(parents=True)
    assert resolve_model_path("models/x/ckpt") == str(tmp_path / "models" / "x" / "ckpt")


def test_result_rows_follow_the_common_schema():
    R = M.resolve_config(_cfg(metrics=["verbmem_f", "privleak"]))
    rows = M.result_rows(R, {"verbmem_f": 60.4, "privleak": -98.0},
                         {"privleak_auc": 0.001, "privleak_auc_retrain": 0.4772,
                          "rouge_backend": "native"}, {"verbmem_f": 100, "privleak": 300})
    got = {(r.task, r.metric): r for r in rows}
    assert got[("muse_news", "verbmem_f")].value == 60.4 and got[("muse_news", "verbmem_f")].n == 100
    assert got[("muse_news", "privleak")].tp is None
    assert got[("muse_news", "privleak_auc")].meta["auc_retrain"] == 0.4772
    assert got[("muse_news", "verbmem_f")].meta["dtype"] == "float32"


def test_results_markdown_shows_reference_and_delta():
    R = M.resolve_config(_cfg(reference="news_target", metrics=["verbmem_f", "privleak"]))
    md = M.results_markdown(R, {"verbmem_f": 60.4, "privleak": -98.0},
                            {"privleak_auc": 0.001, "privleak_auc_retrain": 0.4772}, "cmd")
    assert "| verbmem_f | 60.4 | 58.4 | 2.0 |" in md
    assert "| privleak | -98.0 | -99.8 | 1.8 |" in md
    assert "raw AUC" in md and "0.4772" in md
    np.testing.assert_allclose(0.0, 0.0)
