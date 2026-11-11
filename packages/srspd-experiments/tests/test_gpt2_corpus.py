"""Corpus spec, span disjointness, and the LR schedule.

The disjointness tests are the load-bearing ones. Recovery is measured on Dutch
the model has *not* seen, and every condition is evaluated on a *fixed* held-out
set. Both claims are false if the three spans overlap, and neither would fail
loudly -- the run would simply report a recovery time that is really a
memorisation time. So overlap is made structurally impossible and pinned here.

No torch / transformers / datasets: these cover the pure logic, which is exactly
the part that can be checked without a GPU or a network.
"""
from __future__ import annotations

import pytest

from srspd_experiments.data import gpt2_corpus as pc


def _trainer():
    from srspd_experiments.eval import trainer
    return trainer


# ── spec validation ──────────────────────────────────────────────────────────

def _bucket(**over):
    b = dict(lang="nld", dataset="HuggingFaceFW/fineweb-2", config="nld_Latn",
             split="train", revision="main", text_field="text", min_chars=200)
    b.update(over)
    return b


def test_revision_is_required():
    """The field people leave out, and the one that decides rebuildability."""
    b = _bucket(); del b["revision"]
    with pytest.raises(ValueError, match="revision"):
        pc.resolve_spec([b])


def test_unknown_field_is_rejected_not_ignored():
    """A typo'd key must not silently do nothing."""
    with pytest.raises(ValueError, match="unknown field"):
        pc.resolve_spec([_bucket(tokens=1000)])


def test_duplicate_language_is_rejected():
    with pytest.raises(ValueError, match="duplicate language"):
        pc.resolve_spec([_bucket(), _bucket()])


def test_empty_buckets_rejected():
    with pytest.raises(ValueError, match="at least one language"):
        pc.resolve_spec([])


# ── span budgets and sequential disjointness ────────────────────────────────

def test_spans_carry_their_budgets_in_order():
    s = pc.resolve_spans(111, 222, 333)
    assert list(s) == list(pc.SPANS) == ["train", "val", "recovery"]
    assert (s["train"]["tokens"], s["val"]["tokens"], s["recovery"]["tokens"]) \
        == (111, 222, 333)


def test_zero_or_negative_budget_rejected():
    with pytest.raises(ValueError, match="positive token budget"):
        pc.resolve_spans(0, 10, 10)


def test_one_iterator_fills_spans_without_overlap():
    """The disjointness argument, exercised on the real control flow.

    Spans are filled sequentially from ONE iterator, so a document consumed by
    `train` can never reach `val`. This drives write_span with a stub tokenizer
    and asserts the three spans partition the document stream.
    """
    class StubTok:
        eos_token_id = 0
        def __call__(self, texts, add_special_tokens=False):
            # one token per character keeps the arithmetic transparent
            return {"input_ids": [[1] * len(t) for t in texts]}

    docs = iter([f"{i:04d}" for i in range(500)])      # 4 tokens each + eot
    import tempfile
    seen_ranges = []
    with tempfile.TemporaryDirectory() as tmp:
        cursor = 0
        for name, budget in (("train", 100), ("val", 40), ("recovery", 60)):
            rec = pc.write_span(StubTok(), docs, {"tokens": budget},
                                f"{tmp}/{name}", eot_id=0, encode_batch=5,
                                doc_offset=cursor, log=lambda *a: None)
            assert rec["tokens"] <= budget
            seen_ranges.append((rec["doc_lo"], rec["doc_hi"]))
            cursor = rec["doc_hi"]
    for (a_lo, a_hi), (b_lo, b_hi) in zip(seen_ranges, seen_ranges[1:]):
        assert a_hi <= b_lo, f"spans overlap: {seen_ranges}"
    assert seen_ranges[0][0] == 0


def test_a_span_never_exceeds_its_budget():
    """Overshoot would let one span borrow tokens from the next, which is how a
    held-out set stops being held out."""
    class StubTok:
        eos_token_id = 0
        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [[1] * 50 for _ in texts]}   # big documents
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        rec = pc.write_span(StubTok(), iter(["x"] * 100), {"tokens": 10},
                            tmp, eot_id=0, encode_batch=4,
                            log=lambda *a: None)
    assert rec["tokens"] == 10


# ── manifest hashing ─────────────────────────────────────────────────────────

def test_manifest_hash_is_order_independent_but_content_sensitive():
    a = pc.manifest_hash({"x": 1, "y": [1, 2]})
    b = pc.manifest_hash({"y": [1, 2], "x": 1})
    assert a == b, "key order must not change the hash"
    assert a != pc.manifest_hash({"x": 1, "y": [2, 1]}), \
        "list order is content and must change the hash"


def test_uint16_holds_the_vocabulary():
    """50257 must fit the shard dtype; a larger vocab would wrap silently."""
    import numpy as np
    assert np.iinfo(pc.DTYPE).max >= 50257


# ── LR schedule (pure arithmetic, no torch needed at import time) ────────────

def test_lr_warms_up_then_decays_to_the_floor():
    TrainConfig, lr_at = _trainer().TrainConfig, _trainer().lr_at
    cfg = TrainConfig(total_tokens=1_000_000, warmup_tokens=100_000,
                      lr=6e-4, min_lr_frac=0.1)
    assert lr_at(0, cfg) < cfg.lr
    assert lr_at(cfg.warmup_tokens, cfg) == pytest.approx(cfg.lr, rel=1e-6)
    end = lr_at(cfg.total_tokens, cfg)
    assert end == pytest.approx(cfg.lr * cfg.min_lr_frac, rel=1e-6)
    mid = lr_at(cfg.total_tokens // 2, cfg)
    assert end < mid < cfg.lr


def test_lr_never_goes_negative_past_the_end():
    """A run that overshoots its budget by a step must not flip the sign."""
    TrainConfig, lr_at = _trainer().TrainConfig, _trainer().lr_at
    cfg = TrainConfig(total_tokens=1000, warmup_tokens=10)
    assert lr_at(10_000, cfg) >= cfg.lr * cfg.min_lr_frac - 1e-12


def test_shipped_corpus_spec_resolves():
    import yaml

    from srspd_experiments.data import spec_path
    cfg = pc.load_corpus_config(spec_path("gpt2_trilingual_corpus"))
    buckets = pc.resolve_spec(cfg["corpus"]["buckets"])
    assert [b["lang"] for b in buckets] == ["eng", "deu", "nld"]
    assert cfg["corpus"]["tokens_per_language"] == 833333334
    assert cfg["tokenizer"]["vocab_size"] == pc.VOCAB_SIZE == 50257
    assert yaml.safe_load(open(spec_path("gpt2_trilingual_corpus")))["name"] == "gpt2_corpus"
