"""The equal-token-budget plumbing (``data.budget``): the ``budget:`` block, the
packed retained stream (every document token once, files round-robin), the
step ranges, ``windows_to_batch`` and the realised-exposure check. CPU."""
from __future__ import annotations

import pytest
import torch

from gpt2_tiny_helpers import corpora_block, tiny_tokenizer, write_sets
from rspd.experiments.data import budget as B
from rspd.experiments.data import corpora as C

BUDGET = {"target_passes": 4, "retain_exposure_tokens": 100, "steps": 2,
          "forget_max_length": 32, "retain_block": 16}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(C, "CORPORA", {k: v for k, v in C.CORPORA.items()
                                       if C.corpus_kind(v) != "local_jsonl"})
    write_sets(tmp_path, n_target=2, n_contrastive=3, n_heldout=1, words=10)
    C.register_config_corpora({"corpora": corpora_block()})
    return tmp_path


def test_resolve_budget():
    assert B.resolve_budget({}) is None
    out = B.resolve_budget({"budget": BUDGET})
    assert out["passes_per_step"] == 2
    with pytest.raises(ValueError, match="multiple"):
        B.resolve_budget({"budget": {**BUDGET, "target_passes": 3}})
    with pytest.raises(ValueError, match="missing"):
        B.resolve_budget({"budget": {"steps": 2}})
    with pytest.raises(ValueError, match="unknown"):
        B.resolve_budget({"budget": {**BUDGET, "lr": 1}})


def test_spread_ranges_cover_every_item_once():
    r = B.spread_ranges(7, 3)
    assert r == [(0, 2), (2, 4), (4, 7)]
    assert sum(b - a for a, b in r) == 7


def test_packed_stream_sees_every_document_token_once_round_robin(env):
    tok = tiny_tokenizer()
    s = B.packed_retain_stream(tok, "contrastive", block=8, sep_token_id=tok.eos_token_id)
    # 3 docs x 10 tokens per file, + 1 separator each = 33 tokens -> 5 blocks per file
    assert s["n_docs"] == 6 and s["n_doc_tokens"] == 60 and s["n_sep_tokens"] == 6
    assert s["block_file"] == [0, 1] * 5
    assert sum(int(b.shape[0]) for b in s["blocks"]) == 66
    assert sum(s["block_n_sep"]) == 6
    plan = B.budget_retain_plan(tok, "contrastive", budget={**BUDGET, "retain_block": 8})
    assert plan["ranges"] == [(0, 5), (5, 10)]
    with pytest.raises(ValueError, match="fewer than"):
        B.budget_retain_plan(tok, "contrastive", budget={**BUDGET, "retain_block": 64, "steps": 4})


def test_windows_to_batch_pads_and_masks():
    w = [torch.tensor([1, 2, 3]), torch.tensor([4])]
    enc = B.windows_to_batch(w, pad_token_id=0, padding_side="left", with_labels=True)
    assert enc["input_ids"].tolist() == [[1, 2, 3], [0, 0, 4]]
    assert enc["labels"].tolist() == [[1, 2, 3], [-100, -100, 4]]


def test_exposure_record_and_assertion():
    e = B.Exposure()
    for _ in range(2):
        e.add_target(torch.ones(1, 10, dtype=torch.long))
        e.add_target(torch.ones(1, 10, dtype=torch.long))
        e.add_retain(torch.ones(2, 25, dtype=torch.long), n_sep=0)
        e.step()
    rec = e.record(budget=BUDGET, target_unique=10, retain_unique=100, retain_sep=0)
    assert rec["target_passes_realised"] == 4.0 and rec["steps"] == 2
    B.assert_budget(rec, BUDGET)
    rec["steps"] = 3
    with pytest.raises(RuntimeError, match="did not realise"):
        B.assert_budget(rec, BUDGET)
    assert "[budget]" in B.Exposure.line(rec)
