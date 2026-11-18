"""``trainer.lm_loss`` scores the NEXT token, not the one after it.

Regression: passing the loader's already-shifted ``y`` to HF as ``labels=``
shifts it twice, so the model trains on predicting t+2.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from rspd.experiments.eval.trainer import lm_loss  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=50, n_positions=32, n_embd=16,
                                  n_layer=1, n_head=2)
    return transformers.GPT2LMHeadModel(cfg).eval()


def test_lm_loss_is_next_token_ce(tiny):
    chunk = torch.randint(0, 50, (2, 17))
    x, y = chunk[:, :-1], chunk[:, 1:]
    logits = tiny(input_ids=x).logits
    want = torch.nn.functional.cross_entropy(logits.reshape(-1, 50), y.reshape(-1))
    assert torch.allclose(lm_loss(tiny, x, y), want)


def test_lm_loss_matches_hf_labels_on_unshifted_input(tiny):
    # HF shifts labels internally, so labels=x is the correct HF spelling of the
    # same objective (up to its dropping the final position, which has no target).
    chunk = torch.randint(0, 50, (2, 17))
    x, y = chunk[:, :-1], chunk[:, 1:]
    hf = tiny(input_ids=chunk, labels=chunk).loss
    assert torch.allclose(lm_loss(tiny, x, y), hf, atol=1e-5)
