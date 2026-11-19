"""The toy of the README animation and the quickstart notebook: the component
carries the directions only the target uses, and the dose sets the shared one."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import rspd  # noqa: E402
import rspd.toy  # noqa: E402


def _fit(toy, m):
    return rspd.core.erased_component(toy.W, toy.X, toy.Y, mu=toy.mu(m))


def test_the_component_carries_what_only_the_target_uses():
    toy = rspd.toy.two_sets()
    assert toy.directions == ["x0", "x1", "shared", "y0", "y1"]
    c = rspd.toy.carried(toy.W, _fit(toy, 0.1))
    assert (c[toy.index("x")] > 0.9).all(), c
    assert (c[toy.index("y")] < 0.05).all(), c
    assert 0.03 < float(c[toy.index("shared")[0]]) < 0.3, c


def test_the_dose_sets_the_shared_direction():
    toy = rspd.toy.two_sets()
    shared = toy.index("shared")[0]
    got = [float(rspd.toy.carried(toy.W, _fit(toy, m))[shared]) for m in (0.01, 0.1, 1.0, 10.0)]
    assert got == sorted(got) and got[0] < 0.05 and got[-1] > 0.8, got


def test_leaks_say_the_same_thing():
    toy = rspd.toy.two_sets()
    M = _fit(toy, 0.1)
    lk = rspd.core.leaks(toy.W, M, toy.X, toy.Y)
    assert lk["L_Y_rel"] < 0.1            # the retained response is barely disturbed
    assert lk["L_X_rel"] < 0.75           # most of the target response is removed


def test_it_is_reproducible_and_sized_as_asked():
    a, b = rspd.toy.two_sets(seed=3), rspd.toy.two_sets(seed=3)
    assert torch.equal(a.X, b.X) and torch.equal(a.W, b.W)
    t = rspd.toy.two_sets(50, 70, n_x_only=3, n_shared=2, n_y_only=1, d_out=6)
    assert t.X.shape == (50, 6) and t.Y.shape == (70, 6) and t.W.shape == (6, 6)
    assert t.groups == ["x"] * 3 + ["shared"] * 2 + ["y"]


def test_a_moving_point_is_active_where_it_stands():
    toy = rspd.toy.two_sets()
    in_x = rspd.toy.activation(toy, 1.0, 0.0)
    in_y = rspd.toy.activation(toy, 0.0, 1.0)
    both = rspd.toy.activation(toy, 1.0, 1.0)
    assert (in_x[toy.index("y")] == 0).all() and (in_x[toy.index("x")] != 0).all()
    assert (in_y[toy.index("x")] == 0).all() and (in_y[toy.index("y")] != 0).all()
    assert (both != 0).all()


def test_the_tiny_language_model_goes_through_the_model_interface():
    pytest.importorskip("transformers")
    from rspd import SRSPD
    model, tokenizer = rspd.toy.tiny_language_model()
    texts = rspd.toy.tiny_texts()
    assert set(texts) == {"nl", "en", "de", "fr"} and len(texts["nl"]) == 12
    ids = tokenizer(texts["nl"][0], return_tensors="pt").input_ids
    assert int(ids.max()) < model.config.vocab_size and 1 not in ids          # no unknown words
    chat = tokenizer.apply_chat_template([{"role": "user", "content": texts["en"][0]}],
                                         tokenize=False)
    assert chat.startswith("en")
    comps = SRSPD(model, tokenizer).fit(target=texts["nl"], retained=texts["en"] + texts["de"],
                                        mu_per_token=0.1)
    assert len(comps) == 14                       # seven projections in each of two blocks
    comps.apply()
    assert comps.applied and comps.restore()
    comps.close()
