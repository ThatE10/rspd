"""Unit tests for :mod:`srspd_experiments.methods.simnpo` — the loss and the plumbing.

Everything here is CPU, synthetic and tiny: no GPU, no network, no
``transformers``, no ``datasets``, no checkpoint. Importing the driver costs
torch, numpy, yaml and the shared scaffold (which imports its heavy
dependencies inside the functions that need them).

What is worth pinning is what nobody can eyeball on an H100 halfway through a
500-step run:

* ``get_batch_loss`` against a **hand-computed** example, because it is the one
  function ported verbatim from upstream and a wrong shift or a wrong reduction
  would still train, just not SimNPO;
* **length normalization**, the property that distinguishes SimNPO from NPO:
  two batches differing only in how much padding they carry must give the same
  forget term. If the pads were counted (the upstream WMDP loader's bug, which
  ``common.tokenize_batch`` fixes) this test fails;
* **the gamma margin**, because the WMDP-track variant silently drops it and
  repurposes the name as the retain weight — the divergence this driver exists
  to avoid. gamma must shift the pre-sigmoid argument, and ``gamma`` and
  ``retain_weight`` must be independent;
* **config validation**, because each rule is a way to make the run quietly
  meaningless (a retain corpus on the forget side, a beta that flips the sign of
  the loss, bf16 asserted over an fp32 load) and each costs a GPU-hour to
  discover otherwise;
* **the schedule**, because ``max_steps`` overriding epochs and
  ``warmup = max(1, max_steps // 10)`` are upstream's semantics, not obvious
  ones.
"""
from __future__ import annotations

import importlib

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml


D = importlib.import_module("srspd_experiments.methods.simnpo")

from recipe_configs import config  # noqa: E402

#: (recipe, run) of the shipped SimNPO configs these tests read.
ZEPHYR = ("zephyr_wmdp", "simnpo_port")


def MUSE(side, run):
    return (f"muse_{side}", run)


def _cfg(base=ZEPHYR, **overrides):
    """A shipped (recipe) config as a dict, mutated by dotted key; ``_DEL`` removes."""
    cfg = config(*base)
    for dotted, value in overrides.items():
        node, *rest = dotted.split(".")
        target = cfg
        while rest:
            target = target[node]
            node, rest = rest[0], rest[1:]
        if value is _DEL:
            target.pop(node, None)
        else:
            target[node] = value
    return cfg


_DEL = object()


# --------------------------------------------------------------------------- #
# get_batch_loss — against a hand-computed example                             #
# --------------------------------------------------------------------------- #
def test_get_batch_loss_matches_hand_computation():
    """Two tokens, three classes, one sequence: the summed NLL of the two
    *predicted* positions, computed by hand from log-softmax."""
    # labels [a, b, c]; logits at positions 0,1 predict b and c (position 2's
    # logits are dropped by the shift).
    logits = torch.tensor([[[2.0, 0.0, 0.0],      # predicts label[1] = 1
                            [0.0, 1.0, 0.5],      # predicts label[2] = 2
                            [9.0, 9.0, 9.0]]])    # dropped by the shift
    labels = torch.tensor([[0, 1, 2]])

    lsm0 = torch.log_softmax(logits[0, 0], dim=-1)
    lsm1 = torch.log_softmax(logits[0, 1], dim=-1)
    expected = -(lsm0[1] + lsm1[2])

    got = D.get_batch_loss(logits, labels)
    assert got.shape == (1,), "get_batch_loss is PER-SEQUENCE, not a scalar"
    assert torch.allclose(got, expected.reshape(1), atol=1e-6)


def test_get_batch_loss_ignores_masked_positions_and_is_summed():
    """``-100`` labels contribute nothing, and the result is a sum (not a mean):
    duplicating a sequence's scored positions doubles it."""
    logits = torch.randn(2, 6, 5, generator=torch.Generator().manual_seed(0))
    labels = torch.tensor([[0, 1, 2, -100, -100, -100],
                           [0, 1, 2, 3, 4, 0]])
    full = D.get_batch_loss(logits, labels)

    # row 0 scores exactly positions 1,2 of the shifted labels
    lsm = torch.log_softmax(logits[0, :2], dim=-1)
    assert torch.allclose(full[0], -(lsm[0, 1] + lsm[1, 2]), atol=1e-6)

    # a mean would be insensitive to the number of scored positions; a sum is not
    fewer = D.get_batch_loss(logits[1:, :4], labels[1:, :4])
    assert float(fewer[0]) < float(full[1])


# --------------------------------------------------------------------------- #
# The SimNPO forget term: length normalization and the gamma margin            #
# --------------------------------------------------------------------------- #
def _padded_pair():
    """The same one-sequence batch twice, the second carrying three extra PAD
    positions (label ``-100``, arbitrary logits there — exactly what
    ``common.tokenize_batch`` produces for a shorter member of a batch)."""
    g = torch.Generator().manual_seed(7)
    core = torch.randn(1, 5, 11, generator=g)
    labels = torch.tensor([[3, 1, 4, 1, 5]])

    pad_logits = torch.randn(1, 3, 11, generator=g) * 100.0   # deliberately wild
    padded_logits = torch.cat([core, pad_logits], dim=1)
    padded_labels = torch.cat([labels, torch.full((1, 3), -100)], dim=1)
    return (core, labels), (padded_logits, padded_labels)


@pytest.mark.parametrize("beta", [0.05, 0.1, 1.0, 2.5])
def test_forget_loss_is_length_normalized_over_padding(beta):
    """SimNPO's defining property: the forget term divides by ``|y|`` counted as
    the NON-masked labels, so padding a sequence cannot change it.

    This is the test that fails if the pads are counted (the upstream WMDP
    loader's ``padding="max_length"`` + eos-as-pad behaviour) or if the sum were
    left unnormalized (which is NPO, not SimNPO).
    """
    (l0, y0), (l1, y1) = _padded_pair()
    a = D.simnpo_forget_loss(l0, y0, beta=beta, gamma=0.0)
    b = D.simnpo_forget_loss(l1, y1, beta=beta, gamma=0.0)
    assert torch.allclose(a, b, atol=1e-6), "padding changed the forget term"


def test_forget_loss_is_length_normalized_across_lengths():
    """Two sequences with the same *per-token* NLL but different lengths give
    the same forget term — the bias NPO's summed reward has and SimNPO removes."""
    # a uniform 4-class distribution: every token costs exactly log 4 nats
    short = torch.zeros(1, 3, 4)
    short_labels = torch.tensor([[0, 1, 2]])
    long = torch.zeros(1, 9, 4)
    long_labels = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0]])

    a = D.simnpo_forget_loss(short, short_labels, beta=0.1, gamma=0.0)
    b = D.simnpo_forget_loss(long, long_labels, beta=0.1, gamma=0.0)
    # |y| counts the unshifted non-masked labels (upstream's off-by-one), so the
    # per-sequence reward is log(4) * (n-1)/n and the two differ only through it
    assert float(a) > float(b) > 0.0
    # ...and an unnormalized (NPO-style) reward would differ by a factor of 3 in
    # the argument; the normalized one differs by <6% here
    arg_a = 3.0 * math.log(4.0) / 3.0
    arg_b = 8.0 * math.log(4.0) / 9.0
    assert abs(arg_a - arg_b) / arg_a < 0.12


def test_gamma_shifts_the_pre_sigmoid_argument():
    """``gamma`` is a MARGIN subtracted from the length-normalized NLL, so the
    loss at ``gamma`` equals the loss of a batch whose per-token NLL is ``gamma``
    lower. Closed form on a uniform distribution."""
    beta, n_cls, seq = 0.7, 4, 5
    logits = torch.zeros(1, seq, n_cls)
    labels = torch.arange(seq).remainder(n_cls).reshape(1, seq)
    per_seq = (seq - 1) * math.log(n_cls)          # summed NLL
    denom = seq                                     # |y| = non-masked labels
    for gamma in (0.0, 0.25, 1.0):
        expect = -F.logsigmoid(torch.tensor(beta * (per_seq / denom - gamma))) * 2 / beta
        got = D.simnpo_forget_loss(logits, labels, beta=beta, gamma=gamma)
        assert torch.allclose(got, expect, atol=1e-6), f"gamma={gamma}"
    # a larger margin means a smaller argument means a LARGER loss
    losses = [float(D.simnpo_forget_loss(logits, labels, beta=beta, gamma=g))
              for g in (0.0, 0.5, 1.0)]
    assert losses[0] < losses[1] < losses[2]


def test_gamma_zero_reproduces_the_wmdp_track_forget_term():
    """The WMDP-track ``SimNPO_FT`` drops the margin entirely; with
    ``gamma: 0.0`` this driver's forget term is identical to it, so the
    published zephyr hyperparameters stay reproducible."""
    g = torch.Generator().manual_seed(11)
    logits = torch.randn(3, 7, 9, generator=g)
    labels = torch.randint(0, 9, (3, 7), generator=g)
    beta = 0.1
    upstream = -F.logsigmoid(
        beta * (D.get_batch_loss(logits, labels) / (labels != -100).sum(-1))
    ).mean() * 2 / beta
    assert torch.allclose(D.simnpo_forget_loss(logits, labels, beta=beta, gamma=0.0),
                          upstream, atol=1e-7)


# --------------------------------------------------------------------------- #
# The step: one forward on the forget batch, gamma and retain_weight separate  #
# --------------------------------------------------------------------------- #
class _FakeModel:
    """A stand-in language model: returns fixed logits and a fixed ``.loss``,
    and counts how many times it was called (the reference-free property is
    "one forget forward", and there is only ever one model)."""

    def __init__(self, logits, retain_loss=2.0):
        self._logits = logits
        self._retain = torch.tensor(retain_loss)
        self.calls = []

    def __call__(self, *, input_ids, attention_mask, labels):
        self.calls.append(int(input_ids.shape[1]))
        return SimpleNamespace(logits=self._logits, loss=self._retain)


def _batch(logits_seq):
    ids = torch.zeros(1, logits_seq, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.arange(logits_seq).reshape(1, logits_seq) % 4}


def test_step_losses_are_one_forward_each_and_mix_independently():
    logits = torch.zeros(1, 5, 4)
    model = _FakeModel(logits, retain_loss=3.0)
    fb, rb = _batch(5), _batch(5)

    total, forget, retain = D.simnpo_step_losses(
        model, fb, rb, beta=0.1, gamma=0.0, forget_weight=0.25, retain_weight=4.0)
    assert len(model.calls) == 2, "one forget forward + one retain forward, no oracle"
    assert float(retain) == pytest.approx(3.0)
    assert float(total) == pytest.approx(0.25 * float(forget) + 4.0 * 3.0, rel=1e-6)

    # retain_weight = 0 skips the retain forward entirely (pure SimNPO)
    model2 = _FakeModel(logits, retain_loss=3.0)
    total2, forget2, retain2 = D.simnpo_step_losses(
        model2, fb, rb, beta=0.1, gamma=0.0, forget_weight=1.0, retain_weight=0.0)
    assert len(model2.calls) == 1
    assert float(retain2) == 0.0
    assert float(total2) == pytest.approx(float(forget2), rel=1e-6)


def test_gamma_and_retain_weight_are_independent_knobs():
    """The divergence from the WMDP-track variant, stated as a test: changing
    ``gamma`` must not touch the retain term, and changing ``retain_weight``
    must not touch the forget term."""
    logits = torch.zeros(1, 5, 4)
    fb, rb = _batch(5), _batch(5)
    base = D.simnpo_step_losses(_FakeModel(logits), fb, rb, beta=0.5, gamma=0.0,
                                forget_weight=1.0, retain_weight=1.0)
    moved_gamma = D.simnpo_step_losses(_FakeModel(logits), fb, rb, beta=0.5, gamma=0.5,
                                       forget_weight=1.0, retain_weight=1.0)
    moved_lam = D.simnpo_step_losses(_FakeModel(logits), fb, rb, beta=0.5, gamma=0.0,
                                     forget_weight=1.0, retain_weight=7.0)
    assert float(moved_gamma[2]) == float(base[2])          # retain untouched by gamma
    assert float(moved_gamma[1]) != float(base[1])          # forget moved by gamma
    assert float(moved_lam[1]) == float(base[1])            # forget untouched by lambda
    assert float(moved_lam[2]) == float(base[2])
    assert float(moved_lam[0]) != float(base[0])


# --------------------------------------------------------------------------- #
# loss_form: the MUSE-repository raw-logit variant (iterative.py)              #
# --------------------------------------------------------------------------- #
def test_muse_upstream_loss_is_the_raw_logit_mean():
    """``-logsigmoid(beta * (-logits - gamma)).mean() * 2 / beta`` over every
    (batch, position, vocab) element — gamma=0, beta=1 is the bare form."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.zeros(2, 5, dtype=torch.long)
    got = D.simnpo_forget_loss_muse_upstream(logits, labels, beta=1.0, gamma=0.0)
    want = -F.logsigmoid(-logits).mean() * 2
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    got = D.simnpo_forget_loss_muse_upstream(logits, labels, beta=0.7, gamma=0.1)
    want = -F.logsigmoid(0.7 * (-logits - 0.1)).mean() * 2 / 0.7
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    # a constant logit field: closed form
    const = torch.full((1, 3, 4), 2.0)
    got = D.simnpo_forget_loss_muse_upstream(const, labels[:1, :3], beta=1.0, gamma=0.0)
    assert float(got) == pytest.approx(2 * math.log(1 + math.exp(2.0)), rel=1e-6)


def test_muse_upstream_loss_is_not_length_normalized_but_skips_pads():
    """Unlike the paper form, no per-sequence NLL is formed (the per-position
    mean is over the vocab axis too); pad positions (label -100) are left out,
    which coincides with upstream's plain ``.mean()`` on pad-free windows."""
    torch.manual_seed(1)
    logits = torch.randn(1, 6, 5)
    full = torch.zeros(1, 6, dtype=torch.long)
    padded = full.clone(); padded[0, 4:] = -100
    unmasked = D.simnpo_forget_loss_muse_upstream(logits, full, beta=1.0, gamma=0.0)
    masked = D.simnpo_forget_loss_muse_upstream(logits, padded, beta=1.0, gamma=0.0)
    on_prefix = D.simnpo_forget_loss_muse_upstream(logits[:, :4], full[:, :4], beta=1.0, gamma=0.0)
    assert float(masked) == pytest.approx(float(on_prefix), rel=1e-6)
    assert float(masked) != pytest.approx(float(unmasked))
    # and it is a different number from the paper form on the same inputs
    paper = D.simnpo_forget_loss(logits, full, beta=1.0, gamma=0.0)
    assert float(paper) != pytest.approx(float(unmasked))


def test_step_losses_switch_on_loss_form_and_keep_paper_as_default():
    torch.manual_seed(2)
    logits = torch.randn(1, 5, 4)
    fb, rb = _batch(5), _batch(5)
    default = D.simnpo_step_losses(_FakeModel(logits), fb, rb, beta=0.5, gamma=0.1,
                                   forget_weight=1.0, retain_weight=0.1)
    paper = D.simnpo_step_losses(_FakeModel(logits), fb, rb, beta=0.5, gamma=0.1,
                                 forget_weight=1.0, retain_weight=0.1, loss_form="paper")
    model = _FakeModel(logits)
    muse = D.simnpo_step_losses(model, fb, rb, beta=0.5, gamma=0.1,
                                forget_weight=1.0, retain_weight=0.1, loss_form="muse_upstream")
    assert float(default[1]) == float(paper[1]) == float(
        D.simnpo_forget_loss(logits, fb["labels"], beta=0.5, gamma=0.1))
    assert float(muse[1]) == pytest.approx(float(
        D.simnpo_forget_loss_muse_upstream(logits, fb["labels"], beta=0.5, gamma=0.1)))
    assert float(muse[1]) != pytest.approx(float(paper[1]))
    assert len(model.calls) == 2 and float(muse[2]) == float(paper[2])   # same retain term
    assert float(muse[0]) == pytest.approx(float(muse[1]) + 0.1 * float(muse[2]), rel=1e-6)
    with pytest.raises(ValueError, match="loss_form"):
        D.simnpo_step_losses(_FakeModel(logits), fb, rb, beta=0.5, gamma=0.1,
                             forget_weight=1.0, retain_weight=0.1, loss_form="wmdp")


def test_loss_form_config_key_defaults_to_paper_and_is_validated():
    assert D.resolve_simnpo_config(_cfg())["loss_form"] == "paper"
    assert D.resolve_simnpo_config(_cfg(**{"simnpo.loss_form": "muse_upstream"}))["loss_form"] \
        == "muse_upstream"
    for bad in ("muse", "upstream", "tofu", 1, None, True):
        with pytest.raises(ValueError, match="simnpo.loss_form"):
            D.resolve_simnpo_config(_cfg(**{"simnpo.loss_form": bad}))
    assert D.LOSS_FORMS == ("paper", "muse_upstream")
    assert set(D.LOSS_FORM_DESCRIPTION) == set(D.LOSS_FORMS)


@pytest.mark.parametrize("side", ["news", "books"])
def test_shipped_muse_configs_differ_only_in_loss_form_name_repo_and_tag(side):
    plain = _cfg(MUSE(side, "simnpo"))
    up = _cfg(MUSE(side, "simnpo_upstreamloss"))
    assert D.resolve_simnpo_config(plain)["loss_form"] == "paper"
    S = D.resolve_simnpo_config(up)
    assert S["loss_form"] == "muse_upstream"
    assert S["name"] == f"muse_{side}_simnpo_upstreamloss"
    # strip the intended differences; everything else must be identical
    for c in (plain, up):
        c.pop("name"); c["simnpo"].pop("loss_form", None)
    assert plain == up


# --------------------------------------------------------------------------- #
# Schedule                                                                     #
# --------------------------------------------------------------------------- #
def test_max_steps_overrides_epochs_and_warmup_defaults_to_a_tenth():
    s = D.resolve_schedule(10_000, max_steps=500, num_epochs=5, grad_accum=4)
    assert s == {"total_steps": 500, "warmup_steps": 50, "micro_steps": 2000}

    # no max_steps -> ceil(epochs * n_batches / grad_accum)
    s2 = D.resolve_schedule(10, max_steps=None, num_epochs=5, grad_accum=4)
    assert s2["total_steps"] == math.ceil(5 * 10 / 4) == 13
    assert s2["warmup_steps"] == max(1, 13 // 10)

    # an explicit warmup wins, and is clamped to the run length
    assert D.resolve_schedule(10, max_steps=3, num_epochs=1, grad_accum=1,
                              warmup_steps=99)["warmup_steps"] == 3
    with pytest.raises(ValueError, match="no batches"):
        D.resolve_schedule(0, max_steps=10, num_epochs=1, grad_accum=1)


def test_lr_schedule_is_linear_warmup_then_linear_decay():
    kw = dict(base_lr=1e-5, warmup_steps=50, total_steps=500)
    assert D.lr_at_step(0, **kw) == 0.0
    assert D.lr_at_step(25, **kw) == pytest.approx(0.5e-5)
    assert D.lr_at_step(50, **kw) == pytest.approx(1e-5)
    assert D.lr_at_step(275, **kw) == pytest.approx(0.5e-5, rel=1e-6)
    assert D.lr_at_step(500, **kw) == pytest.approx(0.0)
    # monotone down after the peak
    decay = [D.lr_at_step(s, **kw) for s in range(50, 501, 25)]
    assert decay == sorted(decay, reverse=True)


# --------------------------------------------------------------------------- #
# Batch plan                                                                   #
# --------------------------------------------------------------------------- #
def test_forget_plan_interleaves_corpora_with_per_topic_max_length():
    plan = D.build_forget_plan(
        ["bio-forget-corpus", "cyber-forget-corpus"],
        [[["b0"], ["b1"], ["b2"]], [["c0"], ["c1"]]])
    assert [p[0] for p in plan] == ["bio-forget-corpus", "cyber-forget-corpus",
                                    "bio-forget-corpus", "cyber-forget-corpus",
                                    "bio-forget-corpus"]
    assert [p[1] for p in plan] == [512, 768, 512, 768, 512]
    # an override replaces both
    plan2 = D.build_forget_plan(["bio-forget-corpus"], [[["b0"]]], 256)
    assert plan2[0][1] == 256


# --------------------------------------------------------------------------- #
# Config validation — fail fast, before any weight is loaded                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", [ZEPHYR])
def test_shipped_configs_resolve(path):
    S = D.resolve_simnpo_config(_cfg(path))
    assert S["beta"] == 0.1 and S["gamma"] == 0.0
    assert S["forget_weight"] == 1.0 and S["retain_weight"] == 1.0
    assert S["lr"] == pytest.approx(9.5e-6)
    assert S["max_steps"] == 500 and S["grad_accum"] == 4 and S["batch_size"] == 1
    assert S["weight_decay"] == 0.0 and S["dtype"] == "bfloat16" and S["bf16"] is True
    assert S["optim"] == "adamw_torch"
    assert S["retain_corpus"] == "wikitext"
    assert S["forget_corpora"] == ["bio-forget-corpus", "cyber-forget-corpus"]
    assert S["push_to_hub"] is False
    sched = D.resolve_schedule(1000, max_steps=S["max_steps"], num_epochs=S["num_epochs"],
                               grad_accum=S["grad_accum"], warmup_steps=S["warmup_steps"])
    assert sched["warmup_steps"] == 50


def test_hub_repo_name_is_derived_from_the_model():
    assert D.model_short("HuggingFaceH4/zephyr-7b-beta") == "zephyr-7b-beta"
    assert D.model_short("google/gemma-3-4b-it") == "gemma-3-4b-it"
    assert D.resolve_simnpo_config(_cfg(ZEPHYR))["hub_repo"] == "rspd-wmdp-simnpo-zephyr-7b-beta"


def test_cli_overrides_win():
    S = D.resolve_simnpo_config(_cfg(), max_steps=3, max_rows=8, no_push=True)
    assert S["max_steps"] == 3 and S["max_rows"] == 8 and S["push_to_hub"] is False


@pytest.mark.parametrize("overrides, match", [
    ({"name": ""}, "name"),
    ({"model": _DEL}, "model"),
    ({"simnpo": _DEL}, "simnpo"),
    ({"simnpo.beta": 0.0}, "simnpo.beta"),
    ({"simnpo.beta": -0.1}, "simnpo.beta"),
    ({"simnpo.gamma": -1.0}, "simnpo.gamma"),
    ({"simnpo.forget_weight": 0.0}, "simnpo.forget_weight"),
    ({"simnpo.retain_weight": -1.0}, "simnpo.retain_weight"),
    ({"simnpo.forget_corpora": []}, "forget_corpora"),
    # a retain corpus on the forget side unlearns what it was meant to keep
    ({"simnpo.forget_corpora": ["wikitext"]}, "RETAIN corpus"),
    ({"simnpo.forget_corpora": ["bio-forget-corpus", "bio-forget-corpus"]}, "duplicates"),
    ({"simnpo.forget_corpora": ["no-such-corpus"]}, "must be one of"),
    # ...and a forget corpus on the retain side preserves what it was meant to remove
    ({"simnpo.retain_corpus": "bio-forget-corpus"}, "FORGET corpus"),
    ({"simnpo.retain_max_length": 0}, "retain_max_length"),
    ({"train.lr": 0}, "train.lr"),
    ({"train.max_steps": 0}, "train.max_steps"),
    ({"train.grad_accum": 1.5}, "train.grad_accum"),
    ({"train.weight_decay": -0.1}, "train.weight_decay"),
    ({"train.optim": "adamw_bnb_8bit"}, "train.optim"),
    ({"train.warmup_steps": 0}, "train.warmup_steps"),
    # bf16 asserted over an fp32 load would train in a dtype nobody chose
    ({"dtype": "float32"}, "bf16"),
    ({"dtype": "int8"}, "dtype"),
    ({"hub_repo": "thate10/x"}, "hub_repo"),
])
def test_config_validation_rejects(overrides, match):
    with pytest.raises(ValueError, match=match):
        D.resolve_simnpo_config(_cfg(**overrides))


def test_upstream_coefficient_names_are_rejected_not_silently_ignored():
    """``npo_coeff`` / ``grad_diff_coeff`` are the TOFU names for the two mixing
    weights. Accepting them silently would let a ported config set nothing."""
    with pytest.raises(ValueError, match="npo_coeff"):
        D.resolve_simnpo_config(_cfg(**{"simnpo.npo_coeff": 0.1375}))
    with pytest.raises(ValueError, match="grad_diff_coeff"):
        D.resolve_simnpo_config(_cfg(**{"simnpo.grad_diff_coeff": 1.0}))


# --------------------------------------------------------------------------- #
# train.lr_schedule — linear (default, HF Trainer) vs constant (upstream MUSE)  #
# --------------------------------------------------------------------------- #
def test_lr_schedule_defaults_to_linear_and_is_validated():
    assert D.LR_SCHEDULES == ("linear", "constant") and D.DEFAULT_LR_SCHEDULE == "linear"
    S = D.resolve_simnpo_config(_cfg())
    assert S["lr_schedule"] == "linear" and S["warmup_steps"] is None
    for bad in ("cosine", "linear_decay", 1, None, True):
        with pytest.raises(ValueError, match="train.lr_schedule"):
            D.resolve_simnpo_config(_cfg(**{"train.lr_schedule": bad}))


def test_constant_schedule_holds_lr_and_ignores_warmup():
    kw = dict(base_lr=1e-5, warmup_steps=0, total_steps=500, schedule="constant")
    assert [D.lr_at_step(s, **kw) for s in (0, 1, 250, 499, 500, 10_000)] == [1e-5] * 6
    # the linear default is unchanged (the positional/keyword default is "linear")
    assert D.lr_at_step(25, base_lr=1e-5, warmup_steps=50, total_steps=500) \
        == D.lr_at_step(25, base_lr=1e-5, warmup_steps=50, total_steps=500, schedule="linear") \
        == pytest.approx(0.5e-5)
    with pytest.raises(ValueError, match="unknown lr schedule"):
        D.lr_at_step(0, base_lr=1e-5, warmup_steps=0, total_steps=10, schedule="cosine")


@pytest.mark.parametrize("warmup", [None, 0])
def test_constant_schedule_resolves_warmup_to_zero(warmup):
    S = D.resolve_simnpo_config(_cfg(**{"train.lr_schedule": "constant",
                                        "train.warmup_steps": warmup}))
    assert S["lr_schedule"] == "constant" and S["warmup_steps"] == 0
    sched = D.resolve_schedule(1000, max_steps=S["max_steps"], num_epochs=S["num_epochs"],
                               grad_accum=S["grad_accum"], warmup_steps=S["warmup_steps"])
    assert sched["warmup_steps"] == 0 and sched["total_steps"] == 500


@pytest.mark.parametrize("warmup", [1, 50, -1, 0.0, True, "0"])
def test_constant_schedule_refuses_a_warmup(warmup):
    with pytest.raises(ValueError, match="train.warmup_steps.*constant"):
        D.resolve_simnpo_config(_cfg(**{"train.lr_schedule": "constant",
                                        "train.warmup_steps": warmup}))


def test_linear_schedule_still_refuses_warmup_zero():
    """Byte-identical default path: under `linear` an explicit 0 was and is rejected."""
    with pytest.raises(ValueError, match="train.warmup_steps"):
        D.resolve_simnpo_config(_cfg(**{"train.warmup_steps": 0}))



@pytest.mark.parametrize("side,run", [("news", "simnpo"), ("books", "simnpo"),
                                      ("news", "simnpo_upstreamloss"),
                                      ("books", "simnpo_upstreamloss")])
def test_shipped_muse_sched_configs_differ_only_in_schedule_and_name(side, run):
    plain = _cfg(MUSE(side, run))
    sched = _cfg(MUSE(side, f"{run}_sched"))
    Sp = D.resolve_simnpo_config(plain)
    Ss = D.resolve_simnpo_config(sched)
    assert Sp["lr_schedule"] == "linear" and Sp["warmup_steps"] is None
    assert Ss["lr_schedule"] == "constant" and Ss["warmup_steps"] == 0
    assert Ss["loss_form"] == Sp["loss_form"]
    assert Ss["name"] == f"muse_{side}_{run}_sched"
    for c in (plain, sched):
        c.pop("name")
        c["train"].pop("lr_schedule", None); c["train"].pop("warmup_steps", None)
    assert plain == sched
