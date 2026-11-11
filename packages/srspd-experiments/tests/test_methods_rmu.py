"""Unit tests for the RMU runner :mod:`srspd_experiments.methods.rmu`.

No GPU, no network, no model: everything here is the part of RMU that is pure
arithmetic or pure validation — the config gate that runs *before* two copies of
a 7B are loaded, the control-vector construction, and the topic-interleaving
index math that decides which corpus batch each optimizer step sees. Getting any
of those wrong is silent on a GPU run and expensive to discover afterwards.

The shipped configs are validated here too, so a typo in a YAML fails the suite
rather than the first ten minutes of a launch.
"""
from __future__ import annotations

import importlib

import copy

import pytest
import yaml


drv = importlib.import_module("srspd_experiments.methods.rmu")

from recipe_configs import config, legacy  # noqa: E402
#: The WMDP RMU configs. The Gemma-4 Dutch-unlearning RMU configs (``*_dutch1k_*``:
#: a Dutch target, the equal token budget, no hub push) are validated by
#: tests/test_wmdp_gemma4_dutch_configs.py instead.


def base_cfg(**over):
    """A minimal, valid config dict (the published zephyr values)."""
    cfg = {
        "name": "hf_rmu_test", "model": "HuggingFaceH4/zephyr-7b-beta",
        "dtype": "bfloat16", "device_map": "cuda", "push_to_hub": False,
        "rmu": {
            "forget_corpora": ["bio-forget-corpus", "cyber-forget-corpus"],
            "retain_corpora": ["wikitext", "wikitext"],
            "steering_coeffs": [6.5, 6.5], "alpha": [1200.0, 1200.0],
            "lr": 5.0e-5, "max_num_batches": 150, "batch_size": 4,
            "layer_id": 7, "layer_ids": [5, 6, 7], "seed": 42,
        },
    }
    cfg.update(over)
    return cfg


# --------------------------------------------------------------------------- #
# Config validation — the good case                                            #
# --------------------------------------------------------------------------- #
def test_resolve_accepts_the_published_hyperparameters():
    R = drv.resolve_rmu_config(base_cfg())
    assert R["forget_corpora"] == ["bio-forget-corpus", "cyber-forget-corpus"]
    assert R["retain_corpora"] == ["wikitext", "wikitext"]
    assert R["n_topics"] == 2
    assert R["steering_coeffs"] == [6.5, 6.5]
    assert R["alpha"] == [1200.0, 1200.0]
    assert R["lr"] == pytest.approx(5e-5)
    assert (R["max_num_batches"], R["batch_size"]) == (150, 4)
    assert (R["layer_id"], R["layer_ids"]) == (7, [5, 6, 7])
    assert R["seed"] == 42
    # per-topic sequence lengths: bio 512, cyber 768; retain always 512
    assert R["forget_max_length"] == [512, 768]
    assert R["retain_max_length"] == 512


def test_defaults_fill_an_almost_empty_rmu_block():
    """Every key but one may be omitted; the defaults are the published repro."""
    R = drv.resolve_rmu_config(base_cfg(rmu={"lr": 1e-4}))
    assert R["lr"] == pytest.approx(1e-4)
    for k in ("steering_coeffs", "alpha", "max_num_batches", "batch_size",
              "layer_id", "layer_ids", "seed"):
        assert R[k] == drv.DEFAULTS[k], k


def test_cli_overrides_win_over_the_config():
    R = drv.resolve_rmu_config(base_cfg(), max_num_batches=3, batch_size=2, seed=7,
                               push_to_hub=True)
    assert (R["max_num_batches"], R["batch_size"], R["seed"]) == (3, 2, 7)
    assert R["push_to_hub"] is True
    # ... and the config still wins when no override is passed
    assert drv.resolve_rmu_config(base_cfg())["push_to_hub"] is False


def test_hub_repo_naming():
    assert drv.model_short("HuggingFaceH4/zephyr-7b-beta") == "zephyr-7b-beta"
    assert drv.model_short("google/gemma-3-4b-it") == "gemma-3-4b-it"
    assert drv.default_repo_name("google/gemma-3-4b-it") == "rspd-wmdp-rmu-gemma-3-4b-it"
    # an explicit hub_repo overrides the derived name
    R = drv.resolve_rmu_config(base_cfg(hub_repo="rspd-wmdp-rmu-zephyr-7b"))
    assert (R["hub_repo"], R["hub_org"]) == ("rspd-wmdp-rmu-zephyr-7b", None)
    assert drv.resolve_rmu_config(base_cfg())["hub_repo"] == "rspd-wmdp-rmu-zephyr-7b-beta"


# --------------------------------------------------------------------------- #
# Config validation — the bad cases (fail fast, and say why)                   #
# --------------------------------------------------------------------------- #
def _bad(mutate, match):
    cfg = base_cfg()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        drv.resolve_rmu_config(cfg)


def test_missing_rmu_block_is_rejected():
    _bad(lambda c: c.pop("rmu"), r"no `rmu:` block")


def test_missing_name_or_model_is_rejected():
    _bad(lambda c: c.pop("name"), r"non-empty `name:`")
    _bad(lambda c: c.__setitem__("model", ""), r"non-empty `model:`")


def test_unknown_rmu_key_is_rejected():
    """A typo must not be silently ignored — it would run the default instead."""
    _bad(lambda c: c["rmu"].__setitem__("steering_coeff", 6.5), r"unknown key")


def test_unknown_corpus_is_rejected():
    _bad(lambda c: c["rmu"].__setitem__("forget_corpora", ["bio-forget-corpuz"]),
         r"not a known corpus")


def test_retain_and_forget_must_be_the_same_length():
    _bad(lambda c: c["rmu"].__setitem__("retain_corpora", ["wikitext"]),
         r"retain_corpora has 1 entries but")


def test_per_topic_lists_must_match_the_topic_count():
    _bad(lambda c: c["rmu"].__setitem__("steering_coeffs", [6.5]),
         r"steering_coeffs has 1 entries")
    _bad(lambda c: c["rmu"].__setitem__("alpha", [1200.0, 1200.0, 1200.0]),
         r"alpha has 3 entries")
    _bad(lambda c: c["rmu"].__setitem__("steering_coeffs", 6.5),
         r"steering_coeffs must be a list")


def test_non_positive_numbers_are_rejected():
    _bad(lambda c: c["rmu"].__setitem__("lr", 0), r"rmu\.lr must be a number > 0")
    _bad(lambda c: c["rmu"].__setitem__("lr", -1e-5), r"rmu\.lr must be a number > 0")
    _bad(lambda c: c["rmu"].__setitem__("batch_size", 0), r"batch_size must be a positive int")
    _bad(lambda c: c["rmu"].__setitem__("max_num_batches", -1),
         r"max_num_batches must be a positive int")
    _bad(lambda c: c["rmu"].__setitem__("steering_coeffs", [0.0, 6.5]),
         r"steering_coeffs\[0\] must be a number > 0")
    _bad(lambda c: c["rmu"].__setitem__("alpha", [-1.0, 1200.0]),
         r"alpha\[0\] must be a number > 0")


def test_layers_after_the_hooked_block_are_rejected():
    """A down_proj above the hooked block cannot influence its output, so it
    would sit in the optimizer collecting no gradient at all."""
    _bad(lambda c: c["rmu"].__setitem__("layer_ids", [5, 6, 7, 8]),
         r"sit after the hooked block")
    _bad(lambda c: c["rmu"].__setitem__("layer_ids", [5, 5, 7]), r"layer_ids has duplicates")
    _bad(lambda c: c["rmu"].__setitem__("layer_ids", []), r"layer_ids must be a non-empty list")
    _bad(lambda c: c["rmu"].__setitem__("layer_id", -1), r"layer_id must be an int >= 0")


def test_param_names_defaults_to_upstreams_single_tensor():
    """The default MUST reproduce upstream exactly — every existing config omits
    the key, and `hf_kuda.py`'s null-space projector assumes ``down_proj``."""
    assert drv.DEFAULTS["param_names"] == ["mlp.down_proj.weight"]
    assert drv.resolve_rmu_config(base_cfg())["param_names"] == ["mlp.down_proj.weight"]


def test_param_names_accepts_several_dotted_paths():
    """The Gemma-3 lever: ``down_proj`` sits behind ``post_feedforward_layernorm``
    and is a null direction there, so other tensors have to be nameable."""
    names = ["mlp.down_proj.weight", "self_attn.o_proj.weight"]
    cfg = base_cfg()
    cfg["rmu"]["param_names"] = list(names)
    assert drv.resolve_rmu_config(cfg)["param_names"] == names


def test_bad_param_names_are_rejected():
    _bad(lambda c: c["rmu"].__setitem__("param_names", []),
         r"param_names must be a non-empty list")
    _bad(lambda c: c["rmu"].__setitem__("param_names", "mlp.down_proj.weight"),
         r"param_names must be a non-empty list")
    _bad(lambda c: c["rmu"].__setitem__("param_names", ["mlp.down_proj.weight", ""]),
         r"param_names\[1\] must be a non-empty string")
    _bad(lambda c: c["rmu"].__setitem__("param_names", ["mlp..weight"]),
         r"is not a dotted attribute path")
    _bad(lambda c: c["rmu"].__setitem__("param_names", ["mlp.down_proj .weight"]),
         r"is not a dotted attribute path")
    _bad(lambda c: c["rmu"].__setitem__("param_names", ["mlp.down_proj.weight"] * 2),
         r"param_names has duplicates")


def test_the_tokenizer_sides_are_resolved_and_recorded():
    """Upstream forces them, RMU's MSE covers pad positions, and the two models
    ship different defaults — so they are part of the objective, and the
    resolved config carries them into ``metrics.json``."""
    R = drv.resolve_rmu_config(base_cfg())
    assert (R["padding_side"], R["truncation_side"]) == ("left", "right")
    assert (drv.TOKENIZER_PADDING_SIDE, drv.TOKENIZER_TRUNCATION_SIDE) == ("left", "right")


def test_booleans_are_not_accepted_as_numbers():
    """``True`` is an int in Python; it is not a batch size."""
    _bad(lambda c: c["rmu"].__setitem__("batch_size", True), r"batch_size must be a positive int")
    _bad(lambda c: c["rmu"].__setitem__("lr", True), r"rmu\.lr must be a number > 0")


# --------------------------------------------------------------------------- #
# The shipped configs                                                          #
# --------------------------------------------------------------------------- #
def _wmdp_rmu_runs():
    from srspd_experiments.recipes import catalog
    out = []
    for res in catalog.list_results():
        R = catalog.load_recipe(res)
        out += [(res, r["id"]) for r in R["runs"] if r["method"] == "rmu"
                and r["config"]["rmu"]["forget_corpora"] == ["bio-forget-corpus",
                                                              "cyber-forget-corpus"]]
    return out


def test_there_are_rmu_recipes():
    assert _wmdp_rmu_runs(), "no WMDP RMU run in the shipped recipes"


@pytest.mark.parametrize("res,run", _wmdp_rmu_runs())
def test_shipped_config_validates(res, run):
    cfg = config(res, run)
    R = drv.resolve_rmu_config(copy.deepcopy(cfg))
    assert R["name"] == cfg["name"]
    assert R["dtype"] == "bfloat16"
    assert R["push_to_hub"] is False and "wandb" not in cfg
    # the benchmark itself is fixed across models
    assert R["forget_corpora"] == ["bio-forget-corpus", "cyber-forget-corpus"]
    assert R["retain_corpora"] == ["wikitext", "wikitext"]
    assert (R["max_num_batches"], R["batch_size"], R["seed"]) == (150, 4, 42)


def test_zephyr_config_is_the_published_repro():
    R = drv.resolve_rmu_config(legacy("zephyr_rmu"))
    assert R["model"] == "HuggingFaceH4/zephyr-7b-beta"
    assert R["steering_coeffs"] == [6.5, 6.5]
    assert R["alpha"] == [1200.0, 1200.0]
    assert R["lr"] == pytest.approx(5e-5)
    assert (R["layer_id"], R["layer_ids"]) == (7, [5, 6, 7])


# --------------------------------------------------------------------------- #
# Control vectors                                                              #
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")


def test_control_vector_shape_and_norm():
    """``c_i = u / ||u|| * steering_coeff_i`` with ``u ~ Uniform``: shape
    ``(1, 1, hidden)`` so it broadcasts over ``(batch, seq, hidden)``, strictly
    positive, and with norm exactly the steering coefficient."""
    hidden, coeffs = 2560, [6.5, 300.0]
    vecs = drv.build_control_vectors(coeffs, hidden, dtype=torch.float32, device="cpu")
    assert len(vecs) == len(coeffs)
    for v, c in zip(vecs, coeffs):
        assert tuple(v.shape) == (1, 1, hidden)
        assert float(torch.norm(v)) == pytest.approx(c, rel=1e-5)
        assert bool((v > 0).all()), "torch.rand is uniform on [0,1): the vector is positive"
    # the two topics get *different* directions
    a = vecs[0] / torch.norm(vecs[0])
    b = vecs[1] / torch.norm(vecs[1])
    assert float(torch.norm(a - b)) > 1e-3


def test_control_vector_broadcasts_over_an_activation_batch():
    v = drv.build_control_vectors([6.5], 8, dtype=torch.float32, device="cpu")[0]
    acts = torch.zeros(3, 5, 8)
    assert tuple(v.expand_as(acts).shape) == (3, 5, 8)
    # expanding before mse_loss is numerically identical to upstream's implicit
    # broadcast — it only drops the UserWarning torch emits for the mismatch
    with pytest.warns(UserWarning, match="target size"):
        implicit = float(torch.nn.functional.mse_loss(acts, v))
    assert float(torch.nn.functional.mse_loss(acts, v.expand_as(acts))) == pytest.approx(
        implicit, rel=1e-6)


def test_control_vectors_are_reproducible_under_the_seed():
    drv.set_seed(42)
    a = drv.build_control_vectors([6.5, 6.5], 64, dtype=torch.float32, device="cpu")
    drv.set_seed(42)
    b = drv.build_control_vectors([6.5, 6.5], 64, dtype=torch.float32, device="cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    drv.set_seed(43)
    c = drv.build_control_vectors([6.5, 6.5], 64, dtype=torch.float32, device="cpu")
    assert not torch.equal(a[0], c[0])


# --------------------------------------------------------------------------- #
# Topic interleaving and the batch budget                                      #
# --------------------------------------------------------------------------- #
def upstream_schedule(num_batches, n_topics):
    """``rmu/unlearn.py``'s loop, transcribed, as the oracle for this module::

        for idx in range(num_batches):
            topic_idx = idx % len(forget_data_list)
            batch_idx = idx // len(forget_data_list)

    ``num_batches`` is the TOTAL step count across all topics — not the count
    per topic. This is deliberately a second, independent statement of the
    semantics: the driver's implementation is checked against it, not against
    itself.
    """
    out = []
    for idx in range(num_batches):
        out.append((idx, idx % n_topics, idx // n_topics))
    return out


def test_topic_schedule_is_upstreams_loop_verbatim():
    """``num_batches`` is the TOTAL number of optimizer steps.

    This test was rewritten deliberately. It used to assert
    ``len(topic_schedule(n, k)) == n * k`` — the driver enumerated
    ``range(num_batches * n_topics)``, which at the published
    ``max_num_batches=150`` ran **300** steps over 300 batches where upstream
    runs **150** steps over 150 batches. The old assertion encoded that bug, so
    it is now written against an independent transcription of upstream's loop
    (:func:`upstream_schedule`) rather than against the driver's own arithmetic.
    """
    for num_batches, n_topics in [(1, 1), (2, 1), (3, 2), (4, 3), (150, 2), (7, 4)]:
        assert drv.topic_schedule(num_batches, n_topics) == \
            upstream_schedule(num_batches, n_topics), (num_batches, n_topics)


def test_topic_schedule_alternates_topics_and_splits_the_budget():
    """``topic_idx = idx % n_topics``, ``batch_idx = idx // n_topics`` — bio and
    cyber strictly alternate, and the ``num_batches`` budget is SPLIT between
    them rather than given to each."""
    assert drv.topic_schedule(6, 2) == [(0, 0, 0), (1, 1, 0), (2, 0, 1),
                                        (3, 1, 1), (4, 0, 2), (5, 1, 2)]
    sched = drv.topic_schedule(12, 3)
    assert len(sched) == 12, "12 steps TOTAL, not 12 per topic"
    assert [t for _, t, _ in sched] == [0, 1, 2] * 4
    assert [b for _, _, b in sched] == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    for topic in range(3):
        seen = [b for _, t, b in sched if t == topic]
        assert seen == [0, 1, 2, 3], "each topic walks its own batches in order, once"


def test_the_published_setting_is_150_steps_not_300():
    """The regression this file exists to pin: ``max_num_batches: 150`` with the
    two published forget topics is 150 optimizer steps, 75 batches of each."""
    sched = drv.topic_schedule(150, 2)
    assert len(sched) == 150
    assert [sum(1 for _, t, _ in sched if t == k) for k in (0, 1)] == [75, 75]
    assert max(b for _, _, b in sched) == 74, "each topic consumes 75 batches, 0..74"


def test_topic_schedule_single_topic_and_empty():
    """With one topic, ``num_batches`` steps walk ``num_batches`` batches — the
    one case where the old and the corrected arithmetic agree."""
    assert drv.topic_schedule(2, 1) == [(0, 0, 0), (1, 0, 1)]
    assert drv.topic_schedule(0, 2) == []
    assert drv.topic_schedule(5, 1) == upstream_schedule(5, 1)


def test_resolve_num_batches_is_the_min_over_every_topic():
    forget = [["b"] * 200, ["b"] * 180]
    retain = [["r"] * 500, ["r"] * 500]
    assert drv.resolve_num_batches(150, forget, retain) == 150      # the cap binds
    assert drv.resolve_num_batches(500, forget, retain) == 180      # a forget topic binds
    assert drv.resolve_num_batches(500, forget, [["r"] * 90, ["r"] * 500]) == 90  # retain binds


def test_resolve_num_batches_rejects_an_empty_corpus():
    with pytest.raises(RuntimeError, match=r"produced no batches"):
        drv.resolve_num_batches(150, [["b"], []], [["r"], ["r"]])


def test_schedule_never_indexes_past_a_corpus():
    forget = [[f"b{i}" for i in range(7)], [f"c{i}" for i in range(5)]]
    retain = [[f"r{i}" for i in range(9)], [f"s{i}" for i in range(9)]]
    n = drv.resolve_num_batches(150, forget, retain)
    assert n == 5
    for _, topic, batch in drv.topic_schedule(n, len(forget)):
        forget[topic][batch]        # would IndexError if the math were wrong
        retain[topic][batch]


# --------------------------------------------------------------------------- #
# Small helpers used inside the loop                                           #
# --------------------------------------------------------------------------- #
def test_forget_max_lengths_are_keyed_on_the_topic_not_the_position():
    assert drv.forget_max_lengths(["bio-forget-corpus", "cyber-forget-corpus"]) == [512, 768]
    # upstream would give [512, 768] here too (position 0 -> 512); the name-keyed
    # form stays correct when the order is reversed
    assert drv.forget_max_lengths(["cyber-forget-corpus", "bio-forget-corpus"]) == [768, 512]


def test_hidden_size_prefers_the_text_config():
    class Cfg:
        hidden_size = 4096

    class TextCfg:
        hidden_size = 2560

    class Multimodal:
        pass

    plain = type("M", (), {"config": Cfg()})()
    assert drv.hidden_size(plain) == 4096

    mm_cfg = Cfg()
    mm_cfg.text_config = TextCfg()
    mm = type("M", (), {"config": mm_cfg})()
    assert drv.hidden_size(mm) == 2560, "the multimodal wrapper's text stack width wins"

    with pytest.raises(AttributeError):
        drv.hidden_size(Multimodal())


def test_forward_with_cache_captures_a_tuple_returning_block():
    """A decoder block returns ``(hidden_states, ...)``; the hook must keep
    element 0, and must be removed again."""
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4)

        def forward(self, x):
            return (self.lin(x), "attn-weights")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()

        def forward(self, x=None):
            return self.block(x)[0].sum()

    m = Model()
    x = torch.ones(2, 3, 4)
    out = drv.forward_with_cache(m, {"x": x}, module=m.block, no_grad=False)
    assert tuple(out.shape) == (2, 3, 4)
    assert out.requires_grad, "no_grad=False must keep the graph for backward()"
    assert torch.allclose(out, m.block.lin(x))
    assert not m.block._forward_hooks, "the hook must be removed before returning"

    with_ng = drv.forward_with_cache(m, {"x": x}, module=m.block, no_grad=True)
    assert not with_ng.requires_grad
    assert not m.block._forward_hooks


def test_forward_with_cache_captures_a_plain_tensor_module():
    lin = torch.nn.Linear(4, 4)
    model = torch.nn.Sequential(lin, torch.nn.ReLU())
    x = torch.ones(1, 4)
    out = drv.forward_with_cache(model, {"input": x}, module=lin, no_grad=True)
    assert torch.allclose(out, lin(x))


def test_module_device_reads_the_blocks_own_parameters():
    lin = torch.nn.Linear(2, 2)
    assert drv.module_device(lin) == next(lin.parameters()).device
    with pytest.raises(AttributeError):
        drv.module_device(torch.nn.ReLU())


# --------------------------------------------------------------------------- #
# The step-1 null-direction diagnostic                                         #
# --------------------------------------------------------------------------- #
def test_grad_weight_cosines_separates_radial_from_orthogonal():
    """``cos(grad, W)`` is 1 when the gradient is parallel to the weight (RMU's
    mechanism on Zephyr: grow ``|h|`` by scaling ``down_proj``) and 0 when the
    loss is flat along ``+W`` (the Gemma-3 RMSNorm-on-branch null direction)."""
    W = torch.tensor([[1.0, 0.0], [0.0, 2.0]])

    radial = torch.nn.Parameter(W.clone())
    radial.grad = W.clone() * -0.5                      # antiparallel: cos = -1
    orthogonal = torch.nn.Parameter(W.clone())
    orthogonal.grad = torch.tensor([[0.0, 1.0], [3.0, 0.0]])   # <g, W> = 0
    zero_grad = torch.nn.Parameter(W.clone())
    zero_grad.grad = torch.zeros_like(W)
    no_grad = torch.nn.Parameter(W.clone())             # never backwarded into

    out = drv.grad_weight_cosines({"radial": radial, "orthogonal": orthogonal,
                                   "zero": zero_grad, "none": no_grad})
    assert out["radial"] == pytest.approx(-1.0, abs=1e-6)
    assert out["orthogonal"] == pytest.approx(0.0, abs=1e-6)
    assert out["zero"] == 0.0
    assert out["none"] is None
    # ... and the diagnostic's own threshold classifies them the way the driver does
    assert abs(out["orthogonal"]) <= drv.NULL_DIRECTION_COS
    assert abs(out["radial"]) > drv.NULL_DIRECTION_COS


def test_grad_weight_cosines_leaves_the_parameters_alone():
    """It runs between ``backward()`` and ``step()``; it must not perturb either
    the weight or the gradient."""
    p = torch.nn.Parameter(torch.randn(3, 4))
    p.grad = torch.randn(3, 4)
    w0, g0 = p.detach().clone(), p.grad.clone()
    drv.grad_weight_cosines({"p": p})
    assert torch.equal(p.detach(), w0) and torch.equal(p.grad, g0)
