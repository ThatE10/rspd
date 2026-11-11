"""The HF harness: discovery, ``Conv1D`` layout and bias, biased ``nn.Linear``
opt-in, edit / restore, input-only capture, and the layer selector."""
from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

import tiny_models as tm  # noqa: E402
from srspd.capture.hf import harvest_inputs_concat, harvest_inputs_native  # noqa: E402
from srspd.models import harness as hh  # noqa: E402
from srspd.models.harness import (HFLinearHarness, as_dout_din, is_conv1d,  # noqa: E402
                                  write_dout_din)
from srspd.models.selector import (DEFAULT_EXCLUDE_REGEX, DEFAULT_LAYERS_REGEX,  # noqa: E402
                                   GPT2_LAYERS_REGEX, LayerSelector, build_layers_regex,
                                   resolve_selector)


def gpt2(n_layer=2, n_embd=32):
    return tm.gpt2(n_layer=n_layer, n_embd=n_embd, n_positions=32, vocab=64)


def harness(model):
    return HFLinearHarness(model, layers_regex=GPT2_LAYERS_REGEX, exclude_regex="")


# ── Conv1D: discovery and layout ─────────────────────────────────────────────
def test_discovers_four_matrices_per_block():
    h = harness(gpt2(n_layer=3))
    assert len(h.all_names) == 12 and sum("c_attn" in n for n in h.all_names) == 3
    assert all(is_conv1d(m) for m in h.module_by_name.values())


def test_gpt2_preset_is_picked_automatically():
    h = HFLinearHarness(gpt2())
    assert h.selector.preset == "gpt2" and len(h.all_names) == 8


def test_conv1d_bias_is_accepted_but_linear_bias_is_still_refused():
    h = harness(gpt2())
    assert any(getattr(m, "bias", None) is not None for m in h.module_by_name.values())

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.h = torch.nn.ModuleDict({"0": torch.nn.ModuleDict({"attn": torch.nn.ModuleDict(
                {"c_attn": torch.nn.Linear(4, 4, bias=True)})})})
    with pytest.raises(ValueError, match="subtract_bias"):
        HFLinearHarness(Tiny(), layers_regex=GPT2_LAYERS_REGEX, exclude_regex="")


def test_get_weight_returns_d_out_by_d_in():
    m = gpt2(n_embd=32)
    h = harness(m)
    assert h.get_weight("L00.mlp.c_fc").shape == (128, 32)
    assert h.get_weight("L00.attn.c_attn").shape == (96, 32)       # fused QKV, one matrix


def test_write_round_trips_through_the_transpose():
    mod = gpt2().transformer.h[0].mlp.c_fc
    target = torch.randn_like(as_dout_din(mod))
    write_dout_din(mod, target)
    assert torch.equal(as_dout_din(mod), target)


def test_harvest_brackets_the_matmul_with_the_bias_subtracted():
    m = gpt2()
    with torch.no_grad():
        for blk in m.transformer.h:
            blk.mlp.c_fc.bias.add_(1.0)
    h = harness(m)
    data = h.harvest([torch.randint(0, 64, (2, 16))])
    assert max(h.verify_bracketing(data, rel_tol=1e-4).values()) < 1e-4
    A, T = data["L00.mlp.c_fc"]
    W = h.get_weight("L00.mlp.c_fc").float()
    b = m.transformer.h[0].mlp.c_fc.bias.detach().float()
    assert ((A @ W.T + b) - T).norm() / T.norm() > 1e-3     # the guard is sensitive


def test_apply_then_restore_is_bitwise_exact():
    m = gpt2()
    h = harness(m)
    before = {n: h.module_by_name[n].weight.detach().clone() for n in h.all_names}
    h.apply_weights(h.all_names, {n: torch.zeros_like(h.get_weight(n)) for n in h.all_names})
    assert not torch.equal(h.module_by_name["L00.mlp.c_fc"].weight, before["L00.mlp.c_fc"])
    h.restore_all()
    assert all(torch.equal(h.module_by_name[n].weight, w) for n, w in before.items())


def test_an_edit_lands_in_the_right_basis():
    m = gpt2()
    h = harness(m)
    name = "L00.mlp.c_fc"
    ids = torch.randint(0, 64, (2, 16))
    W = h.get_weight(name).float().clone()
    delta = torch.randn_like(W) * 0.01
    h.apply_weights([name], {name: W - delta})
    A2, T2 = h.harvest([ids])[name]
    W2 = h.get_weight(name).float().clone()
    assert torch.allclose(W2, W - delta, atol=1e-6)
    assert torch.allclose(A2 @ W2.T, T2, atol=1e-4)
    h.restore_all()


def test_get_weight_aliases_the_live_parameter():
    h = harness(gpt2())
    view = h.get_weight("L00.mlp.c_fc")
    h.apply_weights(["L00.mlp.c_fc"], {"L00.mlp.c_fc": torch.zeros_like(view)})
    assert torch.equal(view, torch.zeros_like(view))
    h.restore_all()


def test_nn_linear_path_has_no_transpose():
    class Block(torch.nn.Module):
        def __init__(self, d=8):
            super().__init__()
            self.self_attn = torch.nn.ModuleDict(
                {k: torch.nn.Linear(d, d, bias=False) for k in ("q_proj", "k_proj", "v_proj",
                                                                 "o_proj")})

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block()])
            self.config = SimpleNamespace(hidden_size=8)

    h = HFLinearHarness(M())
    assert len(h.all_names) == 4
    mod = h.module_by_name["L00.self_attn.q_proj"]
    assert not is_conv1d(mod) and torch.equal(h.get_weight("L00.self_attn.q_proj"),
                                              mod.weight.detach())


# ── biased nn.Linear (opt-in) ────────────────────────────────────────────────
class _BiasedBlock(torch.nn.Module):
    def __init__(self, d=8, inner=12):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.up_proj = torch.nn.Linear(d, inner, bias=True)
        self.mlp.down_proj = torch.nn.Linear(inner, d, bias=True)

    def forward(self, x):
        return x + self.mlp.down_proj(torch.relu(self.mlp.up_proj(x)))


class _BiasedNet(torch.nn.Module):
    def __init__(self, n=2, d=8):
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([_BiasedBlock(d) for _ in range(n)])
        self.config = SimpleNamespace()

    def forward(self, hidden_states):
        x = hidden_states
        for b in self.layers:
            x = b(x)
        return x


def test_biased_linear_is_refused_by_default_and_names_the_way_out():
    with pytest.raises(ValueError, match="subtract_bias=True"):
        HFLinearHarness(_BiasedNet())


def test_biased_linear_opt_in_subtracts_the_bias_and_never_edits_it():
    m = _BiasedNet()
    with torch.no_grad():
        m.layers[0].mlp.down_proj.bias.add_(1.0)
    h = HFLinearHarness(m, subtract_bias=True, expect_d_in={"mlp.down_proj": 12})
    assert h.all_names == ["L00.mlp.up_proj", "L00.mlp.down_proj", "L01.mlp.up_proj",
                           "L01.mlp.down_proj"]
    assert sorted(h.biased_names) == sorted(h.all_names)
    kw = {"hidden_states": torch.randn(2, 5, 8)}
    data = h.harvest([kw])
    assert max(h.verify_bracketing(data, rel_tol=1e-5).values()) < 1e-5
    A, T = data["L00.mlp.down_proj"]
    b = m.layers[0].mlp.down_proj.bias.detach()
    assert ((A @ h.get_weight("L00.mlp.down_proj").T + b) - T).norm() / T.norm() > 1e-3
    biases = {n: h.module_by_name[n].bias.detach().clone() for n in h.all_names}
    h.apply_weights(h.all_names, {n: torch.zeros_like(h.get_weight(n)) for n in h.all_names})
    assert all(torch.equal(h.module_by_name[n].bias, b) for n, b in biases.items())
    h.restore_all()


def test_expect_d_in_catches_a_wrong_width():
    with pytest.raises(ValueError, match="expect_d_in"):
        HFLinearHarness(_BiasedNet(), subtract_bias=True, expect_d_in={"mlp.down_proj": 13})


# ── stream_chunk_inputs: token batches and kwargs batches ────────────────────
class _StubHarness:
    def __init__(self, d_in=6, d_out=4, n=3):
        self.mods = {f"L{i:02d}.m": torch.nn.Linear(d_in, d_out, bias=False) for i in range(n)}
        self.module_by_name = self.mods
        self.calls = []
        harness = self

        class _Model:
            def __call__(self, **kw):
                harness.calls.append(kw)
                x = kw.get("input_ids")
                if x is None:
                    x = kw["hidden_states"]
                for m in harness.mods.values():
                    m(x.float())
                return x

        self.model = _Model()


def test_stream_tensor_batch_goes_through_input_ids():
    h = _StubHarness()
    x = torch.randn(5, 6)
    out = hh.stream_chunk_inputs(h, x, list(h.mods), device="cpu")
    assert set(out) == set(h.mods) and list(h.calls[0]) == ["input_ids"]
    assert torch.equal(out["L00.m"], x)


def test_stream_dict_batch_is_splatted_as_kwargs():
    h = _StubHarness()
    kw = {"hidden_states": torch.randn(2, 4, 6), "timestep": torch.full((2,), 5.0)}
    out = hh.stream_chunk_inputs(h, kw, list(h.mods), device="cpu")
    assert sorted(h.calls[0]) == ["hidden_states", "timestep"]
    assert torch.equal(out["L00.m"], kw["hidden_states"].flatten(0, -2))


def test_stream_hooks_are_removed_even_when_the_forward_raises():
    h = _StubHarness()
    with pytest.raises(KeyError):
        hh.stream_chunk_inputs(h, {"nope": torch.randn(2, 6)}, list(h.mods), device="cpu")
    assert all(not m._forward_hooks for m in h.mods.values())


def test_stream_only_the_requested_names_are_captured():
    h = _StubHarness()
    assert list(hh.stream_chunk_inputs(h, torch.randn(5, 6), ["L01.m"], device="cpu")) == ["L01.m"]


# ── multi-sequence harvest ───────────────────────────────────────────────────
def test_harvest_inputs_concat_matches_per_sequence_harvests():
    torch.manual_seed(0)
    m = transformers.MistralForCausalLM(transformers.MistralConfig(
        vocab_size=64, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64)).eval()
    h = HFLinearHarness(m)
    seqs = [torch.randint(0, 64, (1, n)) for n in (5, 9, 3)]
    data, n = harvest_inputs_concat(h, seqs)
    assert n == 17
    for name in h.all_names:
        want = torch.cat([harvest_inputs_native(h, s)[0][name] for s in seqs], 0)
        assert torch.equal(data[name], want), name
    assert data["L00.self_attn.q_proj"] is data["L00.self_attn.v_proj"]
    assert data["L01.mlp.gate_proj"] is data["L01.mlp.up_proj"]
    assert data["L01.mlp.up_proj"] is not data["L01.mlp.down_proj"]


# ── selector ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("path,hit", [
    ("model.layers.3.self_attn.q_proj", True),
    ("model.layers.31.mlp.down_proj", True),
    ("model.language_model.layers.7.self_attn.o_proj", True),
    ("language_model.model.layers.7.mlp.gate_proj", True),
    ("model.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj", False),
    ("model.vision_tower.encoder.layers.0.self_attn.q_proj.linear", False),
    ("model.multi_modal_projector.mm_input_projection", False),
])
def test_default_regexes_select_the_text_decoder(path, hit):
    got = (re.search(DEFAULT_LAYERS_REGEX, path) is not None
           and re.search(DEFAULT_EXCLUDE_REGEX, path) is None)
    assert got is hit


def test_build_layers_regex_is_exact():
    rx = build_layers_regex([1, 3])
    assert re.search(rx, "model.layers.1.mlp.up_proj")
    assert not re.search(rx, "model.layers.17.mlp.up_proj")
    assert re.search(build_layers_regex([2], GPT2_LAYERS_REGEX), "transformer.h.2.mlp.c_fc")


def test_selector_forms():
    m = tm.mistral()
    assert resolve_selector(m, None).preset == "decoder"
    s = resolve_selector(m, [1])
    h = HFLinearHarness(m, selector=s)
    assert {n[:3] for n in h.all_names} == {"L01"}
    h = HFLinearHarness(m, selector=resolve_selector(m, {"types": ["down_proj", "q_proj"]}))
    assert h.all_names == ["L00.self_attn.q_proj", "L00.mlp.down_proj", "L01.self_attn.q_proj",
                           "L01.mlp.down_proj"]
    with pytest.raises(ValueError, match="unknown layer-selector keys"):
        resolve_selector(m, {"typo": 1})
    with pytest.raises(ValueError, match="two capture groups"):
        LayerSelector(r"layers\.\d+")
    with pytest.raises(ValueError, match="no target projections"):
        HFLinearHarness(m, selector=resolve_selector(m, {"blocks": [9]}))
