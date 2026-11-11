"""Model loading, block-parameter resolution and export (srspd_experiments.models).
CPU, no network: tiny random GPT-2 / Mistral, a fake processor."""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from srspd_experiments import models as M  # noqa: E402
from gpt2_tiny_helpers import save_checkpoint, tiny_gpt2, tiny_tokenizer  # noqa: E402


def _mistral():
    torch.manual_seed(0)
    cfg = transformers.MistralConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                     num_hidden_layers=3, num_attention_heads=2,
                                     num_key_value_heads=1)
    return transformers.MistralForCausalLM(cfg).eval()


def test_decoder_layers_resolve_on_both_layouts():
    assert len(M.get_decoder_layers(tiny_gpt2())) == 3
    assert len(M.get_decoder_layers(_mistral())) == 3
    with pytest.raises(AttributeError, match="tried"):
        M.get_decoder_layers(torch.nn.Linear(2, 2))


def test_block_params_are_selected_by_name_with_model_level_keys():
    m = _mistral()
    got = M.get_named_block_params(m, [0, 2])
    assert sorted(got) == ["model.layers.0.mlp.down_proj.weight",
                           "model.layers.2.mlp.down_proj.weight"]
    assert got["model.layers.2.mlp.down_proj.weight"] is m.model.layers[2].mlp.down_proj.weight
    two = M.get_named_block_params(m, [1], ["self_attn.o_proj.weight", "mlp.gate_proj.weight"])
    assert set(two) == {"model.layers.1.self_attn.o_proj.weight",
                        "model.layers.1.mlp.gate_proj.weight"}
    with pytest.raises(AttributeError, match="has no"):
        M.get_named_block_params(m, [0], ["mlp.nope.weight"])
    with pytest.raises(AttributeError, match="not an nn.Parameter"):
        M.get_named_block_params(m, [0], ["mlp.down_proj"])
    with pytest.raises(ValueError):
        M.get_named_block_params(m, [0], [])
    with pytest.raises(IndexError):
        M.get_block_module(m, 3)


def test_mlp_out_module_and_weight_layout():
    g = tiny_gpt2()
    assert M.weight_layout(M.get_mlp_out_module(g, 0, "mlp.c_proj")) == "in_out"
    assert M.weight_layout(M.get_down_proj_module(_mistral(), 0)) == "out_in"
    with pytest.raises(ValueError):
        M.get_mlp_out_module(g, 0, "mlp.fc2")


def test_resolve_model_source_only_rewrites_existing_data_root_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    rel, abs_ = save_checkpoint(tmp_path)
    assert M.resolve_model_source(rel) == abs_
    assert M.resolve_model_source("models/absent/x") == "models/absent/x"
    assert M.resolve_model_source("org/name") == "org/name"
    fp = M.model_fingerprint(abs_)
    assert isinstance(fp, str) and len(fp) == 16 and M.model_fingerprint("org/name") is None


def test_load_causal_lm_from_a_data_root_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    rel, _ = save_checkpoint(tmp_path)
    model, tok = M.load_causal_lm(rel, dtype="float32", device_map="cpu",
                                  padding_side="left", truncation_side="right")
    assert type(model).__name__ == "GPT2LMHeadModel"
    assert (tok.padding_side, tok.truncation_side) == ("left", "right")


class _FakeProcessor:
    def save_pretrained(self, d):
        with open(os.path.join(d, "processor_config.json"), "w") as f:
            json.dump({"processor_class": "Fake"}, f)


def test_export_saves_a_given_processor_beside_the_checkpoint(tmp_path):
    m, tok = tiny_gpt2(), tiny_tokenizer()
    info = M.export_checkpoint(m, tok, str(tmp_path / "a"), processor=_FakeProcessor())
    assert info["processor_saved"] is True
    assert (tmp_path / "a" / "processor_config.json").exists()
    assert (tmp_path / "a" / "config.json").exists() and (tmp_path / "a" / "tokenizer.json").exists()
    back = transformers.GPT2LMHeadModel.from_pretrained(str(tmp_path / "a"))
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), back.parameters()))


def test_text_only_export_never_looks_a_processor_up(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        classmethod(lambda cls, *a, **k: calls.append(a) or None))
    m = tiny_gpt2()
    info = M.export_checkpoint(m, tiny_tokenizer(), str(tmp_path / "b"), source="org/name")
    assert info["processor_saved"] is False and calls == [] and not M.is_multimodal(m)
    assert not (tmp_path / "b" / "processor_config.json").exists()


def test_a_processor_identical_to_the_tokenizer_is_not_saved(tmp_path):
    tok = tiny_tokenizer()
    assert M.save_processor(tiny_gpt2(), tok, str(tmp_path), processor=tok) is False


def test_save_model_and_write_metrics(tmp_path):
    out = M.save_model(tiny_gpt2(), tiny_tokenizer(), str(tmp_path / "m"))
    assert os.path.isfile(os.path.join(out, "model.safetensors"))
    p = M.write_metrics(str(tmp_path / "l"), {"a": 1})
    assert json.load(open(p)) == {"a": 1}
