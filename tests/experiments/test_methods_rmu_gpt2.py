"""RMU on GPT-2 (:mod:`rspd.experiments.methods.rmu`): the block-output hook on
``transformer.h``, ``param_names: [mlp.c_proj.weight]``, ``epochs`` over a tiny
forget set and ``position_ids_from_mask`` -- a short ``run_rmu`` on a tiny
GPT-2. CPU, no network."""
from __future__ import annotations

import importlib

import copy

import pytest

from conftest import isolate_corpora  # noqa: E402
import torch
import yaml

from gpt2_tiny_helpers import tiny_gpt2, tiny_tokenizer, write_sets

drv = importlib.import_module("rspd.experiments.methods.rmu")
common = drv.common
from recipe_configs import legacy  # noqa: E402



def test_gpt2_config_resolves():
    cfg = legacy("gpt2_rmu")
    R = drv.resolve_rmu_config(cfg)
    assert R["param_names"] == ["mlp.c_proj.weight"]
    assert R["position_ids_from_mask"] is True and R["epochs"] == 150
    assert (R["layer_id"], R["layer_ids"]) == (5, [3, 4, 5])
    assert R["forget_corpora"] == ["target_nld"] and R["retain_corpora"] == ["contrastive"]
    assert R["forget_max_length"] == [768] and R["dtype"] == "float32"
    assert R["push_to_hub"] is False
    # the defaults stay upstream for every other config
    base = drv.resolve_rmu_config({"name": "t", "model": "m", "rmu": {"lr": 1e-4}})
    assert base["epochs"] == 1 and base["position_ids_from_mask"] is False
    bad = copy.deepcopy(cfg)
    bad["rmu"]["epochs"] = 0
    with pytest.raises(ValueError, match="rmu.epochs"):
        drv.resolve_rmu_config(bad)


def test_hidden_size_and_block_hook_on_gpt2():
    m, tok = tiny_gpt2(), tiny_tokenizer()
    assert drv.hidden_size(m) == 16
    blk = common.get_block_module(m, 1)
    enc = tok(["w1 w2 w3"], return_tensors="pt")
    h = drv.forward_with_cache(m, enc, module=blk, no_grad=True)
    assert tuple(h.shape) == (1, 3, 16)


def test_run_rmu_trains_only_c_proj_and_repeats_the_schedule(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    isolate_corpora(monkeypatch)
    write_sets(tmp_path, n_target=3, n_contrastive=6, n_heldout=2, words=20)
    cfg = legacy("gpt2_rmu")
    cfg["rmu"].update(layer_id=2, layer_ids=[1, 2], max_num_batches=4, batch_size=2,
                      epochs=3, steering_coeffs=[4.0])
    R = drv.resolve_rmu_config(cfg)
    tok = tiny_tokenizer()
    tok.padding_side, tok.truncation_side = R["padding_side"], R["truncation_side"]
    updated, frozen = tiny_gpt2(), tiny_gpt2()
    before = {n: p.detach().clone() for n, p in updated.named_parameters()}
    summary = drv.run_rmu(updated, frozen, tok, R)
    # 3 target docs / batch 2 -> 2 forget batches; contrastive 12 docs -> 4 (capped at
    # max_num_batches 4) -> num_batches 2, x 3 epochs
    assert summary["num_batches"] == 2 and summary["steps"] == 6
    moved = [n for n, p in updated.named_parameters() if not torch.equal(p, before[n])]
    assert moved == ["transformer.h.1.mlp.c_proj.weight", "transformer.h.2.mlp.c_proj.weight"]
    assert summary["step1_null_direction"] == []          # c_proj is a real actuator on GPT-2





def test_equal_budget_config_overrides_the_windows_and_pins_the_schedule():
    cfg = legacy("gpt2_eb_rmu")
    R = drv.resolve_rmu_config(cfg)
    assert R["budget"]["steps"] == 200 and R["budget"]["passes_per_step"] == 1
    assert R["forget_max_length"] == [1024] and R["retain_max_length"] == 1024
    assert (R["batch_size"], R["max_num_batches"], R["epochs"]) == (1, 1, 200)
    assert R["steering_coeffs"] == [100.0] and R["alpha"] == [10.0] and R["lr"] == pytest.approx(1e-4)
    bad = copy.deepcopy(cfg)
    bad["rmu"]["epochs"] = 150
    with pytest.raises(ValueError, match="epochs x rmu.max_num_batches"):
        drv.resolve_rmu_config(bad)
    with pytest.raises(ValueError, match="epochs x rmu.max_num_batches"):
        drv.resolve_rmu_config(cfg, max_num_batches=4)
    bad = copy.deepcopy(cfg)
    bad["rmu"].update(forget_corpora=["target_nld", "target_nld"],
                      retain_corpora=["contrastive", "contrastive"],
                      steering_coeffs=[100.0, 100.0], alpha=[10.0, 10.0])
    with pytest.raises(ValueError, match="one forget topic"):
        drv.resolve_rmu_config(bad)
    # no budget: the sweep config keeps the upstream windows
    R0 = drv.resolve_rmu_config(legacy("gpt2_rmu"))
    assert R0["budget"] is None and R0["forget_max_length"] == [768] and R0["retain_max_length"] == 512


def test_equal_budget_run_on_a_tiny_gpt2_sees_the_target_per_step_and_the_retained_set_once(tmp_path, monkeypatch):
    from gpt2_tiny_helpers import save_checkpoint
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    isolate_corpora(monkeypatch)
    save_checkpoint(tmp_path)
    # 1 target doc of 20 tokens; 2 x 5 retained docs of 20 (+1 sep) -> 14 blocks of 16
    write_sets(tmp_path, n_target=1, n_contrastive=5, n_heldout=2, words=20)
    cfg = legacy("gpt2_eb_rmu")
    cfg.update(name="rmu_gpt2_eb", device_map="cpu", wandb={"enabled": False},
               budget={"target_passes": 3, "retain_exposure_tokens": 200, "steps": 3,
                       "forget_max_length": 1024, "retain_block": 16})
    cfg["rmu"].update(layer_id=2, layer_ids=[1, 2], epochs=3, steering_coeffs=[4.0])
    cp = tmp_path / "eb.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    before = {n: p.detach().clone() for n, p in tiny_gpt2().named_parameters()}
    m = drv.run(str(cp), push_to_hub=False)
    assert m["steps"] == 3 and m["num_batches"] == 1
    assert m["hyperparameters"]["forget_max_length"] == [1024]
    b = m["budget"]
    assert b["steps"] == 3
    assert b["target_tokens_unique"] == 20 and b["target_tokens_exposed"] == 3 * 20
    assert b["retain_tokens_unique"] == 200 == b["retain_tokens_exposed_docs"]
    assert b["retain_separator_tokens"] == 10 and b["retain_blocks"] == 14
    assert b["retain_tokens_covariance"] == 0
    assert (m["retain_plan"]["blocks_per_step_min"], m["retain_plan"]["blocks_per_step_max"]) == (4, 5)
    assert m["retain_batches_per_topic"] == [0]              # the document loader was not used
    from transformers import AutoModelForCausalLM
    after = dict(AutoModelForCausalLM.from_pretrained(str(tmp_path / "models" / "methods" / "rmu_gpt2_eb")).named_parameters())
    moved = [n for n, p in after.items() if not torch.equal(p, before[n])]
    assert moved == ["transformer.h.1.mlp.c_proj.weight", "transformer.h.2.mlp.c_proj.weight"]
    import json
    metrics = json.loads((tmp_path / "logs" / "methods" / "rmu_gpt2_eb" / "metrics.json").read_text())
    assert metrics["budget"]["retain_tokens_exposed_docs"] == metrics["budget"]["retain_tokens_unique"]
