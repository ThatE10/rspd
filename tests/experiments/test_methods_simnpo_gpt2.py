"""SimNPO on GPT-2 (:mod:`rspd.experiments.methods.simnpo`): the shipped config, the
two-file ``local_jsonl`` retain corpus, and a short full fine-tune of a tiny
GPT-2 in fp32. CPU, no network."""
from __future__ import annotations

import importlib

import json

import pytest

from conftest import isolate_corpora  # noqa: E402
import yaml

from gpt2_tiny_helpers import save_checkpoint, tiny_gpt2, tiny_tokenizer, write_sets

D = importlib.import_module("rspd.experiments.methods.simnpo")
C = D.C
from recipe_configs import legacy  # noqa: E402



def test_gpt2_config_resolves_with_the_concatenated_retain_corpus():
    S = D.resolve_simnpo_config(legacy("gpt2_simnpo"))
    assert S["retain_corpus"] == "contrastive" and S["forget_corpora"] == ["target_nld"]
    assert C.CORPORA["contrastive"]["path"] == [
        "datasets/gpt2_unlearning/gpt2_unlearning_1m/contrastive_eng.jsonl",
        "datasets/gpt2_unlearning/gpt2_unlearning_1m/contrastive_deu.jsonl"]
    assert S["dtype"] == "float32" and S["bf16"] is False and S["push_to_hub"] is False
    assert S["forget_max_length"] == 1024 and S["retain_max_length"] == 1024
    assert S["max_steps"] == 200 and S["gradient_checkpointing"] is False


def test_short_run_on_a_tiny_gpt2(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    isolate_corpora(monkeypatch)
    rel, _ = save_checkpoint(tmp_path)
    write_sets(tmp_path, n_target=3, n_contrastive=5, n_heldout=2, words=20)
    cfg = legacy("gpt2_simnpo")
    cfg.update(name="simnpo_gpt2_e2e", device_map="cpu", wandb={"enabled": False})
    cfg["train"].update(max_steps=3, batch_size=2, grad_accum=1, log_every=1)
    cp = tmp_path / "s.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    m = D.run(str(cp), no_push=True)
    assert m["data"]["forget_batches"] == 2            # 3 target docs / batch 2
    assert m["data"]["retain_batches"] == 5            # 10 contrastive docs (eng + deu) / 2
    assert m["schedule"]["total_steps"] == 3 and m["failure"] is None
    md = tmp_path / "models" / "methods" / "simnpo_gpt2_e2e"
    assert (md / "config.json").exists() and (md / "tokenizer.json").exists()
    metrics = json.loads((tmp_path / "logs" / "methods" / "simnpo_gpt2_e2e" / "metrics.json").read_text())
    assert metrics["n_params"] == sum(p.numel() for p in tiny_gpt2().parameters())





def test_equal_budget_config_resolves_and_refuses_a_schedule_that_is_not_the_protocol():
    cfg = legacy("gpt2_eb_simnpo")
    S = D.resolve_simnpo_config(cfg)
    assert S["budget"] == {"target_passes": 200, "retain_exposure_tokens": 1_000_000, "steps": 200,
                           "forget_max_length": 1024, "retain_block": 1024, "passes_per_step": 1}
    assert (S["batch_size"], S["grad_accum"], S["max_steps"]) == (1, 1, 200)
    assert S["forget_max_length"] == 1024 and S["retain_max_length"] == 1024
    for key, val, msg in (("batch_size", 2, "batch_size"), ("grad_accum", 2, "grad_accum"),
                          ("max_steps", 150, "max_steps")):
        bad = legacy("gpt2_eb_simnpo")
        bad["train"][key] = val
        with pytest.raises(ValueError, match=msg):
            D.resolve_simnpo_config(bad)
    bad = legacy("gpt2_eb_simnpo")
    bad["simnpo"]["retain_max_length"] = 512
    with pytest.raises(ValueError, match="retain_max_length"):
        D.resolve_simnpo_config(bad)
    bad = legacy("gpt2_eb_simnpo")
    bad["simnpo"]["forget_max_length"] = 768
    with pytest.raises(ValueError, match="forget_max_length"):
        D.resolve_simnpo_config(bad)
    with pytest.raises(ValueError, match="max_steps"):
        D.resolve_simnpo_config(cfg, max_steps=3)          # the smoke override disagrees too
    # no budget: the sweep config resolves exactly as before
    assert D.resolve_simnpo_config(legacy("gpt2_simnpo"))["budget"] is None


def test_equal_budget_run_on_a_tiny_gpt2_sees_the_target_per_step_and_the_retained_set_once(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    isolate_corpora(monkeypatch)
    save_checkpoint(tmp_path)
    # 1 target doc of 20 tokens; 2 x 5 retained docs of 20 tokens (+1 sep each)
    # -> 2 x 105 tokens -> 2 x 7 blocks of 16 -> 14 blocks over 3 steps (4-5 each)
    write_sets(tmp_path, n_target=1, n_contrastive=5, n_heldout=2, words=20)
    cfg = legacy("gpt2_eb_simnpo")
    cfg.update(name="simnpo_gpt2_eb", device_map="cpu", wandb={"enabled": False},
               budget={"target_passes": 3, "retain_exposure_tokens": 200, "steps": 3,
                       "forget_max_length": 1024, "retain_block": 16})
    cfg["simnpo"]["retain_max_length"] = 16
    cfg["train"].update(max_steps=3, log_every=1)
    cp = tmp_path / "eb.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    m = D.run(str(cp), no_push=True)
    assert m["failure"] is None and m["schedule"]["total_steps"] == 3
    assert m["data"]["retain_batches"] == 0                  # the sampler was not used
    b = m["budget"]
    assert b["steps"] == 3
    assert b["target_tokens_unique"] == 20 and b["target_tokens_exposed"] == 3 * 20
    assert b["target_passes_realised"] == 3.0
    assert b["retain_tokens_unique"] == 200 == b["retain_tokens_exposed_docs"]
    assert b["retain_separator_tokens"] == 10 and b["retain_tokens_exposed"] == 210
    assert b["retain_blocks"] == 14 and b["retain_tokens_covariance"] == 0
    rp = m["schedule"]["retain_plan"]
    assert (rp["n_blocks"], rp["blocks_per_step_min"], rp["blocks_per_step_max"]) == (14, 4, 5)
    metrics = json.loads((tmp_path / "logs" / "methods" / "simnpo_gpt2_eb" / "metrics.json").read_text())
    assert metrics["budget"]["retain_tokens_exposed_docs"] == metrics["budget"]["retain_tokens_unique"]
    assert metrics["budget"]["configured"]["steps"] == 3
    # a budget the data cannot realise fails AFTER metrics.json is written
    cfg.update(name="simnpo_gpt2_eb_bad")
    cfg["budget"]["retain_exposure_tokens"] = 400
    cp.write_text(yaml.safe_dump(cfg))
    with pytest.raises(RuntimeError, match="retain_exposure_tokens"):
        D.run(str(cp), no_push=True)
    assert (tmp_path / "logs" / "methods" / "simnpo_gpt2_eb_bad" / "metrics.json").exists()
