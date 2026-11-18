"""KUDA on GPT-2 (:mod:`rspd.experiments.methods.kuda`): ``module_name: mlp.c_proj``,
the ``Conv1D`` ``[d_in, d_out]`` gradient projection (``P @ grad``), the
covariance pre-hook on ``c_proj``'s input, and a short end-to-end run on a tiny
GPT-2 over ``local_jsonl`` corpora. CPU, no network."""
from __future__ import annotations

import importlib

import copy
import json

import pytest

from conftest import isolate_corpora  # noqa: E402
import torch
import yaml

from gpt2_tiny_helpers import corpora_block, save_checkpoint, tiny_gpt2, tiny_tokenizer, write_sets

d = importlib.import_module("rspd.experiments.methods.kuda")
C = d.C
from recipe_configs import legacy  # noqa: E402



def _projector(n=8, big=4, tau=1e-2):
    g = torch.Generator().manual_seed(0)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    s = torch.tensor([10.0, 7.0, 4.0, 1.0][:big] + [1e-4, 5e-5, 2e-5, 1e-6][:n - big],
                     dtype=torch.float64)
    cov = Q @ torch.diag(s) @ Q.T
    P, _ = d.build_projector(cov, tau=tau, dtype=torch.float64)
    retained = Q[:, [i for i in range(n) if s[i] >= tau]]       # the retained-input subspace
    return P, retained


@pytest.mark.parametrize("layout", ["out_in", "in_out"])
def test_projected_gradient_is_orthogonal_to_the_retained_input_subspace(layout):
    """nn.Linear [out, in] -> grad @ P; Conv1D [in, out] -> P @ grad. Either way
    no component of the update is left along a direction the retain corpus uses."""
    n = 8
    P, retained = _projector(n)
    lin = torch.nn.Linear(n, 3, bias=False).double()
    conv = torch.nn.Parameter(torch.randn(n, 3, dtype=torch.float64))   # Conv1D weight [d_in, d_out]
    w = lin.weight if layout == "out_in" else conv
    w.grad = torch.randn_like(w)
    raw = w.grad.clone()
    assert d.project_gradients({"w": w}, {"w": P}, {"w": layout}) == 1
    assert w.grad.shape == raw.shape and not torch.allclose(w.grad, raw)
    if layout == "out_in":
        assert torch.allclose(w.grad @ retained, torch.zeros(3, retained.shape[1], dtype=torch.float64), atol=1e-9)
        assert torch.allclose(w.grad, raw @ P, atol=1e-9)
    else:
        assert torch.allclose(retained.T @ w.grad, torch.zeros(retained.shape[1], 3, dtype=torch.float64), atol=1e-9)
        assert torch.allclose(w.grad, P @ raw, atol=1e-9)
    # idempotent
    before = w.grad.clone()
    d.project_gradients({"w": w}, {"w": P}, {"w": layout})
    assert torch.allclose(w.grad, before, atol=1e-9)


def test_default_layout_is_out_in_and_unknown_layouts_are_refused():
    P, _ = _projector()
    w = torch.nn.Parameter(torch.randn(3, 8, dtype=torch.float64))
    w.grad = torch.randn_like(w)
    raw = w.grad.clone()
    d.project_gradients({"w": w}, {"w": P})
    assert torch.allclose(w.grad, raw @ P)
    with pytest.raises(ValueError, match="layout"):
        d.project_gradients({"w": w}, {"w": P}, {"w": "sideways"})


def test_resolve_accepts_c_proj_and_refuses_other_modules():
    cfg = legacy("gpt2_kuda")
    R = d.resolve_kuda_config(cfg)
    assert R["module_name"] == "mlp.c_proj" and R["position_ids_from_mask"] is True
    assert R["dtype"] == "float32" and R["push_to_hub"] is False
    assert R["forget_corpora"] == ["target_nld"] and R["retain_corpora"] == ["contrastive"]
    assert R["layer_id"] == 6 and R["layer_ids"] == [3, 4, 5, 6]
    assert R["nullspace"]["max_length"] == 1024 and R["nullspace"]["cov_dtype"] == "float32"
    assert R["init_jitter"] == 0.0
    bad = copy.deepcopy(cfg)
    bad["kuda"]["module_name"] = "mlp.c_fc"
    with pytest.raises(ValueError, match="module_name"):
        d.resolve_kuda_config(bad)
    bad = copy.deepcopy(cfg)
    bad["kuda"]["position_ids_from_mask"] = "yes"
    with pytest.raises(ValueError, match="position_ids_from_mask"):
        d.resolve_kuda_config(bad)
    # the module name is part of the covariance key
    other = dict(R, module_name="mlp.down_proj")
    assert d._cov_key(R, 3) != d._cov_key(other, 3)


def test_covariance_hooks_c_proj_input_on_gpt2():
    m, tok = tiny_gpt2(), tiny_tokenizer()
    texts = ["w1 w2 w3 w4", "w5 w6", "w7 w8 w9 w10 w11"]
    covs, n_tok = d.accumulate_down_proj_cov(m, tok, [0, 2], texts, batch_size=2, max_length=8,
                                             cov_dtype="float64", module_name="mlp.c_proj")
    assert n_tok == 11
    for lid in (0, 2):
        assert tuple(covs[lid].shape) == (64, 64)        # the 4*n_embd c_proj INPUT width
        assert torch.allclose(covs[lid], covs[lid].T)
    with pytest.raises(AttributeError, match="mlp.down_proj"):
        d.accumulate_down_proj_cov(m, tok, [0], texts, batch_size=2, max_length=8,
                                   module_name="mlp.down_proj")


def test_end_to_end_on_a_tiny_gpt2(tmp_path, monkeypatch):
    """The shipped GPT-2 config, shrunk, on a tiny GPT-2 over the local sets:
    only the c_proj weights of the target blocks move, the projectors are
    64 x 64 (the c_proj input), and the checkpoint is the run dir."""
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    isolate_corpora(monkeypatch)
    rel, _ = save_checkpoint(tmp_path)
    write_sets(tmp_path, n_target=3, n_contrastive=6, n_heldout=2, words=20)
    cfg = legacy("gpt2_kuda")
    cfg.update(name="kuda_gpt2_e2e", device_map="cpu", wandb={"enabled": False})
    cfg["kuda"].update(layer_id=2, layer_ids=[1, 2], max_num_batches=2, epoch=2,
                       batch_size=2, tau=1e-2)
    cfg["kuda"]["nullspace"].update(max_length=16, batch_size=4, cov_dtype="float64")
    cp = tmp_path / "k.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    before = {n: p.detach().clone() for n, p in tiny_gpt2().named_parameters()}
    m = d.run(str(cp), no_push=True)
    assert m["name"] == "kuda_gpt2_e2e"
    md = tmp_path / "models" / "methods" / "kuda_gpt2_e2e"
    assert (md / "config.json").exists() and list(md.glob("*.safetensors"))
    assert m["target_parameters"] == ["transformer.h.1.mlp.c_proj.weight",
                                      "transformer.h.2.mlp.c_proj.weight"]
    assert set(m["weight_layouts"].values()) == {"in_out"}
    for lid, info in m["nullspace"].items():
        assert info["d"] == 64
    from transformers import AutoModelForCausalLM
    after = dict(AutoModelForCausalLM.from_pretrained(str(md)).named_parameters())
    moved = [n for n, p in after.items() if not torch.equal(p, before[n])]
    assert moved == ["transformer.h.1.mlp.c_proj.weight", "transformer.h.2.mlp.c_proj.weight"]
    assert m["steps"] == 4                            # 2 batches x 2 epochs
    assert m["grad_norm_projected_last"] is not None





def test_equal_budget_config_overrides_the_windows_pins_the_schedule_and_shares_the_cache():
    cfg = legacy("gpt2_eb_kuda")
    R = d.resolve_kuda_config(cfg)
    assert R["budget"]["steps"] == 200 and R["forget_max_length"] == [1024]
    assert R["retain_max_length"] == 1024 and (R["max_num_batches"], R["epoch"]) == (1, 200)
    assert R["tau"] == pytest.approx(1e-2) and R["lr"] == pytest.approx(3e-4)
    # the same covariance key as the un-budgeted point: the spectrum cache is shared
    R0 = d.resolve_kuda_config(legacy("gpt2_kuda"))
    assert R0["budget"] is None and R0["forget_max_length"] == [768]
    assert d._cov_key(R, 3) == d._cov_key(R0, 3)
    bad = copy.deepcopy(cfg)
    bad["kuda"]["epoch"] = 100
    with pytest.raises(ValueError, match="epoch x kuda.max_num_batches"):
        d.resolve_kuda_config(bad)
    with pytest.raises(ValueError, match="epoch x kuda.max_num_batches"):
        d.resolve_kuda_config(cfg, max_num_batches=2)
    bad = copy.deepcopy(cfg)
    bad["kuda"]["retain_max_length"] = 512
    with pytest.raises(ValueError, match="retain_max_length"):
        d.resolve_kuda_config(bad)


def test_equal_budget_run_on_a_tiny_gpt2_sees_the_target_per_step_and_the_retained_set_once(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    isolate_corpora(monkeypatch)
    save_checkpoint(tmp_path)
    # 1 target doc of 20 tokens; 2 x 5 retained docs of 20 (+1 sep) -> 14 blocks of 16
    write_sets(tmp_path, n_target=1, n_contrastive=5, n_heldout=2, words=20)
    cfg = legacy("gpt2_eb_kuda")
    cfg.update(name="kuda_gpt2_eb", device_map="cpu", wandb={"enabled": False},
               budget={"target_passes": 3, "retain_exposure_tokens": 200, "steps": 3,
                       "forget_max_length": 1024, "retain_block": 16})
    cfg["kuda"].update(layer_id=2, layer_ids=[1, 2], epoch=3, retain_max_length=16)
    cfg["kuda"]["nullspace"].update(max_length=16, batch_size=4, cov_dtype="float64")
    cp = tmp_path / "eb.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    m = d.run(str(cp), no_push=True)
    assert m["steps"] == 3 and m["num_batches"] == 1
    assert m["resolved"]["forget_max_length"] == [1024] and m["forget_max_length"] == [1024]
    b = m["budget"]
    assert b["steps"] == 3
    assert b["target_tokens_unique"] == 20 and b["target_tokens_exposed"] == 3 * 20
    assert b["retain_tokens_unique"] == 200 == b["retain_tokens_exposed_docs"]
    assert b["retain_separator_tokens"] == 10 and b["retain_blocks"] == 14
    # the covariance pass read the retained set outside the loss (10 docs x 16-token window)
    assert b["retain_tokens_covariance"] == max(info["n_tokens"] for info in m["nullspace"].values()) > 0
    assert (m["retain_plan"]["blocks_per_step_min"], m["retain_plan"]["blocks_per_step_max"]) == (4, 5)
    metrics = json.loads((tmp_path / "logs" / "methods" / "kuda_gpt2_eb" / "metrics.json").read_text())
    assert metrics["budget"]["retain_tokens_exposed_docs"] == metrics["budget"]["retain_tokens_unique"]
    # a second run hits the projector cache and still reports the covariance tokens
    cfg.update(name="kuda_gpt2_eb2")
    cp.write_text(yaml.safe_dump(cfg))
    m2 = d.run(str(cp), no_push=True)
    assert all(info["cached"] for info in m2["nullspace"].values())
    assert m2["budget"]["retain_tokens_covariance"] == b["retain_tokens_covariance"]
    # --skip-train records a zero exposure and does not assert
    cfg.update(name="kuda_gpt2_eb3")
    cp.write_text(yaml.safe_dump(cfg))
    m3 = d.run(str(cp), no_push=True, skip_train=True)
    assert m3["steps"] == 0 and m3["budget"]["steps"] == 0
