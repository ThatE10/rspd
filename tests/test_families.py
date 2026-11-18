"""End to end on a tiny random-weight model of every verified family:
``inspect_model`` reports the expected input keys and quirks; ``erase`` runs
under both solvers with leaks for every edited matrix; the two solvers agree;
``restore`` is bit for bit; Gemma-4 excludes its towers and backfills the
KV-shared tensors on export."""
from __future__ import annotations

import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import rspd  # noqa: E402
import tiny_models as tm  # noqa: E402

DECODER_KEYS = {"attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "o_in": ["self_attn.o_proj"], "mlp_in": ["mlp.gate_proj", "mlp.up_proj"],
                "down_in": ["mlp.down_proj"]}
GPT2_TAILS = ["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"]

#: family -> (blocks, KV-shared blocks (no k/v_proj), expected quirks)
SPEC = {
    "gpt2": (3, (), {"conv1d_transposed"}),
    "mistral": (2, (), {"shared_input_check"}),
    "llama": (2, (), {"shared_input_check"}),
    "qwen3": (2, (), {"shared_input_check"}),
    "gemma3_text": (2, (), {"shared_input_check"}),
    "gemma3_multimodal": (2, (), {"shared_input_check", "multimodal_towers"}),
    "gemma4_text": (4, (2, 3), {"shared_input_check", "kv_shared_layers"}),
    "gemma4_multimodal": (4, (2, 3), {"shared_input_check", "kv_shared_layers",
                                      "multimodal_towers"}),
}


def _expected_keys(family):
    blocks, shared, _ = SPEC[family]
    out = {}
    for b in range(blocks):
        if family == "gpt2":
            for t in GPT2_TAILS:
                out[f"L{b:02d}.{t}"] = [f"L{b:02d}.{t}"]
            continue
        for key, tails in DECODER_KEYS.items():
            ts = [t for t in tails if not (b in shared and t in ("self_attn.k_proj",
                                                                  "self_attn.v_proj"))]
            out[f"L{b:02d}.{key}"] = [f"L{b:02d}.{t}" for t in ts]
    return out


def _params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _same(m, before):
    return all(torch.equal(before[n], p) for n, p in m.named_parameters())


@pytest.mark.parametrize("family", list(SPEC))
def test_inspect_reports_keys_and_quirks(family):
    m = tm.FAMILIES[family]()
    info = rspd.inspect_model(m)
    json.dumps(info)                                     # JSON-serialisable
    keys = {k: v["members"] for k, v in info["input_keys"].items()}
    assert keys == _expected_keys(family)
    assert set(info["quirks"]) == SPEC[family][2], info["quirks"]
    for k, v in info["input_keys"].items():
        assert v["gram_bytes_readout"] == 8 * v["d_in"] ** 2
        assert all(info["matrices"][n]["d_in"] == v["d_in"] for n in v["members"])
    assert info["totals"]["blocks"] == SPEC[family][0] and not info["warnings"]
    if "multimodal" in family:
        assert info["quirks"]["multimodal_towers"]["excluded_modules"] > 0
        assert not any("vision" in v["path"] for v in info["matrices"].values())
    if family.startswith("gemma4"):
        assert info["quirks"]["kv_shared_layers"]["first_shared_layer"] == 2


@pytest.mark.parametrize("family", list(SPEC))
def test_erase_both_solvers_leaks_and_restore(family):
    m = tm.FAMILIES[family]()
    before = _params(m)
    tgt, ret = tm.token_sets()
    reports = {}
    for solver in ("readout", "grams"):
        rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0, solver=solver,
                          ridge_rel=1e-10)
        names = {n for ms in _expected_keys(family).values() for n in ms}
        assert set(rep.matrices) == names                 # leaks for EVERY edited matrix
        for n, r in rep.matrices.items():
            assert math.isfinite(r["L_X_rel"]) and math.isfinite(r["L_Y_rel"]), n
            assert 0.0 <= r["L_X_rel"] <= 1.0 + 1e-9 and r["L_Y_rel"] >= 0.0, n
            assert r["edit_fro"] > 0.0 and r["solve_path"] in ("cholesky", "cholesky_ridge", "eigh"), n
        assert rep.applied and rep.dose["mu_per_token"] == 1.0
        assert rep.dose["n_x"] == 3 * 24 and rep.dose["n_y"] == 6 * 32
        assert rep.edit["survival"] == pytest.approx(1.0, abs=1e-5)       # fp32 weights
        assert set(rep.quirks) == SPEC[family][2]
        d = json.loads(rep.to_json())
        assert d["solver"] == solver and set(d["matrices"]) == names
        assert not _same(m, before)
        assert sorted(rspd.api.restore(m)) == sorted(names)
        assert _same(m, before)                           # bit for bit
        reports[solver] = rep
    for n, r in reports["readout"].matrices.items():
        g = reports["grams"].matrices[n]
        for f in ("L_X_rel", "L_Y_rel"):
            assert g[f] == pytest.approx(r[f], rel=1e-4, abs=1e-7), (n, f)
        assert g["delta_fro"] == pytest.approx(r["delta_fro"], rel=1e-4)


@pytest.mark.parametrize("family", ["gemma4_text", "gemma4_multimodal"])
def test_gemma4_export_backfills_kv_shared_tensors(tmp_path, family):
    from safetensors import safe_open
    from safetensors.torch import save_file

    from rspd.models.export import KV_SHARED_SHARD, export_checkpoint
    m = tm.FAMILIES[family]()
    # the "source" checkpoint: what save_pretrained writes, plus the dead tensors a hub
    # checkpoint carries for the KV-shared layers (2, 3)
    src = tmp_path / "source"
    m.save_pretrained(src)
    prefix = ("model.language_model.layers" if "multimodal" in family else "model.layers")
    dead = {}
    for i in (2, 3):
        kv = 16 if i % 2 else 8
        dead[f"{prefix}.{i}.self_attn.k_proj.weight"] = torch.randn(kv, 16)
        dead[f"{prefix}.{i}.self_attn.v_proj.weight"] = torch.randn(kv, 16)
        dead[f"{prefix}.{i}.self_attn.k_norm.weight"] = torch.randn(kv // 2)
    save_file(dead, str(src / "model-extra.safetensors"), metadata={"format": "pt"})

    tgt, ret = tm.token_sets()
    rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=1.0)
    assert "kv_shared_layers" in rep.quirks
    info = export_checkpoint(m, None, tmp_path / "out", source=str(src))
    assert sorted(info["kv_shared_backfilled"]) == sorted(dead)
    with safe_open(str(tmp_path / "out" / KV_SHARED_SHARD), framework="pt") as f:
        for k in dead:
            assert torch.equal(f.get_tensor(k), dead[k])
    # the edited weights are what was exported
    from safetensors.torch import load_file
    w = {}
    for p in (tmp_path / "out").glob("model*.safetensors"):
        w.update(load_file(str(p)))
    q = f"{prefix}.0.self_attn.q_proj.weight"
    assert torch.equal(w[q], dict(m.named_parameters())[q].detach())
    rspd.api.restore(m)
    # without a source the quirk is reported as a warning, not silently dropped
    info2 = export_checkpoint(m, None, tmp_path / "out2")
    assert info2["kv_shared_backfilled"] == [] and any("KV-shared" in w for w in info2["warnings"])


def test_bf16_model_reports_survival_and_restores():
    m = tm.mistral(dtype=torch.bfloat16)
    before = _params(m)
    tgt, ret = tm.token_sets()
    rep = rspd.api.erase(m, target=tgt, retained=ret, mu_per_token=[0.1, 1.0],
                      apply_mu_per_token=0.1)
    assert "bf16_edit_in_fp32" in rep.quirks
    assert 0.9 < rep.edit["survival"] < 1.1 and rep.edit["rel_err"] < 0.05
    assert set(rep.by_dose) == {repr(0.1), repr(1.0)}
    rspd.api.restore(m)
    assert _same(m, before)
