"""``srspd.report``: per-tensor relative Frobenius change of an edit,
its aggregates, and the two writers. A tiny random GPT-2 (2 blocks, 32 dims,
vocab 64) is enough: the arithmetic does not care about scale."""
from __future__ import annotations

import csv
import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from srspd import report as WD  # noqa: E402


def _tiny(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=16, n_embd=32,
                                  n_layer=2, n_head=2)
    return transformers.GPT2LMHeadModel(cfg).eval()


def _clone(model):
    m = _tiny()
    m.load_state_dict(model.state_dict())
    return m


def test_classify_gpt2_and_fallback_names():
    assert WD.classify("transformer.h.3.attn.c_attn.weight") == (3, "attn.c_attn")
    assert WD.classify("transformer.h.11.mlp.c_proj.weight") == (11, "mlp.c_proj")
    assert WD.classify("model.layers.7.mlp.down_proj.weight") == (7, "mlp.down_proj")
    assert WD.classify("transformer.wte.weight") == (None, "embeddings")
    assert WD.classify("lm_head.weight") == (None, "embeddings")
    assert WD.classify("transformer.ln_f.weight") == (None, "other")


def test_zero_delta_is_all_unchanged():
    base = _tiny()
    rep = WD.weight_delta_report(base, _clone(base), erank=True)
    rows = rep["tensors"]
    # 2 blocks x 4 projections + wte + wpe (lm_head is tied to wte: counted once).
    assert len(rows) == 2 * 4 + 2
    assert all(r["rel"] == 0.0 and not r["changed"] for r in rows)
    assert all(r["erank_delta"] is None for r in rows)
    assert all(r["erank_base"] is not None for r in rows if r["block"] is not None)
    t = rep["totals"]
    assert t["n_changed"] == 0 and t["changed_param_fraction"] == 0.0
    assert t["grad_mass"] == 0.0 and t["grad_mass_proj"] == 0.0
    assert t["sum_rel"] == 0.0 and t["global_rel"] == 0.0 and t["fro_delta"] == 0.0
    assert t["erank_delta_mean"] is None
    assert set(rep["per_proj"]) == {"attn.c_attn", "attn.c_proj", "mlp.c_fc",
                                    "mlp.c_proj", "embeddings"}
    assert set(rep["per_block"]) == {"0", "1"}


def test_single_edited_matrix_is_localised():
    base = _tiny()
    edited = _clone(base)
    with torch.no_grad():
        w = edited.transformer.h[1].mlp.c_fc.weight
        w.add_(0.5 * torch.randn_like(w))
        want_rel = float(torch.linalg.norm(w - base.transformer.h[1].mlp.c_fc.weight)
                         / torch.linalg.norm(base.transformer.h[1].mlp.c_fc.weight))
    rep = WD.weight_delta_report(base, edited, erank=True)
    changed = [r for r in rep["tensors"] if r["changed"]]
    assert [r["name"] for r in changed] == ["transformer.h.1.mlp.c_fc.weight"]
    r = changed[0]
    assert r["rel"] == pytest.approx(want_rel, rel=1e-5)
    assert r["block"] == 1 and r["proj"] == "mlp.c_fc"
    assert 1.0 <= r["erank_delta"] <= min(r["shape"])
    t = rep["totals"]
    assert t["n_changed"] == 1
    # grad mass of a single edited matrix is that matrix's squared Frobenius delta
    assert r["fro_delta_sq"] == pytest.approx(r["fro_delta"] ** 2, rel=1e-6)
    assert t["grad_mass"] == pytest.approx(r["fro_delta"] ** 2, rel=1e-6)
    assert t["grad_mass_proj"] == pytest.approx(t["grad_mass"])
    assert rep["per_block"]["1"]["grad_mass"] == pytest.approx(t["grad_mass"])
    assert rep["per_block"]["0"]["grad_mass"] == 0.0
    assert rep["per_proj"]["mlp.c_fc"]["grad_mass"] == pytest.approx(t["grad_mass"])
    assert t["changed_param_fraction"] == pytest.approx(r["numel"] / t["numel"])
    assert t["sum_rel"] == pytest.approx(want_rel, rel=1e-5)
    assert t["mean_rel_weighted"] == pytest.approx(want_rel * r["numel"] / t["numel"], rel=1e-5)
    assert t["erank_delta_mean"] == pytest.approx(r["erank_delta"])
    assert rep["per_block"]["0"]["n_changed"] == 0
    assert rep["per_block"]["1"]["n_changed"] == 1
    assert rep["per_proj"]["mlp.c_fc"]["n_changed"] == 1
    assert rep["per_proj"]["attn.c_attn"]["fro_delta"] == 0.0


def test_aggregates_add_up_and_global_rel_is_a_norm():
    base = _tiny()
    edited = _clone(base)
    with torch.no_grad():
        for p in edited.parameters():
            if p.ndim == 2:
                p.add_(0.1 * torch.randn_like(p))
    rep = WD.weight_delta_report(base, edited, erank=False)
    rows = rep["tensors"]
    t = rep["totals"]
    assert t["n_tensors"] == len(rows) and t["n_changed"] == len(rows)
    assert t["numel"] == sum(r["numel"] for r in rows)
    assert t["sum_rel"] == pytest.approx(sum(r["rel"] for r in rows))
    assert t["fro_delta"] == pytest.approx(sum(r["fro_delta"] ** 2 for r in rows) ** 0.5)
    assert t["global_rel"] == pytest.approx(t["fro_delta"] / t["fro_base"])
    assert sum(b["numel"] for b in rep["per_block"].values()) + \
        rep["per_proj"]["embeddings"]["numel"] == t["numel"]
    assert sum(p["numel"] for p in rep["per_proj"].values()) == t["numel"]
    assert all(r["erank_delta"] is None for r in rows)       # flag off: skipped


def test_grad_mass_adds_over_matrices_and_embeddings_count_once():
    base = _tiny()
    edited = _clone(base)
    with torch.no_grad():
        w1 = edited.transformer.h[0].attn.c_proj.weight
        w2 = edited.transformer.h[1].mlp.c_proj.weight
        w1.add_(0.3 * torch.randn_like(w1))
        w2.add_(0.7 * torch.randn_like(w2))
        d1 = float(torch.sum((w1 - base.transformer.h[0].attn.c_proj.weight) ** 2))
        d2 = float(torch.sum((w2 - base.transformer.h[1].mlp.c_proj.weight) ** 2))
    rep = WD.weight_delta_report(base, edited)
    t = rep["totals"]
    assert t["grad_mass"] == pytest.approx(d1 + d2, rel=1e-5)
    assert t["grad_mass_proj"] == pytest.approx(d1 + d2, rel=1e-5)
    assert rep["per_block"]["0"]["grad_mass"] == pytest.approx(d1, rel=1e-5)
    assert rep["per_block"]["1"]["grad_mass"] == pytest.approx(d2, rel=1e-5)
    # grad_mass is the sum of the per-tensor squares, i.e. fro_delta**2 of the totals
    assert t["grad_mass"] == pytest.approx(t["fro_delta"] ** 2, rel=1e-5)
    # an embedding edit counts once (lm_head is tied to wte) and is outside grad_mass_proj
    with torch.no_grad():
        wte = edited.transformer.wte.weight
        wte.add_(0.1 * torch.randn_like(wte))
        d3 = float(torch.sum((wte - base.transformer.wte.weight) ** 2))
    rep = WD.weight_delta_report(base, edited)
    t = rep["totals"]
    assert t["grad_mass"] == pytest.approx(d1 + d2 + d3, rel=1e-5)
    assert t["grad_mass_proj"] == pytest.approx(d1 + d2, rel=1e-5)
    assert rep["per_proj"]["embeddings"]["grad_mass"] == pytest.approx(d3, rel=1e-5)
    assert sum(1 for r in rep["tensors"] if r["proj"] == "embeddings" and r["changed"]) == 1
    # ordering: grad_mass is the first totals key and the first summary column
    assert list(t)[0] == "grad_mass"
    assert list(WD.totals_summary(rep))[0] == "weight_delta/grad_mass"


def test_state_dicts_and_modules_are_interchangeable():
    base = _tiny()
    edited = _clone(base)
    a = WD.weight_delta_report(base, edited)
    b = WD.weight_delta_report(base.state_dict(), edited.state_dict())
    assert [r["name"] for r in a["tensors"]] == [r["name"] for r in b["tensors"]]
    with pytest.raises(TypeError):
        WD.weight_delta_report(42, edited)


def test_writers(tmp_path):
    base = _tiny()
    edited = _clone(base)
    with torch.no_grad():
        edited.transformer.h[0].attn.c_attn.weight.add_(1.0)
    rep = WD.weight_delta_report(base, edited, erank=True)
    paths = WD.write_weight_delta(rep, str(tmp_path))
    j = json.loads((tmp_path / "weight_delta.json").read_text())
    assert j["totals"]["n_changed"] == 1
    rows = list(csv.DictReader(open(paths["csv"])))
    assert len(rows) == len(rep["tensors"])
    assert set(rows[0]) == set(WD.CSV_FIELDS)
    hit = [r for r in rows if r["changed"] == "1"]
    assert len(hit) == 1 and hit[0]["name"] == "transformer.h.0.attn.c_attn.weight"
    assert hit[0]["shape"] == "32x96"
    s = WD.totals_summary(rep)
    assert s["weight_delta/n_changed"] == 1
