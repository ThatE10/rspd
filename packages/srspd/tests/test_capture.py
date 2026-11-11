"""Gram capture: the pass planner, the fp64 sink, the HF hook source (early exit,
boundary resume, shared-input check), spills and the pristine store."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import tiny_models as tm  # noqa: E402
from srspd.capture import (BoundaryCache, Fp64GramSink, GramSink, HFHookSource,  # noqa: E402
                           MemorySpill, TensorSpill, gram_bytes, plan_gram_groups)
from srspd.capture.hf import decoder_layer_list  # noqa: E402
from srspd.models import HFLinearHarness, PristineStore, group_by_input  # noqa: E402


# ── planner ──────────────────────────────────────────────────────────────────
def test_plan_is_greedy_contiguous_and_complete():
    items = [(f"L{i:02d}.k", 10, "cpu") for i in range(5)]
    plan = plan_gram_groups(items, 2 * gram_bytes(10))
    assert plan == [["L00.k", "L01.k"], ["L02.k", "L03.k"], ["L04.k"]]
    assert plan_gram_groups(items, 10 ** 9) == [[k for k, _, _ in items]]
    two = plan_gram_groups(items, 2 * gram_bytes(10, 16), bytes_per_elem=16)
    assert two == plan


def test_plan_other_devices_may_still_join():
    items = [("a", 10, "cuda:0"), ("b", 10, "cuda:0"), ("c", 10, "cuda:1")]
    assert plan_gram_groups(items, gram_bytes(10)) == [["a", "c"], ["b"]]


def test_plan_refuses_an_oversized_key_and_duplicates():
    with pytest.raises(ValueError, match="no pass can hold it"):
        plan_gram_groups([("big", 100, "cpu")], gram_bytes(10))
    with pytest.raises(ValueError, match="twice"):
        plan_gram_groups([("a", 1, "cpu"), ("a", 1, "cpu")], 100)


# ── sink ─────────────────────────────────────────────────────────────────────
def test_sink_accumulates_in_float64_in_place():
    s = Fp64GramSink({"k": 3}, {"k": torch.device("cpu")})
    assert isinstance(s, GramSink)
    xs = [torch.randn(4, 3, dtype=torch.bfloat16) for _ in range(3)]
    G0 = s.grams["k"]
    for x in xs:
        s.add("k", x)
    G, n = s.result()["k"]
    assert G is G0 and G.dtype == torch.float64 and n == 12
    X = torch.cat([x.double() for x in xs])
    assert torch.allclose(G, X.T @ X, rtol=1e-12, atol=1e-12)


# ── hook source ──────────────────────────────────────────────────────────────
def _setup(model):
    h = HFLinearHarness(model)
    km = group_by_input(h.all_names)
    rep = {k: m[0] for k, m in km.items()}
    dims = {k: h.d_in(rep[k]) for k in km}
    return h, km, rep, dims


def _run(h, km, rep, dims, seqs, keys, **kw):
    sink = Fp64GramSink({k: dims[k] for k in keys}, {k: torch.device("cpu") for k in keys})
    stats = HFHookSource(h).accumulate(iter(seqs), {k: rep[k] for k in keys}, sink,
                                       members_of_key={k: km[k] for k in keys}, **kw)
    return sink.result(), stats


@pytest.mark.parametrize("build", [tm.mistral, tm.gpt2], ids=["mistral", "gpt2"])
def test_resumed_passes_give_bitwise_the_same_grams(build):
    h, km, rep, dims = _setup(build())
    seqs = [torch.randint(0, tm.VOCAB, (1, n)) for n in (7, 11, 5)]
    keys = list(km)
    full, st = _run(h, km, rep, dims, seqs, keys)
    assert st["chunks"] == 3 and st["tokens"] == 23 and not st["resumed"]
    layers = decoder_layer_list(h)
    first = [k for k in keys if k.startswith("L00")]
    rest = [k for k in keys if not k.startswith("L00")]
    cache = BoundaryCache()
    g0, s0 = _run(h, km, rep, dims, seqs, first, layers=layers, cache=cache, next_start=1)
    assert cache.start == 1 and len(cache.chunks) == 3
    g1, s1 = _run(h, km, rep, dims, seqs, rest, layers=layers, cache=cache)
    assert s1["resumed"] and s1["start_layer"] == 1
    for k in keys:
        G, n = (g0 if k in g0 else g1)[k]
        assert n == full[k][1] and torch.equal(G, full[k][0]), k
    # every hook and patched forward is gone
    assert all(not m._forward_hooks for m in h.module_by_name.values())
    assert all("forward" not in lay.__dict__ for lay in layers)


def test_early_exit_skips_deeper_layers():
    h, km, rep, dims = _setup(tm.mistral())
    seen = []
    deep = h.module_by_name["L01.mlp.down_proj"]
    hd = deep.register_forward_hook(lambda *a: seen.append(1))
    try:
        _run(h, km, rep, dims, [torch.randint(0, tm.VOCAB, (1, 6))], ["L00.attn_in"])
    finally:
        hd.remove()
    assert not seen


def test_shared_input_mismatch_raises():
    m = tm.mistral()
    h, km, rep, dims = _setup(m)
    # make k_proj see a different input than q_proj by perturbing it in a pre-hook
    k_mod = h.module_by_name["L00.self_attn.k_proj"]
    hd = k_mod.register_forward_pre_hook(lambda mod, args: (args[0] + 1.0,))
    try:
        with pytest.raises(AssertionError, match="bitwise-identical"):
            _run(h, km, rep, dims, [torch.randint(0, tm.VOCAB, (1, 6))], ["L00.attn_in"])
    finally:
        hd.remove()
    assert all(not mm._forward_hooks for mm in h.module_by_name.values())


# ── spills and the pristine store ────────────────────────────────────────────
@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])
def test_spills_round_trip_bitwise(tmp_path, disk):
    sp = TensorSpill(tmp_path / "s") if disk else MemorySpill()
    t = torch.randn(3, 4, dtype=torch.float64)
    sp.put("a/b:c", t)
    assert sp.has("a/b:c") and torch.equal(sp.get("a/b:c"), t)
    assert sp.get("a/b:c", dtype=torch.float32).dtype == torch.float32


@pytest.mark.parametrize("disk", [False, True], ids=["memory", "disk"])
@pytest.mark.parametrize("build", [tm.mistral, tm.gpt2], ids=["mistral", "gpt2"])
def test_pristine_store_restores_bit_for_bit(tmp_path, disk, build):
    m = build()
    h = HFLinearHarness(m)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    st = PristineStore(h, tmp_path if disk else None)
    for n in h.all_names:
        p = st.pristine(n)
        assert torch.equal(p, h.get_weight(n))
        st.write(n, p * 0.5)
    assert st.is_edited
    for n in h.all_names:          # pristine() still returns the original after the edit
        assert torch.equal(st.pristine(n) * 0.5, h.get_weight(n))
    assert sorted(st.restore_all()) == sorted(h.all_names)
    assert all(torch.equal(before[n], p) for n, p in m.named_parameters())
    assert st.restore_all() == []
