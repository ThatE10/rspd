"""``srspd.truncate_component`` -- several rank truncations of one erased
component from ONE spectrum (R2), against ``erased_component_gram(...,
lam_rel=l)`` taken one ``lam_rel`` at a time.

CPU, no model. X = TARGET (erased), Y = RETAINED (kept), batch-first. One SVD
of ``M X^T`` per matrix serves every ``lam_rel``; rebuilding the component per
``lam_rel`` would repeat the costliest step.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from srspd.solve import (ErasedComponent, ErasureConventionError,  # noqa: E402
                        apply_erasure, erased_component_gram, leaks_gram, readout_gram,
                        truncate_component)

F64 = torch.float64
LAMS = [1e-3, 1e-2, 0.1, 0.5]


def _problem(seed=0, d_out=14, d_in=12, n_x=9, n_y=30, mu=2.5):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    W, X, Y = rn(d_out, d_in), rn(n_x, d_in), rn(n_y, d_in)
    # a graded target spectrum, so every lam in LAMS cuts somewhere different
    U, _, Vh = torch.linalg.svd(X, full_matrices=False)
    X = (U * torch.logspace(0, -2.0, min(n_x, d_in), dtype=F64)) @ Vh
    G_Y = Y.T @ Y
    S, _ = readout_gram(G_Y, X, mu)
    return W, X, G_Y, S, mu


def _rel(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


@pytest.mark.parametrize("shape", [(14, 12, 9), (6, 12, 9), (14, 5, 9), (14, 12, 3)])
def test_float64_matches_erased_component_gram_per_lam(shape):
    d_out, d_in, n_x = shape
    W, X, G_Y, S, mu = _problem(d_out=d_out, d_in=d_in, n_x=n_x)
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    comps = truncate_component(full, X, LAMS, svd_dtype=torch.float64)
    assert len(comps) == len(LAMS)
    for l, c in zip(LAMS, comps):
        ref = erased_component_gram(W, X, mu=mu, S=S, lam_rel=l)
        assert isinstance(c, ErasedComponent) and c.info["truncated"] is True
        assert c.info["rank"] == ref.info["rank"] and c.info["k"] == ref.info["k"] == min(d_out, n_x)
        assert c.A.shape == (d_out, c.info["rank"]) and c.B.shape == (c.info["rank"], d_in)
        assert c.A.dtype == c.B.dtype == F64
        assert _rel(c.dense(), ref.dense()) < 1e-9
        assert c.info["fit"] == pytest.approx(ref.info["fit"], rel=1e-9, abs=1e-12)
        assert torch.allclose(c.info["sigma"], ref.info["sigma"], rtol=1e-9, atol=1e-12)
        assert c.info["formula"] == ref.info["formula"]
        assert c.info["lam_rel"] == l and c.info["svd_dtype"] == "float64"
        for k in ("solver", "mu", "ridge_delta", "n_x", "d_out", "d_in"):
            assert c.info[k] == ref.info[k] or (c.info[k] != c.info[k] and ref.info[k] != ref.info[k]), k
        # the same edit and the same leaks
        assert _rel(apply_erasure(W, c), apply_erasure(W, ref)) < 1e-9
        a, b = leaks_gram(W, c, X, G_Y), leaks_gram(W, ref, X, G_Y)
        for k in ("L_X_rel", "L_Y_rel", "delta_fro"):
            assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12)


def test_float32_matches_to_1e4_relative():
    W, X, G_Y, S, mu = _problem(d_out=20, d_in=16, n_x=12)
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    comps = truncate_component(full, X, LAMS)                    # the default: float32
    for l, c in zip(LAMS, comps):
        ref = erased_component_gram(W, X, mu=mu, S=S, lam_rel=l)
        assert c.info["rank"] == ref.info["rank"] and c.info["svd_dtype"] == "float32"
        assert c.info["sigma"].dtype == torch.float32
        assert c.A.dtype == c.B.dtype == F64                     # factors in the component's dtype
        assert _rel(c.dense(), ref.dense()) < 1e-4
        assert c.info["fit"] == pytest.approx(ref.info["fit"], rel=1e-4, abs=1e-6)
        assert _rel(apply_erasure(W, c), apply_erasure(W, ref)) < 1e-4


def test_ranks_are_monotone_in_lam_rel_and_the_cut_is_the_threshold():
    W, X, G_Y, S, mu = _problem()
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    comps = truncate_component(full, X, LAMS, svd_dtype=torch.float64)
    ranks = [c.info["rank"] for c in comps]
    assert ranks == sorted(ranks, reverse=True) and ranks[0] > ranks[-1] >= 1
    s2 = comps[0].info["sigma"] ** 2
    for l, c in zip(LAMS, comps):
        assert c.info["rank"] == int((s2 > l * s2[0]).sum())
        assert c.info["fit"] == pytest.approx(float(s2[c.info["rank"]:].sum()))
    # the order of lam_rels is the order of the result, not sorted
    rev = truncate_component(full, X, LAMS[::-1], svd_dtype=torch.float64)
    assert [c.info["rank"] for c in rev] == ranks[::-1]
    # lam_rel 0 keeps every nonzero singular value: M_r = M
    (c0,) = truncate_component(full, X, [0.0], svd_dtype=torch.float64)
    assert c0.info["rank"] == min(X.shape[0], W.shape[0]) and _rel(c0.dense(), full.dense()) < 1e-9


def test_one_svd_per_call(monkeypatch):
    W, X, G_Y, S, mu = _problem()
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    calls = []
    real = torch.linalg.svd

    def counting(*a, **kw):
        calls.append(tuple(a[0].shape))
        return real(*a, **kw)

    monkeypatch.setattr(torch.linalg, "svd", counting)
    truncate_component(full, X, LAMS)
    assert calls == [(W.shape[0], X.shape[0])]                   # one thin SVD of M X^T


def test_spectrum_false_skips_the_svd_and_keeps_the_factors(monkeypatch):
    W, X, G_Y, S, mu = _problem()
    with_ = erased_component_gram(W, X, mu=mu, S=S)
    calls = []
    real = torch.linalg.svd
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    without = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    assert calls == []
    assert torch.equal(with_.A, without.A) and torch.equal(with_.B, without.B)
    assert without.info["sigma"] is None and without.info["truncated"] is False
    assert without.info["rank"] == with_.info["rank"] == X.shape[0]
    assert without.info["k"] == with_.info["k"] and without.info["fit"] == with_.info["fit"] == 0.0
    assert without.info["formula"] == with_.info["formula"]
    # a truncation request ignores the flag
    tr = erased_component_gram(W, X, mu=mu, S=S, lam_rel=0.1, spectrum=False)
    assert tr.info["truncated"] and tr.info["sigma"] is not None and calls == [1]


def test_empty_target():
    W = torch.randn(5, 4, dtype=F64)
    X = torch.zeros(0, 4, dtype=F64)
    G_Y = torch.eye(4, dtype=F64)
    full = erased_component_gram(W, X, G_Y, mu=1.0, spectrum=False)
    comps = truncate_component(full, X, [0.01, 0.1])
    for c in comps:
        assert c.info["rank"] == 0 and c.info["k"] == 0 and c.info["fit"] == 0.0
        assert c.A.shape == (5, 0) and c.B.shape == (0, 4)
        assert torch.equal(apply_erasure(W, c), W)


def test_refusals():
    W, X, G_Y, S, mu = _problem()
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    with pytest.raises(ErasureConventionError):
        truncate_component((full.A, full.B), X, [0.1])
    with pytest.raises(ValueError, match="FULL-RANK"):
        truncate_component(erased_component_gram(W, X, mu=mu, S=S, lam_rel=0.1), X, [0.1])
    with pytest.raises(ValueError, match="lam_rel"):
        truncate_component(full, X, [1.0])
    with pytest.raises(ValueError, match="svd_dtype"):
        truncate_component(full, X, [0.1], svd_dtype=torch.bfloat16)


# --------------------------------------------------------------------------- #
# The effective-rank rule: r = max(1, round(erank(M X^T))) from the SAME SVD  #
# --------------------------------------------------------------------------- #
def _erank_direct(R):
    """Roy-Vetterli on the singular values, written out independently (fp64):
    exp(-sum p ln p), p = sigma / sum sigma."""
    s = torch.linalg.svdvals(R.to(F64))
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


@pytest.mark.parametrize("seed,shape", [(0, (14, 12, 9)), (1, (6, 12, 9)), (2, (14, 5, 9)),
                                        (3, (14, 12, 3)), (4, (40, 30, 25))])
def test_erank_rule_matches_a_direct_computation(seed, shape):
    d_out, d_in, n_x = shape
    W, X, G_Y, S, mu = _problem(seed=seed, d_out=d_out, d_in=d_in, n_x=n_x)
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    (c,) = truncate_component(full, X, ["erank"], svd_dtype=torch.float64)
    R = full.dense() @ X.T                                       # M X^T
    er = _erank_direct(R)
    k = min(d_out, n_x)
    assert c.info["erank"] == pytest.approx(er, rel=1e-10)
    assert c.info["rule"] == "erank" and c.info["lam_rel"] is None and c.info["k"] == k
    assert c.info["rank"] == min(k, max(1, round(er))) and 1 <= c.info["rank"] <= k
    # the truncation itself: U_r U_r^T M at that rank, as erased_component_gram(rank=r)
    ref = erased_component_gram(W, X, mu=mu, S=S, rank=c.info["rank"])
    assert _rel(c.dense(), ref.dense()) < 1e-9


def test_erank_recorded_on_every_rule_and_lam_rel_unchanged():
    W, X, G_Y, S, mu = _problem()
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    plain = truncate_component(full, X, LAMS, svd_dtype=torch.float64)
    mixed = truncate_component(full, X, ["erank"] + LAMS, svd_dtype=torch.float64)
    er = mixed[0].info["erank"]
    for a, b in zip(plain, mixed[1:]):                           # the lam_rel rows are bit-identical
        assert torch.equal(a.A, b.A) and torch.equal(a.B, b.B)
        assert a.info["rank"] == b.info["rank"] and a.info["fit"] == b.info["fit"]
        assert b.info["rule"] == "lam_rel" and b.info["erank"] == er
    assert all(c.info["erank"] == er for c in plain)


def test_erank_rule_shares_the_one_svd(monkeypatch):
    W, X, G_Y, S, mu = _problem()
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    calls, real = [], torch.linalg.svd
    monkeypatch.setattr(torch.linalg, "svd", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    monkeypatch.setattr(torch.linalg, "svdvals", lambda *a, **kw: calls.append(2) or 1 / 0)
    truncate_component(full, X, ["erank", 0.01])
    assert calls == [1]                                          # one thin SVD, no svdvals


def test_erank_rule_refuses_other_names():
    W, X, G_Y, S, mu = _problem()
    full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
    with pytest.raises(ValueError, match="erank"):
        truncate_component(full, X, ["entropy"])


def test_erank_rank_floor_and_zero_spectrum():
    from srspd.solve import erank_truncation_rank
    assert erank_truncation_rank(torch.tensor([1.0, 0.0, 0.0], dtype=F64)) == 1
    assert erank_truncation_rank(torch.tensor([1e-30, 1e-31], dtype=F64)) == 1
    assert erank_truncation_rank(torch.ones(5, dtype=F64)) == 5   # flat -> erank = k
    assert erank_truncation_rank(torch.zeros(4, dtype=F64)) == 0
    assert erank_truncation_rank(torch.zeros(0, dtype=F64)) == 0


def test_shared_formula_with_effective_rank():
    from srspd.erank import effective_rank, effective_rank_from_singular_values
    g = torch.Generator().manual_seed(7)
    for shape in [(9, 4), (20, 31), (3, 3)]:
        A = torch.randn(*shape, generator=g)
        s32 = torch.linalg.svdvals(A)
        assert effective_rank(A) == effective_rank_from_singular_values(s32)
        assert effective_rank(s32) == effective_rank_from_singular_values(s32)
        # the fp64 path is the same formula at higher precision
        s64 = torch.linalg.svdvals(A.to(F64))
        assert effective_rank_from_singular_values(s64) == pytest.approx(effective_rank(A), rel=1e-5)
        assert effective_rank_from_singular_values(s64) == pytest.approx(_erank_direct(A), rel=1e-12)
    with pytest.raises(ValueError):
        effective_rank_from_singular_values(torch.zeros(3))
