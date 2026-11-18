"""``rspd.solve`` -- erase by subtraction, the core table as tests.

CPU, float64, no model. X = TARGET (erased), Y = RETAINED (kept), batch-first.

* the off-span preservation test (the test that would have caught the
  kept-map confusion): X and Y span a strict subspace; for a probe
  ``z ⊥ span(X ∪ Y)``, ``(W - M_r) z = W z`` at EVERY rank and mu, while the
  kept map gives 0 on z -- the core table's last column, both rows;
* the table's first row on the batch: ``(W - M) X = L_X`` and
  ``(W - M) Y = W Y - L_Y``, disentangled and overlapping variants;
* leaks: reported by every entry point, zero on an exactly separable toy,
  nonzero on an overlapping one; the kept map loses retained response under
  truncation where ``W - M_r`` does not;
* ``M + M_keep = W P`` and the closed forms ``M = W X^T X G_mu^+``;
* full rank == the lam = 0 edit ``W - (W X^T) S`` exactly, and the truncation
  formula ``M_r = U_r U_r^T M`` with ``U_r`` from the SVD of ``M X^T``;
* the Gram form equals the stacked form; ``leaks_gram`` equals ``leaks``;
* ``apply_erasure`` refuses a ``KeptMap`` and a bare tensor, printing the table.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rspd import solve as erase  # noqa: E402
from rspd.solve import (CORE_TABLE, ErasedComponent, ErasureConventionError,  # noqa: E402
                        KeptMap, apply_erasure, erased_component, erased_component_gram,
                        kept_map, kept_map_gram, leaks, leaks_gram, readout_gram)

F64 = torch.float64


def _toy(seed=0, d_out=14, d_in=12, n_x=9, n_y=30, k_x=2, k_y=4, overlap=0.0, noise=0.0):
    """``X`` from a ``k_x``-dim generator, ``Y`` from a ``k_y``-dim one; the two
    generators are orthogonal (``overlap = 0``, exactly separable) or tilted
    into each other by ``overlap``; together they span ``k_x + k_y < d_in``
    directions, so a probe ``z`` orthogonal to both exists. ``noise`` adds a
    full-rank jitter inside the SAME ``k_x + k_y`` subspace (keeps the strict
    span) so the Gram is well conditioned on it."""
    g = torch.Generator().manual_seed(seed)
    rn = lambda *s: torch.randn(*s, generator=g, dtype=F64)
    Q, _ = torch.linalg.qr(rn(d_in, d_in))
    Sx, Sy = Q[:, :k_x], Q[:, k_x:k_x + k_y]
    Sx_t = (1 - overlap ** 2) ** 0.5 * Sx + overlap * Sy[:, :k_x]
    span = Q[:, :k_x + k_y]
    W = rn(d_out, d_in)
    X = rn(n_x, k_x) @ Sx_t.T + noise * (rn(n_x, k_x + k_y) @ span.T)
    Y = rn(n_y, k_y) @ Sy.T + noise * (rn(n_y, k_x + k_y) @ span.T)
    z = Q[:, k_x + k_y:]                                 # every column ⊥ span(X ∪ Y)
    return W, X, Y, z


def _closed_forms(W, X, Y, mu):
    G = X.T @ X + mu * (Y.T @ Y)
    Gp = torch.linalg.pinv(G, hermitian=True)
    M = W @ (X.T @ X) @ Gp
    Mk = W @ (mu * (Y.T @ Y)) @ Gp
    P = G @ Gp
    return M, Mk, P


MUS = [0.1, 1.0, 10.0]
VARIANTS = {"disentangled": dict(overlap=0.0), "overlapping": dict(overlap=0.3, noise=0.05)}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("mu", MUS)
def test_off_span_preservation_at_every_rank_and_mu(variant, mu):
    """Last column of the core table: W - M_r acts as W off the batch span at
    every rank; the kept map is zero there."""
    W, X, Y, z = _toy(**VARIANTS[variant])
    Wz = W @ z
    for rank in [None] + list(range(0, X.shape[0] + 1)):
        comp = erased_component(W, X, Y, mu=mu, rank=rank)
        Wn = apply_erasure(W, comp)
        assert float(((Wn @ z) - Wz).norm()) / float(Wz.norm()) < 1e-6, (rank, mu)
        assert comp.info["rank"] == (X.shape[0] if rank is None else min(rank, comp.info["k"]))
    for rank in [None, 1, 3]:
        km = kept_map(W, X, Y, mu=mu, rank=rank)
        assert float((km.dense() @ z).norm()) < 1e-8 * float(W.norm()), (rank, mu)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("mu", MUS)
def test_first_row_of_the_table_on_the_batch(variant, mu):
    """``(W - M) X = L_X`` and ``(W - M) Y = W Y - L_Y`` with ``L_X = M_keep X``,
    ``L_Y = M Y``; ``M + M_keep = W P``; both maps vanish on ``I - P``."""
    W, X, Y, z = _toy(**VARIANTS[variant])
    M, Mk, P = _closed_forms(W, X, Y, mu)
    comp = erased_component(W, X, Y, mu=mu)
    assert torch.allclose(comp.dense(), M, atol=1e-8)
    Wn = apply_erasure(W, comp)
    L_X, L_Y = Mk @ X.T, M @ Y.T
    assert torch.allclose(Wn @ X.T, L_X, atol=1e-8)
    assert torch.allclose(Wn @ Y.T, W @ Y.T - L_Y, atol=1e-8)
    assert torch.allclose(M + Mk, W @ P, atol=1e-8)
    I = torch.eye(W.shape[1], dtype=F64)
    assert float((M @ (I - P)).norm()) < 1e-8 and float((Mk @ (I - P)).norm()) < 1e-8
    assert torch.allclose(Wn - Mk, W @ (I - P), atol=1e-8)      # (W - M) - M_keep = W (I - P)
    assert float((W @ (I - P)).norm()) > 1e-3 * float(W.norm())  # ... and it is NOT zero
    assert torch.allclose(kept_map(W, X, Y, mu=mu).dense(), Mk, atol=1e-8)


def test_leaks_zero_when_disentangled_nonzero_when_overlapping():
    W, X, Y, _ = _toy(**VARIANTS["disentangled"])
    comp = erased_component(W, X, Y, mu=1.0)
    lk = leaks(W, comp, X, Y)
    assert set(lk) >= {"L_X_rel", "L_Y_rel", "retained_rel", "erased_rel", "delta_fro", "w_fro"}
    assert lk["L_X_rel"] < 1e-9 and lk["L_Y_rel"] < 1e-9
    W, X, Y, _ = _toy(**VARIANTS["overlapping"])
    comp = erased_component(W, X, Y, mu=1.0)
    lk = leaks(W, comp, X, Y)
    assert lk["L_X_rel"] > 1e-3 and lk["L_Y_rel"] > 1e-3
    assert lk["retained_rel"] == pytest.approx(lk["L_Y_rel"] ** 2)
    assert lk["erased_rel"] == pytest.approx(lk["L_X_rel"] ** 2)
    # mu trades them: larger mu shrinks L_Y and grows L_X
    lk_hi = leaks(W, erased_component(W, X, Y, mu=100.0), X, Y)
    assert lk_hi["L_Y_rel"] < lk["L_Y_rel"] and lk_hi["L_X_rel"] > lk["L_X_rel"]


def test_truncated_kept_map_loses_retained_response_where_subtraction_does_not():
    W, X, Y, _ = _toy(**VARIANTS["disentangled"], d_out=16, d_in=14, k_y=6, n_y=40)
    WY = W @ Y.T
    for r in (1, 2):
        comp_r = erased_component(W, X, Y, mu=1.0, rank=r)
        ret_sub = float(((apply_erasure(W, comp_r) @ Y.T) - WY).norm()) / float(WY.norm())
        km_r = kept_map(W, X, Y, mu=1.0, rank=r)
        ret_keep = float(((km_r.dense() @ Y.T) - WY).norm()) / float(WY.norm())
        assert ret_sub < 1e-9, r          # W - M_r keeps W Y up to the (zero) leak
        assert ret_keep > 0.3, r          # the rank-r kept map reproduces W on <= r directions


def test_full_rank_equals_the_lam0_edit_exactly_and_the_truncation_formula():
    for variant in VARIANTS:
        W, X, Y, _ = _toy(**{**VARIANTS[variant], 'noise': 0.1})
        mu = 2.5
        G_Y = Y.T @ Y
        S, info = readout_gram(G_Y, X, mu)
        comp = erased_component_gram(W, X, G_Y, mu=mu)
        assert not comp.info["truncated"]
        assert torch.equal(apply_erasure(W, comp), W - (W @ X.T) @ S)      # the same formula
        assert torch.allclose(S, X @ torch.linalg.pinv(X.T @ X + mu * G_Y), atol=1e-8)
        M = comp.dense()
        U, s, _ = torch.linalg.svd(M @ X.T, full_matrices=False)
        assert torch.allclose(comp.info["sigma"], s, atol=1e-9)
        for r in (1, 2, 4):
            cr = erased_component_gram(W, X, G_Y, mu=mu, rank=r)
            assert cr.A.shape == (W.shape[0], r) and cr.B.shape == (r, W.shape[1])
            assert torch.allclose(cr.dense(), U[:, :r] @ (U[:, :r].T @ M), atol=1e-8)
            assert torch.allclose(cr.dense() @ X.T, (U[:, :r] * s[:r]) @ _vh(M @ X.T)[:r], atol=1e-8)
            assert cr.info["fit"] == pytest.approx(float((s[r:] ** 2).sum()), abs=1e-8)
        # at r = k (the full spectrum) the truncation IS M: range(M) = range(M X^T)
        ck = erased_component_gram(W, X, G_Y, mu=mu, rank=X.shape[0])
        assert torch.allclose(ck.dense(), M, atol=1e-8)
        # the rank penalty is the hard threshold on that spectrum
        lam = float(s[2] ** 2) * 1.0001
        assert erased_component_gram(W, X, G_Y, mu=mu, lam=lam).info["rank"] == 2
        assert erased_component_gram(W, X, G_Y, mu=mu, lam_rel=float((s[2] / s[0]) ** 2) * 1.0001).info["rank"] == 2


def _vh(A):
    return torch.linalg.svd(A, full_matrices=False)[2]


def test_gram_form_equals_stacked_form_and_leaks_gram_equals_leaks():
    W, X, Y, _ = _toy(**VARIANTS["overlapping"])
    for rank in (None, 3):
        a = erased_component(W, X, Y, mu=3.0, rank=rank)
        b = erased_component_gram(W, X, Y.T @ Y, mu=3.0, rank=rank)
        assert torch.allclose(a.dense(), b.dense(), atol=1e-9)
        la, lb = leaks(W, a, X, Y), leaks_gram(W, b, X, Y.T @ Y)
        for key in ("L_X_rel", "L_Y_rel", "retained_rel", "erased_rel", "delta_fro", "w_fro"):
            assert la[key] == pytest.approx(lb[key], rel=1e-8, abs=1e-12), key
    # a precomputed S skips the Cholesky and gives the same component
    S, _ = readout_gram(Y.T @ Y, X, 3.0)
    c = erased_component_gram(W, X, None, mu=3.0, S=S, rank=3)
    assert torch.allclose(c.dense(), erased_component(W, X, Y, mu=3.0, rank=3).dense(), atol=1e-9)
    assert c.info["solver"] == "given"


def test_ridge_keeps_W_minus_M_equal_to_the_kept_map():
    W, X, Y, z = _toy(**VARIANTS["overlapping"])
    comp = erased_component(W, X, Y, mu=1.0, ridge_rel=1e-6)
    km = kept_map(W, X, Y, mu=1.0, ridge_rel=1e-6)
    assert torch.allclose(apply_erasure(W, comp), km.dense(), atol=1e-9)
    assert comp.info["ridge_delta"] > 0 and comp.info["solver"] == "cholesky_ridge"
    assert torch.allclose(apply_erasure(W, comp) @ z, W @ z, atol=1e-9)


def test_singular_gram_takes_the_eigh_path_and_still_preserves_off_span():
    W, X, Y, z = _toy(**VARIANTS["disentangled"])          # exactly low-rank batch
    comp = erased_component(W, X, Y, mu=1.0)
    assert comp.info["solver"] == "eigh"
    assert torch.allclose(apply_erasure(W, comp) @ z, W @ z, atol=1e-9)


def test_empty_target_erases_nothing():
    W, X, Y, _ = _toy()
    comp = erased_component(W, X[:0], Y, mu=1.0)
    assert comp.info["solver"] == "none" and comp.info["rank"] == 0
    assert torch.equal(apply_erasure(W, comp), W)


def test_apply_erasure_refuses_the_kept_map_and_bare_tensors():
    W, X, Y, _ = _toy()
    with pytest.raises(ErasureConventionError) as ei:
        apply_erasure(W, kept_map(W, X, Y, mu=1.0))
    msg = str(ei.value)
    assert "W_new = W − M" in msg and "W_new = M_keep" in msg and "W (I − P)" in msg
    assert CORE_TABLE in msg
    with pytest.raises(ErasureConventionError):
        apply_erasure(W, W)                                  # a fitted/bare map
    with pytest.raises(ErasureConventionError):
        apply_erasure(W, (W[:, :2], W[:2]))                  # an anonymous factor pair
    with pytest.raises(ErasureConventionError):
        leaks(W, KeptMap(W, torch.eye(W.shape[1], dtype=F64), {}), X, Y)
    with pytest.raises(ValueError):
        apply_erasure(W, ErasedComponent(W[:, :2], W[:3], {}))
    assert isinstance(ErasureConventionError(""), TypeError)


def test_apply_erasure_dtype_and_copy_semantics():
    W, X, Y, _ = _toy()
    comp = erased_component(W, X, Y, mu=1.0)
    W32 = W.to(torch.float32)
    out = apply_erasure(W32, comp)
    assert out.dtype == torch.float32 and out.data_ptr() != W32.data_ptr()
    assert torch.allclose(out.double(), W - comp.dense(), atol=1e-5)
    assert torch.equal(W32, W.to(torch.float32))              # W untouched


def test_specificity_matches_generalized_eigensolve():
    scipy_linalg = pytest.importorskip("scipy.linalg")
    import numpy as np
    W, X, Y, _ = _toy(**VARIANTS['overlapping'], n_x=5)
    g = torch.Generator().manual_seed(7)
    Y = Y + 0.05 * torch.randn(*Y.shape, generator=g, dtype=F64)   # full-rank retained Gram
    n_x, n_y = X.shape[0], Y.shape[0]
    G_Y = Y.T @ Y
    sp = erase.specificity_gram(G_Y, X, n_y, k=4)
    ref = scipy_linalg.eigh((X.T @ X / n_x).numpy(), (G_Y / n_y).numpy(), eigvals_only=True)
    ref = np.sort(ref)[::-1][:min(n_x, X.shape[1])]
    assert sp["solver"] == "cholesky" and sp["n"] == len(ref)
    assert np.allclose(sp["top"], ref[:4], rtol=1e-8, atol=1e-10)


# --------------------------------------------------------------------------- #
# the two-Gram form: a target too large to hold                               #
# --------------------------------------------------------------------------- #
from rspd.solve import erased_component_grams, factor_grams, leaks_grams  # noqa: E402

GRAM_MUS = [0.01, 1.0, 100.0]
GRAM_RIDGES = [0.0, 1e-10, 1e-6]


def _gram_toy(seed=0, d_out=9, d_in=12, n_x=30, n_y=7, full_rank_y=False):
    """``n_x > d_in`` (the target spans the input space, so ``M`` is full rank and
    ``G_mu`` is PD) with a RANK-DEFICIENT retained Gram (``n_y < d_in``), or a
    full-rank ``Y``."""
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_out, d_in, generator=g, dtype=F64)
    X = torch.randn(n_x, d_in, generator=g, dtype=F64)
    Y = torch.randn(d_in + 5 if full_rank_y else n_y, d_in, generator=g, dtype=F64)
    return W, X, Y


@pytest.mark.parametrize("mu", GRAM_MUS)
@pytest.mark.parametrize("ridge_rel", GRAM_RIDGES)
@pytest.mark.parametrize("full_rank_y", [False, True])
def test_two_gram_form_equals_the_readout_form_and_the_stacked_form(mu, ridge_rel, full_rank_y):
    W, X, Y = _gram_toy(full_rank_y=full_rank_y)
    G_X, G_Y = X.T @ X, Y.T @ Y
    a = erased_component_gram(W, X, G_Y, mu=mu, ridge_rel=ridge_rel)
    b = erased_component_grams(W, G_X, G_Y, mu=mu, ridge_rel=ridge_rel)
    c = erased_component(W, X, Y, mu=mu, ridge_rel=ridge_rel)
    Ma, Mb, Mc = a.dense(), b.dense(), c.dense()
    assert float((Ma - Mb).norm()) / float(Ma.norm()) < 1e-9
    assert float((Mc - Mb).norm()) / float(Mc.norm()) < 1e-9
    # the same ridge convention to the letter: delta = ridge_rel * max diag(mu G_Y)
    assert b.info["ridge_delta"] == pytest.approx(a.info["ridge_delta"], rel=1e-12, abs=0.0)
    assert b.info["ridge_delta"] == pytest.approx(ridge_rel * mu * float(G_Y.diagonal().max()))
    assert b.info["solver"] == a.info["solver"]
    # the dense full-rank representation the type expects: A = M, B = I
    assert b.info["B_is_identity"] and b.A.shape == W.shape
    assert torch.equal(b.B, torch.eye(W.shape[1], dtype=F64))
    assert b.info["rank"] == W.shape[1] and not b.info["truncated"]
    # (b) apply_erasure accepts it and subtracts exactly M
    Wn = apply_erasure(W, b)
    assert torch.allclose(Wn, W - Mb, atol=1e-12)
    assert torch.allclose(Wn, apply_erasure(W, a), atol=1e-9)
    # (c) the leaks from the two Grams equal leaks_gram / leaks
    lb = leaks_grams(W, b, G_X, G_Y)
    la, lc = leaks_gram(W, a, X, G_Y), leaks(W, c, X, Y)
    for key in ("L_X_rel", "L_Y_rel", "retained_rel", "erased_rel", "delta_fro", "w_fro",
                "wx_energy", "lx_energy", "wy_energy", "ly_energy"):
        assert lb[key] == pytest.approx(la[key], rel=1e-9, abs=1e-12), key
        assert lb[key] == pytest.approx(lc[key], rel=1e-9, abs=1e-12), key
    assert lb["retained_rel"] == pytest.approx(lb["L_Y_rel"] ** 2)
    assert lb["erased_rel"] == pytest.approx(lb["L_X_rel"] ** 2)
    # mx_energy = ||M X^T||^2 = tr(M G_X M^T)
    assert lb["mx_energy"] == pytest.approx(float((Mb @ X.T).norm()) ** 2, rel=1e-9)
    # the squared-form identities the keys stand for
    assert lb["L_X_rel"] ** 2 == pytest.approx(
        float(((W - Mb) @ G_X * (W - Mb)).sum()) / float((W @ G_X * W).sum()), rel=1e-9)
    assert lb["L_Y_rel"] ** 2 == pytest.approx(
        float((Mb @ G_Y * Mb).sum()) / float((W @ G_Y * W).sum()), rel=1e-9)


@pytest.mark.parametrize("mu", GRAM_MUS)
def test_two_gram_form_on_a_singular_batch_gram(mu):
    """``n_x < d_in`` AND a rank-deficient ``Y``: ``G_mu`` is singular. At ridge 0
    both forms take the eigh pseudo-inverse and agree to 1e-9; with a tiny ridge
    the two forms factor ``G_mu + delta I`` (kappa ~ 1e10) from arithmetic that
    differs at one ulp, so they are compared through the normal equations they
    both solve, ``M (G_mu + delta I) = W G_X``, not against each other."""
    W, X, Y = _gram_toy(n_x=6, n_y=4)            # span 10 < d_in 12
    G_X, G_Y = X.T @ X, Y.T @ Y
    a = erased_component_gram(W, X, G_Y, mu=mu)
    b = erased_component_grams(W, G_X, G_Y, mu=mu)
    assert a.info["solver"] == b.info["solver"] == "eigh"
    assert float((a.dense() - b.dense()).norm()) / float(a.dense().norm()) < 1e-9
    for ridge_rel in (1e-10, 1e-6):
        b = erased_component_grams(W, G_X, G_Y, mu=mu, ridge_rel=ridge_rel)
        assert b.info["solver"] == "cholesky_ridge"
        G = G_X + mu * G_Y + b.info["ridge_delta"] * torch.eye(12, dtype=F64)
        resid = float((b.dense() @ G - W @ G_X).norm()) / float((W @ G_X).norm())
        assert resid < 1e-8, (ridge_rel, resid)


def test_two_gram_form_shares_one_factor_per_input():
    W1, X, Y = _gram_toy()
    W2 = torch.randn(5, 12, generator=torch.Generator().manual_seed(9), dtype=F64)
    G_X, G_Y = X.T @ X, Y.T @ Y
    factor, info = factor_grams(G_X, G_Y, 3.0, ridge_rel=1e-8)
    assert factor.kind == "cholesky" and info["solver"] == "cholesky_ridge"
    for Wk in (W1, W2):
        shared = erased_component_grams(Wk, G_X, G_Y, mu=3.0, factor=factor, ridge_rel=1e-8)
        own = erased_component_grams(Wk, G_X, G_Y, mu=3.0, ridge_rel=1e-8)
        assert torch.allclose(shared.dense(), own.dense(), atol=1e-12)
        assert shared.info["ridge_delta"] == own.info["ridge_delta"]
    with pytest.raises(ValueError, match="built at mu"):
        erased_component_grams(W1, G_X, G_Y, mu=4.0, factor=factor)
    with pytest.raises(ValueError, match="mu must be > 0"):
        erased_component_grams(W1, G_X, G_Y, mu=0.0)
    with pytest.raises(ValueError, match="d_in"):
        erased_component_grams(W1[:, :5], G_X, G_Y, mu=1.0)


def test_two_gram_form_core_table_on_a_disentangled_toy():
    """(d) X ⊥ Y: ``(W - M) X^T = 0``, ``(W - M) Y^T = W Y^T``, and ``(W - M) z =
    W z`` for ``z ⊥ span(X ∪ Y)`` -- the first row of the core table with zero
    leaks, from the two Grams alone."""
    W, X, Y, z = _toy(**VARIANTS["disentangled"], n_x=20)
    G_X, G_Y = X.T @ X, Y.T @ Y
    for mu in GRAM_MUS:
        comp = erased_component_grams(W, G_X, G_Y, mu=mu)
        Wn = apply_erasure(W, comp)
        assert float((Wn @ X.T).norm()) < 1e-8 * float((W @ X.T).norm())
        assert torch.allclose(Wn @ Y.T, W @ Y.T, atol=1e-8)
        assert torch.allclose(Wn @ z, W @ z, atol=1e-8)
        lk = leaks_grams(W, comp, G_X, G_Y)
        assert lk["L_X_rel"] < 1e-8 and lk["L_Y_rel"] < 1e-8
        # the closed form M = W G_X G_mu^+
        Gp = torch.linalg.pinv(G_X + mu * G_Y, hermitian=True)
        assert torch.allclose(comp.dense(), W @ G_X @ Gp, atol=1e-8)


def test_leaks_grams_refuses_non_components():
    W, X, Y = _gram_toy()
    with pytest.raises(ErasureConventionError):
        leaks_grams(W, KeptMap(W, torch.eye(12, dtype=F64), {}), X.T @ X, Y.T @ Y)
    with pytest.raises(ErasureConventionError):
        leaks_grams(W, W, X.T @ X, Y.T @ Y)
