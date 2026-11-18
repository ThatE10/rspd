"""``rspd.core.coala.fit_coala`` -- the COALA projector (Parkina & Rakhuba,
arXiv:2507.07580, Proposition 1) -- against the same projector computed by an
independent route, written inline here: for a thin QR ``X^T = Q R``,
``W X = (W R^T) Q^T``, so the left singular vectors and singular values of the
``d_out x min(n, d_in)`` matrix ``W R^T`` are those of the response ``W X``.

* the rank-r products ``A_fac[:, :r] @ B_fac[:r]`` agree, for several r;
* the singular spectra agree;
* column signs of ``U`` may differ, so the factors themselves are not compared,
  only sign-invariant quantities (products, sigma).

Rank-deficient responses: beyond ``rank(WX)`` the extra columns of ``U`` span the
null space of the response and are an arbitrary choice inside it -- a different
but equally valid choice for each route -- so intermediate ranks ``rank(WX) < r
< k`` are NOT comparable and are skipped. Every r <= rank(WX) is compared, and so
is full rank r = k = d_out (where ``U U^T = I`` regardless of the basis choice).

CPU-only; small random matrices; the measured max abs differences are printed
(``pytest -s``) and asserted below float tolerance. Also: the compression entry
points materialise exactly the COALA prefix.
"""
import pytest

torch = pytest.importorskip("torch")

from rspd.core import coala  # noqa: E402


def fit_factors(W, A, T):
    """The QR route to the same projector: ``(U, U^T W, sigma)`` from the SVD of
    ``W R^T`` with ``A = Q R`` (only ``R`` is formed). ``T`` is not read."""
    R = torch.linalg.qr(A, mode="r").R
    U, sigma, _Vh = torch.linalg.svd(W @ R.T, full_matrices=False)
    return U, U.T @ W, sigma

# (name, d_out, d_in, n_cols, activation rank or None for full)
CASES = [
    ("wide-full-rank", 6, 8, 40, None),
    ("square", 8, 8, 30, None),
    ("tall-d_out>d_in", 10, 6, 30, None),
    ("fewer-rows-than-d_in", 6, 12, 5, None),     # n_cols < d_in
    ("exactly-low-rank-X", 6, 10, 40, 3),         # rank(X) = 3 < d_in
]
# float64: tight; float32: LAPACK single precision on well-conditioned toys
TOL = {torch.float64: 1e-10, torch.float32: 2e-4}

_MEASURED = []   # (case, dtype, max |Δ product|, max |Δ sigma|) for the -s report


def _problem(seed, d_out, d_in, n_cols, act_rank, dtype):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_out, d_in, generator=g, dtype=torch.float64)
    A = torch.randn(n_cols, d_in, generator=g, dtype=torch.float64)
    if act_rank is not None:
        A = A[:, :act_rank] @ torch.randn(act_rank, d_in, generator=g, dtype=torch.float64)
    T = A @ W.T
    return W.to(dtype), A.to(dtype), T.to(dtype)


def _ranks_to_compare(k, num_rank):
    """r = 1..num_rank plus full rank k (deduplicated, sorted)."""
    return sorted({*range(1, num_rank + 1), k})


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_fit_coala_matches_fit_factors(case, dtype):
    name, d_out, d_in, n_cols, act_rank = case
    W, A, T = _problem(7, d_out, d_in, n_cols, act_rank, dtype)
    Ac, Bc, sc = coala.fit_coala(W, A, T)
    Ar, Br, sr = fit_factors(W, A, T)

    # Widths: fit_coala keeps min(d_out, n_cols) singular triples, the solver
    # min(d_out, d_in) (it SVDs the d_out x d_in matrix W R^T). They differ only
    # when d_out > d_in and n_cols > d_in ("tall"): COALA then carries
    # d_out - d_in extra columns whose singular values are exactly zero
    # (rank(WX) <= d_in), an arbitrary basis of the response's null space.
    assert Ac.shape[1] == sc.numel() == min(d_out, n_cols)
    assert Ar.shape[1] == sr.numel() == min(d_out, d_in, n_cols)
    k = sr.numel()                          # comparable width
    assert Bc.shape[1] == Br.shape[1] == d_in and Br.shape[0] == k
    if sc.numel() > k:                      # the dead COALA columns
        assert float(sc[k:].abs().max()) <= 1e-5 * float(sc.max())
    sc = sc[:k]

    # numerical rank of the response, from the spectrum (both solvers' spectra agree)
    tol_rank = float(sc.max()) * max(d_out, n_cols) * torch.finfo(dtype).eps * 10
    num_rank = int((sc > tol_rank).sum())

    tol = TOL[dtype]
    scale = float(W.abs().max())
    d_sigma = float((sc - sr).abs().max())
    d_prod = 0.0
    for r in _ranks_to_compare(k, num_rank):
        Wc = Ac[:, :r] @ Bc[:r]
        Wr = Ar[:, :r] @ Br[:r]
        d_prod = max(d_prod, float((Wc - Wr).abs().max()))
    _MEASURED.append((name, str(dtype).split(".")[-1], d_prod, d_sigma))
    print(f"[coala-vs-qr-route] {name:22s} {str(dtype):14s} "
          f"max|d products|={d_prod:.3e}  max|d sigma|={d_sigma:.3e}  "
          f"(numerical rank {num_rank}/{k})")

    assert d_sigma <= tol * float(sc.max()), f"sigma differs by {d_sigma:.3e}"
    assert d_prod <= tol * scale * max(1.0, float(sc.max()) / scale), \
        f"rank-r products differ by {d_prod:.3e}"
    # the prefix property of both: full rank reproduces the rank-k projector of W
    if num_rank == min(d_out, d_in):        # rank(WX) = rank(W): the projector is exact
        assert float((Ac @ Bc - W).abs().max()) <= tol * scale * 10   # COALA full-rank anchor
        assert float((Ar @ Br - W).abs().max()) <= tol * scale * 10   # QR route


def test_fit_coala_is_the_projector_and_the_qr_route_is_the_same_map():
    """``fit_coala`` is the projector: same SVD, ``(U, U.T @ W, S)``. The QR
    route returns the same map (float32 here, so to the float32 tolerance)."""
    W, A, T = _problem(3, 6, 8, 20, None, torch.float32)
    U, S, _ = torch.linalg.svd(T.T, full_matrices=False)
    Ac, Bc, sc = coala.fit_coala(W, A, T)
    assert torch.equal(Ac, U) and torch.equal(sc, S) and torch.equal(Bc, U.T @ W)
    A0, B0, s0 = fit_factors(W, A, T)
    assert torch.allclose(sc[:s0.numel()], s0, rtol=1e-4, atol=1e-5)
    for r in range(1, s0.numel() + 1):
        Pc, P0 = Ac[:, :r] @ Bc[:r], A0[:, :r] @ B0[:r]
        assert float((Pc - P0).norm() / Pc.norm()) < 1e-4, r


def test_coala_prefix_property_and_orthonormal_U():
    W, A, T = _problem(5, 7, 9, 30, None, torch.float64)
    Ac, Bc, sc = coala.fit_coala(W, A, T)
    assert torch.allclose(Ac.T @ Ac, torch.eye(Ac.shape[1], dtype=Ac.dtype), atol=1e-12)
    assert bool((sc[:-1] >= sc[1:]).all())
    # on span(X): U_r U_r^T (WX) == [WX]_r  (best rank-r approximation of the response)
    U, S, Vh = torch.linalg.svd(T.T, full_matrices=False)
    for r in (1, 3, 5):
        surrogate = Ac[:, :r] @ Bc[:r]
        best = (U[:, :r] * S[:r]) @ Vh[:r]
        assert torch.allclose(surrogate @ A.T, best, atol=1e-10)


def test_compression_materialises_the_coala_prefix():
    W, A, T = _problem(11, 7, 9, 30, None, torch.float64)
    Ac, Bc, sc = coala.fit_coala(W, A, T)
    Af, Bf, sf = coala.compress_factors(W, A, T)
    assert torch.equal(Af, Ac) and torch.equal(Bf, Bc) and torch.equal(sf, sc)
    for r in (0, 1, 3, 7, 50):
        want = Ac[:, :min(r, Ac.shape[1])] @ Bc[:min(r, Ac.shape[1])]
        assert torch.equal(coala.materialize_compressed(Ac, Bc, r), want)
        assert torch.equal(coala.compress_weight(W, A, T, r), want)
    assert coala.materialize_compressed(Ac, Bc, 3, dtype=torch.float32).dtype == torch.float32
    # full rank reproduces W (rank(W X) = d_out here)
    assert torch.allclose(coala.compress_weight(W, A, T, 7), W, atol=1e-10)


def test_truncation_uses_the_coala_basis_of_M_over_X():
    """The erasure truncation is ``fit_coala`` applied to ``M`` with the
    target as context: ``U_r U_r^T M`` from the SVD of ``M X^T``."""
    from rspd.core import erased_component, truncate_component
    g = torch.Generator().manual_seed(2)
    W = torch.randn(10, 8, generator=g, dtype=torch.float64)
    X = torch.randn(6, 8, generator=g, dtype=torch.float64)
    Y = torch.randn(20, 8, generator=g, dtype=torch.float64)
    full = erased_component(W, X, Y, mu=1.5)
    M = full.dense()
    Ac, Bc, sc = coala.fit_coala(M, X, X @ M.T)
    for r in (1, 2, 4):
        cr = erased_component(W, X, Y, mu=1.5, rank=r)
        assert torch.allclose(cr.dense(), Ac[:, :r] @ Bc[:r], atol=1e-10)
    (c,) = truncate_component(full, X, [0.0], svd_dtype=torch.float64)
    assert torch.allclose(c.info["sigma"], sc[:c.info["k"]], rtol=1e-10, atol=1e-12)
