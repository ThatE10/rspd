"""Erase by subtraction: the closed-form solve, the edit, and the leaks.

Notation (fixed; never swap)
----------------------------
``X``      the TARGET set -- activations of the behaviour being erased,
           batch-first ``[n_x, d_in]`` (its rows are the columns of the
           mathematical ``X``);
``Y``      the RETAINED set -- activations of the behaviour being preserved,
           batch-first ``[n_y, d_in]``; at scale only its Gram
           ``G_Y = Y^T Y`` (``[d_in, d_in]``) is ever formed;
``W``      the pristine weight ``[d_out, d_in]``;
``mu``     the weight on the retained term of the objective (> 0). Users give
           the dose as ``mu_per_token`` and convert with :mod:`srspd.dose`;
``G_mu``   ``= X^T X + mu Y^T Y``, the weighted batch Gram;
``P``      ``= G_mu G_mu^+``, the orthogonal projector onto span(X ∪ Y);
``M``      the ERASED COMPONENT
           ``= argmin_M ||W X^T - M X^T||^2 + mu ||M Y^T||^2 = W X^T X G_mu^+``;
``M_keep`` the complementary KEPT MAP ``= W mu Y^T Y G_mu^+``;
``L_X``    ``= M_keep X^T`` and ``L_Y = M Y^T``: the LEAKS, which measure
           imperfect disentanglement and are NOT assumed zero.
Facts: ``M (I - P) = 0`` and ``M_keep (I - P) = 0`` (both live on the batch span
only); ``M + M_keep = W P`` exactly.

The core table -- identical on the batch, different off it::

  | edit               | on the target X | on the retained set Y | off the batch (z ⊥ span(X ∪ Y)) |
  | W_new = W − M      |      L_X        |       W Y − L_Y       |             W z                 |
  | W_new = M_keep     |      L_X        |       W Y − L_Y       |              0                  |

``(W − M) − M_keep = W (I − P)``: the original weight's action on every input
direction the batch does not span. Neither ``X`` nor ``Y`` spans the input
space, so this term is NOT zero. Writing the kept map throws it away; under
rank truncation the kept map ALSO loses retained behaviour inside the span,
whereas ``W − M_r`` keeps ``W Y`` up to the leak ``L_Y``.

The rules
---------
R1  The edit is ALWAYS ``W_new = W - M`` (:func:`apply_erasure`). Never
    ``W_new = M_keep``, never ``W_new = (anything fitted)``.
R2  Rank truncation / the rank penalty ``lam`` act on ``M``, never on the
    kept map (:func:`erased_component`, ``rank`` / ``lam``).
R3  Low-rank factor pairs ``A B``, ranks and truncation curves are components
    of ``M``, never of ``W`` and never of ``M_keep``.
R4  The leaks are measured and reported per edited matrix (:func:`leaks`,
    :func:`leaks_gram`, :func:`leaks_grams`); disentanglement is a measured
    quantity.
R5  ``X`` means target and ``Y`` means retained, everywhere.

Rank truncation (R2) -- the COALA projector on M over X
-------------------------------------------------------
``M_r = U_r U_r^T M`` with ``U_r`` the leading ``r`` left singular vectors of
the erased component's response on the target, ``M X^T`` (Eckart--Young on
``M X^T``: ``M_r X^T = [M X^T]_r``). This is the context-aware projector of
COALA (Parkina & Rakhuba, arXiv:2507.07580, NeurIPS 2025, Proposition 1)
applied to ``M`` with the target as context; see :mod:`srspd.coala`.
``range(M) = range(M X^T)``, so at full rank ``M_r = M`` and ``W - M_r`` equals
the lam = 0 edit ``W - (W X^T) S`` exactly. The hard-threshold rank rule of the
rank penalty is ``r* = #{j : sigma_j^2 > lam}`` on that spectrum
(``lam = lam_rel sigma_1^2`` for the relative rule), or the effective-rank rule
``r = max(1, round(erank(sigma)))``.

The readout
-----------
``S = X G_mu^+`` (``[n_x, d_in]``) is the least-squares readout of the target
that is blind to the retained set; ``M = (W X^T) S``. One ``S`` serves every
matrix that reads the same input.

The two-Gram form (a large target)
----------------------------------
When the target is too large to hold as activations only its Gram
``G_X = X^T X`` is formed, by the same streaming pass that forms ``G_Y``, and
``M = W G_X (G_X + mu G_Y + delta I)^+`` directly
(:func:`erased_component_grams`, one :func:`factor_grams` per input and
``mu``; leaks from the Grams by :func:`leaks_grams`). Same ``M``, same ridge
convention, no ``S``; returned dense (``A = M``, ``B = I``), no truncation.

Regularisation
--------------
``ridge_rel > 0`` solves with ``G_mu + delta I``, ``delta = ridge_rel *
max(diag(mu Y^T Y))``. On ``M`` this is a ridge toward zero, ``+ delta ||M||^2``;
on the kept map it is the W-anchored ridge ``+ delta ||M_keep - W||^2``, and
``W - M = M_keep`` holds exactly with or without it. If Cholesky fails, the
ridge escalates x10 up to ``1e-6`` relative before an ``eigh`` pseudo-inverse
fallback; the ``delta`` actually used is reported as ``ridge_delta``.

Numerics: every solve is float64 internally (the Gram squares the condition
number, ``kappa(G) = kappa([X; Y])^2``); returned factors are float64 and the
caller casts.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from .coala import coala_basis

__all__ = [
    "CORE_TABLE", "TRUNCATION_FORMULA", "ERANK_RULE",
    "ErasedComponent", "KeptMap", "ErasureConventionError", "GramMuFactor",
    "readout_gram", "erased_component_gram", "erased_component", "truncate_component",
    "erank_truncation_rank", "apply_erasure", "leaks", "leaks_gram",
    "factor_grams", "erased_component_grams", "leaks_grams",
    "specificity", "specificity_gram", "kept_map", "kept_map_gram", "gram_eig_extremes",
]

CORE_TABLE = """\
X = TARGET set (erased), Y = RETAINED set (kept), W = the pristine weight,
G_mu = X X^T + mu Y Y^T, P = G_mu G_mu^+ (projector onto span(X ∪ Y)),
M = W X X^T G_mu^+ (the ERASED COMPONENT), M_keep = W mu Y Y^T G_mu^+ (the KEPT MAP),
leaks L_X = M_keep X, L_Y = M Y (measured, never assumed zero); M + M_keep = W P.

  | edit               | on the target X | on the retained set Y | off the batch (z ⊥ span(X ∪ Y)) |
  | W_new = W − M      |      L_X        |       W Y − L_Y       |             W z                 |
  | W_new = M_keep     |      L_X        |       W Y − L_Y       |              0                  |

(W − M) − M_keep = W (I − P): the original weight's action on every input
direction the batch does not span, which is NOT zero. Writing the kept map
throws it away; under rank truncation it also loses retained behaviour inside
the span. The edit is ALWAYS W_new = W − M (srspd.apply_erasure)."""


class ErasedComponent(NamedTuple):
    """The erased component ``M = A @ B`` (or its rank-``r`` truncation
    ``M_r``), ``A [d_out, r]``, ``B [r, d_in]``. The ONLY thing
    :func:`apply_erasure` subtracts from a weight. Components of ``M`` (R3):
    a factor pair, a truncation of ``M``, a sum of components of ``M`` are all
    ``ErasedComponent``s. ``info`` carries the solve diagnostics."""
    A: torch.Tensor
    B: torch.Tensor
    info: dict

    def dense(self) -> torch.Tensor:
        return self.A @ self.B


class KeptMap(NamedTuple):
    """The complementary kept map ``M_keep = A @ B`` (``= W P - M``; at full
    rank ``W - M``, with ``B = I``). DIAGNOSTICS ONLY: it is the second row of
    the core table and :func:`apply_erasure` refuses it."""
    A: torch.Tensor
    B: torch.Tensor
    info: dict

    def dense(self) -> torch.Tensor:
        return self.A @ self.B


class ErasureConventionError(TypeError):
    """Raised when something other than an :class:`ErasedComponent` is about
    to be written into a model or measured as one. The message carries the
    core table."""


def _default_gram_rcond(d_in):
    """Relative eigenvalue floor below which a float64 Gram counts as singular:
    ``max(d_in * eps, 1e-12)``. The ``d_in * eps`` term is ``torch.linalg.pinv``'s
    convention; the 1e-12 floor keeps a small problem's roundoff-level pivot (an
    exactly singular G factored in floating point) from passing as positive
    definite, and it is the ``lambda_min / lambda_max`` level at which float64
    conditioning is at risk anyway."""
    return max(int(d_in) * torch.finfo(torch.float64).eps, 1e-12)


def _eigh_oom_safe(G):
    """``torch.linalg.eigh(G)``, redone on the CPU if the GPU cannot hold its
    workspace. cuSOLVER's ``syevd`` wants ~2 d^2 doubles on top of ``G`` --
    13.8 GiB for a 21504-wide Gram, which a GPU already carrying a large model
    may not have. Only the non-PD fallbacks call this. Results come back on
    ``G``'s device."""
    try:
        return torch.linalg.eigh(G)
    except torch.OutOfMemoryError:           # pragma: no cover - needs a full GPU
        dev = G.device
        Gc = G.cpu()
        del G
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ev, V = torch.linalg.eigh(Gc)
        return ev.to(dev), V.to(dev)


def _ridge_ladder(ridge_rel):
    """The relative ridges tried in order: ``[0]`` without a ridge, else
    ``ridge_rel, 10 ridge_rel, ...`` up to ``1e-6``."""
    ladder = [float(ridge_rel)] if float(ridge_rel) > 0.0 else [0.0]
    if float(ridge_rel) > 0.0:
        r = float(ridge_rel) * 10.0
        while r <= 1e-6 * (1 + 1e-9):
            ladder.append(r)
            r *= 10.0
    return ladder


def _gram_solve(G_Y, X, mu, *, rcond=None, ridge_rel=0.0):
    """The shared core: ``(S, Lf, solver, delta)`` in float64 on ``G_Y``'s device.

    ``S = X (G_mu + delta I)^+`` (``[n_x, d_in]``, the target readout) and
    ``Lf`` a square-root factor of ``G_mu + delta I = Lf Lf^T`` (the lower
    Cholesky factor when numerically positive definite, else
    ``V_+ sqrt(Lambda_+)`` from ``eigh`` restricted to the eigenvalues above
    ``rcond * lambda_max``, and ``S`` then uses the matching pseudo-inverse).
    ``solver`` is ``"cholesky"`` | ``"cholesky_ridge"`` | ``"eigh"`` | ``"none"``
    (an empty target: nothing to erase, ``S`` has no rows).

    The PD test is not just ``cholesky_ex``'s ``info``: a singular PSD matrix
    often *passes* Cholesky in floating point with a roundoff-sized pivot, and
    ``G^{-1}`` then amplifies that roundoff by ~1e16. The squared pivot ratio
    ``min(diag L)^2 / max(diag L)^2`` is checked against ``rcond`` as well --
    a squared pivot is a Schur-complement diagonal, bounded below by
    ``lambda_min(G)``, so an exactly singular G shows up at roundoff level.

    Ridge: ``delta = ridge_rel * max(diag(mu G_Y))``. If Cholesky still fails,
    ``delta`` escalates x10 (to ``1e-6`` relative) before the eigh fallback;
    the ``delta`` actually used is returned.

    The Gram is factored IN PLACE: at d_in 21504 one float64 copy is 3.4 GiB,
    and a single large GPU has room for one transient of that size, not two.
    """
    G_Y = G_Y.to(torch.float64)
    d_in = int(G_Y.shape[0])
    dev = G_Y.device
    X = X.to(device=dev, dtype=torch.float64)
    n_x = int(X.shape[0])
    if rcond is None:
        rcond = _default_gram_rcond(d_in)
    mu = float(mu)
    scale = mu * float(G_Y.diagonal().max().clamp_min(0.0))

    def build_G(delta=0.0):
        G = mu * G_Y                                               # mu Y^T Y
        if n_x:
            G.addmm_(X.T, X)                                       # + X^T X
        if delta > 0.0:
            G.diagonal().add_(delta)                               # + delta I
        return G

    ladder = _ridge_ladder(ridge_rel)
    for rr in ladder:
        delta = rr * scale
        L = build_G(delta)
        info = torch.zeros((), dtype=torch.int32, device=dev)
        try:
            torch.linalg.cholesky_ex(L, out=(L, info))
        except torch.OutOfMemoryError:                            # pragma: no cover
            raise               # the non-aliasing retry below needs strictly MORE memory
        except RuntimeError:                                      # pragma: no cover
            L, info = torch.linalg.cholesky_ex(L)                 # backend refuses aliasing
        piv = L.diagonal()
        ok = (int(info) == 0 and bool(torch.isfinite(piv).all())
              and float(piv.min()) > 0.0
              and float(piv.min()) ** 2 > rcond * float(piv.max()) ** 2)
        del piv        # a VIEW of L: consumed into `ok` above; keeping the name across
        #              # the next ladder rung would pin the failed d x d factor alive
        #              # beside the rebuilt one
        if ok:
            L.tril_()                # cholesky_ex leaves the upper triangle unspecified
            if n_x == 0:
                return torch.zeros(0, d_in, dtype=torch.float64, device=dev), L, "none", delta
            St = torch.cholesky_solve(X.T, L)                     # (G_mu + delta I)^{-1} X^T
            return (St.T.contiguous(), L,
                    ("cholesky" if delta == 0.0 else "cholesky_ridge"), delta)
        del L                                                     # a failed factor; rebuild
    delta = ladder[-1] * scale
    G = build_G(delta)
    evals, V = _eigh_oom_safe(G)
    del G
    keep = evals > rcond * float(evals.max().clamp_min(0.0))
    Vk, ek = V[:, keep], evals[keep]
    Lf = Vk * ek.sqrt()                                           # G ~= Lf Lf^T
    if n_x == 0:
        return torch.zeros(0, d_in, dtype=torch.float64, device=dev), Lf, "none", delta
    St = Vk @ ((Vk.T @ X.T) / ek[:, None])                        # G^+ X^T
    return St.T.contiguous(), Lf, "eigh", delta


def readout_gram(G_Y, X, mu, *, rcond=None, ridge_rel=0.0):
    """The target readout ``S = X G_mu^+`` from the retained Gram. Returns
    ``(S, info)``.

    ``S`` (``[n_x, d_in]``) is everything the full-rank erased component needs:
    ``M = (W X^T) S`` exactly, so at scale a whole weight edit is one
    ``d_out x n_x`` by ``n_x x d_in`` product per matrix, and every matrix that
    reads the same input shares one ``S``.

    Parameters
    ----------
    G_Y : [d_in, d_in] retained Gram ``Y^T Y`` (batch-first ``Y``); any float
        dtype -- the solve is float64 internally (the Gram squares the
        condition number: ``kappa(G) = kappa([X; Y])^2``, decisive in float32).
    X : [n_x, d_in] target activations (batch-first; may have zero rows).
    mu : float > 0, the weight on ``||M Y^T||^2`` in the erased-component
        objective (from :func:`srspd.dose_from_mu_per_token`).
    rcond : relative eigenvalue floor for the PD test and the eigh fallback
        (default ``max(d_in * eps_float64, 1e-12)``).
    ridge_rel : relative ridge, ``delta = ridge_rel * max(diag(mu Y^T Y))``.

    Returns
    -------
    S : [n_x, d_in] in ``G_Y``'s dtype (``0 x d_in`` when ``n_x = 0``).
    info : ``solver`` (``"cholesky"`` | ``"cholesky_ridge"`` | ``"eigh"`` |
        ``"none"``), ``mu``, ``n_x``, ``d_in``, ``ridge_delta``, ``ridge_rel``.
    """
    if not float(mu) > 0.0:
        raise ValueError(f"mu must be > 0, got {mu!r}")
    S, _Lf, solver, delta = _gram_solve(G_Y, X, mu, rcond=rcond, ridge_rel=ridge_rel)
    return S.to(G_Y.dtype), dict(solver=solver, mu=float(mu), n_x=int(X.shape[0]),
                                 d_in=int(G_Y.shape[0]), ridge_delta=float(delta),
                                 ridge_rel=float(ridge_rel))


#: The truncated edit's formula, as every truncated component's ``info`` reports it.
TRUNCATION_FORMULA = "W - U_r U_r^T (W X^T) S,  U_r = leading left singular vectors of M X^T"

_FULL_FORMULA = "W - (W X^T) S,  S = X (X^T X + mu Y^T Y)^+"


def _truncation_rank(sigma, *, rank, lam, lam_rel):
    k = int(sigma.numel())
    if rank is not None:
        if int(rank) < 0:
            raise ValueError(f"rank must be >= 0, got {rank!r}")
        return min(int(rank), k)
    if lam_rel is not None:
        if not 0.0 <= float(lam_rel) < 1.0:
            raise ValueError(f"lam_rel must be in [0, 1), got {lam_rel!r}")
        lam = float(lam_rel) * float(sigma[0] ** 2) if k else 0.0
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam!r}")
    return int((sigma ** 2 > lam).sum())


def erased_component_gram(W, X, G_Y=None, *, mu=None, rank=None, lam=0.0, lam_rel=None,
                          ridge_rel=0.0, rcond=None, S=None, spectrum=True) -> ErasedComponent:
    """The erased component from the retained Gram (the form that streams at
    scale). ``M = (W X^T) S`` with ``S = X G_mu^+``.

    Truncation (R2) is the COALA projector on ``M`` over the target (Parkina &
    Rakhuba, arXiv:2507.07580, NeurIPS 2025, Proposition 1;
    :func:`srspd.coala.coala_basis`): ``M_r = U_r U_r^T M``, ``U_r`` the leading
    left singular vectors of the target response ``M X^T = (W X^T)(S X^T)`` (a
    ``d_out x n_x`` SVD, cheap). ``r`` is ``rank`` when given, else
    ``#{j : sigma_j^2 > lam}`` with ``lam = lam_rel * sigma_1^2`` when
    ``lam_rel`` is given (each matrix's own spectrum). With no truncation
    requested (``rank=None, lam=0, lam_rel=None``) the factors are
    ``A = W X^T``, ``B = S`` -- the full-rank ``M`` without any projection, so
    ``W - A B`` is the lam = 0 edit ``W - (W X^T) S`` to the last bit of its
    formula.

    Parameters
    ----------
    W : [d_out, d_in] pristine weight.
    X : [n_x, d_in] target activations, batch-first.
    G_Y : [d_in, d_in] retained Gram ``Y^T Y``; may be ``None`` when ``S`` is given.
    mu : float > 0, the retained weight (from :func:`srspd.dose_from_mu_per_token`).
    rank, lam, lam_rel : the truncation request (at most one is meaningful;
        ``rank`` wins, then ``lam_rel``, then ``lam``).
    ridge_rel, rcond : see :func:`readout_gram`.
    S : [n_x, d_in] a precomputed :func:`readout_gram` for this
        ``(G_Y, X, mu, ridge_rel)``; skips the Cholesky.
    spectrum : ``False`` skips the SVD of ``M X^T`` when NO truncation is
        requested; then ``info["sigma"]`` is ``None``, ``rank = n_x``,
        ``k = min(d_out, n_x)`` and ``fit = 0.0``. The factors are
        bit-identical to ``spectrum=True``. A truncation request ignores the
        flag. Several truncations of one component share ONE spectrum through
        :func:`truncate_component`.

    Returns
    -------
    ErasedComponent with float64 factors; ``info``: ``sigma`` (singular values
    of ``M X^T``, descending), ``rank``, ``k``, ``fit`` (``sum_{j>r}
    sigma_j^2``, the target response the truncation discards), ``solver``
    (``"given"`` when ``S`` was passed), ``mu``, ``ridge_delta``, ``n_x``,
    ``d_out``, ``d_in``, ``truncated`` (bool), ``formula``.
    """
    if mu is None or not float(mu) > 0.0:
        raise ValueError(f"mu must be > 0, got {mu!r}")
    if S is None:
        if G_Y is None:
            raise ValueError("erased_component_gram needs G_Y (the retained Gram) or S")
        S, _Lf, solver, delta = _gram_solve(G_Y, X, mu, rcond=rcond, ridge_rel=ridge_rel)
    else:
        solver, delta = "given", float("nan")
    dev = S.device
    W64 = W.to(device=dev, dtype=torch.float64)
    X64 = X.to(device=dev, dtype=torch.float64)
    S64 = S.to(device=dev, dtype=torch.float64)
    d_out, d_in = int(W64.shape[0]), int(W64.shape[1])
    n_x = int(X64.shape[0])
    WXt = W64 @ X64.T                                              # d_out x n_x
    truncated = rank is not None or lam_rel is not None or float(lam) > 0.0
    if not truncated and not spectrum:
        info = dict(sigma=None, rank=n_x, k=min(d_out, n_x), fit=0.0, solver=solver,
                    mu=float(mu), ridge_delta=float(delta), n_x=n_x, d_out=d_out, d_in=d_in,
                    truncated=False, formula=_FULL_FORMULA)
        return ErasedComponent(WXt.contiguous(), S64.contiguous(), info)
    resp = WXt @ (S64 @ X64.T) if n_x else WXt                     # M X^T
    if n_x:
        U, sigma = coala_basis(resp)                               # COALA basis of M over X
    else:
        U = torch.zeros(d_out, 0, dtype=torch.float64, device=dev)
        sigma = torch.zeros(0, dtype=torch.float64, device=dev)
    r = _truncation_rank(sigma, rank=rank, lam=lam, lam_rel=lam_rel) if truncated else n_x
    if truncated:
        M = WXt @ S64                                              # d_out x d_in
        A, B = U[:, :r].contiguous(), (U[:, :r].T @ M).contiguous()
        del M
        formula = TRUNCATION_FORMULA
    else:
        A, B = WXt.contiguous(), S64.contiguous()
        formula = _FULL_FORMULA
    s2 = sigma ** 2
    info = dict(sigma=sigma, rank=int(r), k=int(sigma.numel()), fit=float(s2[r:].sum()),
                solver=solver, mu=float(mu), ridge_delta=float(delta), n_x=n_x, d_out=d_out,
                d_in=d_in, truncated=bool(truncated), formula=formula)
    return ErasedComponent(A, B, info)


#: The string spec of :func:`truncate_component`'s effective-rank rule.
ERANK_RULE = "erank"


def erank_truncation_rank(sigma) -> int:
    """The effective-rank rule's kept rank ``r = max(1, round(erank(sigma)))``,
    capped at ``k = len(sigma)`` (``erank <= k`` already; the cap is a guard);
    ``0`` for an empty or all-zero spectrum (nothing to keep). ``erank`` is
    :func:`srspd.erank.effective_rank_from_singular_values`."""
    k = int(sigma.numel())
    if k == 0 or float(sigma.max()) <= 0.0:
        return 0
    from .erank import effective_rank_from_singular_values
    return min(k, max(1, int(round(effective_rank_from_singular_values(sigma)))))


def truncate_component(comp_full, X, specs, *, svd_dtype=torch.float32) -> list:
    """Several rank truncations of ONE full-rank erased component from ONE
    spectrum (R2), by the COALA projector on ``M`` over the target (Parkina &
    Rakhuba, arXiv:2507.07580, NeurIPS 2025, Proposition 1;
    :func:`srspd.coala.coala_basis`).

    Parameters
    ----------
    comp_full : the full-rank readout-form :class:`ErasedComponent`
        (``A = W X^T [d_out, n_x]``, ``B = S [n_x, d_in]``, ``M = A B``) from
        :func:`erased_component_gram` with no truncation requested.
    X : [n_x, d_in] the target activations it was solved on.
    specs : list; each entry is a ``lam_rel`` float in ``[0, 1)`` or the string
        ``"erank"``.
    svd_dtype : ``torch.float32`` (default) or ``torch.float64``.

    Returns
    -------
    list of truncated :class:`ErasedComponent`, one per spec, in the order
    given: ``A_r = U_r [d_out, r]``, ``B_r = U_r^T M = (U_r^T A) B [r, d_in]``,
    with ``r = #{j : sigma_j^2 > lam_rel sigma_1^2}`` or, for ``"erank"``,
    ``r = max(1, round(erank(sigma)))``. Each equals
    ``erased_component_gram(..., lam_rel=l)`` (or ``rank=r``) up to the
    precision of the SVD.

    The target response ``R = M X^T = A (B X^T)`` is formed ONCE (its two
    matmuls in the component's own dtype) and decomposed ONCE by a thin SVD of
    ``R.to(svd_dtype)`` on the component's device; every spec slices the same
    ``U`` and ``sigma``. The SVD rather than an eigendecomposition of the
    smaller Gram (``R R^T`` or ``R^T R``): the Gram squares the condition
    number, so in float32 the directions near the cut lose another ~30x of
    accuracy in ``U_r``. ``U_r`` is cast back to the component's dtype for the
    factor products, so ``B_r`` carries ``M`` at the component's precision
    projected through an ``svd_dtype``-accurate ``U_r``.

    ``svd_dtype`` defaults to float32: the SVD is the only step whose cost is
    not linear in the matrix size (``O(d_out n_x^2)``), float32 runs it several
    times faster than float64 on a GPU, and edits are usually applied in
    float32 anyway. The full-rank readout solve ``S`` stays float64. Pass
    ``svd_dtype=torch.float64`` to reproduce :func:`erased_component_gram` to
    ~1e-12.

    ``info`` per component: :func:`erased_component_gram`'s keys (``sigma`` in
    ``svd_dtype``, ``rank``, ``k``, ``fit = sum_{j>r} sigma_j^2``, ``solver``,
    ``mu``, ``ridge_delta``, ``n_x``, ``d_out``, ``d_in``, ``truncated=True``,
    ``formula``) plus ``rule`` (``"lam_rel"`` | ``"erank"``), ``lam_rel`` (the
    float; ``None`` under ``"erank"``), ``erank`` (of the float64 spectrum,
    recorded whatever the rule; ``0.0`` for an all-zero response) and
    ``svd_dtype`` (``"float32"`` / ``"float64"``).
    """
    if not isinstance(comp_full, ErasedComponent):
        raise ErasureConventionError("truncate_component truncates an ErasedComponent, got "
                                     f"{type(comp_full).__name__}\n\n" + CORE_TABLE)
    if comp_full.info.get("truncated", False):
        raise ValueError("truncate_component needs the FULL-RANK component (A = W X^T, B = S); "
                         f"got one already truncated to rank {comp_full.info.get('rank')}")
    if svd_dtype not in (torch.float32, torch.float64):
        raise ValueError(f"svd_dtype must be torch.float32 or torch.float64, got {svd_dtype!r}")
    rules = []
    for l in specs:
        if isinstance(l, str):
            if l != ERANK_RULE:
                raise ValueError(f"truncation spec must be a lam_rel in [0, 1) or {ERANK_RULE!r}, "
                                 f"got {l!r}")
            rules.append((ERANK_RULE, None))
            continue
        l = float(l)
        if not 0.0 <= l < 1.0:
            raise ValueError(f"lam_rel must be in [0, 1), got {l!r}")
        rules.append(("lam_rel", l))
    A, B = comp_full.A, comp_full.B
    dev, cdt = A.device, A.dtype
    d_out, n_x = int(A.shape[0]), int(A.shape[1])
    d_in = int(B.shape[1])
    if n_x:
        Xc = X.to(device=dev, dtype=cdt)
        R = A @ (B @ Xc.T)                                         # M X^T, d_out x n_x
        del Xc
        U, sigma = coala_basis(R.to(svd_dtype))                    # COALA basis of M over X
        del R
    else:
        U = torch.zeros(d_out, 0, dtype=svd_dtype, device=dev)
        sigma = torch.zeros(0, dtype=svd_dtype, device=dev)
    s64 = sigma.to(torch.float64)
    s2 = s64 ** 2
    if s64.numel() and float(s64.max()) > 0.0:
        from .erank import effective_rank_from_singular_values
        erank = float(effective_rank_from_singular_values(s64))
    else:
        erank = 0.0
    base = {k: comp_full.info.get(k) for k in ("solver", "mu", "ridge_delta")}
    out = []
    for rule, l in rules:
        if rule == ERANK_RULE:
            r = erank_truncation_rank(s64)
        else:
            r = _truncation_rank(sigma, rank=None, lam=0.0, lam_rel=l)
        Ur = U[:, :r].to(cdt)
        A_r = Ur.contiguous()
        B_r = ((Ur.T @ A) @ B).contiguous()                        # U_r^T M, r x d_in
        info = dict(sigma=sigma, rank=int(r), k=int(sigma.numel()), fit=float(s2[r:].sum()),
                    **base, n_x=n_x, d_out=d_out, d_in=d_in, truncated=True,
                    formula=TRUNCATION_FORMULA, rule=rule, lam_rel=l, erank=erank,
                    svd_dtype=str(svd_dtype).replace("torch.", ""))
        out.append(ErasedComponent(A_r, B_r, info))
    return out


def erased_component(W, X, Y, *, mu, rank=None, lam=0.0, lam_rel=None, ridge_rel=0.0,
                     rcond=None) -> ErasedComponent:
    """The erased component from stacked activations:

        M = argmin_M ||W X^T - M X^T||^2 + mu ||M Y^T||^2  =  W X^T X G_mu^+,
        G_mu = X^T X + mu Y^T Y

    (``X`` target ``[n_x, d_in]``, ``Y`` retained ``[n_y, d_in]``, batch-first;
    ``W [d_out, d_in]``). Forms ``G_Y = Y^T Y`` in float64 and calls
    :func:`erased_component_gram`; see it for ``rank`` / ``lam`` / ``lam_rel``
    (truncation of ``M`` by the COALA projector over ``X``, R2), ``ridge_rel``
    and the returned ``info``. Apply with :func:`apply_erasure`; measure the
    leaks with :func:`leaks`.
    """
    Y64 = Y.to(torch.float64)
    G_Y = Y64.T @ Y64
    return erased_component_gram(W, X, G_Y, mu=mu, rank=rank, lam=lam, lam_rel=lam_rel,
                                 ridge_rel=ridge_rel, rcond=rcond)


def kept_map_gram(W, X, G_Y, *, mu, rank=None, ridge_rel=0.0, rcond=None) -> KeptMap:
    """DIAGNOSTICS ONLY -- the complementary kept map

        M_keep = W (mu Y^T Y + delta I) (G_mu + delta I)^+     (= W - M on span, W P - M)

    (the second row of the core table; ``delta`` is the ridge, under which
    ``W - M = M_keep`` holds exactly when ``G_mu`` is nonsingular) and, with
    ``rank``, its Eckart--Young truncation on the whole weighted batch:
    ``U_r U_r^T M_keep`` with ``U_r`` the leading left singular vectors of
    ``M_keep Lf`` (``Lf Lf^T = G_mu + delta I``). :func:`apply_erasure` refuses
    the result. Dense float64 ``eigh``: a diagnostic routine, not the
    streaming solve.
    """
    if not float(mu) > 0.0:
        raise ValueError(f"mu must be > 0, got {mu!r}")
    G_Y = G_Y.to(torch.float64)
    dev = G_Y.device
    d_in = int(G_Y.shape[0])
    W64 = W.to(device=dev, dtype=torch.float64)
    X64 = X.to(device=dev, dtype=torch.float64)
    if rcond is None:
        rcond = _default_gram_rcond(d_in)
    mu = float(mu)
    delta = float(ridge_rel) * mu * float(G_Y.diagonal().max().clamp_min(0.0))
    G = mu * G_Y + X64.T @ X64
    G.diagonal().add_(delta)
    ev, V = torch.linalg.eigh(0.5 * (G + G.T))
    keep = ev > rcond * float(ev.max().clamp_min(0.0))
    Vk, ek = V[:, keep], ev[keep]
    num = mu * G_Y
    num.diagonal().add_(delta)
    Mk = ((W64 @ num) @ Vk / ek) @ Vk.T                           # W num G^+
    Lf = Vk * ek.sqrt()
    U, sigma, _Vh = torch.linalg.svd(Mk @ Lf, full_matrices=False)
    k = int(sigma.numel())
    info = dict(sigma=sigma, solver="eigh", mu=mu, ridge_delta=delta, k=k,
                note="kept map: the second row of the core table; never written into a model")
    if rank is None:
        info["rank"] = k
        return KeptMap(Mk, torch.eye(d_in, dtype=torch.float64, device=dev), info)
    r = min(int(rank), k)
    info["rank"] = r
    return KeptMap(U[:, :r].contiguous(), (U[:, :r].T @ Mk).contiguous(), info)


def kept_map(W, X, Y, *, mu, rank=None, ridge_rel=0.0, rcond=None) -> KeptMap:
    """DIAGNOSTICS ONLY: :func:`kept_map_gram` from stacked activations."""
    Y64 = Y.to(torch.float64)
    return kept_map_gram(W, X, Y64.T @ Y64, mu=mu, rank=rank, ridge_rel=ridge_rel, rcond=rcond)


def apply_erasure(W, component) -> torch.Tensor:
    """R1: ``W_new = W - A @ B`` for an :class:`ErasedComponent` -- the only way
    a weight is edited for erasure. Returns a NEW tensor in ``W``'s dtype and
    device (the product is formed in the wider of ``W``'s and the factors'
    dtypes, then cast; pass an fp32 copy of a bf16 weight to keep the edit
    above bf16's resolution).

    Anything else -- a :class:`KeptMap`, a bare tensor, a factor pair of
    ``W`` -- raises :class:`ErasureConventionError` with the core table; a
    shape mismatch raises ``ValueError``.
    """
    if isinstance(component, KeptMap):
        raise ErasureConventionError(
            "apply_erasure was handed a KeptMap. The kept map is never written into a "
            "model: it is zero off the batch span and, truncated, loses retained "
            "behaviour inside it. Subtract the erased component instead "
            "(srspd.erased_component -> srspd.apply_erasure).\n\n" + CORE_TABLE)
    if not isinstance(component, ErasedComponent):
        raise ErasureConventionError(
            f"apply_erasure takes an ErasedComponent (A, B) with M = A @ B the erased "
            f"component; got {type(component).__name__}. Never write a fitted map, a "
            f"kept map or a factor pair of W into a model.\n\n" + CORE_TABLE)
    A, B = component.A, component.B
    if A.shape[1] != B.shape[0] or A.shape[0] != W.shape[0] or B.shape[1] != W.shape[1]:
        raise ValueError(f"shape mismatch: W {tuple(W.shape)}, A {tuple(A.shape)}, "
                         f"B {tuple(B.shape)}")
    dt = torch.promote_types(W.dtype, torch.promote_types(A.dtype, B.dtype))
    out = W.to(dtype=dt, copy=True)
    out.sub_(A.to(device=W.device, dtype=dt) @ B.to(device=W.device, dtype=dt))
    return out.to(W.dtype)


#: Deprecated leak-energy keys -> their descriptive names (same value). ``erased_rel``
#: is ``L_X_rel^2``, the target energy that SURVIVES the edit, despite its name.
ENERGY_ALIASES = {"erased_rel": "target_surviving_rel", "retained_rel": "retained_disturbed_rel"}


def _with_energy_names(d: dict) -> dict:
    """Add ``target_surviving_rel`` (= ``L_X_rel^2``) and ``retained_disturbed_rel``
    (= ``L_Y_rel^2``) beside their deprecated aliases ``erased_rel`` /
    ``retained_rel`` -- the same float objects, nothing recomputed."""
    for old, new in ENERGY_ALIASES.items():
        d[new] = d[old]
    return d


def leaks(W, component, X, Y) -> dict:
    """R4: the two leaks of ``W_new = W - M`` on stacked activations, relative:

        L_X_rel = ||(W - M) X^T||_F / ||W X^T||_F   (target response surviving)
        L_Y_rel = ||M Y^T||_F / ||W Y^T||_F         (retained response disturbed)

    Both are exactly zero iff span(X) and span(Y) are disentangled (and
    ``M`` is untruncated); for a truncated ``M_r`` they are the first row of
    the core table with ``M_r`` in place of ``M``.

    Returns a dict of floats: ``L_X_rel``, ``L_Y_rel``, their squares
    ``target_surviving_rel = L_X_rel^2`` (target energy left) and
    ``retained_disturbed_rel = L_Y_rel^2`` (retained energy disturbed), the same
    two values under the deprecated names ``erased_rel`` / ``retained_rel``, ``delta_fro =
    ||M||_F``, ``w_fro = ||W||_F``, and the energies ``wx_energy =
    ||W X^T||^2``, ``lx_energy = ||(W - M) X^T||^2``, ``wy_energy``,
    ``ly_energy``.
    """
    if not isinstance(component, ErasedComponent):
        raise ErasureConventionError("leaks measures an ErasedComponent, got "
                                     f"{type(component).__name__}\n\n" + CORE_TABLE)
    W64 = W.to(torch.float64)
    A, B = component.A.to(W64.device, torch.float64), component.B.to(W64.device, torch.float64)
    X64, Y64 = X.to(W64.device, torch.float64), Y.to(W64.device, torch.float64)
    M = A @ B
    wx = float((W64 @ X64.T).norm())
    wy = float((W64 @ Y64.T).norm())
    lx = float(((W64 - M) @ X64.T).norm())
    ly = float((M @ Y64.T).norm())
    return _with_energy_names(dict(
        L_X_rel=lx / max(wx, 1e-300), L_Y_rel=ly / max(wy, 1e-300),
        retained_rel=(ly / max(wy, 1e-300)) ** 2, erased_rel=(lx / max(wx, 1e-300)) ** 2,
        delta_fro=float(M.norm()), w_fro=float(W64.norm()),
        wx_energy=wx ** 2, lx_energy=lx ** 2, wy_energy=wy ** 2, ly_energy=ly ** 2))


def leaks_gram(W, component, X, G_Y, *, ret_energy=None) -> dict:
    """:func:`leaks` from the retained Gram, never forming ``M Y^T``:
    ``||M Y^T||^2 = tr(M G_Y M^T) = tr(A (B G_Y B^T) A^T)`` (an ``r x r`` core)
    and ``||W Y^T||^2 = tr(W G_Y W^T)`` (``ret_energy``; pass it precomputed
    when scoring several ``mu`` for one matrix -- it is the one
    ``d_out x d_in^2`` product here). Same keys as :func:`leaks` plus
    ``ret_energy``.
    """
    if not isinstance(component, ErasedComponent):
        raise ErasureConventionError("leaks_gram measures an ErasedComponent, got "
                                     f"{type(component).__name__}\n\n" + CORE_TABLE)
    dev = G_Y.device
    G = G_Y.to(torch.float64)
    W64 = W.to(device=dev, dtype=torch.float64)
    A = component.A.to(device=dev, dtype=torch.float64)
    B = component.B.to(device=dev, dtype=torch.float64)
    X64 = X.to(device=dev, dtype=torch.float64)
    if ret_energy is None:
        ret_energy = float(((W64 @ G) * W64).sum())
    wy = max(float(ret_energy), 0.0)
    BG = B @ G                                                     # r x d_in
    C = BG @ B.T                                                   # r x r = B G_Y B^T
    ly2 = max(float(((A @ C) * A).sum()), 0.0)                     # tr(A C A^T)
    WXt = W64 @ X64.T
    MXt = A @ (B @ X64.T)
    wx2 = float((WXt ** 2).sum())
    lx2 = float(((WXt - MXt) ** 2).sum())
    delta_fro = float(((A @ (B @ B.T)) * A).sum()) ** 0.5           # ||A B||_F
    return _with_energy_names(dict(
        L_X_rel=(lx2 / max(wx2, 1e-300)) ** 0.5, L_Y_rel=(ly2 / max(wy, 1e-300)) ** 0.5,
        retained_rel=ly2 / max(wy, 1e-300), erased_rel=lx2 / max(wx2, 1e-300),
        delta_fro=delta_fro, w_fro=float(W64.norm()), ret_energy=wy,
        wx_energy=wx2, lx_energy=lx2, wy_energy=wy, ly_energy=ly2))


# --------------------------------------------------------------------------- #
# The Gram form in BOTH sets: a target too large to hold as activations         #
# --------------------------------------------------------------------------- #
class GramMuFactor(NamedTuple):
    """A factorisation of ``G_mu + delta I = X^T X + mu Y^T Y + delta I`` built
    from the two Grams by :func:`factor_grams`, shared by every matrix that
    reads the same input (one Cholesky per input and ``mu``, as ``S`` is
    shared in the readout form). ``kind`` is ``"cholesky"`` (``L`` the lower
    factor, ``V``/``e`` ``None``) or ``"eigh"`` (``V`` the eigenvectors above
    the ``rcond`` floor, ``e`` their eigenvalues; ``L`` ``None``)."""
    kind: str
    L: torch.Tensor | None
    V: torch.Tensor | None
    e: torch.Tensor | None
    delta: float
    solver: str
    mu: float
    d_in: int


def factor_grams(G_X, G_Y, mu, *, rcond=None, ridge_rel=0.0):
    """Factor ``G_mu + delta I = G_X + mu G_Y + delta I`` from the TARGET Gram
    ``G_X = X^T X`` and the RETAINED Gram ``G_Y = Y^T Y`` (both ``[d_in, d_in]``,
    any float dtype; float64 inside, on ``G_Y``'s device). Returns
    ``(GramMuFactor, info)`` with ``info`` = ``solver``, ``mu``, ``d_in``,
    ``ridge_delta``, ``ridge_rel``.

    This is the readout solve with ``X^T X`` handed in as a Gram instead of
    accumulated from ``X``, and the SAME regularisation convention to the
    letter: ``delta = ridge_rel * max(diag(mu G_Y))``, the same x10 ladder to
    ``1e-6`` relative when Cholesky fails, the same PD test on the squared
    pivot ratio against ``rcond`` (default ``max(d_in * eps, 1e-12)``), the
    same ``eigh`` pseudo-inverse fallback restricted to eigenvalues above
    ``rcond * lambda_max``. So for the same ``(X, Y, mu, ridge_rel)`` the two
    forms factor the same matrix and :func:`erased_component_grams` equals
    :func:`erased_component_gram` to float64 roundoff.

    The Gram is factored in place (one ``d_in^2`` float64 transient beside the
    two resident Grams).
    """
    if not float(mu) > 0.0:
        raise ValueError(f"mu must be > 0, got {mu!r}")
    G_Y = G_Y.to(torch.float64)
    dev = G_Y.device
    G_X = G_X.to(device=dev, dtype=torch.float64)
    d_in = int(G_Y.shape[0])
    if tuple(G_X.shape) != (d_in, d_in):
        raise ValueError(f"G_X {tuple(G_X.shape)} and G_Y {tuple(G_Y.shape)} must both be "
                         f"[d_in, d_in]")
    if rcond is None:
        rcond = _default_gram_rcond(d_in)
    mu = float(mu)
    scale = mu * float(G_Y.diagonal().max().clamp_min(0.0))      # as the readout solve

    def build_G(delta=0.0):
        G = mu * G_Y                                               # mu Y^T Y
        G.add_(G_X)                                                # + X^T X
        if delta > 0.0:
            G.diagonal().add_(delta)                               # + delta I
        return G

    ladder = _ridge_ladder(ridge_rel)
    for rr in ladder:
        delta = rr * scale
        L = build_G(delta)
        info = torch.zeros((), dtype=torch.int32, device=dev)
        try:
            torch.linalg.cholesky_ex(L, out=(L, info))
        except torch.OutOfMemoryError:                            # pragma: no cover
            raise
        except RuntimeError:                                      # pragma: no cover
            L, info = torch.linalg.cholesky_ex(L)
        piv = L.diagonal()
        ok = (int(info) == 0 and bool(torch.isfinite(piv).all())
              and float(piv.min()) > 0.0
              and float(piv.min()) ** 2 > rcond * float(piv.max()) ** 2)
        del piv                                                    # a view of L; free it
        if ok:
            L.tril_()
            solver = "cholesky" if delta == 0.0 else "cholesky_ridge"
            f = GramMuFactor("cholesky", L, None, None, float(delta), solver, mu, d_in)
            return f, dict(solver=solver, mu=mu, d_in=d_in, ridge_delta=float(delta),
                           ridge_rel=float(ridge_rel))
        del L
    delta = ladder[-1] * scale
    G = build_G(delta)
    evals, V = _eigh_oom_safe(G)
    del G
    keep = evals > rcond * float(evals.max().clamp_min(0.0))
    f = GramMuFactor("eigh", None, V[:, keep].contiguous(), evals[keep].contiguous(),
                     float(delta), "eigh", mu, d_in)
    return f, dict(solver="eigh", mu=mu, d_in=d_in, ridge_delta=float(delta),
                   ridge_rel=float(ridge_rel))


def _apply_gram_inverse_right(T, f: GramMuFactor):
    """``T (G_mu + delta I)^+`` for a ``[d_out, d_in]`` ``T`` from the factor."""
    if f.kind == "cholesky":
        return torch.cholesky_solve(T.T.contiguous(), f.L).T.contiguous()
    return (((T @ f.V) / f.e) @ f.V.T).contiguous()


def erased_component_grams(W, G_X, G_Y, *, mu, ridge_rel=0.0, rcond=None,
                           factor=None) -> ErasedComponent:
    """The erased component from the two Grams -- the form for a TARGET set too
    large to hold as activations:

        M = W G_X (G_X + mu G_Y + delta I)^+,    G_X = X^T X,  G_Y = Y^T Y

    (``X`` the target, ``Y`` the retained set, batch-first; ``G_X``, ``G_Y``
    ``[d_in, d_in]``, float64 inside). It is the same ``M`` as
    :func:`erased_component_gram` -- ``W X^T X G_mu^+`` with ``S = X G_mu^+``
    never formed -- and the global minimiser of
    ``||W X^T - M X^T||^2 + mu ||M Y^T||^2`` over ALL ``M``. ``delta`` is the
    ridge in exactly the readout solve's convention, ``delta = ridge_rel *
    max(diag(mu G_Y))`` with the same escalation ladder and ``eigh`` fallback
    (:func:`factor_grams`).

    Representation. ``M`` is dense and in general full rank (``n_x > d_in``:
    ``range(M) = range(M X^T)`` spans all of ``R^{d_out}`` as soon as the target
    does), so the component is returned in the full-rank form ``A = M``
    (``[d_out, d_in]``), ``B = I_{d_in}`` (float64 identity on ``M``'s device),
    ``info["B_is_identity"] = True``; :func:`apply_erasure` then subtracts
    ``A @ B = M`` as for any component. No rank truncation: the COALA
    truncation needs the target response ``M X^T``, which this form does not
    hold.

    Pass ``factor`` (from :func:`factor_grams` for this ``(G_X, G_Y, mu,
    ridge_rel)``) to share one factorisation across every matrix that reads the
    same input; then ``G_Y`` is only used for its device.

    ``info``: ``solver``, ``mu``, ``ridge_delta``, ``ridge_rel``, ``d_out``,
    ``d_in``, ``rank`` = ``k`` = ``d_in`` (the representation's inner
    dimension), ``truncated`` (``False``), ``B_is_identity``, ``formula``.
    """
    if mu is None or not float(mu) > 0.0:
        raise ValueError(f"mu must be > 0, got {mu!r}")
    if factor is None:
        factor, finfo = factor_grams(G_X, G_Y, mu, rcond=rcond, ridge_rel=ridge_rel)
    else:
        if abs(float(factor.mu) - float(mu)) > 1e-12 * max(1.0, float(mu)):
            raise ValueError(f"factor was built at mu {factor.mu!r}, not {mu!r}")
        finfo = dict(solver=factor.solver, ridge_delta=factor.delta, ridge_rel=float(ridge_rel))
    dev = G_Y.device
    W64 = W.to(device=dev, dtype=torch.float64)
    G_X64 = G_X.to(device=dev, dtype=torch.float64)
    d_out, d_in = int(W64.shape[0]), int(W64.shape[1])
    if int(G_X64.shape[0]) != d_in:
        raise ValueError(f"W {tuple(W.shape)} does not read a d_in {int(G_X64.shape[0])} input")
    M = _apply_gram_inverse_right(W64 @ G_X64, factor)              # W G_X (G_mu + delta I)^+
    info = dict(solver=finfo["solver"], mu=float(mu), ridge_delta=float(finfo["ridge_delta"]),
                ridge_rel=float(ridge_rel), d_out=d_out, d_in=d_in, rank=d_in, k=d_in,
                truncated=False, B_is_identity=True,
                formula="W - W G_X (G_X + mu G_Y + delta I)^+,  G_X = X^T X, G_Y = Y^T Y")
    return ErasedComponent(M, torch.eye(d_in, dtype=torch.float64, device=dev), info)


def _quad_trace(D, G, block=4096):
    """``tr(D G D^T)`` in row blocks of ``D`` (never a second ``d_out x d_in``)."""
    return float(sum(((D[i:i + block] @ G) * D[i:i + block]).sum()
                     for i in range(0, int(D.shape[0]), block)))


def _dense_component(component, dev):
    if component.info.get("B_is_identity"):
        A = component.A
        if tuple(component.B.shape) != (int(A.shape[1]), int(A.shape[1])):
            raise ValueError("B_is_identity set but B is not [d_in, d_in]")
        return A.to(device=dev, dtype=torch.float64)
    return component.A.to(dev, torch.float64) @ component.B.to(dev, torch.float64)


def leaks_grams(W, component, G_X, G_Y, *, ret_energy=None, target_energy=None) -> dict:
    """R4 from the two Grams, for a target held only as ``G_X = X^T X``:

        L_X_rel^2 = tr((W - M) G_X (W - M)^T) / tr(W G_X W^T)   (= ||(W - M) X^T||^2 / ||W X^T||^2)
        L_Y_rel^2 = tr(M G_Y M^T) / tr(W G_Y W^T)               (= ||M Y^T||^2 / ||W Y^T||^2)

    -- the same quantities as :func:`leaks` / :func:`leaks_gram` under the same
    keys: ``L_X_rel``, ``L_Y_rel``, their squares ``target_surviving_rel`` /
    ``retained_disturbed_rel`` (deprecated aliases ``erased_rel`` /
    ``retained_rel``), ``delta_fro`` (``||M||_F``), ``w_fro``, ``ret_energy``,
    and the energies ``wx_energy``, ``lx_energy``, ``wy_energy``, ``ly_energy``
    plus ``mx_energy = tr(M G_X M^T)`` (the component's own target response).
    ``ret_energy`` / ``target_energy`` take the two ``tr(W G W^T)`` precomputed
    when several ``mu`` are scored for one matrix. ``M`` is formed dense
    (``component.A`` when ``B`` is the identity).
    """
    if not isinstance(component, ErasedComponent):
        raise ErasureConventionError("leaks_grams measures an ErasedComponent, got "
                                     f"{type(component).__name__}\n\n" + CORE_TABLE)
    dev = G_Y.device
    GY = G_Y.to(torch.float64)
    GX = G_X.to(device=dev, dtype=torch.float64)
    W64 = W.to(device=dev, dtype=torch.float64)
    M = _dense_component(component, dev)
    wy = max(float(_quad_trace(W64, GY)) if ret_energy is None else float(ret_energy), 0.0)
    wx = max(float(_quad_trace(W64, GX)) if target_energy is None else float(target_energy), 0.0)
    D = W64 - M
    lx2 = max(_quad_trace(D, GX), 0.0)
    del D
    ly2 = max(_quad_trace(M, GY), 0.0)
    mx2 = max(_quad_trace(M, GX), 0.0)
    return _with_energy_names(dict(
        L_X_rel=(lx2 / max(wx, 1e-300)) ** 0.5, L_Y_rel=(ly2 / max(wy, 1e-300)) ** 0.5,
        retained_rel=ly2 / max(wy, 1e-300), erased_rel=lx2 / max(wx, 1e-300),
        delta_fro=float(M.norm()), w_fro=float(W64.norm()), ret_energy=wy,
        wx_energy=wx, lx_energy=lx2, wy_energy=wy, ly_energy=ly2, mx_energy=mx2))


# --------------------------------------------------------------------------- #
# Specificity: which directions a dose erases, before any edit                 #
# --------------------------------------------------------------------------- #
def specificity(G_Y, X, n_y, k=8, *, ridge_rel=0.0, full=False):
    """The generalized eigenvalues ``gamma`` of the per-token target covariance
    ``X^T X / n_x`` against the per-token retained covariance ``G_Y / n_y``.

    Why this spectrum: in the generalized eigenbasis the full-rank edit is
    diagonal -- the kept map ``W - M = W mu Y^T Y G_mu^{-1}`` scales
    generalized eigendirection ``i`` by ``1 / (1 + gamma_i m)`` with
    ``m = mu_per_token = n_x / (mu n_y)``, so the erased fraction of that
    direction is ``gamma_i m / (1 + gamma_i m)`` (:mod:`srspd.dose`). So
    ``gamma`` says, before any edit is made, which directions a given dose
    erases: a direction with ``gamma`` = 100 is 100x more target than retained
    per token and is half-erased already at ``m = 0.01``; one with
    ``gamma`` << 1 survives every moderate dose. Only ``min(n_x, d_in)`` of the
    ``d_in`` values can be nonzero (``X`` spans at most ``n_x`` directions);
    the rest are exactly 0 and are not reported.

    Computed as the squared singular values of ``X L_y^{-T} / sqrt(n_x)`` with
    ``L_y = chol(G_Y / n_y)`` -- via ``eigvalsh`` of the ``n_x x n_x`` Gram of
    that matrix, so the cost is one triangular solve. ``G_Y`` not PD falls
    back to an ``eigh`` pseudo-inverse square root (``solver`` records it).
    ``ridge_rel`` adds ``ridge_rel * max(diag(G_Y / n_y))`` to ``G_Y / n_y``.

    Parameters
    ----------
    G_Y : [d_in, d_in] retained Gram ``Y^T Y``.
    X : [n_x, d_in] target activations, batch-first.
    n_y : int, the number of retained rows (tokens) behind ``G_Y``.
    k : int, how many of the largest values to report in ``top``.
    ridge_rel : float >= 0.
    full : bool, also return the whole spectrum under ``"spectrum"``.

    Returns
    -------
    dict: ``top`` (the ``k`` largest, descending, list of float), ``median``,
    ``n_above`` (``{"1", "10", "100"}`` -> count above each threshold), ``n``
    (number of values), ``solver`` (``"cholesky"`` | ``"cholesky_ridge"`` |
    ``"eigh"`` | ``"none"``), and with ``full=True`` ``spectrum`` (all ``n``
    values, descending, list of float).
    """
    dev = G_Y.device
    C = G_Y.to(torch.float64) / float(n_y)
    X64 = X.to(device=dev, dtype=torch.float64)
    n_x = int(X64.shape[0])
    if n_x == 0:
        out = dict(top=[], median=0.0, n_above={"1": 0, "10": 0, "100": 0}, n=0,
                   solver="none")
        if full:
            out["spectrum"] = []
        return out
    d_in = int(C.shape[0])
    rcond = _default_gram_rcond(d_in)
    if float(ridge_rel) > 0.0:
        C.diagonal().add_(float(ridge_rel) * float(C.diagonal().max().clamp_min(0.0)))
    Ly = C                                     # factored in place: one d_in^2 transient
    info = torch.zeros((), dtype=torch.int32, device=dev)
    try:
        torch.linalg.cholesky_ex(Ly, out=(Ly, info))
    except RuntimeError:                                          # pragma: no cover
        Ly, info = torch.linalg.cholesky_ex(Ly)
    piv = Ly.diagonal()
    piv_ok = (int(info) == 0 and float(piv.min()) > 0.0
              and float(piv.min()) ** 2 > rcond * float(piv.max()) ** 2)
    del piv        # a VIEW of Ly -- dropping the name lets `del Ly, C` below actually
    #              # free the d x d factor before eigvalsh / the eigh fallback
    if piv_ok:
        Ly.tril_()
        Zt = torch.linalg.solve_triangular(Ly, X64.T, upper=False)   # L_y^{-1} X^T
        solver = "cholesky" if float(ridge_rel) == 0.0 else "cholesky_ridge"
    else:
        Ly = C = None                                  # a failed factor; rebuild C
        Cr = G_Y.to(torch.float64) / float(n_y)
        if float(ridge_rel) > 0.0:
            Cr.diagonal().add_(float(ridge_rel) * float(Cr.diagonal().max().clamp_min(0.0)))
        ev, V = _eigh_oom_safe(Cr)
        del Cr
        keep = ev > rcond * float(ev.max().clamp_min(0.0))
        Zt = (V[:, keep].T @ X64.T) / ev[keep].sqrt()[:, None]
        solver = "eigh"
        del V
    del Ly, C
    g = torch.linalg.eigvalsh((Zt.T @ Zt) / n_x).flip(0).clamp_min(0.0)
    g = g[: min(n_x, d_in)]
    out = dict(top=[float(v) for v in g[:k].tolist()],
               median=float(torch.quantile(g, 0.5)) if g.numel() else 0.0,
               n_above={str(t): int((g > t).sum()) for t in (1, 10, 100)},
               n=int(g.numel()), solver=solver)
    if full:
        out["spectrum"] = [float(v) for v in g.tolist()]
    return out


#: Alias kept for callers that know the Gram-form name.
specificity_gram = specificity


# --------------------------------------------------------------------------- #
# Conditioning check                                                           #
# --------------------------------------------------------------------------- #
def gram_eig_extremes(G, *, iters=200, tol=1e-7, seed=0):
    """``(lam_min, lam_max, info)`` of a symmetric PSD Gram, iteratively.

    The per-input conditioning check. A dense ``eigvalsh`` would give both ends
    exactly, but cuSOLVER's ``syevd`` wants ~2 d^2 doubles of workspace -- 13.8
    GiB for a 21504-wide Gram, which a GPU carrying a large model may not have.
    Here: ``lam_max`` by power iteration on ``G``; ``lam_min`` by power
    iteration on ``G^{-1}`` through its Cholesky factor (one in-place ``d x d``
    copy -- the same transient the readout solve already takes). Each step is a
    matvec or a pair of triangular solves, O(d^2).

    Both are Rayleigh-quotient estimates, so ``lam_max`` is approached from
    below and ``lam_min`` from above: the reported ratio can only *overstate*
    how well conditioned G is, never understate it.

    Returns ``(lam_min, lam_max, info)``; ``info``: ``pd`` (bool),
    ``iters_max``, ``rel_max`` and, when PD, ``iters_min``, ``rel_min`` (the
    iteration counts and final relative changes, so a reader can see
    convergence). ``G`` not numerically PD (Cholesky fails) reports
    ``lam_min = 0`` with ``info["pd"] = False``.
    """
    G = G.to(torch.float64)
    d = int(G.shape[0])
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    def power(apply):
        v = torch.randn(d, 1, dtype=torch.float64, generator=gen).to(G.device)
        v /= v.norm()
        lam, it, rel = 0.0, 0, float("inf")
        for it in range(1, int(iters) + 1):
            w = apply(v)
            new = float((v * w).sum())                 # Rayleigh quotient, |v| = 1
            nw = float(w.norm())
            if nw == 0.0:
                return 0.0, it, 0.0
            v = w / nw
            rel = abs(new - lam) / max(abs(new), 1e-300)
            lam = new
            if rel < tol:
                break
        return lam, it, rel

    lam_max, it_max, rel_max = power(lambda v: G @ v)
    L = G.clone()
    info_t = torch.zeros((), dtype=torch.int32, device=G.device)
    try:
        torch.linalg.cholesky_ex(L, out=(L, info_t))
    except RuntimeError:                                          # pragma: no cover
        L, info_t = torch.linalg.cholesky_ex(L)
    pd = int(info_t) == 0 and bool(torch.isfinite(L.diagonal()).all()) \
        and float(L.diagonal().min()) > 0.0
    if not pd:
        return 0.0, lam_max, dict(pd=False, iters_max=it_max, rel_max=rel_max)
    L.tril_()
    inv_top, it_min, rel_min = power(lambda v: torch.cholesky_solve(v, L))
    del L
    lam_min = 1.0 / inv_top if inv_top > 0 else 0.0
    return lam_min, lam_max, dict(pd=True, iters_max=it_max, rel_max=rel_max,
                                  iters_min=it_min, rel_min=rel_min)
