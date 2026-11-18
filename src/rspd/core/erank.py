"""Effective rank (erank): Roy & Vetterli's spectral effective dimensionality.

The metric is from Roy and Vetterli, *The effective rank: A measure of
effective dimensionality* (EUSIPCO, 2007). For a nonzero matrix ``A`` with
singular values ``s_1 >= ... >= s_Q``, ``Q = min(M, N)``, normalise the spectrum
into a probability distribution ``p_i = s_i / sum_j s_j`` (division by the
nuclear norm) and define

    erank(A) = exp( - sum_i p_i ln p_i ),      0 ln 0 := 0 .

Properties: ``1 <= erank(A) <= rank(A)``, with equality at ``rank(A)`` iff the
nonzero singular values are all equal, and invariance to scalar rescaling of
``A``. The value is real (not an integer); callers that need an integer cutoff
round it themselves (:func:`rspd.core.solve.erank_truncation_rank`).
"""
from __future__ import annotations

import torch

__all__ = ["effective_rank", "effective_rank_from_singular_values"]


@torch.no_grad()
def effective_rank(A: torch.Tensor) -> float:
    """Roy-Vetterli effective rank of a matrix (or of a 1-D singular spectrum).

    Parameters
    ----------
    A : torch.Tensor
        Either a 2-D matrix -- its singular values are computed internally via
        :func:`torch.linalg.svdvals` in float32 -- or a 1-D tensor of
        (non-negative) singular values, so a caller who already holds a
        spectrum does not pay for a second SVD.

    Returns
    -------
    float
        ``exp(-sum_i p_i ln p_i)`` with ``p_i = s_i / sum_j s_j`` and the
        ``0 ln 0 := 0`` convention (zero singular values contribute nothing).

    Raises
    ------
    ValueError
        If the matrix (or spectrum) is all-zero (the definition requires a
        nonzero matrix), if the spectrum is empty or contains NaN/Inf (a NaN
        comparison would otherwise slip past the positivity gate and return a
        plausible-looking 1.0), or if the input is not 1-D/2-D.
    """
    A = torch.as_tensor(A).detach()
    if A.ndim == 2:
        s = torch.linalg.svdvals(A.to(torch.float32))
    elif A.ndim == 1:
        s = A.to(torch.float32)
    else:
        raise ValueError(f"expected a 2-D matrix or 1-D singular spectrum; got ndim={A.ndim}")
    return effective_rank_from_singular_values(s)


@torch.no_grad()
def effective_rank_from_singular_values(sigma: torch.Tensor) -> float:
    """The one Roy-Vetterli formula behind :func:`effective_rank`, on a 1-D
    spectrum of (non-negative) singular values, in the spectrum's own floating
    dtype (float64 stays float64; :func:`effective_rank` passes float32):
    ``exp(-sum_i p_i ln p_i)``, ``p_i = s_i / sum_j s_j`` (the singular values
    normalised by the nuclear norm -- NOT ``s_i^2 / sum s_j^2``), ``0 ln 0 :=
    0``. Raises ``ValueError`` on an empty, non-finite or all-zero spectrum.
    Callers that already hold a spectrum (:func:`rspd.core.truncate_component`)
    use this so no second SVD is taken and no second definition exists."""
    s = torch.as_tensor(sigma).detach()
    if s.ndim != 1:
        raise ValueError(f"expected a 1-D singular spectrum; got ndim={s.ndim}")
    if not s.is_floating_point():
        s = s.to(torch.float32)
    if s.numel() == 0 or not bool(torch.isfinite(s).all()):
        raise ValueError("effective_rank requires a nonempty, finite singular spectrum")
    scale = s.max()
    if float(scale) <= 0.0:
        raise ValueError("effective_rank requires a nonzero matrix (nuclear norm > 0)")

    # Pre-divide by the top singular value (erank is scale-invariant) so the
    # nuclear-norm sum cannot overflow float32 for large spectra.
    p = s / scale
    p = p / p.sum()
    mask = p > 0                                  # 0 ln 0 := 0
    entropy = -(p[mask] * torch.log(p[mask])).sum()
    return float(torch.exp(entropy))
