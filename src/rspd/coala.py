"""COALA: the context-aware low-rank projector, and the compression built on it.

Credit
------
The projector used here is from **COALA**: Uliana Parkina and Maxim Rakhuba,
*COALA: Numerically Stable and Efficient Framework for Context-Aware Low-Rank
Approximation*, arXiv:2507.07580, NeurIPS 2025 -- Proposition 1, the projector
form of the context-aware rank-``r`` approximation:

    W_r = U_r U_r^T W,      U_r = the top-r left singular vectors of the response W X.

This module is that projector form written independently from the paper's
statement; it is **not** a port of the authors' code and copies none of it.

The math
--------
``W`` is ``[d_out, d_in]``; ``A`` the batch of its inputs, batch-first
``[n, d_in]`` (``X = A^T`` are the columns); ``T = A W^T`` the batch-first
response, so ``T^T = W X``. For any thin QR ``X^T = Q R``, ``W X = (W R^T) Q^T``,
so ``W X`` and ``W R^T`` share left singular vectors and singular values: the
``U`` of the SVD of the response IS the basis COALA wants, and no QR,
pseudo-inverse or Gram matrix is needed.

* On span(X) the rank-``r`` surrogate reaches the Eckart--Young optimum of the
  response: ``U_r U_r^T W X = [W X]_r``. Off span(X) it preserves ``W``'s action
  (it is an orthogonal projection of ``W``'s output) instead of zeroing it.
* At full rank ``r = k`` with ``rank(W X) = d_out``, ``U U^T = I`` and the
  surrogate reproduces ``W`` exactly, up to float error.
* Prefix property: ``A_fac[:, :r] @ B_fac[:r]`` is the rank-``r`` surrogate for
  every ``r``, so one SVD serves a whole rank sweep.

Two uses in this package
------------------------
1. **Erasure rank truncation** acts on the erased component ``M``, never on
   ``W``: ``M_r = U_r U_r^T M`` with ``U_r`` from the response ``M X^T`` on the
   target set (:func:`rspd.truncate_component`,
   :func:`rspd.erased_component_gram` with ``rank`` / ``lam`` / ``lam_rel``).
   That is COALA applied to ``M`` over ``X``; both call :func:`coala_basis`,
   and the edit is still ``W_new = W - M_r`` (:func:`rspd.apply_erasure`).
2. **Compression** (:func:`compress_factors`, :func:`materialize_compressed`,
   :func:`compress_weight`) applies the projector to ``W`` itself and the
   rank-``r`` surrogate REPLACES the weight. This is the only legitimate weight
   replacement in the package and it is never an erasure: an erasure edit is
   always ``W_new = W - M``.
"""
from __future__ import annotations

import torch

__all__ = ["coala_basis", "fit_coala", "compress_factors", "materialize_compressed",
           "compress_weight"]


def coala_basis(response: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(U, sigma)``: the COALA basis of a ``[d_out, n]`` column response.

    ``U`` (``[d_out, k]``, orthonormal columns, ``k = min(d_out, n)``) and
    ``sigma`` (``[k]``, descending) are the left singular vectors and singular
    values of ``response`` from one thin ``torch.linalg.svd`` in the response's
    own dtype and device. The rank-``r`` COALA projector of any matrix ``Z``
    whose response this is, is ``U[:, :r] U[:, :r]^T Z`` (Parkina & Rakhuba,
    arXiv:2507.07580, Proposition 1). Shared by :func:`fit_coala` (response
    ``W X``) and the erasure truncation (response ``M X^T``).
    """
    U, sigma, _Vh = torch.linalg.svd(response, full_matrices=False)
    return U, sigma


def fit_coala(W: torch.Tensor, A: torch.Tensor, T: torch.Tensor):
    """The COALA factor pair of one linear map; returns ``(A_fac, B_fac, sigma)``.

    Parameters
    ----------
    W : [d_out, d_in] the matrix.
    A : [n, d_in] the batch-first inputs it is applied to. Not read: COALA
        needs only the response; accepted so every factoriser has one
        signature.
    T : [n, d_out] the batch-first response ``T = A @ W.T`` (``T.T = W X``).

    Returns
    -------
    A_fac : [d_out, k] ``U``, orthonormal columns, unscaled.
    B_fac : [k, d_in] ``U^T W``.
    sigma : [k] the singular values of ``W X``, descending.

    ``k = min(d_out, n)``. ``A_fac[:, :r] @ B_fac[:r] = U_r U_r^T W`` is the
    rank-``r`` COALA surrogate for every ``r`` (Parkina & Rakhuba,
    arXiv:2507.07580, Proposition 1). Beyond ``rank(W X)`` the trailing columns
    of ``U`` are an arbitrary orthonormal completion, and column signs are
    arbitrary: compare products, not factors.
    """
    U, S = coala_basis(T.T)
    return U, U.T @ W, S


# --------------------------------------------------------------------------- #
# Compression: the one legitimate REPLACEMENT of a weight (never an erasure)   #
# --------------------------------------------------------------------------- #
def compress_factors(W: torch.Tensor, A: torch.Tensor, T: torch.Tensor):
    """``(A_fac, B_fac, sigma)`` for compressing ``W`` on its inputs ``A``
    (response ``T = A @ W.T``): :func:`fit_coala`. The prefix
    ``A_fac[:, :r] @ B_fac[:r]`` is the rank-``r`` replacement. Compression
    only -- an erasure edit is ``W - M`` (:func:`rspd.apply_erasure`)."""
    return fit_coala(W, A, T)


def materialize_compressed(A_fac: torch.Tensor, B_fac: torch.Tensor, r: int, *,
                           device=None, dtype=None) -> torch.Tensor:
    """The dense rank-``r`` replacement ``A_fac[:, :r] @ B_fac[:r]`` from a
    factor pair -- the tensor a compression writes into the model. ``r`` is
    clamped to ``[0, A_fac.shape[1]]``; ``device`` / ``dtype`` default to the
    factors' own."""
    r = max(0, min(int(r), int(A_fac.shape[1])))
    Af = A_fac[:, :r].to(device=device or A_fac.device, dtype=dtype or A_fac.dtype)
    Bf = B_fac[:r].to(device=device or B_fac.device, dtype=dtype or B_fac.dtype)
    return Af @ Bf


def compress_weight(W: torch.Tensor, A: torch.Tensor, T: torch.Tensor, r: int) -> torch.Tensor:
    """``W_new = U_r U_r^T W``: fit (:func:`compress_factors`) and materialise
    (:func:`materialize_compressed`) in one call. A replacement of ``W`` by its
    rank-``r`` COALA surrogate (Parkina & Rakhuba, arXiv:2507.07580); never
    use it to erase."""
    A_fac, B_fac, _sigma = compress_factors(W, A, T)
    return materialize_compressed(A_fac, B_fac, r)
