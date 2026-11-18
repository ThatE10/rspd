"""The dose: ``mu_per_token`` is the one user-facing dose of an erasure.

Definitions
-----------
``X`` (target, ``n_x`` rows/tokens) and ``Y`` (retained, ``n_y`` rows/tokens)
enter the solve (:func:`rspd.core.erased_component`) as

    min_M  ||W X^T - M X^T||^2 + mu ||M Y^T||^2,

with ``mu`` the weight on the retained term. Its value depends on how many
tokens each set happens to hold, so it is not the dose a user gives. Divide the
objective by ``n_x``; it becomes the per-token Lagrangian

    E_x ||(W - M) x||^2  +  (1/m) E_y ||M y||^2,        m = mu_per_token,

where ``E_x``, ``E_y`` are means over the target and retained tokens and
``1/m`` is the per-token Lagrange multiplier on the retained disturbance. The
two parametrisations are tied by

    mu_per_token = mu_raw * n_x / n_y,      mu_raw = 1 / mu,
    mu           = n_x / (mu_per_token * n_y).

``mu_raw`` is the weight on the target term when the retained term has unit
weight; it and ``mu`` are DERIVED from ``mu_per_token`` and the set sizes and
are only reported (:class:`Dose`). Every user-facing entry point takes
``mu_per_token``.

What a dose erases
------------------
Let ``gamma_i`` be the generalised eigenvalues of the per-token target
covariance ``X^T X / n_x`` against the per-token retained covariance
``Y^T Y / n_y`` (:func:`rspd.core.specificity`). In that eigenbasis the full-rank
solve is diagonal and the erased fraction of direction ``i`` is

    f_i = gamma_i m / (1 + gamma_i m),

i.e. the eigenvalues of ``X^T X (X^T X + mu Y^T Y)^{-1}``. Half-erasure is at
``gamma m = 1``; a direction ``gamma`` times more target than retained per
token is erased by fraction ``f`` at ``m = f / ((1 - f) gamma)``. This holds
exactly at full rank with no ridge (``ridge_rel = 0``, no truncation, ``G_Y``
nonsingular); a ridge or a rank truncation of ``M`` makes it approximate.
:func:`suggest_mu_per_token` inverts it at chosen quantiles of the ``gamma``
spectrum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch

__all__ = ["Dose", "dose_from_mu_per_token", "erased_fraction", "suggest_mu_per_token",
           "DOSE_FORMULA"]

#: The dose relations, as :meth:`Dose.as_dict` reports them.
DOSE_FORMULA = ("mu_per_token = mu_raw * n_x / n_y, mu_raw = 1 / mu; "
                "per-direction erased fraction f = gamma m / (1 + gamma m)")


def _pos_float(v, name):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number, got {v!r}") from None
    if not (math.isfinite(f) and f > 0.0):
        raise ValueError(f"{name} must be finite and > 0, got {v!r}")
    return f


def _pos_int(v, name):
    if isinstance(v, bool) or int(v) != v or int(v) <= 0:
        raise ValueError(f"{name} must be a positive integer (a row/token count), got {v!r}")
    return int(v)


@dataclass(frozen=True)
class Dose:
    """One erasure dose: ``mu_per_token`` and the set sizes it was resolved at.

    Fields
    ------
    mu_per_token : float > 0, the dose (per-token weight on the target
        response; ``1 / mu_per_token`` is the per-token Lagrange multiplier on
        the retained disturbance).
    n_x : int > 0, target rows (tokens) behind ``X``.
    n_y : int > 0, retained rows (tokens) behind ``G_Y``.

    Derived (properties, reported, never inputs): ``mu`` (pass it to the
    solve) and ``mu_raw = 1 / mu``.
    """
    mu_per_token: float
    n_x: int
    n_y: int

    def __post_init__(self):
        object.__setattr__(self, "mu_per_token", _pos_float(self.mu_per_token, "mu_per_token"))
        object.__setattr__(self, "n_x", _pos_int(self.n_x, "n_x"))
        object.__setattr__(self, "n_y", _pos_int(self.n_y, "n_y"))

    @property
    def mu_raw(self) -> float:
        """``mu_per_token * n_y / n_x``: the weight on the target term at unit
        retained weight (reported only)."""
        return self.mu_per_token * self.n_y / self.n_x

    @property
    def mu(self) -> float:
        """``1 / mu_raw = n_x / (mu_per_token * n_y)``: the solve's weight on
        the retained term (the ``mu`` argument of :func:`rspd.core.erased_component`)."""
        return 1.0 / self.mu_raw

    def erased_fraction(self, gamma):
        """``gamma m / (1 + gamma m)`` at this dose (see :func:`erased_fraction`)."""
        return erased_fraction(gamma, self.mu_per_token)

    def as_dict(self) -> dict:
        """``{"mu_per_token", "mu_raw", "mu", "n_x", "n_y", "formula"}``."""
        return {"mu_per_token": self.mu_per_token, "mu_raw": self.mu_raw, "mu": self.mu,
                "n_x": self.n_x, "n_y": self.n_y, "formula": DOSE_FORMULA}


def dose_from_mu_per_token(m, n_x, n_y) -> Dose:
    """The :class:`Dose` for ``mu_per_token = m`` on ``n_x`` target and ``n_y``
    retained tokens. ``dose.mu`` is what the solve takes."""
    return Dose(mu_per_token=m, n_x=n_x, n_y=n_y)


def erased_fraction(gamma, mu_per_token):
    """Per-direction erased fraction ``f = gamma m / (1 + gamma m)`` (exact at
    full rank with no ridge). ``gamma`` a float, sequence or tensor; returns
    the same shape (float or float64 tensor)."""
    m = _pos_float(mu_per_token, "mu_per_token")
    if isinstance(gamma, (int, float)):
        g = float(gamma)
        return g * m / (1.0 + g * m)
    g = torch.as_tensor(gamma, dtype=torch.float64)
    return g * m / (1.0 + g * m)


def _gamma_tensor(gamma):
    if isinstance(gamma, Mapping):
        if "spectrum" not in gamma:
            raise ValueError("suggest_mu_per_token needs the full gamma spectrum: pass "
                             "rspd.core.specificity(..., full=True), an entry of "
                             "rspd.api.erase(..., spectrum='full').specificity, or the spectrum "
                             "itself")
        gamma = gamma["spectrum"]
    g = torch.as_tensor(gamma, dtype=torch.float64).reshape(-1)
    if g.numel() == 0:
        raise ValueError("suggest_mu_per_token needs a nonempty gamma spectrum")
    if not bool(torch.isfinite(g).all()) or bool((g < 0).any()):
        raise ValueError("gamma must be finite and >= 0 (generalised eigenvalues of two PSD "
                         "covariances)")
    return g


def suggest_mu_per_token(gamma, fractions: Iterable[float] = (0.1, 0.5, 0.9),
                         quantiles: Sequence[float] = (0.5, 0.9, 0.99)) -> dict:
    """Doses that erase the ``q``-quantile direction of the ``gamma`` spectrum
    by fraction ``f``, for every ``(q, f)``: ``m = f / ((1 - f) gamma_q)``.

    Parameters
    ----------
    gamma : the generalised-eigenvalue spectrum -- a sequence/tensor of
        ``gamma >= 0``, or the dict from ``rspd.core.specificity(..., full=True)``
        (also ``report.specificity[key]["specificity"]`` of
        ``rspd.api.erase(..., spectrum="full")``).
    fractions : erased fractions, each in ``(0, 1)``.
    quantiles : quantile levels of ``gamma``, each in ``[0, 1]``
        (``torch.quantile``, linear interpolation, over every value given).

    Returns
    -------
    dict:
      ``rows`` -- list of ``{"quantile", "gamma", "fraction", "mu_per_token"}``,
      quantile-major in the order given; ``mu_per_token`` is ``math.inf`` when
      ``gamma_q = 0`` (no finite dose erases a direction with no target
      excess);
      ``mu_per_token`` -- the sorted distinct finite doses (a ready grid);
      ``formula`` -- ``"m = f / ((1 - f) gamma_q)"``.

    Exact at full rank with no ridge, where ``f(gamma_q, m) = gamma_q m /
    (1 + gamma_q m)`` (:func:`erased_fraction`); approximate otherwise.
    """
    g = _gamma_tensor(gamma)
    fr = [float(f) for f in fractions]
    for f in fr:
        if not 0.0 < f < 1.0:
            raise ValueError(f"fractions must be in (0, 1), got {f!r}")
    qs = [float(q) for q in quantiles]
    for q in qs:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantiles must be in [0, 1], got {q!r}")
    rows = []
    for q in qs:
        gq = float(torch.quantile(g, q))
        for f in fr:
            m = f / ((1.0 - f) * gq) if gq > 0.0 else math.inf
            rows.append(dict(quantile=q, gamma=gq, fraction=f, mu_per_token=m))
    grid = sorted({r["mu_per_token"] for r in rows if math.isfinite(r["mu_per_token"])})
    return dict(rows=rows, mu_per_token=grid, formula="m = f / ((1 - f) gamma_q)")
