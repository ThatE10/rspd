"""A toy small enough to see: two sets of activations that share a direction.

The input activation space has a few named directions. The **target** set
``X`` is active on its own directions and on a shared one; the **retained**
set ``Y`` is active on the shared one and on its own:

    direction   x0   x1   shared   y0   y1
    X active     *    *     *
    Y active                *       *    *

One linear layer ``W`` reads the space. The component ``M`` that SRSPD finds
carries what only ``X`` uses: the ``x`` directions pass through ``M`` almost
whole, the ``y`` directions not at all, and the shared one in proportion to
the dose. ``W - M`` is what is left for ``Y``.

    import rspd, rspd.toy

    toy = rspd.toy.two_sets()
    M = rspd.core.erased_component(toy.W, toy.X, toy.Y, mu=toy.mu(mu_per_token=0.1))
    rspd.toy.carried(toy.W, M)     # per direction: ~1, ~1, ~0.09, ~0, ~0

Tensors only; the README animation (``assets/make_demo.py``) and
``examples/quickstart.ipynb`` are drawn from this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .core.dose import dose_from_mu_per_token

__all__ = ["TwoSets", "two_sets", "carried", "activation"]


@dataclass
class TwoSets:
    """``W [d_out, d_in]``, target ``X [n_x, d_in]``, retained ``Y [n_y, d_in]``
    and the direction names. ``groups[i]`` is ``"x"``, ``"shared"`` or ``"y"``."""
    W: torch.Tensor
    X: torch.Tensor
    Y: torch.Tensor
    directions: list
    groups: list
    noise: float

    def mu(self, mu_per_token: float) -> float:
        """The solver's ``mu`` for a dose ``mu_per_token`` on these two sets."""
        return dose_from_mu_per_token(mu_per_token, self.X.shape[0], self.Y.shape[0]).mu

    def index(self, group: str) -> list:
        return [i for i, g in enumerate(self.groups) if g == group]


def _features(n, k, gen):
    """``n`` samples of ``k`` independent features: sign-changing, about unit scale."""
    phase = 2 * torch.pi * torch.rand(n, k, generator=gen, dtype=torch.float64)
    amp = 0.6 + 0.4 * torch.rand(n, k, generator=gen, dtype=torch.float64)
    return amp * torch.sin(phase)


def two_sets(n_x=1000, n_y=1000, *, n_x_only=2, n_shared=1, n_y_only=2, d_out=4, noise=0.02,
             seed=0) -> TwoSets:
    """Sample the toy. ``noise`` is the small activity every sample has on
    every direction (so nothing is exactly zero, as in a real model)."""
    gen = torch.Generator().manual_seed(seed)
    d = n_x_only + n_shared + n_y_only
    groups = ["x"] * n_x_only + ["shared"] * n_shared + ["y"] * n_y_only
    names = ([f"x{i}" for i in range(n_x_only)]
             + (["shared"] if n_shared == 1 else [f"shared{i}" for i in range(n_shared)])
             + [f"y{i}" for i in range(n_y_only)])
    X = noise * torch.randn(n_x, d, generator=gen, dtype=torch.float64)
    Y = noise * torch.randn(n_y, d, generator=gen, dtype=torch.float64)
    X[:, :n_x_only + n_shared] += _features(n_x, n_x_only + n_shared, gen)
    Y[:, n_x_only:] += _features(n_y, n_shared + n_y_only, gen)
    W = torch.randn(d_out, d, generator=gen, dtype=torch.float64) / d ** 0.5
    return TwoSets(W=W, X=X, Y=Y, directions=names, groups=groups, noise=noise)


def carried(W, component) -> torch.Tensor:
    """Per input direction, the share of ``W``'s output that the component
    carries: ``|M e_i| / |W e_i|``. About 1 where the component keeps a
    direction, about 0 where it leaves it to ``W - M``."""
    M = component.dense() if hasattr(component, "dense") else component
    return M.norm(dim=0) / W.norm(dim=0).clamp_min(1e-300)


def activation(toy: TwoSets, in_x: float, in_y: float, phase: float = 0.0) -> torch.Tensor:
    """The activation of one point that is ``in_x`` inside the target set and
    ``in_y`` inside the retained set (each in [0, 1]); ``phase`` moves it along
    the sets. A point in both is active on every direction."""
    d = len(toy.groups)
    k = torch.arange(1, d + 1, dtype=torch.float64)
    base = (0.6 + 0.25 * torch.sin(1.7 * k + phase)) * torch.where(k % 2 == 0, -1.0, 1.0)
    w = torch.tensor([in_x if g == "x" else in_y if g == "y" else max(in_x, in_y)
                      for g in toy.groups], dtype=torch.float64)
    return w * base
