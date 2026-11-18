"""SRSPD: the supervised variant.

Given examples of a behaviour to remove (the **target** set) and examples of
behaviour to keep (the **retained** set), SRSPD finds, for every selected
linear layer ``W`` of a model, the component ``M`` of ``W`` that the target
uses and the retained set does not,

    M = W X^T X (X^T X + mu Y^T Y)^+

in closed form from the two sets' activations ``X`` and ``Y``. Subtracting it,
``W <- W - M``, removes the behaviour.

    from rspd import SRSPD

    srspd = SRSPD(model, tokenizer)                      # attach
    components = srspd.fit(target=dutch, retained=english, mu_per_token=1.0)
    components.apply()                                   # W <- W - M
    components.leaks                                     # per matrix
    components.restore()                                 # bit-for-bit undo
"""
from __future__ import annotations

from .api import components_of, erase
from .core.types import Components
from .models.inspect import inspect_model

__all__ = ["SRSPD"]


class SRSPD:
    """Supervised RSPD on a Hugging Face model.

    Parameters
    ----------
    model : a loaded model in ``eval()`` mode, on any device and dtype. Nothing
        is changed until :meth:`Components.apply`.
    tokenizer : needed only when the sets are given as text.
    layers : which linear layers to decompose (``None``: every block's
        attention and MLP projections; see :meth:`inspect`).
    solver : ``"readout"`` (the target activations are held; supports rank
        rules) or ``"grams"`` (for a target set too large to hold).
    device, gram_budget_gib, scratch_dir, subtract_bias, progress : where the
        solves run, how much memory the accumulated statistics may take,
        where to spill to disk, whether biased layers are accepted, and a
        ``progress(event, info)`` callback.
    """

    def __init__(self, model, tokenizer=None, *, layers=None, solver="readout", device=None,
                 gram_budget_gib=None, scratch_dir=None, subtract_bias=False, progress=None):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.solver = solver
        self.options = dict(device=device, gram_budget_gib=gram_budget_gib,
                            scratch_dir=scratch_dir, subtract_bias=subtract_bias,
                            progress=progress)

    def inspect(self) -> dict:
        """What :meth:`fit` would decompose on this model: the matrices, the
        inputs they share, the memory the statistics need, and anything unusual
        about the architecture."""
        return inspect_model(self.model, self.layers,
                             subtract_bias=self.options["subtract_bias"],
                             gram_budget_gib=self.options["gram_budget_gib"])

    def fit(self, target, retained, mu_per_token, *, rank_rule=None, ridge_rel=0.0,
            spectrum="top", **solve_options) -> Components:
        """Solve for the component of every selected matrix that ``target``
        uses and ``retained`` does not. The model is not changed.

        Parameters
        ----------
        target, retained : the behaviour to remove and the behaviour to keep:
            lists of strings (needs the tokenizer), or token ids.
        mu_per_token : the dose, a number > 0 or a list of them; larger removes
            more. Every dose in the list is solved from one pass over the data.
        rank_rule : ``None`` (full rank), a relative threshold in (0, 1),
            ``"erank"``, or a list of these: also solve rank-truncated
            components.
        ridge_rel, spectrum, solve_options : passed to the solver (see
            ``rspd.api.erase``).

        Returns
        -------
        :class:`rspd.Components`, one :class:`rspd.Component` per matrix. Its
        settings are ``(mu_per_token, rank_rule)`` pairs; ``components.apply()``
        subtracts the first dose (pass ``apply(mu_per_token)`` or
        ``apply((mu_per_token, rank_rule))`` for another), ``components.report``
        is the full record.
        """
        report = erase(self.model, self.tokenizer, target=target, retained=retained,
                       mu_per_token=mu_per_token, apply=False, layers=self.layers,
                       solver=self.solver, rank_rule=rank_rule, ridge_rel=ridge_rel,
                       spectrum=spectrum, **self.options, **solve_options)
        return components_of(report)
