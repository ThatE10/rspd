"""The output type of every RSPD variant: :class:`Component` and :class:`Components`.

A :class:`Component` is one low-rank piece ``A @ B`` of one weight matrix. A
:class:`Components` is the ordered collection a fit returns. The supervised
variant (SRSPD) returns one component per matrix, the erased component ``M``;
a variant that splits a matrix into several pieces returns several per matrix
in the same container.

Nothing here knows about models: tensors in, tensors out. A fit on a live
model attaches a *backend* (any object with ``component``, ``leaks``,
``apply``, ``restore``, ``close``), which is how :meth:`Components.apply`
reaches the weights.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch

__all__ = ["Component", "Components"]

_UNSET = object()


class Component(NamedTuple):
    """One component ``M = A @ B`` of a weight matrix ``W [d_out, d_in]``:
    ``A [d_out, r]``, ``B [r, d_in]``. For SRSPD it is the erased component (or
    its rank-``r`` truncation) and the only thing ``apply_erasure`` subtracts
    from a weight. ``info`` carries the solve diagnostics and, when the
    component came from a fit on a model, ``name`` (the matrix), the dose and
    the leaks."""
    A: torch.Tensor
    B: torch.Tensor
    info: dict

    def dense(self) -> torch.Tensor:
        """``A @ B`` as one ``[d_out, d_in]`` tensor."""
        return self.A @ self.B

    @property
    def name(self) -> str | None:
        """The weight matrix this component belongs to."""
        return self.info.get("name")

    @property
    def rank(self) -> int:
        """Number of factor columns (an upper bound on the rank of ``A @ B``)."""
        return int(self.A.shape[1])

    @property
    def leaks(self) -> dict:
        """``{"L_X_rel", "L_Y_rel"}`` when measured: the share of the target
        response that survives subtracting this component, and the share of
        the retained response it disturbs."""
        return dict(self.info.get("leaks") or {})


class Components:
    """The ordered components of a fit.

    ``components[name]`` is the :class:`Component` of matrix ``name`` (a list
    when the matrix has several), ``components[i]`` the ``i``-th in order;
    iterating yields them one at a time, so a large model is never held twice.
    ``names`` lists the matrices.

    A fit may solve several *settings* (doses, rank rules). ``setting`` is the
    one this object reads and applies; :meth:`at` returns the same components
    at another setting without solving again.

    On a live model :meth:`apply` subtracts the components from the weights
    (``W <- W - A @ B``) and :meth:`restore` puts the weights back bit for bit.
    Without a model, :meth:`apply_to` does the same on tensors.
    """

    def __init__(self, entries, *, backend=None, settings=(None,), setting=_UNSET, report=None):
        #: ``[(matrix name, index within the matrix)]`` in order.
        self._entries = [(e, 0) if isinstance(e, str) else tuple(e) for e in entries]
        self._backend = backend
        self._store: dict = {}
        self.settings = list(settings)
        self.setting = self.settings[0] if setting is _UNSET else setting
        #: the full record of the fit (variant-specific), or ``None``.
        self.report = report

    # -- construction from tensors -------------------------------------------
    @classmethod
    def from_components(cls, components, **kw) -> "Components":
        """A collection holding ``components`` (each with ``info["name"]``) in memory."""
        comps = list(components)
        count: dict = {}
        entries = []
        for c in comps:
            if c.name is None:
                raise ValueError("every Component needs info['name'] (its weight matrix)")
            entries.append((c.name, count.get(c.name, 0)))
            count[c.name] = count.get(c.name, 0) + 1
        out = cls(entries, **kw)
        out._store = dict(zip(entries, comps))
        return out

    # -- the collection --------------------------------------------------------
    @property
    def names(self) -> list:
        """The weight matrices, in order, each once."""
        return list(dict.fromkeys(n for n, _ in self._entries))

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name) -> bool:
        return any(n == name for n, _ in self._entries)

    def _load(self, entry) -> Component:
        if entry in self._store:
            return self._store[entry]
        if self._backend is None:
            raise RuntimeError("these Components hold no tensors and no live fit")
        return self._backend.component(entry[0], entry[1], self.setting)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._load(self._entries[key])
        hits = [e for e in self._entries if e[0] == key]
        if not hits:
            raise KeyError(f"{key!r} is not a fitted matrix; fitted: {self.names[:6]}"
                           f"{' ...' if len(self.names) > 6 else ''}")
        return self._load(hits[0]) if len(hits) == 1 else [self._load(e) for e in hits]

    def __iter__(self):
        for e in self._entries:
            yield self._load(e)

    def __repr__(self) -> str:
        return (f"Components({len(self)} components of {len(self.names)} matrices, "
                f"setting={self.setting!r}, applied={self.applied})")

    # -- settings --------------------------------------------------------------
    def _resolve(self, setting):
        if setting in self.settings:
            return setting
        resolve = getattr(self._backend, "resolve", None)
        if resolve is not None:
            return resolve(setting, self.settings)
        raise ValueError(f"setting {setting!r} was not solved; solved: {self.settings}")

    def at(self, setting) -> "Components":
        """These components at another solved ``setting`` (shares the fit)."""
        setting = self._resolve(setting)
        out = Components(self._entries, backend=self._backend, settings=self.settings,
                         setting=setting, report=self.report)
        out._store = self._store if len(self.settings) == 1 else {}
        return out

    # -- measurements ----------------------------------------------------------
    @property
    def leaks(self) -> dict:
        """``{matrix: {"L_X_rel", "L_Y_rel"}}`` at :attr:`setting`."""
        if self._backend is not None:
            return self._backend.leaks(self.setting)
        return {c.name: c.leaks for c in self}

    # -- editing ---------------------------------------------------------------
    @property
    def applied(self) -> bool:
        """Whether these components are currently subtracted from a live model."""
        return bool(self._backend is not None and self._backend.applied_setting() is not None)

    def _need_backend(self):
        if self._backend is None:
            raise RuntimeError("these Components are not attached to a model; use "
                               "apply_to({name: W}) on tensors")
        return self._backend

    def apply(self, setting=_UNSET) -> "Components":
        """Subtract the components from the model's weights, ``W <- W - A @ B``
        (restoring any edit this fit applied before). ``setting`` selects
        another solved setting and becomes :attr:`setting`. Returns ``self``."""
        if setting is not _UNSET:
            self.setting = self._resolve(setting)
        self._need_backend().apply(self.setting)
        return self

    def restore(self) -> list:
        """Put the model's weights back, bit for bit. Returns the restored names."""
        return self._need_backend().restore()

    def close(self) -> None:
        """Restore the model and free what the fit holds."""
        if self._backend is not None:
            self._backend.close()

    def apply_to(self, weights: dict) -> dict:
        """``{name: W - sum of the components of name}`` for the tensors in
        ``weights`` (new tensors, in each ``W``'s dtype). No model involved."""
        out = {}
        for name, W in weights.items():
            comps = self[name]
            comps = comps if isinstance(comps, list) else [comps]
            dt = torch.promote_types(W.dtype, comps[0].A.dtype)
            new = W.to(dtype=dt, copy=True)
            for c in comps:
                new.sub_(c.A.to(device=W.device, dtype=dt) @ c.B.to(device=W.device, dtype=dt))
            out[name] = new.to(W.dtype)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        self.close()
