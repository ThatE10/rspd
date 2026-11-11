"""Weight edits with a bit-for-bit restore.

:class:`PristineStore` backs each edited matrix up ONCE, before its first
write, in the module's native layout and dtype (so a restore is a byte copy),
either in host RAM (``scratch_dir=None``) or on disk under
``<scratch_dir>/pristine/`` (one file per matrix; for a model whose weights do
not fit twice in host RAM). Every weight write for erasure goes through
:meth:`PristineStore.write`, and :meth:`PristineStore.restore_all` puts every
edited matrix back exactly.
"""
from __future__ import annotations

import os

import torch

from ..capture.grams import MemorySpill, TensorSpill
from .harness import as_dout_din, is_conv1d, write_dout_din

__all__ = ["PristineStore"]


class PristineStore:
    """Disk- or RAM-backed pristine weights for a harness's matrices.

    Parameters
    ----------
    harness : :class:`~srspd.models.harness.HFLinearHarness`.
    scratch_dir : directory for the disk backup, or ``None`` for host RAM.
    """

    def __init__(self, harness, scratch_dir=None):
        self.h = harness
        self.scratch_dir = None if scratch_dir is None else str(scratch_dir)
        self.spill = (MemorySpill() if scratch_dir is None
                      else TensorSpill(os.path.join(str(scratch_dir), "pristine")))
        self.saved: set = set()
        self.edited: set = set()

    def write(self, name, value) -> None:
        """Copy ``value`` (``[d_out, d_in]``) into the live parameter (cast to its
        dtype), backing the pristine weight up first if this is its first edit."""
        mod = self.h.module_by_name[name]
        if name not in self.saved:
            self.spill.put(name, mod.weight.detach())
            self.saved.add(name)
        with torch.no_grad():
            write_dout_din(mod, value)
        self.edited.add(name)

    def pristine(self, name, *, device=None, dtype=None):
        """The pristine ``[d_out, d_in]`` weight, always a fresh copy (never a view)."""
        mod = self.h.module_by_name[name]
        if name in self.saved:
            w = self.spill.get(name)
            w = w.T if is_conv1d(mod) else w
        else:
            w = as_dout_din(mod)
        return w.to(device=device or w.device, dtype=dtype or w.dtype, copy=True)

    def restore_all(self) -> list:
        """Restore every matrix edited since the last restore (idempotent);
        returns their names."""
        names = sorted(self.edited)
        with torch.no_grad():
            for n in names:
                p = self.h.module_by_name[n].weight
                p.data.copy_(self.spill.get(n).to(p.device, p.dtype))
        self.edited.clear()
        return names

    @property
    def is_edited(self) -> bool:
        return bool(self.edited)
