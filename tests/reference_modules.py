"""Keep this package and the reference research package apart in one process.

Both are imported as ``rspd``. The parity tests run the reference driver and
this package side by side, so ``sys.modules`` has to hold one or the other:
:class:`ModuleSwap` stashes the modules of whichever side is not running and
:meth:`ModuleSwap.reference` switches to the reference for the length of a
``with`` block.
"""
from __future__ import annotations

import contextlib
import sys

PKG = "rspd"


def _is_pkg(name, extra):
    return name == PKG or name.startswith(PKG + ".") or name in extra


class ModuleSwap:
    def __init__(self, extra=()):
        self.extra = tuple(extra)
        self._ours = {}
        self._ref = {}

    def _take(self) -> dict:
        names = [m for m in sys.modules if _is_pkg(m, self.extra)]
        return {m: sys.modules.pop(m) for m in names}

    def begin(self):
        """Stash this package's modules; the reference may now be imported."""
        self._ours = self._take()

    def end_import(self):
        """The reference is imported: stash it and put this package back."""
        self._ref = self._take()
        sys.modules.update(self._ours)

    @contextlib.contextmanager
    def reference(self):
        """Run the block with the reference's modules installed."""
        self._ours = self._take()
        sys.modules.update(self._ref)
        try:
            yield
        finally:
            self._ref = self._take()
            sys.modules.update(self._ours)

    def close(self):
        self._take()
        sys.modules.update(self._ours)
        self._ref = {}
