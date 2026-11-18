"""Gram bookkeeping: sizes, the pass planner, the fp64 Gram sink, tensor spills.

A float64 Gram is ``8 d_in^2`` bytes; several projections share one input
(:func:`rspd.models.harness.input_key`), so one Gram is kept per INPUT, not per
matrix. When the Grams of every input do not fit the memory left beside a
resident model, the corpus is streamed once per *pass*, each pass
accumulating a contiguous group of inputs (:func:`plan_gram_groups`).
"""
from __future__ import annotations

import json
import os
import re
import time

import torch

__all__ = ["gram_bytes", "plan_gram_groups", "Fp64GramSink", "TensorSpill", "MemorySpill"]


def gram_bytes(d_in: int, bytes_per_elem: int = 8) -> int:
    """Bytes of one ``d_in x d_in`` Gram at ``bytes_per_elem`` (8 = float64)."""
    return int(d_in) * int(d_in) * int(bytes_per_elem)


def plan_gram_groups(keys_with_dim_and_device, budget_bytes_per_device,
                     bytes_per_elem: int = 8) -> list:
    """Split the inputs into passes whose Grams fit a per-device byte budget.

    ``keys_with_dim_and_device``: iterable of ``(key, d_in, device)`` in LAYER
    ORDER (the order the forward reaches them). Greedy in that order: a pass
    takes keys until the next key on some device would overflow that device's
    budget; that device is then closed for the pass (every pass is a
    contiguous run of each device's keys, so its early exit is as shallow as
    possible) while keys on other devices may still join. Every key lands in
    exactly one pass; a key larger than the budget raises (no pass can hold
    it). ``bytes_per_elem`` is 8 for one float64 Gram per key, 16 for two
    (the ``grams`` solver holds ``G_X`` and ``G_Y``).

    Returns a list of passes, each a list of keys in layer order.
    """
    items = [(str(k), int(d), str(dev)) for k, d, dev in keys_with_dim_and_device]
    budget = int(budget_bytes_per_device)
    seen = set()
    for k, d, dev in items:
        if k in seen:
            raise ValueError(f"key {k!r} listed twice")
        seen.add(k)
        b = gram_bytes(d, bytes_per_elem)
        if b > budget:
            raise ValueError(
                f"input {k!r} (d_in {d}) needs {b} bytes ({b / 2**30:.3g} GiB) of Gram, more "
                f"than the {budget} bytes ({budget / 2**30:.3g} GiB) per-device budget on "
                f"{dev}; no pass can hold it. "
                f"Raise gram_budget_gib or free device memory.")
    remaining = items
    passes = []
    while remaining:
        used: dict = {}
        closed = set()
        cur, rest = [], []
        for k, d, dev in remaining:
            b = gram_bytes(d, bytes_per_elem)
            if dev not in closed and used.get(dev, 0) + b <= budget:
                cur.append(k)
                used[dev] = used.get(dev, 0) + b
            else:
                closed.add(dev)
                rest.append((k, d, dev))
        passes.append(cur)
        remaining = rest
    return passes


class Fp64GramSink:
    """The backend-agnostic Gram sink (:class:`rspd.capture.base.GramSink`):
    ``add(key, x)`` with ``x`` the ``[tokens, d_in]`` activations of one
    input; accumulates ``G[key] += x^T x`` IN PLACE in float64 on the key's
    device (``x`` is upcast there; nothing outlives the call).

    ``grams[key]`` is the float64 ``[d_in, d_in]`` Gram, ``tokens[key]`` the
    row count; :meth:`result` returns ``{key: (G, n_tokens)}``. ``gram_s`` is
    the time spent inside the updates (CUDA events on GPU, synchronised by
    :meth:`sync`).
    """

    def __init__(self, dims, device_of_key):
        self.grams = {k: torch.zeros(int(d), int(d), dtype=torch.float64,
                                     device=device_of_key[k]) for k, d in dims.items()}
        self.tokens = {k: 0 for k in dims}
        self.gram_s = 0.0
        self._events: list = []
        self._cuda = any(g.device.type == "cuda" for g in self.grams.values())

    def add(self, key, x) -> None:
        g = self.grams[key]
        if self._cuda and g.device.type == "cuda":
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            xd = x.to(device=g.device, dtype=torch.float64)
            g.addmm_(xd.T, xd)
            e1.record()
            self._events.append((e0, e1))
        else:
            t0 = time.perf_counter()
            xd = x.to(device=g.device, dtype=torch.float64)
            g.addmm_(xd.T, xd)
            self.gram_s += time.perf_counter() - t0
        del xd
        self.tokens[key] += int(x.shape[0])

    def sync(self) -> None:
        if self._events:
            torch.cuda.synchronize()
            self.gram_s += sum(e0.elapsed_time(e1) for e0, e1 in self._events) / 1000.0
            self._events.clear()

    def result(self) -> dict:
        self.sync()
        return {k: (self.grams[k], self.tokens[k]) for k in self.grams}


class TensorSpill:
    """One ``.pt`` file per tensor under ``root``, keyed by a tag (atomic
    writes), plus an optional ``manifest.json``. Keeps readouts, truncated
    factors and dense components off the device and out of host RAM."""

    def __init__(self, root):
        self.root = str(root)
        os.makedirs(self.root, exist_ok=True)

    def path(self, tag: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.+\-]", "_", str(tag))
        return os.path.join(self.root, safe + ".pt")

    def put(self, tag, tensor) -> str:
        p = self.path(tag)
        tmp = p + ".tmp"
        torch.save(tensor.detach().to("cpu").contiguous(), tmp)
        os.replace(tmp, p)
        return p

    def get(self, tag, *, device=None, dtype=None):
        t = torch.load(self.path(tag), map_location="cpu", weights_only=True)
        if device is not None or dtype is not None:
            t = t.to(device=device or t.device, dtype=dtype or t.dtype)
        return t

    def has(self, tag) -> bool:
        return os.path.exists(self.path(tag))

    def write_manifest(self, manifest: dict) -> str:
        p = os.path.join(self.root, "manifest.json")
        tmp = p + ".tmp"
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1, default=str)
        os.replace(tmp, p)
        return p

    def read_manifest(self):
        p = os.path.join(self.root, "manifest.json")
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)


class MemorySpill:
    """:class:`TensorSpill`'s interface over a dict of CPU tensors (no
    ``scratch_dir``). A CPU round trip, like the disk one: bit-for-bit."""

    def __init__(self):
        self._d: dict = {}

    def put(self, tag, tensor):
        self._d[str(tag)] = tensor.detach().to("cpu", copy=True).contiguous()
        return str(tag)

    def get(self, tag, *, device=None, dtype=None):
        t = self._d[str(tag)]
        return t.to(device=device or t.device, dtype=dtype or t.dtype, copy=True)

    def has(self, tag) -> bool:
        return str(tag) in self._d

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self._d.values())
