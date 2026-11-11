"""The HF forward-hook activation source.

* :func:`harvest_inputs_native` / :func:`harvest_inputs_concat` -- the target
  activations ``X`` held on the host, input-only, in the dtype the layer
  received (lossless), one tensor per shared input.
* :class:`HFHookSource` -- streams sequences through the model and feeds each
  shared input's activations to a :class:`~srspd.capture.base.GramSink`, with

  - **early exit**: the forward stops (:class:`_StopForward`) at the deepest
    hooked input of the pass;
  - **boundary resume**: the residual stream entering the next pass's first
    block is cached per sequence (:class:`BoundaryCache`), and the next pass
    patches the earlier blocks to identities and restarts from it, so every
    block runs once per sequence over the whole plan. Chunk 0 is re-run as an
    ordinary forward and must reproduce the resumed inputs bitwise, else
    :class:`ResumeGuardError`;
  - **shared-input check** (quirk ``shared_input_check``): on the first
    sequence every member of every key is hooked and must receive a bitwise
    identical input to the key's representative;
  - TF32 forced off for the duration, every hook and patch removed in a
    ``finally``.
"""
from __future__ import annotations

import itertools
import re
import time

import torch

from ..models.harness import group_by_input, key_layer

__all__ = ["harvest_inputs_native", "harvest_inputs_concat", "decoder_layer_list",
           "BoundaryCache", "ResumeGuardError", "HFHookSource", "forward_ids"]


def forward_ids(model, ids, **kwargs):
    """``model(input_ids=ids, **kwargs)`` for a tensor, ``model(**ids, **kwargs)`` for a dict."""
    if isinstance(ids, torch.Tensor):
        return model(input_ids=ids, **kwargs)
    return model(**ids, **kwargs)


def harvest_inputs_native(harness, ids, names=None, *, expect_dtype=None) -> tuple:
    """One forward over ``ids``; ``{name: X [tokens, d_in]}`` on CPU in the dtype
    the layer actually received, plus a dedup report.

    The members of one input key are stored ONCE when their inputs are bitwise
    identical. ``expect_dtype`` asserts the captured dtype (storing a bf16
    activation as bf16 is lossless). Returns ``(data, report)`` with
    ``report = {"shared": {key: [names]}, "mismatch": {key: [names whose input
    differed from the key's first]}, "dtype": str}``.
    """
    names = list(names if names is not None else harness.all_names)
    store: dict = {}
    dtypes = set()

    def make_hook(n):
        def hook(_mod, inputs, _output):
            x = inputs[0].detach().flatten(0, -2)
            dtypes.add(x.dtype)
            # a CPU model hands over the live tensor; never keep a view of it
            store[n] = x.clone() if x.device.type == "cpu" else x.to("cpu")
        return hook

    handles = [harness.module_by_name[n].register_forward_hook(make_hook(n)) for n in names]
    try:
        with torch.no_grad():
            harness.model(input_ids=ids)
    finally:
        for hd in handles:
            hd.remove()
    missing = [n for n in names if n not in store]
    if missing:
        raise RuntimeError(f"harvest hooks never fired for {missing[:4]}")
    if expect_dtype is not None and dtypes != {expect_dtype}:
        raise AssertionError(f"captured activation dtype(s) {sorted(map(str, dtypes))} != "
                             f"expected {expect_dtype}: storing them as {expect_dtype} would "
                             f"not be lossless")
    shared, mismatch = {}, {}
    for key, members in group_by_input(names).items():
        first = store[members[0]]
        for m in members[1:]:
            if torch.equal(store[m], first):
                store[m] = first                     # alias: one copy per shared input
                shared.setdefault(key, [members[0]]).append(m)
            else:
                mismatch.setdefault(key, [members[0]]).append(m)
    return store, {"shared": shared, "mismatch": mismatch,
                   "dtype": ",".join(sorted(map(str, dtypes)))}


def harvest_inputs_concat(harness, ids_list, *, expect_dtype=None, names=None) -> tuple:
    """:func:`harvest_inputs_native` over several sequences, each its own
    forward (no padding), the rows concatenated in order. Returns
    ``(data, n_tokens)``; members of one input key share ONE tensor. A sequence
    on which a key's members saw different inputs raises. An empty ``ids_list``
    returns ``({}, 0)``."""
    ids_list = list(ids_list)
    if not ids_list:
        return {}, 0
    names = list(names if names is not None else harness.all_names)
    groups = group_by_input(names)
    parts = {members[0]: [] for members in groups.values()}
    n = 0
    for ids in ids_list:
        data, report = harvest_inputs_native(harness, ids, names, expect_dtype=expect_dtype)
        if report["mismatch"]:
            raise AssertionError(f"harvest: matrices sharing an input key received different "
                                 f"inputs {report['mismatch']}; one Gram / one readout would "
                                 f"be wrong for them")
        for members in groups.values():
            parts[members[0]].append(data[members[0]])
        n += int(ids.shape[-1])
        del data
    out = {}
    for members in groups.values():
        X = torch.cat(parts.pop(members[0]), dim=0)
        for m in members:
            out[m] = X
    return out, n


class _StopForward(Exception):
    """Raised from the last hook a pass needs; caught by the pass loop."""


class ResumeGuardError(RuntimeError):
    """The forward resumed from the cached boundary did not reproduce the full
    forward bitwise. Raised, never swallowed: pass ``resume_from_cache=False``."""


def decoder_layer_list(harness):
    """The ``ModuleList`` holding the blocks the harness's matrices sit in
    (``...layers.<N>.`` or GPT-2's ``...h.<N>.``), indexed by block number.
    Every target must live in the same container."""
    containers = set()
    for n in harness.all_names:
        m = re.match(r"^(.*?(?:layers|(?:^|\.)h))\.(\d+)\.", harness.path_by_name[n])
        if m is None:
            raise ValueError(f"cannot find the block container of {n} "
                             f"({harness.path_by_name[n]}); pass resume_from_cache=False")
        containers.add(m.group(1))
    if len(containers) != 1:
        raise ValueError(f"targets span several block containers: {sorted(containers)}")
    return harness.model.get_submodule(containers.pop())


class BoundaryCache:
    """The residual stream entering block ``start``, one CPU tensor per
    sequence, in order -- exactly the tensor that block received (same dtype
    and shape), so a restart from it is bitwise. Written by pass ``k``, read
    by pass ``k + 1``; slot ``j`` is read before it is overwritten, so only one
    boundary is resident. ``out_kind`` records what a block returns (a tensor
    or a tuple) so a skipped block can mimic it."""

    def __init__(self):
        self.start = None
        self.chunks: list = []
        self.out_kind = None

    @property
    def nbytes(self) -> int:
        return sum(int(t.numel()) * t.element_size() for t in self.chunks if t is not None)

    def clear(self):
        self.start = None
        self.chunks = []


def _identity_forward(kind):
    def forward(*args, **kwargs):
        hs = args[0] if args else kwargs["hidden_states"]
        if kind and kind[0] == "tuple":
            return (hs,) + (None,) * (int(kind[1]) - 1)
        return hs
    return forward


def _patch_identity(layers, upto, kind) -> list:
    """Make ``layers[:upto]`` identities (instance attributes only; a
    pre-existing instance ``forward`` is restored by :func:`_unpatch`)."""
    saved = []
    for li in range(int(upto)):
        lay = layers[li]
        saved.append((lay, "forward" in lay.__dict__, lay.__dict__.get("forward")))
        lay.forward = _identity_forward(kind)
    return saved


def _unpatch(saved) -> None:
    for lay, had, orig in saved:
        if had:
            lay.forward = orig
        else:
            lay.__dict__.pop("forward", None)


def _hidden_arg(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


def _with_hidden(args, kwargs, value):
    if args:
        return (value,) + tuple(args[1:]), kwargs
    kw = dict(kwargs)
    kw["hidden_states"] = value
    return args, kw


def _noop(_event, _info):
    return None


class HFHookSource:
    """:class:`~srspd.capture.base.ActivationSource` over an
    :class:`~srspd.models.harness.HFLinearHarness` by forward hooks."""

    def __init__(self, harness):
        self.harness = harness

    def accumulate(self, sequences, keys_to_rep, sink, *, members_of_key=None,
                   verify_shared=True, n_chunks=None, label="gram", forward_kwargs=None,
                   layers=None, cache=None, next_start=None, progress=None,
                   progress_every=100) -> dict:
        """Stream ``sequences`` once, calling ``sink.add(key, x)`` per key and
        sequence with the input of ``keys_to_rep[key]``. Returns stats
        (``chunks``, ``tokens``, ``wall_s``, ``forward_s``, ``guard_s``,
        ``hooked``, ``resumed``, ``start_layer``, ``deepest_layer``,
        ``layers_forwarded``, ``cache_gib``).

        ``layers`` + ``cache`` enable the boundary resume (see the module
        docstring); ``next_start`` makes this pass store the stream entering
        block ``next_start`` for the next pass. ``forward_kwargs`` default
        ``{"use_cache": False}``. ``progress(event, info)`` receives
        ``"chunk"`` events every ``progress_every`` sequences.
        """
        h = self.harness
        progress = progress or _noop
        keys = list(keys_to_rep)
        reps = {k: str(keys_to_rep[k]) for k in keys}
        if members_of_key is None:
            grouped = group_by_input(h.all_names)
            members_of_key = {k: grouped.get(k, [reps[k]]) for k in keys}
        fkw = {"use_cache": False} if forward_kwargs is None else dict(forward_kwargs)
        resume = bool(layers is not None and cache is not None and cache.start
                      and int(cache.start) > 0)
        start = int(cache.start) if resume else 0
        if resume and len(cache.chunks) == 0:
            raise ValueError("resume requested but the boundary cache is empty")
        rep_layers = [key_layer(k) for k in keys]
        if resume and min(rep_layers) < start:
            raise ValueError(f"pass reads layer {min(rep_layers)} but resumes at layer {start}")
        capture_at = None
        if layers is not None and cache is not None and next_start is not None:
            if int(next_start) - 1 >= start:
                capture_at = int(next_start) - 1
            elif int(next_start) != start:
                raise ValueError(f"next pass starts at layer {next_start}, before this one "
                                 f"({start})")

        pending: set = set()
        fired: set = set()
        stash: dict = {}
        ntok = {k: 0 for k in keys}
        phase = {"chunk": 0}
        BOUNDARY = "__boundary__"

        def maybe_stop():
            if not pending:
                raise _StopForward

        def rep_hook(k):
            rep = reps[k]

            def hook(_mod, inputs, _output):
                if rep in fired:
                    raise RuntimeError(f"{rep} ran twice in one forward; its Gram would "
                                       f"double count the sequence")
                fired.add(rep)
                x = inputs[0].detach().flatten(0, -2)
                if phase["chunk"] == 0 and (verify_shared or resume):
                    stash[rep] = x.clone()
                sink.add(k, x)
                ntok[k] += int(x.shape[0])
                pending.discard(rep)
                maybe_stop()
            return hook

        def member_hook(n):
            def hook(_mod, inputs, _output):
                if n not in stash:
                    stash[n] = inputs[0].detach().flatten(0, -2).clone()
                pending.discard(n)
                maybe_stop()
            return hook

        def boundary_hook(_mod, _inputs, output):
            ci = phase["chunk"]
            if isinstance(output, tuple):
                cache.out_kind = ("tuple", len(output))
                hs = output[0]
            else:
                cache.out_kind = ("tensor",)
                hs = output
            t = hs.detach().to("cpu", copy=True)
            if ci < len(cache.chunks):
                cache.chunks[ci] = t
            else:
                cache.chunks.append(t)
            pending.discard(BOUNDARY)
            maybe_stop()

        def resume_pre_hook(_mod, args, kwargs):
            cur = _hidden_arg(args, kwargs)
            t = cache.chunks[phase["chunk"]]
            if tuple(t.shape) != tuple(cur.shape) or t.dtype != cur.dtype:
                raise ResumeGuardError(
                    f"[{label}] cached boundary for sequence {phase['chunk']} is "
                    f"{tuple(t.shape)} {t.dtype}, layer {start} expects {tuple(cur.shape)} "
                    f"{cur.dtype}")
            return _with_hidden(args, kwargs, t.to(cur.device))

        def run_forward(ids):
            try:
                forward_ids(h.model, ids, **fkw)
            except _StopForward:
                pass

        it = iter(sequences)
        guard_ref, guard_s = {}, 0.0
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        handles, extra, patched = [], [], []
        t_start = time.perf_counter()
        n_done = 0
        try:
            with torch.no_grad():
                if resume:
                    first = next(it)
                    it = itertools.chain([first], it)
                    tg = time.perf_counter()

                    def ref_hook(n):
                        def hook(_mod, inputs, _output):
                            guard_ref.setdefault(n, inputs[0].detach().flatten(0, -2).clone())
                            pending.discard(n)
                            maybe_stop()
                        return hook
                    pending.clear()
                    pending.update(reps.values())
                    gh = [h.module_by_name[r].register_forward_hook(ref_hook(r))
                          for r in reps.values()]
                    try:
                        run_forward(first)
                    finally:
                        for hd in gh:
                            hd.remove()
                    if pending:
                        raise RuntimeError(f"[{label}] guard forward never reached "
                                           f"{sorted(pending)}")
                    guard_s = time.perf_counter() - tg
                    patched = _patch_identity(layers, start, cache.out_kind)
                    handles.append(layers[start].register_forward_pre_hook(
                        resume_pre_hook, with_kwargs=True))
                handles += [h.module_by_name[reps[k]].register_forward_hook(rep_hook(k))
                            for k in keys]
                if capture_at is not None:
                    handles.append(layers[capture_at].register_forward_hook(boundary_hook))
                if verify_shared:
                    for k in keys:
                        for m in members_of_key.get(k, []):
                            if m != reps[k]:
                                extra.append(h.module_by_name[m].register_forward_hook(
                                    member_hook(m)))
                hooked_all = set(reps.values()) | (
                    {m for k in keys for m in members_of_key.get(k, [])}
                    if verify_shared else set())
                if capture_at is not None:
                    hooked_all.add(BOUNDARY)
                for ci, ids in enumerate(it):
                    phase["chunk"] = ci
                    pending.clear()
                    pending.update(hooked_all if ci == 0 else set(reps.values())
                                   | ({BOUNDARY} if capture_at is not None else set()))
                    fired.clear()
                    run_forward(ids)
                    if pending:
                        raise RuntimeError(f"[{label}] hooks never fired on sequence {ci}: "
                                           f"{sorted(pending)[:6]}")
                    if hasattr(sink, "sync"):
                        sink.sync()
                    if ci == 0:
                        if resume:
                            bad = [r for r in reps.values() if not torch.equal(
                                stash[r], guard_ref[r].to(stash[r].device))]
                            if bad:
                                raise ResumeGuardError(
                                    f"[{label}] resuming at layer {start} from the cached "
                                    f"boundary did NOT reproduce the full forward bitwise for "
                                    f"{bad[:4]} (sequence 0); pass resume_from_cache=False")
                        if verify_shared:
                            bad = []
                            for k in keys:
                                ref = stash[reps[k]]
                                for m in members_of_key.get(k, []):
                                    if m != reps[k] and not torch.equal(stash[m].to(ref.device),
                                                                        ref):
                                        bad.append((k, m))
                            for hd in extra:
                                hd.remove()
                            extra = []
                            if bad:
                                raise AssertionError(
                                    f"[{label}] matrices sharing an input key did NOT receive "
                                    f"bitwise-identical inputs: {bad[:6]}; one Gram cannot "
                                    f"stand in for them (srspd.models.harness.input_key)")
                        stash.clear()
                        guard_ref.clear()
                    n_done = ci + 1
                    if progress_every and n_done % int(progress_every) == 0:
                        progress("chunk", {"label": label, "done": n_done, "of": n_chunks,
                                           "wall_s": time.perf_counter() - t_start})
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32
            for hd in handles + extra:
                hd.remove()
            _unpatch(patched)
        if cache is not None and capture_at is not None:
            if len(cache.chunks) != n_done:
                raise RuntimeError(f"[{label}] boundary cache holds {len(cache.chunks)} "
                                   f"sequences after {n_done}")
            cache.start = int(next_start)
        wall = time.perf_counter() - t_start
        gram_s = float(getattr(sink, "gram_s", 0.0))
        deepest = max(rep_layers + ([capture_at] if capture_at is not None else []))
        return dict(chunks=n_done, tokens=(ntok[keys[0]] if keys else 0), wall_s=round(wall, 3),
                    gram_s=round(gram_s, 3),
                    forward_s=round(max(wall - gram_s - guard_s, 0.0), 3),
                    guard_s=round(guard_s, 3), hooked=len(keys), resumed=bool(resume),
                    start_layer=start, deepest_layer=deepest,
                    layers_forwarded=deepest - start + 1,
                    cache_gib=(round(cache.nbytes / 2 ** 30, 3) if cache is not None else 0.0))
