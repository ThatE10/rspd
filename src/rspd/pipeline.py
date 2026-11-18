"""SRSPD orchestration on a resident HF model: Gram passes, per-input solves,
the edit ``W - M`` of every selected matrix at once, leaks per matrix.

Notation (fixed): ``X`` is the TARGET set (erased), ``Y`` the RETAINED set
(kept, known at scale only through its per-input Gram ``G_Y = Y^T Y``), ``W``
the pristine weight, ``M`` the erased component

    M = argmin_M ||W X^T - M X^T||^2 + mu ||M Y^T||^2 = W X^T X G_mu^+,
    G_mu = X^T X + mu Y^T Y.

The dose is a :class:`rspd.dose.Dose` (``mu_per_token`` plus the set sizes);
``Dose.mu`` is what the solves take. Every weight write is ``W_new = W - M``
through :func:`rspd.apply_erasure`, and the leaks
``L_X_rel = ||(W - M) X^T|| / ||W X^T||`` and ``L_Y_rel = ||M Y^T|| / ||W Y^T||``
are measured for every edited matrix.

Two solvers (:data:`SOLVERS`):

``"readout"``
    The target is held as activations ``X`` per shared input; the per-input
    readout ``S = X G_mu^+`` (:func:`rspd.readout_gram`) is spilled in fp32 per
    dose, and ``M = (W X^T) S`` is formed in fp32 at apply time. Supports rank
    truncation of ``M`` (``rank_rule``: COALA projector on ``M`` over ``X``).
``"grams"``
    For a target too large to hold: the target sequences are streamed through
    the same Gram passes, so per input only ``G_X = X^T X`` and ``G_Y`` exist,
    ``M = W G_X (G_X + mu G_Y + delta I)^+`` is solved per matrix
    (:func:`rspd.erased_component_grams`) and spilled dense in fp32 per
    (dose, matrix). No rank truncation (it needs ``M X^T``).

The Gram passes are planned under a per-device byte budget
(:func:`rspd.capture.grams.plan_gram_groups`); with more than one pass, each
pass resumes from the residual stream cached at the previous pass's boundary
(disabled on models with KV-shared layers, quirk ``kv_shared_layers``).

Progress is reported through an optional ``progress(event, info)`` callback
(default: none); nothing here prints.
"""
from __future__ import annotations

import gc
import shutil
import statistics
import time

import torch

from .capture.grams import Fp64GramSink, gram_bytes, plan_gram_groups
from .capture.hf import BoundaryCache, HFHookSource, decoder_layer_list
from .models.harness import key_layer
from .models.selector import matrix_type
from .solve import (ERANK_RULE, ErasedComponent, apply_erasure, erased_component_gram,
                    erased_component_grams, factor_grams, gram_eig_extremes, leaks_gram,
                    leaks_grams, readout_gram, specificity, truncate_component)

__all__ = ["SOLVERS", "CONVENTION", "LEAK_FIELDS", "LAM_SVD_DTYPES", "run_gram_passes",
           "solve_pass", "solve_key_order", "apply_dose", "quad_trace", "leaks_per_matrix",
           "summarize_by_type", "matrix_type", "rank_rule_spec", "scratch_bytes_needed",
           "check_scratch_space", "SolveState", "non_pd_summary", "NON_PD_MESSAGE"]

SOLVERS = ("readout", "grams")
CONVENTION = "erase by subtraction: W_new = W - M (X = target, Y = retained)"
LEAK_FIELDS = ("L_X_rel", "L_Y_rel")
#: dtypes of the one thin SVD of ``M X^T`` that every rank rule of a matrix shares.
LAM_SVD_DTYPES = ("float32", "float64")


#: The phrase every non-PD warning carries.
NON_PD_MESSAGE = "G_mu is not numerically positive definite at the requested ridge"


def _noop(_event, _info):
    return None


def non_pd_summary(records) -> str | None:
    """ONE warning for every non-PD solve of a run (``SolveState.non_pd``):
    counts by solver path, over matrices and over (input key, dose) solves.
    ``None`` when every solve was PD."""
    if not records:
        return None
    paths: dict = {}
    for r in records:
        paths[r["solve_path"]] = paths.get(r["solve_path"], 0) + 1
    keys = sorted({r["input_key"] for r in records})
    n_solves = len({(r["input_key"], r["mu_per_token"]) for r in records})
    by_path = ", ".join(f"{p}: {c}" for p, c in sorted(paths.items()))
    return (f"{NON_PD_MESSAGE} for {len(records)} matrix solve(s) ({by_path}) over "
            f"{n_solves} (input key, dose) solve(s) at {len(keys)} input key(s), e.g. "
            f"{keys[:3]} (common for LayerNorm-fed inputs); per-matrix detail in "
            f"report.non_pd and matrices[name]['solve_path' / 'ridge_delta']")


def _check_solver(solver):
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")


def rank_rule_spec(rule):
    """Validate one rank rule: a ``lam_rel`` float in (0, 1) (keep
    ``r = #{j : sigma_j^2 > lam_rel sigma_1^2}`` of ``M X^T``) or ``"erank"``
    (``r = max(1, round(erank(M X^T)))``). Returns the spec
    :func:`rspd.truncate_component` takes."""
    if rule == ERANK_RULE:
        return ERANK_RULE
    if isinstance(rule, bool) or not isinstance(rule, (int, float)):
        raise ValueError(f"rank_rule must be a lam_rel in (0, 1) or {ERANK_RULE!r}, got {rule!r}")
    v = float(rule)
    if not 0.0 < v < 1.0:
        raise ValueError(f"rank_rule lam_rel must be in (0, 1) (0 is the full-rank edit, "
                         f">= 1 keeps no singular value), got {v:g}")
    return v


def summarize_by_type(per_matrix: dict, fields) -> dict:
    """``{type: {"n", field: {median, min, max}}}`` over the matrices of each type."""
    groups: dict = {}
    for name, d in per_matrix.items():
        groups.setdefault(matrix_type(name), []).append(d)
    out = {}
    for t, ds in sorted(groups.items()):
        out[t] = {"n": len(ds)}
        for f in fields:
            v = [float(d[f]) for d in ds if d.get(f) is not None]
            if v:
                out[t][f] = {"median": float(statistics.median(v)), "min": float(min(v)),
                             "max": float(max(v))}
    return out


def leaks_per_matrix(diags: dict) -> dict:
    """``{matrix: {"L_X_rel", "L_Y_rel"}}`` from per-matrix diagnostics."""
    return {n: {f: d[f] for f in LEAK_FIELDS if d.get(f) is not None}
            for n, d in diags.items() if any(d.get(f) is not None for f in LEAK_FIELDS)}


def scratch_bytes_needed(params_by_name: dict, *, n_lam: int, elem_bytes: int = 2,
                         margin: float = 1.05, n_dense: int = 0,
                         dense_elem_bytes: int = 4) -> int:
    """Scratch bytes a run fills: one pristine copy of every matrix (weight
    dtype, ``elem_bytes``), one weight-sized copy per rank-truncated setting,
    and under ``grams`` one dense fp32 component per dose (``n_dense``)."""
    n = sum(params_by_name.values())
    return int(n * (elem_bytes * (1 + int(n_lam)) + int(dense_elem_bytes) * int(n_dense))
               * margin)


def check_scratch_space(path, need: int, *, free=None) -> int:
    """Raise if ``path``'s filesystem has fewer than ``need`` bytes free; return
    the free byte count (``free`` injectable for tests)."""
    free = shutil.disk_usage(path).free if free is None else int(free)
    if free < need:
        raise RuntimeError(
            f"scratch_dir {path} has {free / 2 ** 30:.2f} GiB free but this erasure spills "
            f"{need / 2 ** 30:.2f} GiB (pristine weights + truncated factors / dense grams-"
            f"solver components). Point scratch_dir at a larger disk, drop rank_rule, or pass "
            f"fewer doses.")
    return free


def quad_trace(D, G, block=4096):
    """``tr(D G D^T)`` in row blocks of ``D`` (never more than ``block x d_in`` of
    temporary)."""
    return float(sum(((D[i:i + block] @ G) * D[i:i + block]).sum()
                     for i in range(0, int(D.shape[0]), block)))


def solve_key_order(pkeys, d_of):
    """Order in which a pass's keys are solved: ascending input width, stable.
    Each key's solve is independent, so results are bitwise identical under any
    order; ascending width frees the smaller Grams before the largest key's
    solve transient is taken."""
    order = {k: i for i, k in enumerate(pkeys)}
    return sorted(pkeys, key=lambda k: (int(d_of[k]), order[k]))


def readout_tag(key, di):
    return f"S__{key}__dose{int(di)}"


def component_tag(di, name):
    return f"M__dose{int(di)}__{name}"


def lam_tags(li, name):
    return f"A__lam{int(li)}__{name}", f"B__lam{int(li)}__{name}"


class SolveState:
    """Everything the solve produced that the edit and the report need.

    ``doses`` : list of :class:`~rspd.dose.Dose`; ``lam_settings`` : list of
    ``{"dose_index", "rule"}``; ``diag_by_dose[i][name]`` : the full-rank leaks
    record of matrix ``name`` at dose ``i``; ``lam_info[li][name]`` : the
    truncated component's record; ``specificity[key]`` : per-input
    conditioning; ``pass_plan``; ``stats``; the spills holding ``S`` /
    truncated factors / dense components; ``X_of`` (readout solver).
    """

    def __init__(self, *, solver, doses, lam_settings, key_members, X_of, readouts,
                 lam_spill, component_spill):
        self.solver = solver
        self.doses = list(doses)
        self.lam_settings = list(lam_settings)
        self.key_members = key_members
        self.X_of = X_of
        self.readouts = readouts
        self.lam_spill = lam_spill
        self.component_spill = component_spill
        self.diag_by_dose = {i: {} for i in range(len(self.doses))}
        self.lam_info = [dict() for _ in self.lam_settings]
        self.specificity: dict = {}
        self.pass_plan: list = []
        self.stats: dict = {"passes": []}
        self.warnings: list = []
        #: one record per (matrix, dose) whose ``G_mu`` was not numerically PD at the
        #: requested ridge (summarised by :func:`non_pd_summary`).
        self.non_pd: list = []


def run_gram_passes(h, state, *, retained, n_y, keys, rep, d_in_of, device_of_key,
                    gram_budget_bytes, resume_from_cache=True, eig_extremes=True,
                    ridge_rel=0.0, lam_svd_dtype="float32", target=None, n_x=None,
                    progress=None, progress_every=100, full_spectrum=False) -> None:
    """Plan the Gram passes under ``gram_budget_bytes`` per device, accumulate
    each pass's float64 retained Grams ``G_Y`` (and, under ``grams``, the target
    Grams ``G_X``) over the sequences, and solve them (:func:`solve_pass`) before
    the next pass starts. Results land in ``state``.

    ``retained`` / ``target`` are zero-argument callables returning a fresh
    iterator of ``[1, T]`` input-id tensors on the model's input device (called
    once per pass); ``n_y`` / ``n_x`` their total token counts.
    ``full_spectrum`` keeps the whole ``gamma`` spectrum per input key
    (readout solver). Every non-PD solve is recorded in ``state.non_pd`` and
    summarised in ONE warning after the last pass.
    """
    progress = progress or _noop
    solver = state.solver
    _check_solver(solver)
    grams_mode = solver == "grams"
    if grams_mode:
        if state.lam_settings:
            raise ValueError("solver 'grams' has no rank truncation (a rank rule needs the "
                             "target response M X^T); drop rank_rule or use solver='readout'")
        if target is None:
            raise ValueError("solver 'grams' needs the target sequences")
    bytes_per_elem = 16 if grams_mode else 8
    plan = plan_gram_groups([(k, d_in_of[k], str(device_of_key[k])) for k in keys],
                            int(gram_budget_bytes), bytes_per_elem=bytes_per_elem)
    state.pass_plan = [{"keys": p, "gram_bytes": sum(gram_bytes(d_in_of[k], bytes_per_elem)
                                                     for k in p),
                        "deepest_layer": max(key_layer(k) for k in p)} for p in plan]
    progress("plan", {"passes": len(plan), "inputs": len(keys), "solver": solver})

    resume = bool(resume_from_cache) and len(plan) > 1
    from .models.quirks import num_kv_shared_layers
    if resume and num_kv_shared_layers(h.model) > 0:
        resume = False
        state.warnings.append("boundary resume disabled: the model shares K/V across its last "
                              f"{num_kv_shared_layers(h.model)} layers (quirk kv_shared_layers)")
    layers = decoder_layer_list(h) if resume else None
    boundary = BoundaryCache() if resume else None
    boundary_X = BoundaryCache() if (resume and grams_mode) else None
    starts = [min(key_layer(k) for k in p) for p in plan]
    state.stats["resume_from_cache"] = resume
    source = HFHookSource(h)

    def accumulate(pkeys, it, label, cache, nxt):
        sink = Fp64GramSink({k: d_in_of[k] for k in pkeys}, device_of_key)
        stats = source.accumulate(it(), {k: rep[k] for k in pkeys}, sink,
                                  members_of_key={k: state.key_members[k] for k in pkeys},
                                  label=label, layers=layers, cache=cache, next_start=nxt,
                                  progress=progress, progress_every=progress_every)
        return sink.result(), stats

    for pi, pkeys in enumerate(plan):
        nxt = starts[pi + 1] if (resume and pi + 1 < len(plan)) else None
        progress("pass_start", {"pass": pi, "of": len(plan), "keys": list(pkeys)})
        grams, stats = accumulate(pkeys, retained, f"gram p{pi + 1}", boundary, nxt)
        rec = {"pass": pi, "keys": list(pkeys), "retained": stats}
        grams_X = None
        if grams_mode:
            grams_X, stats_X = accumulate(pkeys, target, f"gram_X p{pi + 1}", boundary_X, nxt)
            rec["target"] = stats_X
        t0 = time.perf_counter()
        solve_pass(h, state, pkeys, grams, n_x=n_x, ridge_rel=ridge_rel,
                   eig_extremes=eig_extremes, lam_svd_dtype=lam_svd_dtype, grams_X=grams_X,
                   progress=progress, full_spectrum=full_spectrum)
        rec["solve_s"] = round(time.perf_counter() - t0, 3)
        state.stats["passes"].append(rec)
        del grams, grams_X
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        progress("pass_done", {"pass": pi, "of": len(plan)})
    for b in (boundary, boundary_X):
        if b is not None:
            b.clear()
    msg = non_pd_summary(state.non_pd)
    if msg is not None:
        state.warnings.append(msg)
        progress("warning", {"message": msg})


def solve_pass(h, state, pkeys, grams, *, n_x, ridge_rel=0.0, eig_extremes=True,
               lam_svd_dtype="float32", grams_X=None, progress=None,
               full_spectrum=False) -> None:
    """Every solve that needs the retained Gram ``G_Y``, per input of one pass;
    each Gram is dropped as soon as its input is done.

    Per input and dose (``mu = dose.mu``):

    * ``readout``: the readout ``S = X G_mu^+`` spilled in fp32; per member the
      full-rank component's leaks (:func:`rspd.leaks_gram`) and its own target
      energy ``mx_energy = ||M X^T||^2``; then, per rank-truncated setting of the
      dose, ONE thin SVD of ``M X^T`` (``lam_svd_dtype``) from which every rule
      slices ``M_r`` (:func:`rspd.truncate_component`), its factors spilled in
      fp32.
    * ``grams``: one :func:`rspd.factor_grams` per input and dose; per member
      the dense component (:func:`rspd.erased_component_grams`) spilled in fp32
      and its leaks (:func:`rspd.leaks_grams`).
    """
    progress = progress or _noop
    solver = state.solver
    grams_mode = solver == "grams"
    if lam_svd_dtype not in LAM_SVD_DTYPES:
        raise ValueError(f"lam_svd_dtype must be one of {LAM_SVD_DTYPES}, got {lam_svd_dtype!r}")
    svd_dtype = getattr(torch, lam_svd_dtype)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    pkeys = solve_key_order(pkeys, {k: int(g[0].shape[0]) for k, g in grams.items()})
    for k in pkeys:
        G_Y, n_y = grams.pop(k)
        G_X = None
        if grams_mode:
            G_X, n_x_key = grams_X.pop(k)
            if int(n_x_key) != int(n_x):
                raise RuntimeError(f"{k}: the target Gram saw {n_x_key} tokens, the target "
                                   f"sequences hold {n_x}")
        d = int(G_Y.shape[0])
        dev = G_Y.device
        X = None if grams_mode else state.X_of[k]
        entry = {"n_Y_retained_tokens": int(n_y), "n_X_target_tokens": int(n_x), "d_in": d,
                 "solver": solver, "gram_bytes": gram_bytes(d) * (2 if grams_mode else 1)}
        w64 = lambda n: h.get_weight(n).to(device=dev, dtype=torch.float64)  # noqa: E731
        if eig_extremes:
            lo, hi, einfo = gram_eig_extremes(G_Y)
            entry.update(lam_min=lo, lam_max=hi, lam_ratio=lo / max(hi, 1e-300),
                         eig_estimate=einfo)
        if not grams_mode:
            entry["specificity"] = specificity(G_Y, X, n_y, ridge_rel=ridge_rel,
                                               full=bool(full_spectrum))
        state.specificity[k] = entry
        members = state.key_members[k]
        ret_e, tgt_e = {}, {}
        for n in members:
            W = w64(n)
            ret_e[n] = quad_trace(W, G_Y)                        # ||W Y^T||^2
            if grams_mode:
                tgt_e[n] = quad_trace(W, G_X)                    # ||W X^T||^2
            del W
        base_delta_unit = float(ridge_rel) * float(G_Y.diagonal().max())
        S_keep = {}
        X64 = None if grams_mode else X.to(device=dev, dtype=torch.float64)
        lam_by_dose: dict = {}
        for li, ls in enumerate(state.lam_settings):
            lam_by_dose.setdefault(ls["dose_index"], []).append((li, ls["rule"]))
        for di, dose in enumerate(state.doses):
            mu = dose.mu
            base_delta = base_delta_unit * mu
            if not grams_mode:
                S, info = readout_gram(G_Y, X, mu, ridge_rel=ridge_rel)
                state.readouts.put(readout_tag(k, di), S.to(torch.float32))
                S64 = S.to(device=dev, dtype=torch.float64)
                SXt = S64 @ X64.T                                        # n_x x n_x
            else:
                factor, info = factor_grams(G_X, G_Y, mu, ridge_rel=ridge_rel)
            entry.setdefault("ridge_delta", {})[str(di)] = info.get("ridge_delta", 0.0)
            entry.setdefault("solve_path", {})[str(di)] = info["solver"]
            non_pd = (info["solver"] == "eigh"
                      or info.get("ridge_delta", 0.0) > 1.01 * base_delta)
            for n in members:
                W = w64(n)
                if not grams_mode:
                    comp = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
                    dg = leaks_gram(W, comp, X, G_Y, ret_energy=ret_e[n])
                    dg["mx_energy"] = float(((comp.A @ SXt) ** 2).sum())
                    dg["rank_bound"] = int(min(W.shape[0], W.shape[1], X.shape[0]))
                else:
                    comp = erased_component_grams(W, G_X, G_Y, mu=mu, factor=factor,
                                                  ridge_rel=ridge_rel)
                    dg = leaks_grams(W, comp, G_X, G_Y, ret_energy=ret_e[n],
                                     target_energy=tgt_e[n])
                    state.component_spill.put(component_tag(di, n), comp.A.to(torch.float32))
                    dg["rank_bound"] = int(min(W.shape[0], W.shape[1], int(n_x)))
                dg.update(solve_path=info["solver"], ridge_delta=info.get("ridge_delta", 0.0),
                          mu=mu, mu_per_token=dose.mu_per_token, n_X=int(n_x), n_Y=int(n_y),
                          input_key=k, d_out=int(W.shape[0]), d_in=int(W.shape[1]),
                          retained_mse_per_token=dg["ly_energy"] / float(n_y),
                          erased_mse_per_token=dg["lx_energy"] / max(int(n_x), 1))
                state.diag_by_dose[di][n] = dg
                if non_pd:
                    state.non_pd.append({
                        "matrix": n, "input_key": k, "mu_per_token": dose.mu_per_token,
                        "solve_path": info["solver"],
                        "ridge_delta": info.get("ridge_delta", 0.0),
                        "requested_ridge_delta": base_delta})
                del W, comp
            if not grams_mode:
                del S64, SXt
                if di in lam_by_dose:
                    S_keep[di] = S
                del S
            else:
                del factor
            progress("solve", {"key": k, "dose": dose.mu_per_token})
        for di, lis in lam_by_dose.items():
            S = S_keep[di]
            mu = state.doses[di].mu
            for n in members:
                W = w64(n)
                full = erased_component_gram(W, X, mu=mu, S=S, spectrum=False)
                comps = truncate_component(full, X, [r for _, r in lis], svd_dtype=svd_dtype)
                del full
                state.diag_by_dose[di][n]["erank"] = float(comps[0].info["erank"])
                for (li, _r), comp in zip(lis, comps):
                    ta, tb = lam_tags(li, n)
                    state.lam_spill.put(ta, comp.A.to(torch.float32))
                    state.lam_spill.put(tb, comp.B.to(torch.float32))
                    dg = leaks_gram(W, comp, X, G_Y, ret_energy=ret_e[n])
                    k_ = int(comp.info["k"])
                    r = int(comp.info["rank"])
                    state.lam_info[li][n] = {
                        "rank": r, "k": k_, "r_frac": r / max(k_, 1), "rule": comp.info["rule"],
                        "erank": float(comp.info["erank"]), "d_out": int(W.shape[0]),
                        "d_in": d, "fit": float(comp.info["fit"]), "input_key": k,
                        "L_X_rel": dg["L_X_rel"], "L_Y_rel": dg["L_Y_rel"],
                        "retained_rel": dg["retained_rel"], "erased_rel": dg["erased_rel"],
                        "retained_disturbed_rel": dg["retained_disturbed_rel"],
                        "target_surviving_rel": dg["target_surviving_rel"],
                        "delta_fro": dg["delta_fro"], "w_fro": dg["w_fro"],
                        "solve_path": state.diag_by_dose[di][n]["solve_path"],
                        "ridge_delta": state.diag_by_dose[di][n]["ridge_delta"],
                        "formula": comp.info["formula"], "svd_dtype": comp.info["svd_dtype"]}
                del comps, W
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            del S
        del S_keep, G_Y, G_X, X64
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def apply_dose(h, store, state, dose_index, *, lam_index=None) -> dict:
    """Subtract one setting's erased component from every matrix, one matrix at
    a time: ``W_new = W - M`` through :func:`rspd.apply_erasure`, written by
    ``store`` (:class:`~rspd.models.pristine.PristineStore`, which backs the
    pristine weight up first). Returns the edit accounting.

    ``readout``, full rank: ``M = (W X^T) S`` in fp32 from the live (pristine)
    weight and the spilled readout. Truncated (``lam_index``): the spilled fp32
    factors of ``M_r``. ``grams``: the spilled dense fp32 component. The edit is
    formed in fp32 and cast once to the weight dtype (quirk
    ``bf16_edit_in_fp32``); ``survival = sum ||W_new - W_old||_F / sum ||M||_F``
    and ``rel_err = sum ||(W_new - W_old) + M||_F / sum ||M||_F`` say how much of
    the intended change ``-M`` the cast kept. Every matrix must be pristine when
    this is called (restore first).
    """
    grams_mode = state.solver == "grams"
    if lam_index is not None and grams_mode:
        raise ValueError("solver 'grams' has no truncated settings")
    num = den = err = 0.0
    by_type: dict = {}
    per_matrix: dict = {}
    for k, members in state.key_members.items():
        for n in members:
            # explicit copy: get_weight aliases the live parameter
            Wf = h.get_weight(n).to(dtype=torch.float32, copy=True)
            dev = Wf.device
            if grams_mode:
                Md = state.component_spill.get(component_tag(dose_index, n), device=dev,
                                               dtype=torch.float32)
                comp = ErasedComponent(Md, torch.eye(int(Wf.shape[1]), dtype=torch.float32,
                                                     device=dev),
                                       {"formula": "W G_X (G_X + mu G_Y + delta I)^+",
                                        "B_is_identity": True})
            elif lam_index is None:
                S = state.readouts.get(readout_tag(k, dose_index), device=dev,
                                       dtype=torch.float32)
                Xd = state.X_of[k].to(device=dev, dtype=torch.float32)
                comp = ErasedComponent(Wf @ Xd.T, S, {"formula": "(W X^T) S", "rank": None})
                del S, Xd
            else:
                ta, tb = lam_tags(lam_index, n)
                comp = ErasedComponent(
                    state.lam_spill.get(ta, device=dev, dtype=torch.float32),
                    state.lam_spill.get(tb, device=dev, dtype=torch.float32),
                    {"formula": "U_r U_r^T (W X^T) S", "lam_index": lam_index})
            store.write(n, apply_erasure(Wf, comp))
            M = comp.A if comp.info.get("B_is_identity") else comp.A @ comp.B
            dW = h.get_weight(n).to(dtype=torch.float32, copy=True).sub_(Wf)
            a, b = float(dW.norm()), float(M.norm())
            e = float(dW.add_(M).norm())
            num += a
            den += b
            err += e
            t = by_type.setdefault(matrix_type(n), [0.0, 0.0])
            t[0] += a
            t[1] += b
            per_matrix[n] = {"edit_fro": a, "intended_fro": b, "survival": a / max(b, 1e-300)}
            del Wf, M, dW, comp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"survival": num / max(den, 1e-300), "rel_err": err / max(den, 1e-300),
            "delta_fro_sum": den,
            "by_type": {t: v[0] / max(v[1], 1e-300) for t, v in sorted(by_type.items())},
            "per_matrix": per_matrix}
