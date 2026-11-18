"""The one-call entry point: :func:`erase` on a resident HF model, :func:`restore`.

Design (agent-facing)
---------------------
``erase`` SOLVES at every requested dose and APPLIES exactly one:

* ``mu_per_token`` a number -> that dose is solved and applied;
* ``mu_per_token`` a list -> every dose is solved (one set of Gram passes);
  pass ``apply_mu_per_token=<one of them>`` to apply it, or ``apply=False`` to
  apply none. The returned :class:`EditReport` carries the leaks of every
  dose (``by_dose``), so a caller can pick one afterwards and call
  ``report.apply_dose(m)``, which restores the pristine weights and applies
  dose ``m`` -- no second pass over the data.

``rank_rule`` works the same way: ``None`` (full rank only), one rule (a
``lam_rel`` in (0, 1) or ``"erank"``) or a list of them. Every rule is
truncated, per matrix and dose, from ONE thin SVD of ``M X^T`` shared by all
rules, so a list costs one set of Gram passes and one solve per dose. The
full-rank edit is always solved beside the rules. With more than one rule,
``apply=True`` needs ``apply_rank_rule=`` (a solved rule, or ``None`` for the
full-rank edit); afterwards ``report.apply_dose(m, rule)`` applies any solved
``(dose, rule)`` setting (``rule=None`` is the full-rank edit).

The model is edited in place (``W_new = W - M`` for every selected matrix).
The edit lock belongs to the MODEL (an :class:`_EditLock` attached to it): at
most one report's edit is applied at a time, and while one is, every other
report's ``apply_dose`` and every new ``erase`` on that model refuse, naming
the holder -- so no edit is ever stacked on another and no solve starts from
edited weights. :func:`restore` restores whichever report holds the model, bit
for bit, and releases the lock.

``target`` and ``retained`` are token ids -- a ``LongTensor`` (``[T]`` one
sequence, ``[B, T]`` B sequences), a list of 1-D tensors / int lists, or (with
``tokenizer``) strings. Each sequence is its own forward pass (no padding).
``X`` is every token of ``target``, ``Y`` every token of ``retained``;
``n_x`` / ``n_y`` are their token counts and the dose is resolved with
:func:`rspd.core.dose_from_mu_per_token`.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import tempfile
import weakref
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from .capture.grams import MemorySpill, TensorSpill, gram_bytes
from .capture.hf import harvest_inputs_concat
from .core.dose import dose_from_mu_per_token
from .models.harness import HFLinearHarness, group_by_input, key_layer
from .models.pristine import PristineStore
from .models.quirks import detect_quirks
from .models.selector import resolve_selector
from .pipeline import (CONVENTION, SOLVERS, LEAK_FIELDS, SolveState, apply_dose,
                       build_component,
                       check_scratch_space, leaks_per_matrix, rank_rule_spec, run_gram_passes,
                       scratch_bytes_needed, summarize_by_type)

__all__ = ["erase", "restore", "EditReport"]

#: Attribute of the model object that holds its :class:`_EditLock`.
_LOCK_ATTR = "_rspd_edit_lock"
_REPORT_IDS = itertools.count(1)
#: ``spectrum=`` values of :func:`erase`.
SPECTRA = ("top", "full")

#: ``apply_rank_rule`` not given (``None`` is a choice: the full-rank edit).
_UNSET = object()

#: Per-matrix fields an :class:`EditReport` carries for the applied setting.
MATRIX_FIELDS = ("input_key", "d_out", "d_in", "L_X_rel", "L_Y_rel",
                 "target_surviving_rel", "retained_disturbed_rel", "erased_rel",
                 "retained_rel", "delta_fro", "w_fro", "rank", "rank_bound", "solve_path",
                 "ridge_delta", "erank")
#: Deprecated per-matrix keys and the key that replaces each (same value).
DEPRECATED_MATRIX_FIELDS = {"erased_rel": "target_surviving_rel",
                            "retained_rel": "retained_disturbed_rel"}



def _noop(_event, _info):
    return None


# --------------------------------------------------------------------------- #
# inputs                                                                       #
# --------------------------------------------------------------------------- #
def _encode(tokenizer, text, what):
    if tokenizer is None:
        raise TypeError(f"{what} holds text; pass tokenizer= or give token ids")
    ids = tokenizer(text)["input_ids"]
    return torch.as_tensor(ids, dtype=torch.long)


def as_sequences(x, tokenizer=None, what="target") -> list:
    """Normalise ``target`` / ``retained`` into a list of ``[1, T]`` LongTensors (CPU)."""
    if isinstance(x, str):
        seqs = [_encode(tokenizer, x, what)]
    elif isinstance(x, torch.Tensor):
        if x.dim() == 1:
            seqs = [x]
        elif x.dim() == 2:
            seqs = list(x)
        else:
            raise ValueError(f"{what} tensor must be [T] or [B, T], got {tuple(x.shape)}")
    elif isinstance(x, (list, tuple)):
        if x and all(isinstance(v, int) and not isinstance(v, bool) for v in x):
            seqs = [torch.as_tensor(list(x), dtype=torch.long)]
        else:
            seqs = []
            for v in x:
                if isinstance(v, str):
                    seqs.append(_encode(tokenizer, v, what))
                elif isinstance(v, torch.Tensor):
                    if v.dim() == 2 and v.shape[0] == 1:
                        v = v[0]
                    if v.dim() != 1:
                        raise ValueError(f"each {what} sequence must be [T] or [1, T], got "
                                         f"{tuple(v.shape)}")
                    seqs.append(v)
                else:
                    seqs.append(torch.as_tensor(list(v), dtype=torch.long))
    else:
        raise TypeError(f"{what} must be token ids (tensor / list of sequences) or text, got "
                        f"{type(x).__name__}")
    out = []
    for i, s in enumerate(seqs):
        s = torch.as_tensor(s).to("cpu")
        if s.dtype not in (torch.int64, torch.int32, torch.int16, torch.uint8):
            raise TypeError(f"{what} sequence {i} has dtype {s.dtype}; token ids are integers")
        if s.numel() == 0:
            raise ValueError(f"{what} sequence {i} is empty")
        out.append(s.to(torch.long).reshape(1, -1))
    if not out:
        raise ValueError(f"{what} holds no sequences")
    return out


def _input_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _doses(mu_per_token, n_x, n_y):
    vals = list(mu_per_token) if isinstance(mu_per_token, (list, tuple)) else [mu_per_token]
    if not vals:
        raise ValueError("mu_per_token is empty")
    seen, out = set(), []
    for v in vals:
        d = dose_from_mu_per_token(v, n_x, n_y)
        if d.mu_per_token not in seen:
            seen.add(d.mu_per_token)
            out.append(d)
    return out


def _auto_budget(device_of_key, d_in_of, bytes_per_elem):
    """Default per-device Gram budget: every Gram at once on CPU; on CUDA half of
    the smallest free memory over the devices holding Grams."""
    devs = {torch.device(str(d)) for d in device_of_key.values()}
    cuda = [d for d in devs if d.type == "cuda"]
    if not cuda:
        return sum(gram_bytes(d, bytes_per_elem) for d in d_in_of.values())
    return int(0.5 * min(torch.cuda.mem_get_info(d)[0] for d in cuda))


def _dose_key(dose):
    return repr(float(dose.mu_per_token))


def _rank_rules(rank_rule) -> list:
    """``rank_rule`` -> the de-duplicated truncation specs, in the order given
    (``None`` entries of a list are the full-rank edit, always solved)."""
    vals = list(rank_rule) if isinstance(rank_rule, (list, tuple)) else [rank_rule]
    out = []
    for v in vals:
        if v is None:
            continue
        spec = rank_rule_spec(v)
        if spec not in out:
            out.append(spec)
    return out


def _rule_index(rules, rule, what):
    """``None`` -> ``None`` (full rank); a solved rule -> its position in ``rules``."""
    if rule is None:
        return None
    spec = rank_rule_spec(rule)
    if spec not in rules:
        raise ValueError(f"{what} {rule!r} was not solved; solved rank rules: {rules} "
                         f"(None = the full-rank edit)")
    return rules.index(spec)


# --------------------------------------------------------------------------- #
# the model's edit lock                                                        #
# --------------------------------------------------------------------------- #
class _EditLock:
    """Attached to a model (attribute ``_rspd_edit_lock``): ``holder`` is the
    session whose edit is currently applied to it, or ``None``. Copies and
    pickles of the model get a fresh, empty lock (the holder's pristine backup
    belongs to the original)."""

    def __init__(self):
        self.holder = None

    def __reduce__(self):
        return (_EditLock, ())


def _lock_of(model, create=False):
    lock = getattr(model, "__dict__", {}).get(_LOCK_ATTR)
    if lock is None and create:
        lock = _EditLock()
        object.__setattr__(model, _LOCK_ATTR, lock)
    return lock


def _holder(model):
    """The session holding ``model``'s edit (``None`` if the model is pristine)."""
    lock = _lock_of(model)
    if lock is None or lock.holder is None:
        return None
    if lock.holder.closed or not lock.holder.store.is_edited:
        lock.holder = None
        return None
    return lock.holder


def _held_error(holder, me=None):
    who = "" if me is None else f", not by this report ({me.describe()})"
    return RuntimeError(
        f"this model carries an applied erasure held by {holder.describe()}{who}: call "
        f"rspd.api.restore(model) (or that report's close()) first so the solve starts from "
        f"the pristine weights; applying on top would stack two edits")


def _release(s):
    m = s.model_ref()
    lock = None if m is None else _lock_of(m)
    if lock is not None and lock.holder is s:
        lock.holder = None


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class EditReport:
    """The record of one :func:`erase` call (JSON via :meth:`to_dict` / :meth:`to_json`).

    Fields
    ------
    convention : the edit convention (``W_new = W - M``; X target, Y retained).
    solver : ``"readout"`` | ``"grams"``.
    applied : whether an edit is currently applied by this report.
    dose : the applied dose, :meth:`rspd.core.dose.Dose.as_dict` (``mu_per_token``,
        the derived weights, ``n_x``, ``n_y``, ``formula``), or ``None``.
    doses : every solved dose, same form.
    rank_rule : the truncation rule of the applied edit (``None`` = full rank,
        or nothing applied).
    rank_rules : every solved truncation rule (the full-rank edit is always
        solved besides).
    settings : every solved ``(dose, rule)`` setting, ``[{"mu_per_token",
        "rank_rule"}]``, dose-major, full rank (``rank_rule`` ``None``) first.
    ridge_rel : the requested relative ridge (the delta actually applied is per
        matrix, ``matrices[name]["ridge_delta"]``).
    n_x, n_y : target / retained token counts.
    matrices : ``{name: {input_key, d_out, d_in, L_X_rel, L_Y_rel, erased_rel,
        retained_rel, delta_fro (||M||_F), w_fro, rank (None = full),
        rank_bound, solve_path ("cholesky" | "eigh"), ridge_delta, erank,
        edit_fro (||W_new - W||_F as written), survival}}`` at the applied dose.
    leaks_by_type : ``{matrix type: {L_X_rel: {median, min, max}, ...}}``.
    by_dose : ``{repr(mu_per_token): {"dose", "leaks", "leaks_by_type",
        "truncated": {str(rule): {"leaks", "leaks_by_type", "rank_by_type"}}}}``
        for every solved dose and rule.
    edit : ``{survival, rel_err, delta_fro_sum, by_type}`` of the applied edit.
    input_keys : ``{key: {members, d_in, gram_bytes}}``.
    quirks : ``{quirk name: detail}`` (:mod:`rspd.models.quirks`).
    selector, pass_plan, stats.
    specificity : per input key; ``specificity[key]["specificity"]`` holds the
        top eight ``gamma`` and, with ``erase(..., spectrum="full")``, the whole
        spectrum under ``"spectrum"`` (pass the entry, or :meth:`spectrum`, to
        :func:`rspd.core.suggest_mu_per_token`).
    warnings : strings; non-PD solves are ONE summary warning per run.
    non_pd : one record per (matrix, dose) whose ``G_mu`` was not numerically
        PD at the requested ridge: ``{matrix, input_key, mu_per_token,
        solve_path, ridge_delta, requested_ridge_delta}``.

    ``matrices`` (and every per-matrix mapping) is ordered by block, then by
    forward order within the block. ``erased_rel`` / ``retained_rel`` are
    deprecated aliases of ``target_surviving_rel`` / ``retained_disturbed_rel``.
    """
    convention: str
    solver: str
    applied: bool
    dose: dict | None
    doses: list
    rank_rule: Any
    rank_rules: list
    settings: list
    ridge_rel: float
    n_x: int
    n_y: int
    matrices: dict
    leaks_by_type: dict
    by_dose: dict
    edit: dict | None
    input_keys: dict
    quirks: dict
    selector: dict
    pass_plan: list
    specificity: dict
    stats: dict
    warnings: list
    non_pd: list = field(default_factory=list)
    _session: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if not k.startswith("_")}
        return json.loads(json.dumps(d, default=_json_default))

    def to_json(self, path=None, **kw) -> str:
        kw.setdefault("indent", 1)
        s = json.dumps(self.to_dict(), **kw)
        if path is not None:
            with open(path, "w") as f:
                f.write(s)
        return s

    # -- live operations (need the session; not serialised) -------------------
    def _need_session(self):
        if self._session is None or self._session.closed:
            raise RuntimeError("this EditReport has no live solve attached (it was loaded "
                               "from JSON or closed); re-run rspd.api.erase")
        return self._session

    def apply_dose(self, mu_per_token, rule=None) -> "EditReport":
        """Restore the pristine weights and apply the solved setting
        ``(mu_per_token, rule)``: ``mu_per_token`` one of :attr:`doses`, ``rule``
        ``None`` (the full-rank edit ``W - M``) or one of :attr:`rank_rules`
        (``W - M_r``). Updates this report in place and returns it.

        Raises ``RuntimeError`` naming the holder when ANOTHER report's edit is
        applied to the model (restore it first)."""
        s = self._need_session()
        idx = [i for i, d in enumerate(s.state.doses) if d.mu_per_token == float(mu_per_token)]
        if not idx:
            raise ValueError(f"mu_per_token {mu_per_token!r} was not solved; solved: "
                             f"{[d.mu_per_token for d in s.state.doses]}")
        ri = _rule_index(s.rules, rule, "rule")
        _apply_setting(self, s, idx[0], ri)
        return self

    def restore(self) -> list:
        """Restore every weight this report edited, bit for bit, and release the
        model's edit lock. Returns the restored names (``[]`` when this report
        does not hold the model)."""
        s = self._need_session()
        names = s.store.restore_all()
        s.mark_restored()
        _release(s)
        return names

    def close(self) -> None:
        """Restore the model, then free the solve's spills (and its scratch
        directory). The report keeps its numbers; :meth:`apply_dose` stops working."""
        s = self._session
        if s is None or s.closed:
            return
        s.store.restore_all()
        _release(s)
        self.applied, self.dose, self.edit, self.rank_rule = False, None, None, None
        s.close()

    def spectrum(self, name) -> list:
        """The full ``gamma`` spectrum (descending) of input key or matrix
        ``name``; needs ``erase(..., spectrum="full")``. Feed it to
        :func:`rspd.core.suggest_mu_per_token`."""
        key = name if name in self.specificity else (self.matrices.get(name) or {}).get(
            "input_key")
        if key not in self.specificity:
            raise KeyError(f"{name!r} is neither an input key nor an edited matrix of this "
                           f"report; input keys: {list(self.specificity)[:6]}")
        spec = self.specificity[key].get("specificity") or {}
        if "spectrum" not in spec:
            raise ValueError("this report keeps only the top gamma values; re-run "
                             "rspd.api.erase(..., spectrum='full') (readout solver)")
        return list(spec["spectrum"])


def _json_default(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, (torch.dtype, torch.device)):
        return str(o)
    if hasattr(o, "as_dict"):
        return o.as_dict()
    return str(o)


class _Session:
    def __init__(self, model, h, store, state, rules, lam_index_of, scratch, own_scratch):
        self.id = next(_REPORT_IDS)
        self.applied_setting = None                # (mu_per_token, rank rule) when applied
        self.model_ref = weakref.ref(model)
        self.h, self.store, self.state = h, store, state
        self.rules = list(rules)
        self.lam_index_of = lam_index_of           # (dose index, rule index) -> lam index
        self.scratch, self.own_scratch = scratch, own_scratch
        self.closed = False
        self.report_ref = None

    def describe(self) -> str:
        what = ("nothing applied" if self.applied_setting is None else
                f"applied mu_per_token {self.applied_setting[0]!r}, rank_rule "
                f"{self.applied_setting[1]!r}")
        return f"EditReport #{self.id} (solver {self.state.solver!r}, {what})"

    def mark_restored(self):
        self.applied_setting = None
        r = self.report_ref() if self.report_ref is not None else None
        if r is not None:
            r.applied, r.dose, r.edit, r.rank_rule = False, None, None, None

    def close(self):
        self.state.readouts = self.state.lam_spill = self.state.component_spill = None
        self.state.X_of = None
        if self.own_scratch and self.scratch and os.path.isdir(self.scratch):
            shutil.rmtree(self.scratch, ignore_errors=True)
        self.closed = True


def _ordered(h, d: dict) -> dict:
    """``d`` (keyed by matrix name) in block order, then forward order within a
    block (the harness's ``all_names``)."""
    pos = {n: i for i, n in enumerate(h.all_names)}
    return {n: d[n] for n in sorted(d, key=lambda n: (key_layer(n), pos.get(n, len(pos)), n))}


def _matrix_rows(state, di, li, edit, h):
    src = state.diag_by_dose[di] if li is None else state.lam_info[li]
    out = {}
    for n, d in _ordered(h, src).items():
        row = {f: d.get(f) for f in MATRIX_FIELDS}
        if li is not None:
            row["rank_bound"] = d.get("k")
        if edit is not None and n in edit["per_matrix"]:
            row.update(edit_fro=edit["per_matrix"][n]["edit_fro"],
                       survival=edit["per_matrix"][n]["survival"])
        out[n] = row
    return out


def _apply_setting(report, s, di, ri=None):
    """Take the model's edit lock for session ``s`` (refusing when another
    session holds it), restore ``s``'s own edit, apply setting ``(di, ri)``."""
    model = s.model_ref()
    if model is None:                                         # pragma: no cover
        raise RuntimeError("the model of this EditReport no longer exists")
    holder = _holder(model)
    if holder is not None and holder is not s:
        raise _held_error(holder, s)
    s.store.restore_all()
    s.applied_setting = None
    _lock_of(model, create=True).holder = s       # before writing: a failure stays restorable
    _apply_into(report, s, di, ri)
    s.applied_setting = (s.state.doses[di].mu_per_token, None if ri is None else s.rules[ri])


def _apply_into(report, s, di, ri=None):
    li = None if ri is None else s.lam_index_of[(di, ri)]
    edit = apply_dose(s.h, s.store, s.state, di, lam_index=li)
    report.matrices = _matrix_rows(s.state, di, li, edit, s.h)
    report.leaks_by_type = summarize_by_type(report.matrices, LEAK_FIELDS)
    report.edit = {k: v for k, v in edit.items() if k != "per_matrix"}
    report.dose = s.state.doses[di].as_dict()
    report.rank_rule = None if ri is None else s.rules[ri]
    report.applied = True


# --------------------------------------------------------------------------- #
# erase / restore                                                              #
# --------------------------------------------------------------------------- #
def erase(model, tokenizer=None, *, target, retained, mu_per_token, apply=True,
          apply_mu_per_token=None, apply_rank_rule=_UNSET, layers=None, solver="readout", rank_rule=None,
          ridge_rel=0.0, device=None, gram_budget_gib=None, scratch_dir=None,
          progress: Callable | None = None, subtract_bias=False, resume_from_cache=True,
          eig_extremes=True, lam_svd_dtype="float32", check_bracketing=True,
          spectrum="top") -> EditReport:
    """Erase the ``target`` behaviour from ``model`` while preserving
    ``retained``: for every selected linear weight ``W``,
    ``W_new = W - M``, ``M = W X^T X (X^T X + mu Y^T Y + delta I)^+``.

    Parameters
    ----------
    model : a resident HF model (``eval()``; any device / dtype). Edited in place.
    tokenizer : only needed when ``target`` / ``retained`` are text.
    target, retained : token ids (see the module docstring); ``X`` and ``Y``.
    mu_per_token : the dose, a number > 0 or a list of them (``1 / mu_per_token``
        is the per-token Lagrange multiplier on the retained disturbance; larger
        erases more). See :mod:`rspd.core.dose`.
    apply, apply_mu_per_token, apply_rank_rule : which solved ``(dose, rule)``
        setting to apply (module docstring). ``apply_rank_rule`` is ``None``
        (the full-rank edit) or one of the solved rules; it may be left out
        when at most one rule is solved (that rule is then applied).
    layers : the layer selector (``None`` = auto preset; see
        :func:`rspd.models.selector.resolve_selector`).
    solver : ``"readout"`` (target held as activations; supports ``rank_rule``)
        or ``"grams"`` (target streamed into ``G_X``; for a target too large to
        hold).
    rank_rule : ``None`` (full rank), a ``lam_rel`` in (0, 1), ``"erank"``,
        or a list of them: each rule's edit is ``W - M_r`` (COALA projector on
        ``M`` over ``X``), every rule of a matrix and dose sliced from one
        shared SVD of ``M X^T``; ``readout`` only.
    ridge_rel : relative ridge, ``delta = ridge_rel * max(diag(mu G_Y))``
        (escalated automatically when ``G_mu`` is not numerically PD; the delta
        used is reported per matrix).
    device : where the Grams and solves live (default: each weight's device).
    gram_budget_gib : float64 Gram budget per device; default every Gram at
        once on CPU, half the free memory on CUDA. More passes when exceeded.
    scratch_dir : spill pristine weights, readouts and components to a fresh
        subdirectory here instead of host RAM (removed by
        :meth:`EditReport.close`).
    progress : ``progress(event: str, info: dict)``, called on ``"plan"``,
        ``"pass_start"``, ``"chunk"``, ``"solve"``, ``"pass_done"``,
        ``"warning"``, ``"applied"``.
    subtract_bias : accept biased ``nn.Linear`` projections (quirk
        ``biased_projection``).
    resume_from_cache : restart each Gram pass from the cached residual stream
        (bitwise-guarded; off automatically on KV-shared models).
    eig_extremes : record ``lam_min`` / ``lam_max`` of every ``G_Y``.
    lam_svd_dtype : dtype of the one SVD of ``M X^T`` (per matrix and dose)
        that every ``rank_rule`` slices.
    check_bracketing : verify on the first target sequence that every hook
        brackets exactly ``x W^T`` (rel. tol 1e-2).
    spectrum : ``"top"`` (default: the top eight ``gamma`` per input key) or
        ``"full"`` (also the whole spectrum, ``specificity[key]["specificity"]
        ["spectrum"]``, ready for :func:`rspd.core.suggest_mu_per_token`; readout
        solver only).

    Returns
    -------
    :class:`EditReport`.
    """
    progress = progress or _noop
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}, got {solver!r}")
    if spectrum not in SPECTRA:
        raise ValueError(f"spectrum must be one of {SPECTRA}, got {spectrum!r}")
    if spectrum == "full" and solver == "grams":
        raise ValueError("spectrum='full' needs solver='readout': the gamma spectrum needs "
                         "the target activations X, which the grams solver never holds")
    holder = _holder(model)
    if holder is not None:
        raise _held_error(holder)
    rules = _rank_rules(rank_rule)
    if rules and solver == "grams":
        raise ValueError("rank_rule needs solver='readout': truncation acts on the target "
                         "response M X^T, which the grams solver never forms")
    sel = resolve_selector(model, layers)
    h = HFLinearHarness(model, selector=sel, subtract_bias=subtract_bias)
    quirks = detect_quirks(model, h)
    in_dev = _input_device(model)
    tgt = as_sequences(target, tokenizer, "target")
    ret = as_sequences(retained, tokenizer, "retained")
    n_x = sum(int(s.shape[-1]) for s in tgt)
    n_y = sum(int(s.shape[-1]) for s in ret)
    doses = _doses(mu_per_token, n_x, n_y)
    if apply:
        if apply_mu_per_token is None:
            if len(doses) > 1:
                raise ValueError(f"{len(doses)} doses solved; pass apply_mu_per_token=<one of "
                                 f"{[d.mu_per_token for d in doses]}> or apply=False (then "
                                 f"report.apply_dose(m))")
            apply_index = 0
        else:
            m = float(apply_mu_per_token)
            hits = [i for i, d in enumerate(doses) if d.mu_per_token == m]
            if not hits:
                raise ValueError(f"apply_mu_per_token {apply_mu_per_token!r} is not one of the "
                                 f"doses {[d.mu_per_token for d in doses]}")
            apply_index = hits[0]
        if apply_rank_rule is _UNSET:
            if len(rules) > 1:
                raise ValueError(f"{len(rules)} rank rules solved; pass apply_rank_rule=<one of "
                                 f"{rules}, or None for the full-rank edit> or apply=False "
                                 f"(then report.apply_dose(m, rule))")
            apply_rule = rules[0] if rules else None
        else:
            apply_rule = apply_rank_rule
        apply_rule_index = _rule_index(rules, apply_rule, "apply_rank_rule")
    if check_bracketing:
        probe = tgt[0][:, :64].to(in_dev)
        h.verify_bracketing(h.harvest([probe]), "probe", rel_tol=1e-2)

    key_members = group_by_input(h.all_names)
    keys = list(key_members)
    rep = {k: m[0] for k, m in key_members.items()}
    d_in_of = {k: h.d_in(rep[k]) for k in keys}
    device_of_key = {k: (torch.device(device) if device is not None
                         else h.get_weight(rep[k]).device) for k in keys}
    grams_mode = solver == "grams"
    budget = (int(float(gram_budget_gib) * 2 ** 30) if gram_budget_gib is not None
              else _auto_budget(device_of_key, d_in_of, 16 if grams_mode else 8))

    scratch, own = None, False
    if scratch_dir is not None:
        os.makedirs(str(scratch_dir), exist_ok=True)
        params = {n: h.get_weight(n).numel() for n in h.all_names}
        elem = h.get_weight(h.all_names[0]).element_size()
        need = scratch_bytes_needed(params, n_lam=len(rules) * len(doses), elem_bytes=elem,
                                    n_dense=len(doses) if grams_mode else 0)
        if not grams_mode:
            need += len(doses) * sum(4 * n_x * d_in_of[k] for k in keys)
        check_scratch_space(str(scratch_dir), need)
        scratch, own = tempfile.mkdtemp(prefix="rspd-", dir=str(scratch_dir)), True

    def spill(sub):
        return MemorySpill() if scratch is None else TensorSpill(os.path.join(scratch, sub))

    X_of = {}
    if not grams_mode:
        data, n_h = harvest_inputs_concat(h, [s.to(in_dev) for s in tgt])
        assert n_h == n_x
        X_of = {k: data[rep[k]] for k in keys}
        del data
    lam_settings = [{"dose_index": di, "rule": r} for di in range(len(doses)) for r in rules]
    state = SolveState(solver=solver, doses=doses, lam_settings=lam_settings,
                       key_members=key_members, X_of=X_of, readouts=spill("readouts"),
                       lam_spill=spill("truncated"), component_spill=spill("components"))

    def retained_iter():
        for s in ret:
            yield s.to(in_dev)

    def target_iter():
        for s in tgt:
            yield s.to(in_dev)

    run_gram_passes(h, state, retained=retained_iter, n_y=n_y, keys=keys, rep=rep,
                    d_in_of=d_in_of, device_of_key=device_of_key, gram_budget_bytes=budget,
                    resume_from_cache=resume_from_cache, eig_extremes=eig_extremes,
                    ridge_rel=ridge_rel, lam_svd_dtype=lam_svd_dtype,
                    target=target_iter if grams_mode else None, n_x=n_x, progress=progress,
                    full_spectrum=spectrum == "full")

    store = PristineStore(h, scratch)
    lam_index_of = {(ls["dose_index"], rules.index(ls["rule"])): li
                    for li, ls in enumerate(lam_settings)}
    session = _Session(model, h, store, state, rules, lam_index_of, scratch, own)

    by_dose = {}
    for di, d in enumerate(doses):
        diags = state.diag_by_dose[di]
        diags = _ordered(h, diags)
        ent = {"dose": d.as_dict(), "leaks": leaks_per_matrix(diags),
               "leaks_by_type": summarize_by_type(diags, LEAK_FIELDS + ("delta_fro",)),
               "truncated": {}}
        for ri, rule in enumerate(rules):
            info = _ordered(h, state.lam_info[lam_index_of[(di, ri)]])
            ent["truncated"][str(rule)] = {
                "leaks": leaks_per_matrix(info),
                "leaks_by_type": summarize_by_type(info, LEAK_FIELDS + ("delta_fro",)),
                "rank_by_type": summarize_by_type(info, ("rank", "k", "r_frac", "erank"))}
        by_dose[_dose_key(d)] = ent

    report = EditReport(
        convention=CONVENTION, solver=solver, applied=False, dose=None,
        doses=[d.as_dict() for d in doses], rank_rule=None, rank_rules=list(rules),
        settings=[{"mu_per_token": d.mu_per_token, "rank_rule": r}
                  for d in doses for r in [None] + rules],
        ridge_rel=float(ridge_rel), n_x=n_x, n_y=n_y, matrices={}, leaks_by_type={},
        by_dose=by_dose, edit=None,
        input_keys={k: {"members": list(m), "d_in": d_in_of[k],
                        "gram_bytes": gram_bytes(d_in_of[k]) * (2 if grams_mode else 1),
                        "device": str(device_of_key[k])} for k, m in key_members.items()},
        quirks=quirks, selector=sel.as_dict(), pass_plan=state.pass_plan,
        specificity=state.specificity,
        stats={**state.stats, "gram_budget_bytes": int(budget),
               "n_target_sequences": len(tgt), "n_retained_sequences": len(ret),
               "scratch_dir": scratch},
        warnings=list(state.warnings), non_pd=list(state.non_pd), _session=session)
    session.report_ref = weakref.ref(report)
    if apply:
        _apply_setting(report, session, apply_index, apply_rule_index)
        progress("applied", {"mu_per_token": doses[apply_index].mu_per_token,
                             "rank_rule": report.rank_rule,
                             "survival": report.edit["survival"]})
    else:
        report.matrices = _matrix_rows(state, 0, None, None, h)
        report.leaks_by_type = summarize_by_type(report.matrices, LEAK_FIELDS)
    bad = [n for n, r in report.matrices.items()
           if any(r.get(f) is None or not math.isfinite(float(r[f])) for f in LEAK_FIELDS)]
    if bad:
        report.warnings.append(f"non-finite leaks for {bad[:6]}")
    return report


class _FitBackend:
    """What :class:`rspd.core.Components` needs from a live fit: the component
    of a matrix at a setting, the leaks, apply / restore / close. A setting is
    ``(mu_per_token, rank_rule)``."""

    def __init__(self, report):
        self.report = report

    def _session(self):
        return self.report._need_session()

    def _indices(self, setting):
        s = self._session()
        m, rule = setting
        di = [i for i, d in enumerate(s.state.doses) if d.mu_per_token == float(m)]
        if not di:
            raise ValueError(f"mu_per_token {m!r} was not solved; solved: "
                             f"{[d.mu_per_token for d in s.state.doses]}")
        ri = _rule_index(s.rules, rule, "rank_rule")
        return s, di[0], ri

    def resolve(self, setting, settings):
        """A bare ``mu_per_token`` names that dose at the default rank rule (the
        one solved rule, else full rank)."""
        try:
            m = float(setting)
        except (TypeError, ValueError):
            raise ValueError(f"setting {setting!r} was not solved; solved: {settings}") from None
        rules = self._session().rules
        want = (m, rules[0] if len(rules) == 1 else None)
        if want not in settings:
            raise ValueError(f"mu_per_token {setting!r} was not solved; solved settings "
                             f"(mu_per_token, rank_rule): {settings}")
        return want

    def component(self, name, _index, setting):
        from .core.types import Component
        s, di, ri = self._indices(setting)
        li = None if ri is None else s.lam_index_of[(di, ri)]
        key = next(k for k, members in s.state.key_members.items() if name in members)
        Wf = s.store.pristine(name, dtype=torch.float32)
        comp = build_component(s.state, key, name, Wf, di, li)
        row = (s.state.diag_by_dose[di] if li is None else s.state.lam_info[li])[name]
        info = dict(comp.info, name=name, mu_per_token=float(setting[0]), rank_rule=setting[1],
                    leaks={f: row.get(f) for f in LEAK_FIELDS},
                    **{f: row.get(f) for f in ("erank", "solve_path", "ridge_delta")
                       if f in row})
        return Component(comp.A, comp.B, info)

    def leaks(self, setting) -> dict:
        s, di, ri = self._indices(setting)
        li = None if ri is None else s.lam_index_of[(di, ri)]
        rows = _ordered(s.h, s.state.diag_by_dose[di] if li is None else s.state.lam_info[li])
        return {n: {f: r.get(f) for f in LEAK_FIELDS} for n, r in rows.items()}

    def applied_setting(self):
        s = self.report._session
        return None if s is None or s.closed else s.applied_setting

    def apply(self, setting):
        s, di, ri = self._indices(setting)
        _apply_setting(self.report, s, di, ri)

    def restore(self) -> list:
        return self.report.restore()

    def close(self) -> None:
        self.report.close()


def components_of(report: EditReport):
    """The :class:`rspd.core.Components` of a live :class:`EditReport`: one
    :class:`rspd.core.Component` per edited matrix, in block order, at every
    solved ``(mu_per_token, rank_rule)`` setting."""
    from .core.types import Components
    s = report._need_session()
    names = list(_ordered(s.h, s.state.diag_by_dose[0]))
    settings = [(x["mu_per_token"], x["rank_rule"]) for x in report.settings]
    default = (settings[0][0], s.rules[0] if len(s.rules) == 1 else None)
    current = s.applied_setting or default
    return Components(names, backend=_FitBackend(report), settings=settings, setting=current,
                      report=report)


def restore(model) -> list:
    """Restore every weight :func:`erase` edited on ``model``, bit for bit,
    whichever report applied the edit, and release the model's edit lock.
    Returns the restored matrix names (``[]`` if nothing is applied)."""
    lock = _lock_of(model)
    s = None if lock is None else lock.holder
    if s is None:
        return []
    names = s.store.restore_all()
    s.mark_restored()
    lock.holder = None
    return names
