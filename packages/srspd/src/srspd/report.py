"""Edit-size report: how much of a model's weights an edit changed.

Given a base checkpoint and an edited one, every 2-D weight tensor present in
both with the same shape is compared by the squared Frobenius change

    fro_delta_sq = ||W_edit - W_base||_F^2

and the relative Frobenius change

    rel = ||W_edit - W_base||_F / ||W_base||_F ,

and the per-tensor numbers are aggregated per transformer block, per projection
type (``attn.c_attn``, ``attn.c_proj``, ``mlp.c_fc``, ``mlp.c_proj`` on GPT-2;
``embeddings`` and ``other`` for the rest) and in total. The headline number is
the **gradient mass**

    grad_mass = sum_i ||W_i^edit - W_i^base||_F^2

over every compared 2-D tensor (absolute, not relative; an unchanged matrix
contributes 0; tied tensors -- GPT-2's ``lm_head.weight`` is ``wte.weight`` --
are counted once). ``grad_mass_proj`` is the same sum restricted to the
in-block projection matrices. The relative totals answer different questions:

* ``sum_rel`` -- the plain sum of per-tensor ``rel``;
* ``mean_rel_weighted`` -- ``rel`` averaged with each tensor's parameter count
  as the weight;
* ``global_rel`` -- ``||dW||_F / ||W||_F`` over the concatenation of every
  compared tensor (the one number that is a norm of a single vector);
* ``changed_param_fraction`` -- the fraction of compared parameters that sit in
  tensors whose ``rel`` exceeds ``CHANGED_TOL`` (a localisation statistic: an
  edit of a few matrices touches few, a full-model solve touches them all).

Optionally the **effective rank** (:func:`srspd.erank.effective_rank`) of each
changed projection's ``dW`` is recorded beside the effective rank of
``W_base`` -- cheap at small model size, and off by default so a large model
does not pay for an SVD per matrix.

The block index is parsed from the parameter name with a regex that defaults to
HF GPT-2's ``transformer.h.N.`` and falls back to ``layers.N.`` (Llama / Gemma
style); pass ``block_regex`` for anything else. Tied tensors are compared once,
under the first name the state dict lists.

Entry points: :func:`weight_delta_report` returns the dict;
:func:`write_weight_delta` writes it as ``weight_delta.json`` plus a
``weight_delta.csv`` with one row per tensor; :func:`totals_summary` flattens
the totals into ``{prefix/key: value}``, ``grad_mass`` first.
"""
from __future__ import annotations

import csv
import json
import os
import re
from collections import OrderedDict
from typing import Any, Mapping

import torch

from .erank import effective_rank

#: A tensor whose relative change is at or below this is "unchanged" — it
#: absorbs bf16 round-trips and save/load noise without hiding a real edit.
CHANGED_TOL = 1e-12

#: HF GPT-2 (``transformer.h.N.``), then the ``layers.N.`` family.
DEFAULT_BLOCK_REGEX = r"(?:^|\.)h\.(\d+)\."
FALLBACK_BLOCK_REGEX = r"layers\.(\d+)\."

_PARAM_SUFFIX = re.compile(r"\.(weight|bias)$")
_EMBED_HINT = re.compile(r"(^|\.)(wte|wpe|embed|embed_tokens|embed_positions|lm_head)(\.|$)")


def _state_dict(obj) -> Mapping[str, torch.Tensor]:
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict()
    if isinstance(obj, Mapping):
        return obj
    raise TypeError(f"expected a state dict or a module, got {type(obj).__name__}")


def _dedupe_tied(sd: Mapping[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Drop later keys that alias an earlier tensor's storage (tied weights)."""
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    seen: set = set()
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        key = (v.data_ptr(), tuple(v.shape), tuple(v.stride())) if v.numel() else None
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out[k] = v
    return out


def classify(name: str, block_regex: str = DEFAULT_BLOCK_REGEX) -> tuple[int | None, str]:
    """``(block index or None, projection type)`` for a parameter name.

    Inside a block the type is the name with the block prefix and the
    ``.weight`` / ``.bias`` suffix stripped (``attn.c_attn``); outside a block
    it is ``embeddings`` for the token / position tables and the LM head, else
    ``other``.
    """
    for pattern in (block_regex, FALLBACK_BLOCK_REGEX):
        m = re.search(pattern, name)
        if m:
            tail = name[m.end():]
            return int(m.group(1)), _PARAM_SUFFIX.sub("", tail) or "other"
    if _EMBED_HINT.search(name):
        return None, "embeddings"
    return None, "other"


def tensor_deltas(base, edited, *, block_regex: str = DEFAULT_BLOCK_REGEX,
                  erank: bool = False, changed_tol: float = CHANGED_TOL) -> list[dict]:
    """One record per 2-D tensor present in both with the same shape."""
    b_sd, e_sd = _dedupe_tied(_state_dict(base)), _dedupe_tied(_state_dict(edited))
    rows: list[dict] = []
    for name, wb in b_sd.items():
        we = e_sd.get(name)
        if we is None or wb.ndim != 2 or we.ndim != 2 or tuple(wb.shape) != tuple(we.shape):
            continue
        wb32 = wb.detach().to("cpu", torch.float32)
        we32 = we.detach().to("cpu", torch.float32)
        d = we32 - wb32
        fro_d_sq = float(torch.sum(d.double() * d.double()))
        fro_d = fro_d_sq ** 0.5
        fro_b = float(torch.linalg.norm(wb32))
        rel = fro_d / fro_b if fro_b > 0 else (float("inf") if fro_d > 0 else 0.0)
        block, proj = classify(name, block_regex)
        row = {"name": name, "block": block, "proj": proj,
               "shape": list(wb.shape), "numel": int(wb.numel()),
               "fro_delta_sq": fro_d_sq, "fro_delta": fro_d, "fro_base": fro_b,
               "rel": rel,
               "changed": bool(rel > changed_tol),
               "erank_delta": None, "erank_base": None}
        if erank and block is not None:
            row["erank_base"] = effective_rank(wb32) if fro_b > 0 else None
            row["erank_delta"] = effective_rank(d) if row["changed"] else None
        rows.append(row)
    return rows


def _aggregate(rows: list[dict]) -> dict:
    numel = sum(r["numel"] for r in rows)
    sq_d = sum(r["fro_delta_sq"] for r in rows)
    sq_b = sum(r["fro_base"] ** 2 for r in rows)
    changed_numel = sum(r["numel"] for r in rows if r["changed"])
    finite = [r for r in rows if r["rel"] != float("inf")]
    return {"grad_mass": sq_d,
            "grad_mass_proj": sum(r["fro_delta_sq"] for r in rows if r["block"] is not None),
            "n_tensors": len(rows), "n_changed": sum(r["changed"] for r in rows),
            "numel": numel, "changed_numel": changed_numel,
            "changed_param_fraction": (changed_numel / numel) if numel else 0.0,
            "fro_delta": sq_d ** 0.5, "fro_base": sq_b ** 0.5,
            "global_rel": (sq_d ** 0.5 / sq_b ** 0.5) if sq_b > 0 else 0.0,
            "sum_rel": sum(r["rel"] for r in finite),
            "mean_rel_weighted": (sum(r["rel"] * r["numel"] for r in finite)
                                  / sum(r["numel"] for r in finite)) if finite else 0.0}


def weight_delta_report(base, edited, *, block_regex: str = DEFAULT_BLOCK_REGEX,
                        erank: bool = False, changed_tol: float = CHANGED_TOL) -> dict:
    """The full report: per-tensor rows plus per-block, per-projection and
    total aggregates. ``base`` / ``edited`` are state dicts or modules."""
    rows = tensor_deltas(base, edited, block_regex=block_regex, erank=erank,
                         changed_tol=changed_tol)
    blocks: dict[str, list] = {}
    projs: dict[str, list] = {}
    for r in rows:
        if r["block"] is not None:
            blocks.setdefault(str(r["block"]), []).append(r)
        projs.setdefault(r["proj"], []).append(r)
    per_block = {k: _aggregate(v) for k, v in sorted(blocks.items(), key=lambda kv: int(kv[0]))}
    per_proj = {k: _aggregate(v) for k, v in sorted(projs.items())}
    totals = _aggregate(rows)
    eranks = [r["erank_delta"] for r in rows if r.get("erank_delta") is not None]
    totals["erank_delta_mean"] = (sum(eranks) / len(eranks)) if eranks else None
    return {"changed_tol": changed_tol, "block_regex": block_regex, "erank": bool(erank),
            "tensors": rows, "per_block": per_block, "per_proj": per_proj,
            "totals": totals}


CSV_FIELDS = ("name", "block", "proj", "shape", "numel", "fro_delta_sq", "fro_delta",
              "fro_base", "rel", "changed", "erank_delta", "erank_base")

#: The totals in report order: ``grad_mass`` is the headline, the relative
#: measures follow. Drivers use this as their summary-column order.
TOTALS_ORDER = ("grad_mass", "grad_mass_proj", "n_tensors", "n_changed", "numel",
                "changed_param_fraction", "fro_delta", "fro_base", "global_rel",
                "sum_rel", "mean_rel_weighted", "erank_delta_mean")


def write_weight_delta(report: dict, out_dir: str, *, prefix: str = "weight_delta") -> dict:
    """Write ``<prefix>.json`` and ``<prefix>.csv`` into ``out_dir``; returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"{prefix}.json")
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for r in report["tensors"]:
            row: dict[str, Any] = {k: r.get(k) for k in CSV_FIELDS}
            row["shape"] = "x".join(str(s) for s in r["shape"])
            row["changed"] = int(r["changed"])
            w.writerow(row)
    return {"json": json_path, "csv": csv_path}


def totals_summary(report: dict, prefix: str = "weight_delta") -> dict:
    """Flat ``{prefix/<key>: value}`` of the totals, ``grad_mass`` first (for a
    run-summary table)."""
    t = report["totals"]
    keys = [k for k in TOTALS_ORDER if k in t] + [k for k in t if k not in TOTALS_ORDER]
    return {f"{prefix}/{k}": t[k] for k in keys}
