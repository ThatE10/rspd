"""Registry of named model quirks: what each one is, how it is detected, how
it is handled.

A quirk is a property of a model that the plain recipe (hook a projection,
read its input, edit ``W`` as ``[d_out, d_in]``) would get silently wrong.
Each :class:`Quirk` has a ``detect(model, harness)`` that returns a
JSON-serialisable detail dict when the quirk applies (``None`` otherwise) and
names the code that handles it. :func:`detect_quirks` runs every registered
detector; :func:`rspd.inspect_model` and :func:`rspd.erase` report the result
under ``"quirks"``.

Registered quirks
-----------------
``conv1d_transposed``
    ``transformers`` ``Conv1D`` (GPT-2) stores ``W`` as ``[d_in, d_out]``.
    Handled by :func:`rspd.models.harness.as_dout_din` /
    :func:`~rspd.models.harness.write_dout_din`; its bias is subtracted at
    harvest and never edited.
``biased_projection``
    A biased ``nn.Linear``. Refused by default; with ``subtract_bias=True`` the
    linear map is edited and the bias left alone.
``multimodal_towers``
    A wrapper whose vision / audio towers contain modules matching the layers
    regex. Handled by the selector's exclusion regex
    (:data:`rspd.models.selector.DEFAULT_EXCLUDE_REGEX`) and the
    ``hidden_size`` cross-check of the harness.
``kv_shared_layers``
    The last ``num_kv_shared_layers`` decoder layers reuse the K/V of an
    earlier layer and own no ``k_proj`` / ``v_proj`` / ``k_norm`` in
    ``transformers``. Handled twice: the boundary resume of the Gram passes is
    disabled (a resumed later layer would read K/V that a skipped layer
    stores), and the export backfills the missing tensors from the source
    checkpoint so loaders that build them unconditionally accept it
    (:func:`rspd.models.export.backfill_kv_shared_tensors`).
``shared_input_check``
    Several projections read one input (q/k/v, gate/up); one Gram serves them
    all, verified bitwise on the first sequence of every pass
    (:class:`rspd.capture.hf.HFHookSource`).
``bf16_edit_in_fp32``
    Reduced-precision weights (bf16 / fp16): the edit ``W - M`` is formed in
    fp32 and cast once on write; the fraction of ``-M`` surviving the cast is
    reported (``edit.survival``) because at a small dose the edit can fall
    below the weight dtype's resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .harness import group_by_input, is_conv1d

__all__ = ["Quirk", "QUIRKS", "register_quirk", "detect_quirks", "num_kv_shared_layers",
           "text_config"]


@dataclass(frozen=True)
class Quirk:
    name: str
    description: str
    detect: Callable
    handled_by: str


QUIRKS: dict = {}


def register_quirk(q: Quirk) -> Quirk:
    """Add (or replace) a quirk in the registry."""
    QUIRKS[q.name] = q
    return q


def text_config(model_or_config):
    """The language model's config: ``config.text_config`` of a multimodal
    wrapper, else the config itself (``None`` if there is none)."""
    cfg = getattr(model_or_config, "config", model_or_config)
    if cfg is None:
        return None
    return getattr(cfg, "text_config", None) or cfg


def num_kv_shared_layers(model_or_config) -> int:
    tc = text_config(model_or_config)
    return int(getattr(tc, "num_kv_shared_layers", 0) or 0) if tc is not None else 0


def _conv1d(model, h):
    names = [n for n in h.all_names if is_conv1d(h.module_by_name[n])]
    return {"matrices": len(names), "examples": names[:4]} if names else None


def _biased(model, h):
    names = list(getattr(h, "biased_names", []))
    return ({"matrices": len(names), "examples": names[:4], "bias_edited": False}
            if names else None)


def _towers(model, h):
    excl = list(getattr(h, "excluded_paths", []))
    cfg = getattr(model, "config", None)
    towers = [k for k in ("vision_config", "audio_config") if getattr(cfg, k, None) is not None]
    if not excl and not towers:
        return None
    roots = sorted({p.split(".layers.")[0] if ".layers." in p else p.rsplit(".", 1)[0]
                    for p in excl})
    return {"excluded_modules": len(excl), "excluded_roots": roots[:8],
            "config_towers": towers, "exclude_regex": h.selector.exclude_regex}


def _kv_shared(model, h):
    n = num_kv_shared_layers(model)
    if n <= 0:
        return None
    tc = text_config(model)
    n_layers = int(getattr(tc, "num_hidden_layers", 0) or 0)
    return {"num_kv_shared_layers": n, "first_shared_layer": n_layers - n,
            "num_hidden_layers": n_layers,
            "effects": ["boundary resume of the Gram passes disabled",
                        "export backfills k_proj/v_proj/k_norm of the shared layers from the "
                        "source checkpoint"]}


def _shared_input(model, h):
    groups = {k: m for k, m in group_by_input(h.all_names).items() if len(m) > 1}
    if not groups:
        return None
    k0 = next(iter(groups))
    return {"shared_keys": len(groups), "example": {k0: groups[k0]},
            "verified": "bitwise on the first sequence of every Gram pass and every harvest"}


def _reduced_precision(model, h):
    dts = sorted({str(h.module_by_name[n].weight.dtype) for n in h.all_names
                  if h.module_by_name[n].weight.dtype in (torch.bfloat16, torch.float16)})
    return {"weight_dtypes": dts, "edit_dtype": "torch.float32"} if dts else None


for _q in (
    Quirk("conv1d_transposed", "Conv1D stores W as [d_in, d_out]; its bias is subtracted at "
          "harvest and never edited", _conv1d, "rspd.models.harness.as_dout_din"),
    Quirk("biased_projection", "biased nn.Linear projections, accepted with subtract_bias=True; "
          "the bias is never edited", _biased, "rspd.models.harness.HFLinearHarness"),
    Quirk("multimodal_towers", "vision/audio towers excluded from the selection", _towers,
          "rspd.models.selector.DEFAULT_EXCLUDE_REGEX"),
    Quirk("kv_shared_layers", "layers sharing K/V own no k_proj/v_proj/k_norm: boundary resume "
          "off, export backfill", _kv_shared,
          "rspd.models.export.backfill_kv_shared_tensors"),
    Quirk("shared_input_check", "projections sharing one input share one Gram, verified "
          "bitwise", _shared_input, "rspd.capture.hf.HFHookSource"),
    Quirk("bf16_edit_in_fp32", "reduced-precision weights: the edit is formed in fp32 and cast "
          "once; survival reported", _reduced_precision, "rspd.pipeline.apply_dose"),
):
    register_quirk(_q)


def detect_quirks(model, harness) -> dict:
    """``{quirk name: detail}`` for every registered quirk that applies."""
    out = {}
    for name, q in QUIRKS.items():
        d = q.detect(model, harness)
        if d is not None:
            out[name] = d
    return out
