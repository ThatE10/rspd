"""``inspect_model``: what an erasure would touch, before any forward pass.

Reports, as a JSON-serialisable dict: the selector, every selected projection
(path, module type, ``[d_out, d_in]``, dtype, device, input key), the input
keys with their members, ``d_in`` and float64 Gram bytes (one Gram per key for
the ``readout`` solver, two for ``grams``), the quirks that fire
(:mod:`srspd.models.quirks`), and warnings. A model id / local path is
instantiated on the ``meta`` device from its config (no weights are loaded).
"""
from __future__ import annotations

import torch

from .harness import HFLinearHarness, group_by_input, is_conv1d
from .quirks import detect_quirks, text_config
from .selector import resolve_selector

__all__ = ["inspect_model", "load_skeleton"]


def load_skeleton(model_id: str, *, trust_remote_code: bool = False):
    """The module tree of ``model_id`` (hub id or local directory) on the ``meta``
    device: config only, no weights. Tries a causal-LM head, then an
    image-text-to-text wrapper, then the bare model class."""
    try:
        import transformers
    except ImportError as e:
        raise ImportError("inspect_model(<model id>) needs transformers (pip install "
                          "'srspd[hf]'); or pass an instantiated model") from e
    cfg = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    errors = []
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            with torch.device("meta"):
                return cls.from_config(cfg, trust_remote_code=trust_remote_code)
        except Exception as exc:          # unknown mapping for this config: try the next
            errors.append(f"{cls_name}: {type(exc).__name__}: {exc}")
    raise ValueError(f"cannot instantiate {model_id!r} from its config: {errors}")


def inspect_model(model_or_id, layers=None, *, subtract_bias=False, gram_budget_gib=None,
                  trust_remote_code=False) -> dict:
    """Describe what :func:`srspd.erase` would edit on this model.

    Parameters
    ----------
    model_or_id : an ``nn.Module`` (HF model) or a hub id / local path.
    layers : the layer selector (:func:`srspd.models.selector.resolve_selector`).
    subtract_bias : as in :func:`srspd.erase`.
    gram_budget_gib : if given, also plan the Gram passes under this per-device
        budget for both solvers (``pass_plan``).

    Returns
    -------
    dict with ``model`` (class, config type, n_parameters when known),
    ``selector``, ``matrices`` (``{name: {path, module, d_out, d_in, dtype,
    device, bias, input_key}}``), ``input_keys`` (``{key: {members, d_in,
    gram_bytes_readout, gram_bytes_grams, device}}``), ``totals`` (counts and
    total Gram bytes per solver), ``quirks`` (``{name: detail}``), ``warnings``
    and, with ``gram_budget_gib``, ``pass_plan``.
    """
    model = model_or_id
    source = None
    if isinstance(model_or_id, str):
        source = model_or_id
        model = load_skeleton(model_or_id, trust_remote_code=trust_remote_code)
    warnings = []
    sel = resolve_selector(model, layers)
    h = HFLinearHarness(model, selector=sel, subtract_bias=subtract_bias)
    matrices = {}
    for n in h.all_names:
        mod = h.module_by_name[n]
        w = mod.weight
        shape = tuple(w.shape)
        d_out, d_in = (shape[1], shape[0]) if is_conv1d(mod) else (shape[0], shape[1])
        matrices[n] = {"path": h.path_by_name[n],
                       "module": "Conv1D" if is_conv1d(mod) else type(mod).__name__,
                       "d_out": int(d_out), "d_in": int(d_in), "dtype": str(w.dtype),
                       "device": str(w.device), "bias": getattr(mod, "bias", None) is not None,
                       "input_key": None}
    keys = {}
    for k, members in group_by_input(h.all_names).items():
        d = matrices[members[0]]["d_in"]
        for m in members:
            matrices[m]["input_key"] = k
        widths = {matrices[m]["d_in"] for m in members}
        if len(widths) != 1:
            warnings.append(f"input key {k} groups matrices of different d_in {sorted(widths)}")
        keys[k] = {"members": list(members), "d_in": int(d),
                   "gram_bytes_readout": 8 * d * d, "gram_bytes_grams": 16 * d * d,
                   "device": matrices[members[0]]["device"]}
    quirks = detect_quirks(model, h)
    tc = text_config(model)
    n_layers = getattr(tc, "num_hidden_layers", None) if tc is not None else None
    if isinstance(n_layers, int) and h.n_blocks != n_layers and sel.blocks is None:
        warnings.append(f"selected {h.n_blocks} blocks but the config has {n_layers} "
                        f"layers: check the selector")
    try:
        n_params = int(sum(p.numel() for p in model.parameters()))
    except Exception:
        n_params = None
    cfg = getattr(model, "config", None)
    out = {
        "model": {"class": type(model).__name__, "source": source,
                  "config_type": getattr(cfg, "model_type", None),
                  "n_parameters": n_params, "n_layers_config": n_layers},
        "selector": sel.as_dict(),
        "matrices": matrices,
        "input_keys": keys,
        "totals": {"matrices": len(matrices), "blocks": h.n_blocks, "input_keys": len(keys),
                   "edited_parameters": int(sum(v["d_out"] * v["d_in"]
                                                for v in matrices.values())),
                   "gram_bytes_readout": int(sum(v["gram_bytes_readout"] for v in keys.values())),
                   "gram_bytes_grams": int(sum(v["gram_bytes_grams"] for v in keys.values()))},
        "quirks": quirks,
        "warnings": warnings,
    }
    if gram_budget_gib is not None:
        from ..capture.grams import plan_gram_groups
        b = int(float(gram_budget_gib) * 2 ** 30)
        items = [(k, v["d_in"], v["device"]) for k, v in keys.items()]
        plans = {}
        for solver, bpe in (("readout", 8), ("grams", 16)):
            try:
                plans[solver] = plan_gram_groups(items, b, bytes_per_elem=bpe)
            except ValueError as e:
                plans[solver] = {"error": str(e)}
        out["pass_plan"] = plans
    return out
