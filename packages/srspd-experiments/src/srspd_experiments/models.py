"""Model loading, block-parameter resolution and checkpoint export for the
method runners (:mod:`srspd_experiments.methods`).

* :func:`load_config` / :func:`load_model` / :func:`load_tokenizer_and_processor`
  / :func:`require_modern_stack` -- the generic loaders (a causal LM first, an
  image-text-to-text wrapper as the fallback for multimodal checkpoints);
* :func:`load_causal_lm` -- ``(model, tokenizer)`` with the tokenizer rules of
  :func:`srspd_experiments.data.corpora.load_tokenizer` (pad = eos when absent,
  padding / truncation side opt-in, the tokenizer optionally from another repo);
* :func:`resolve_model_source` / :func:`model_fingerprint` -- a data-root
  checkpoint reference (``models/...``) to an absolute path, and a cheap content
  fingerprint of a local checkpoint;
* :func:`get_decoder_layers` / :func:`get_block_module` /
  :func:`get_mlp_out_module` / :func:`weight_layout` /
  :func:`get_named_block_params` -- the one place that knows where a decoder's
  blocks live (selection is by attribute name, never by a positional
  ``.parameters()`` index);
* :func:`save_model` / :func:`save_processor` / :func:`export_checkpoint` --
  ``save_pretrained`` plus, for a multimodal checkpoint, the source model's full
  processor (an inference server that resolves the architecture as multimodal
  needs its image-processor files), plus the KV-shared tensor backfill of
  :func:`srspd.models.export.export_checkpoint`.

``torch`` / ``transformers`` are imported inside the functions.
"""
from __future__ import annotations

import hashlib
import json
import os

from .data.corpora import load_tokenizer
from .paths import data_dir, hf_token

MIN_TRANSFORMERS = (4, 55)

__all__ = ["MIN_TRANSFORMERS", "require_modern_stack", "load_config", "load_model",
           "load_tokenizer_and_processor", "load_tokenizer", "load_causal_lm",
           "resolve_model_source", "model_fingerprint", "get_decoder_layers",
           "get_block_module", "get_mlp_out_module", "get_down_proj_module", "weight_layout",
           "get_named_block_params", "get_named_down_proj", "MLP_OUT_MODULE_NAMES",
           "DEFAULT_MLP_OUT_MODULE", "DEFAULT_BLOCK_PARAM_NAMES", "save_model",
           "save_processor", "export_checkpoint", "is_multimodal", "write_metrics"]


def require_modern_stack():
    """``torch`` after checking ``transformers >= MIN_TRANSFORMERS`` (older
    releases lack the newer decoder architectures); raises ``SystemExit``."""
    import torch  # noqa: F401
    import transformers
    ver = tuple(int(p) for p in transformers.__version__.split(".")[:2])
    if ver < MIN_TRANSFORMERS:
        raise SystemExit(
            f"transformers {transformers.__version__} is too old (need >= "
            f"{'.'.join(map(str, MIN_TRANSFORMERS))}); install the pinned stack of "
            f"srspd-experiments' constraints.txt")
    return torch


def load_config(path: str) -> dict:
    """A YAML run config; ``name`` and ``model`` are required."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("name", "model"):
        if key not in cfg:
            raise ValueError(f"{path}: config needs '{key}'")
    return cfg


def load_model(cfg: dict, source: str):
    """Load ``source`` (hub id or checkpoint dir) per the config's ``dtype`` /
    ``device_map``. ``model_class: auto`` tries ``AutoModelForCausalLM`` first,
    then ``AutoModelForImageTextToText`` (multimodal wrappers)."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, cfg.get("dtype", "float32"))
    kwargs = dict(torch_dtype=dtype, device_map=cfg.get("device_map", "cuda"))
    want = cfg.get("model_class", "auto")
    if want in ("auto", "causal_lm"):
        try:
            return AutoModelForCausalLM.from_pretrained(source, **kwargs)
        except ValueError:
            if want == "causal_lm":
                raise
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(source, **kwargs)


def load_tokenizer_and_processor(model_id: str):
    """``(tokenizer, processor or None)``; the processor is ``None`` for a
    text-only model (where ``AutoProcessor`` returns the tokenizer itself)."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token())
    processor = None
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token())
        if type(processor).__name__ == type(tokenizer).__name__:
            processor = None
    except Exception:
        processor = None
    return tokenizer, processor


def resolve_model_source(model_ref: str, root=None) -> str:
    """A data-root-relative checkpoint reference -- a relative path starting
    ``models/`` that exists as a directory under the data root -- resolved to
    its absolute path; anything else (a hub id, an absolute path) unchanged.
    Only the LOAD uses the resolved path; records keep the config's string."""
    if not os.path.isabs(str(model_ref)) and str(model_ref).startswith("models/"):
        cand = data_dir(root, create=False) / str(model_ref)
        if cand.is_dir():
            return str(cand)
    return model_ref


def model_fingerprint(model_src, sample_bytes: int = 1 << 20):
    """Content fingerprint of a LOCAL checkpoint directory (each weight file's
    name, size and three ``sample_bytes`` samples), or ``None`` for a hub id."""
    if not os.path.isdir(str(model_src)):
        return None
    files = sorted(f for f in os.listdir(model_src)
                   if f.endswith((".safetensors", ".bin", ".pt")))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        path = os.path.join(model_src, f)
        size = os.path.getsize(path)
        h.update(f"{f}:{size}".encode())
        with open(path, "rb") as fh:
            for off in sorted({0, max(0, size // 2 - sample_bytes // 2),
                               max(0, size - sample_bytes)}):
                fh.seek(off)
                h.update(fh.read(sample_bytes))
    return h.hexdigest()[:16]


def load_causal_lm(model_id: str, *, dtype: str = "bfloat16", device_map="auto",
                   padding_side: str | None = None, truncation_side: str | None = None,
                   tokenizer_id: str | None = None):
    """``(model, tokenizer)`` for ``model_id`` (hub id, checkpoint dir or a
    data-root ``models/...`` reference).

    * ``dtype`` by name (``bfloat16`` default);
    * a multimodal wrapper that is not in the causal-LM auto-map loads through
      ``AutoModelForImageTextToText``;
    * the tokenizer follows :func:`srspd_experiments.data.corpora.load_tokenizer`
      (pad = eos when it has none; ``padding_side`` / ``truncation_side`` are
      opt-in -- ``None`` keeps the tokenizer's own default);
    * ``tokenizer_id`` loads the tokenizer from a different repo than the
      weights (checkpoints that ship no tokenizer files).

    The pad side matters to the activation-matching baselines (their losses
    cover pad positions), so they pass both sides explicitly. The effective
    values are printed so a run log records what ran.
    """
    import torch
    from transformers import AutoModelForCausalLM

    token = hf_token()
    torch_dtype = getattr(torch, dtype)
    source = resolve_model_source(model_id)
    if source != model_id:
        print(f"[load] {model_id} -> {source}", flush=True)
    kwargs = dict(torch_dtype=torch_dtype, device_map=device_map, token=token)
    try:
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    except ValueError:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
    print(f"[load] {model_id} loaded as {type(model).__name__}", flush=True)

    tokenizer = load_tokenizer(tokenizer_id or source, padding_side=padding_side,
                               truncation_side=truncation_side)
    print(f"[load] {model_id} tokenizer"
          + (f" (from {tokenizer_id})" if tokenizer_id else "") + ": "
          f"padding_side={getattr(tokenizer, 'padding_side', None)!r} "
          f"truncation_side={getattr(tokenizer, 'truncation_side', None)!r} "
          f"pad_token={getattr(tokenizer, 'pad_token', None)!r}"
          + ("" if (padding_side is None and truncation_side is None)
             else "  (sides set explicitly by the runner)"), flush=True)
    return model, tokenizer


# --------------------------------------------------------------------------- #
# Module-path resolution                                                       #
# --------------------------------------------------------------------------- #
#: Candidate attribute chains from the top-level model down to the decoder's
#: ModuleList of blocks (a causal LM; the two nestings of a multimodal wrapper;
#: defensive variants; GPT-2).
_DECODER_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.model.layers",
    "language_model.layers",
    "transformer.h",
)


def _getattr_path(obj, dotted: str):
    cur = obj
    for attr in dotted.split("."):
        if not hasattr(cur, attr):
            return None
        cur = getattr(cur, attr)
    return cur


def get_decoder_layers(model):
    """The decoder's ``ModuleList`` of blocks, probing :data:`_DECODER_LAYER_PATHS`;
    raises ``AttributeError`` listing every path tried."""
    import torch.nn as nn

    tried = []
    for path in _DECODER_LAYER_PATHS:
        tried.append(path)
        cand = _getattr_path(model, path)
        if isinstance(cand, nn.ModuleList) and len(cand) > 0:
            return cand
    raise AttributeError(
        f"could not locate the decoder layer ModuleList on {type(model).__name__}; "
        f"tried (as attribute chains under the model): {tried}. "
        "Add the correct path to srspd_experiments.models._DECODER_LAYER_PATHS.")


def get_block_module(model, layer_id: int):
    """Decoder block ``layer_id`` -- the module whose output residual stream
    the activation-matching baselines hook."""
    layers = get_decoder_layers(model)
    if not (0 <= layer_id < len(layers)):
        raise IndexError(f"layer_id {layer_id} out of range for {len(layers)} decoder layers")
    return layers[layer_id]


#: The MLP output projection: ``mlp.down_proj`` (an ``nn.Linear``, weight
#: ``[out, in]``) or ``mlp.c_proj`` (GPT-2's ``Conv1D``, weight ``[in, out]``).
MLP_OUT_MODULE_NAMES = ("mlp.down_proj", "mlp.c_proj")
DEFAULT_MLP_OUT_MODULE = "mlp.down_proj"


def get_mlp_out_module(model, layer_id: int, name: str = DEFAULT_MLP_OUT_MODULE):
    """The MLP output projection ``name`` of block ``layer_id``, by attribute name."""
    if name not in MLP_OUT_MODULE_NAMES:
        raise ValueError(f"module_name must be one of {MLP_OUT_MODULE_NAMES}, got {name!r}")
    block = get_block_module(model, layer_id)
    mod = _getattr_path(block, name)
    if mod is None:
        raise AttributeError(
            f"block {layer_id} ({type(block).__name__}) has no {name}; the target "
            f"parameter is selected by name, so this path must exist "
            f"(an nn.Linear decoder has mlp.down_proj, GPT-2 has mlp.c_proj)")
    return mod


def get_down_proj_module(model, layer_id: int):
    """``mlp.down_proj`` of block ``layer_id``."""
    return get_mlp_out_module(model, layer_id, DEFAULT_MLP_OUT_MODULE)


def weight_layout(module) -> str:
    """``"in_out"`` for a ``transformers`` ``Conv1D`` (weight ``[d_in, d_out]``),
    ``"out_in"`` for an ``nn.Linear``."""
    return "in_out" if type(module).__name__ == "Conv1D" else "out_in"


#: The default trained parameter per block (the published RMU / KUDA recipes').
DEFAULT_BLOCK_PARAM_NAMES = ("mlp.down_proj.weight",)


def get_named_block_params(model, layer_ids, param_names=DEFAULT_BLOCK_PARAM_NAMES):
    """``{fully_qualified_name: Parameter}`` for each dotted ``param_names``
    entry (relative to a decoder block) of each block in ``layer_ids``, matched
    by identity against ``model.named_parameters()``.

    Selection is by name, never by a positional index. Note that on a decoder
    whose block normalises the MLP branch before the residual add (an RMSNorm
    after the MLP) and has no MLP bias, ``mlp.down_proj.weight`` alone is a
    scale-null direction of any activation-space loss; name other tensors there.
    """
    import torch.nn as nn

    if isinstance(param_names, str):
        param_names = (param_names,)
    param_names = tuple(param_names)
    if not param_names:
        raise ValueError("param_names must name at least one parameter per block "
                         f"(default {list(DEFAULT_BLOCK_PARAM_NAMES)})")

    named = dict(model.named_parameters())
    out = {}
    for lid in layer_ids:
        block = get_block_module(model, lid)
        for dotted in param_names:
            target = _getattr_path(block, dotted)
            if target is None:
                raise AttributeError(
                    f"block {lid} ({type(block).__name__}) has no {dotted!r}; the "
                    f"target parameters are selected by name, so this attribute path "
                    f"must exist under the block "
                    f"(it has: {sorted(n for n, _ in block.named_parameters())})")
            if not isinstance(target, nn.Parameter):
                raise AttributeError(
                    f"block {lid} ({type(block).__name__}) resolves {dotted!r} to a "
                    f"{type(target).__name__}, not an nn.Parameter; name a *parameter* "
                    f"(e.g. 'mlp.down_proj.weight'), not the module holding it")
            match = next((n for n, p in named.items() if p is target), None)
            if match is None:
                raise AttributeError(
                    f"block {lid} ({type(block).__name__}) resolves {dotted!r} to a "
                    f"parameter that is not in the model's named_parameters()")
            out[match] = target
    return out


def get_named_down_proj(model, layer_ids):
    """:func:`get_named_block_params` at the default ``mlp.down_proj.weight``."""
    return get_named_block_params(model, layer_ids, DEFAULT_BLOCK_PARAM_NAMES)


# --------------------------------------------------------------------------- #
# Saving                                                                       #
# --------------------------------------------------------------------------- #
def save_processor(model, tokenizer, out_dir: str, *, processor=None, source=None) -> bool:
    """Save a processor beside a checkpoint when it is more than the tokenizer
    (a multimodal model's image/audio processor). ``processor`` given: saved as
    is. Otherwise the source checkpoint's ``AutoProcessor`` is loaded
    (``source``, default the model's ``config._name_or_path``), best-effort.
    Returns whether a processor was written; a text-only model is ``False``."""
    if processor is not None:
        if tokenizer is not None and type(processor).__name__ == type(tokenizer).__name__:
            return False
        processor.save_pretrained(out_dir)
        print(f"[save] processor ({type(processor).__name__}) -> {out_dir}", flush=True)
        return True
    src = source or getattr(getattr(model, "config", None), "_name_or_path", None)
    if not src:
        return False
    try:
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(src, token=hf_token())
        if type(proc).__name__ != type(tokenizer).__name__:
            proc.save_pretrained(out_dir)
            print(f"[save] processor ({type(proc).__name__}) -> {out_dir}", flush=True)
            return True
    except Exception:
        pass
    return False


#: ``config`` attributes that mark a checkpoint as multimodal (it then needs its processor).
MULTIMODAL_CONFIG_KEYS = ("vision_config", "audio_config", "image_token_id",
                          "image_token_index", "audio_token_id")


def is_multimodal(model) -> bool:
    """Whether the model's config carries a non-text tower (see
    :data:`MULTIMODAL_CONFIG_KEYS`)."""
    cfg = getattr(model, "config", None)
    return any(getattr(cfg, k, None) is not None for k in MULTIMODAL_CONFIG_KEYS)


def export_checkpoint(model, tokenizer, out_dir, *, source=None, processor=None,
                      save_source_processor="auto", token=None) -> dict:
    """:func:`srspd.models.export.export_checkpoint` (weights in the model's own
    dtype, the tokenizer, the KV-shared backfill from ``source``) plus the
    processor of a multimodal model: ``processor`` when given; otherwise, with
    ``save_source_processor`` (``"auto"`` = only when :func:`is_multimodal`), the
    source checkpoint's own ``AutoProcessor``, best-effort. A text-only model
    never triggers a processor lookup.

    Returns the core record plus ``processor_saved``.
    """
    from srspd.models.export import export_checkpoint as _core_export

    info = _core_export(model, tokenizer, out_dir, source=source,
                        token=token if token is not None else hf_token())
    saved = False
    if save_source_processor == "auto":
        save_source_processor = is_multimodal(model)
    if processor is not None or save_source_processor:
        try:
            saved = save_processor(model, tokenizer, str(out_dir), processor=processor,
                                   source=source)
        except Exception as exc:
            info["warnings"].append(f"processor not saved to {out_dir}: "
                                    f"{type(exc).__name__}: {exc}")
    info["processor_saved"] = bool(saved)
    return info


def save_model(model, tokenizer, out_dir: str, *, processor=None, source=None) -> str:
    """``save_pretrained`` the model and tokenizer under ``out_dir`` (with the
    processor and KV-shared backfill of :func:`export_checkpoint`); returns
    the directory. A backfill failure is a printed warning, never an error."""
    os.makedirs(out_dir, exist_ok=True)
    info = export_checkpoint(model, tokenizer, out_dir, source=source, processor=processor)
    for w in info["warnings"]:
        print(f"[save] WARNING: {w}", flush=True)
    print(f"[save] model + tokenizer -> {out_dir}", flush=True)
    return out_dir


def write_metrics(out_dir: str, metrics: dict) -> str:
    """Write ``metrics.json`` under ``out_dir``; return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "metrics.json")
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
    print(f"[metrics] -> {path}", flush=True)
    return path
