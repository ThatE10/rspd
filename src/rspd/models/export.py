"""Write the currently edited model as a loadable HF checkpoint.

:func:`export_checkpoint` calls ``save_pretrained(safe_serialization=True)`` in
the model's own dtype (nothing is cast; the dtype is recorded), saves the
tokenizer best-effort, and applies the ``kv_shared_layers`` quirk:
:func:`backfill_kv_shared_tensors` copies the ``k_proj`` / ``v_proj`` /
``k_norm`` tensors that ``transformers`` does not build on KV-shared layers
(and so does not save) from the source checkpoint into an extra shard, so a
loader that constructs them unconditionally accepts the export. Neither
framework reads those tensors on a shared layer, so the source values are
copied verbatim.

``safetensors`` / ``huggingface_hub`` are imported lazily (the ``[hf]`` extra).
"""
from __future__ import annotations

import json
import os
import time

__all__ = ["export_checkpoint", "backfill_kv_shared_tensors", "backfill_kv_shared_tensors_safe",
           "KV_SHARED_SHARD", "KV_SHARED_SUFFIXES"]

#: The extra shard :func:`backfill_kv_shared_tensors` writes (matches ``*.safetensors``).
KV_SHARED_SHARD = "model-kv_shared.safetensors"
#: The per-layer attention tensors a KV-shared layer does not own in transformers.
KV_SHARED_SUFFIXES = ("k_proj.weight", "v_proj.weight", "k_norm.weight")


def _dir_bytes(path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _safetensors_key_map(dir_path: str) -> dict:
    """``{tensor name: shard basename}`` over every ``*.safetensors`` in
    ``dir_path`` (header-only reads), merged with ``model.safetensors.index.json``."""
    import glob

    from safetensors import safe_open

    key_map = {}
    for path in sorted(glob.glob(os.path.join(dir_path, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                key_map[k] = os.path.basename(path)
    index = os.path.join(dir_path, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            key_map.update(json.load(f).get("weight_map", {}))
    return key_map


def _source_shard_for_keys(source: str, keys, *, token=None) -> dict:
    """``{key: local safetensors path}`` for ``keys`` in the source checkpoint: a
    local directory (scanned) or a hub id (only the shards holding ``keys`` are
    fetched, normally from the local HF cache). Keys the source lacks are left out."""
    if os.path.isdir(source):
        key_map = _safetensors_key_map(source)
        return {k: os.path.join(source, key_map[k]) for k in keys if k in key_map}
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        index_path = hf_hub_download(source, "model.safetensors.index.json", token=token)
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    except EntryNotFoundError:
        weight_map = None
    out, fetched = {}, {}
    for k in keys:
        fname = "model.safetensors" if weight_map is None else weight_map.get(k)
        if fname is None:
            continue
        if fname not in fetched:
            fetched[fname] = hf_hub_download(source, fname, token=token)
        out[k] = fetched[fname]
    if weight_map is None:
        from safetensors import safe_open

        with safe_open(fetched["model.safetensors"], framework="pt") as f:
            present = set(f.keys())
        out = {k: v for k, v in out.items() if k in present}
    return out


def backfill_kv_shared_tensors(ckpt_dir: str, source: str, token=None) -> list:
    """Copy the KV-shared layers' dead ``k_proj`` / ``v_proj`` / ``k_norm``
    tensors from ``source`` (local safetensors dir or hub id) into
    ``ckpt_dir``; return the backfilled key names.

    The layers ``i >= num_hidden_layers - num_kv_shared_layers`` (read from
    ``ckpt_dir/config.json``, ``text_config`` first) are KV-shared. Key names
    come from the checkpoint's own ``<prefix>.<i>.self_attn.q_proj.weight``
    family; keys the source lacks (e.g. ``v_proj`` when K = V) are skipped. The
    copies go into :data:`KV_SHARED_SHARD`; ``model.safetensors.index.json`` is
    written / extended so ``weight_map`` covers every shard (existing shards
    are not rewritten). No-op (``[]``) without positive ``num_kv_shared_layers``
    or when nothing is missing, so it is idempotent.
    """
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return []
    with open(cfg_path) as f:
        cfg = json.load(f)
    text = cfg.get("text_config") or cfg
    n_shared = int(text.get("num_kv_shared_layers") or 0)
    if n_shared <= 0:
        return []
    n_layers = int(text["num_hidden_layers"])
    first_shared = n_layers - n_shared

    key_map = _safetensors_key_map(ckpt_dir)
    if not key_map:
        return []
    prefixes = {k[: -len(".self_attn.q_proj.weight")].rsplit(".", 1)[0]
                for k in key_map if k.endswith(".self_attn.q_proj.weight")}
    prefixes = {p for p in prefixes if f"{p}.{n_layers - 1}.self_attn.q_proj.weight" in key_map}
    if len(prefixes) != 1:
        raise ValueError(f"backfill_kv_shared_tensors: expected one `<prefix>.<i>` decoder "
                         f"family in {ckpt_dir}, found {sorted(prefixes)}")
    prefix = prefixes.pop()
    expected = [f"{prefix}.{i}.self_attn.{suffix}"
                for i in range(first_shared, n_layers) for suffix in KV_SHARED_SUFFIXES]
    missing = [k for k in expected if k not in key_map]
    if not missing:
        return []

    src_files = _source_shard_for_keys(source, missing, token=token)
    copied = [k for k in missing if k in src_files]
    if not copied:
        return []

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    for path in sorted(set(src_files.values())):
        with safe_open(path, framework="pt") as f:
            for k in copied:
                if src_files[k] == path:
                    tensors[k] = f.get_tensor(k)
    save_file(tensors, os.path.join(ckpt_dir, KV_SHARED_SHARD), metadata={"format": "pt"})

    index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
    weight_map = dict(key_map)
    weight_map.update({k: KV_SHARED_SHARD for k in copied})
    total = sum(os.path.getsize(os.path.join(ckpt_dir, fname))
                for fname in set(weight_map.values()))
    index = {"metadata": {"total_size": int(total)},
             "weight_map": dict(sorted(weight_map.items()))}
    if os.path.isfile(index_path):
        with open(index_path) as f:
            old = json.load(f)
        index["metadata"] = {**old.get("metadata", {}), **index["metadata"]}
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    return copied


def backfill_kv_shared_tensors_safe(ckpt_dir: str, source, *, token=None) -> tuple:
    """:func:`backfill_kv_shared_tensors` that never raises (HF loading does not
    need the tensors). Returns ``(keys, warning or None)``."""
    if not source:
        return [], None
    try:
        return backfill_kv_shared_tensors(ckpt_dir, source, token=token), None
    except Exception as exc:
        return [], (f"KV-shared backfill of {ckpt_dir} from {source} failed "
                    f"({type(exc).__name__}: {exc}); HF loading is unaffected")


def export_checkpoint(model, tokenizer, out_dir, *, source=None, token=None) -> dict:
    """``save_pretrained`` the model as it is now (edited, if an erasure is
    applied) to ``out_dir`` in its own dtype, the tokenizer beside it
    (best-effort), and the KV-shared backfill from ``source`` (default the
    model's ``config._name_or_path``).

    Returns ``{"dir", "bytes", "wall_s", "dtype", "tokenizer_saved",
    "kv_shared_backfilled", "warnings"}``.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    warnings = []
    try:
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
    except Exception:
        dtypes = []
    model.save_pretrained(out_dir, safe_serialization=True)
    tok_ok = tokenizer is not None
    if tokenizer is not None:
        try:
            tokenizer.save_pretrained(out_dir)
        except Exception as exc:
            tok_ok = False
            warnings.append(f"tokenizer not saved to {out_dir}: {type(exc).__name__}: {exc}")
    src = source or getattr(getattr(model, "config", None), "_name_or_path", None) or None
    keys, warn = backfill_kv_shared_tensors_safe(out_dir, src, token=token)
    if warn:
        warnings.append(warn)
    from .quirks import num_kv_shared_layers
    if num_kv_shared_layers(model) > 0 and not keys and not src:
        warnings.append("model has KV-shared layers but no source checkpoint was given; their "
                        "k_proj/v_proj/k_norm were not backfilled (pass source=)")
    return {"dir": str(out_dir), "bytes": int(_dir_bytes(out_dir)),
            "wall_s": round(time.time() - t0, 2),
            "dtype": (None if not dtypes else dtypes[0] if len(dtypes) == 1 else dtypes),
            "tokenizer_saved": tok_ok, "kv_shared_backfilled": keys, "warnings": warnings}
