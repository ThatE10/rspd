"""``backfill_kv_shared_tensors`` on a tiny synthetic checkpoint: the KV-shared
layers' ``k_proj`` / ``v_proj`` / ``k_norm`` are copied from the source into an
extra shard and the index covers every file. The source is a local
safetensors directory, so no hub access is needed."""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from srspd.models import export as ex  # noqa: E402

PREFIX = "model.language_model.layers"
N_LAYERS, N_SHARED, HIDDEN = 4, 2, 8


def _layer_tensors(i, *, seed):
    g = torch.Generator().manual_seed(seed * 100 + i)
    kv = 4 if i % 2 else 2
    return {
        f"{PREFIX}.{i}.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN, generator=g),
        f"{PREFIX}.{i}.self_attn.o_proj.weight": torch.randn(HIDDEN, HIDDEN, generator=g),
        f"{PREFIX}.{i}.self_attn.k_proj.weight": torch.randn(kv, HIDDEN, generator=g),
        f"{PREFIX}.{i}.self_attn.v_proj.weight": torch.randn(kv, HIDDEN, generator=g),
        f"{PREFIX}.{i}.self_attn.k_norm.weight": torch.randn(kv // 2, generator=g),
    }


def _write_config(d, *, n_shared):
    text = {"num_hidden_layers": N_LAYERS, "hidden_size": HIDDEN}
    if n_shared:
        text["num_kv_shared_layers"] = n_shared
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"model_type": "toy", "text_config": text}, f)


@pytest.fixture
def source(tmp_path):
    d = tmp_path / "source"
    d.mkdir()
    tensors = {}
    for i in range(N_LAYERS):
        tensors.update(_layer_tensors(i, seed=1))
    save_file(tensors, str(d / "model.safetensors"), metadata={"format": "pt"})
    return d


def _saved_checkpoint(tmp_path, *, n_shared, sharded=False):
    d = tmp_path / "ckpt"
    d.mkdir()
    tensors = {}
    for i in range(N_LAYERS):
        t = _layer_tensors(i, seed=2)
        if i >= N_LAYERS - n_shared:
            for suffix in ex.KV_SHARED_SUFFIXES:
                t.pop(f"{PREFIX}.{i}.self_attn.{suffix}")
        tensors.update(t)
    if sharded:
        keys = sorted(tensors)
        half = len(keys) // 2
        parts = {"model-00001-of-00002.safetensors": keys[:half],
                 "model-00002-of-00002.safetensors": keys[half:]}
        for fname, ks in parts.items():
            save_file({k: tensors[k] for k in ks}, str(d / fname), metadata={"format": "pt"})
        with open(d / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": 0},
                       "weight_map": {k: fn for fn, ks in parts.items() for k in ks}}, f)
    else:
        save_file(tensors, str(d / "model.safetensors"), metadata={"format": "pt"})
    _write_config(d, n_shared=n_shared)
    return d


def _expected_missing():
    return sorted(f"{PREFIX}.{i}.self_attn.{s}" for i in range(N_LAYERS - N_SHARED, N_LAYERS)
                  for s in ex.KV_SHARED_SUFFIXES)


def _assert_repaired(ckpt, source):
    with safe_open(str(ckpt / ex.KV_SHARED_SHARD), framework="pt") as f, \
            safe_open(str(source / "model.safetensors"), framework="pt") as src:
        assert sorted(f.keys()) == _expected_missing()
        for k in f.keys():
            assert torch.equal(f.get_tensor(k), src.get_tensor(k)), k
    with open(ckpt / "model.safetensors.index.json") as f:
        index = json.load(f)
    wm = index["weight_map"]
    all_keys = {}
    for fname in sorted(os.listdir(ckpt)):
        if fname.endswith(".safetensors"):
            with safe_open(str(ckpt / fname), framework="pt") as f:
                all_keys.update({k: fname for k in f.keys()})
    assert wm == all_keys
    assert index["metadata"]["total_size"] == sum(os.path.getsize(ckpt / fn)
                                                  for fn in set(wm.values()))


def test_backfills_missing_shared_layer_tensors_from_source(tmp_path, source):
    ckpt = _saved_checkpoint(tmp_path, n_shared=N_SHARED)
    before = os.path.getmtime(ckpt / "model.safetensors")
    keys = ex.backfill_kv_shared_tensors(str(ckpt), str(source))
    assert sorted(keys) == _expected_missing()
    assert os.path.getmtime(ckpt / "model.safetensors") == before      # not rewritten
    _assert_repaired(ckpt, source)


def test_sharded_checkpoint_index_is_extended(tmp_path, source):
    ckpt = _saved_checkpoint(tmp_path, n_shared=N_SHARED, sharded=True)
    assert sorted(ex.backfill_kv_shared_tensors(str(ckpt), str(source))) == _expected_missing()
    _assert_repaired(ckpt, source)


def test_rerun_is_idempotent(tmp_path, source):
    ckpt = _saved_checkpoint(tmp_path, n_shared=N_SHARED)
    assert ex.backfill_kv_shared_tensors(str(ckpt), str(source))
    snapshot = {fn: os.path.getmtime(ckpt / fn) for fn in os.listdir(ckpt)}
    assert ex.backfill_kv_shared_tensors(str(ckpt), str(source)) == []
    assert {fn: os.path.getmtime(ckpt / fn) for fn in os.listdir(ckpt)} == snapshot


def test_no_kv_sharing_is_a_noop(tmp_path, source):
    ckpt = _saved_checkpoint(tmp_path, n_shared=0)
    files = sorted(os.listdir(ckpt))
    assert ex.backfill_kv_shared_tensors(str(ckpt), str(source)) == []
    assert sorted(os.listdir(ckpt)) == files
    with open(ckpt / "config.json", "w") as f:
        json.dump({"model_type": "toy", "num_hidden_layers": N_LAYERS}, f)
    assert ex.backfill_kv_shared_tensors(str(ckpt), str(source)) == []


def test_safe_wrapper_never_raises(tmp_path):
    ckpt = _saved_checkpoint(tmp_path, n_shared=N_SHARED)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert ex.backfill_kv_shared_tensors_safe(str(ckpt), str(empty)) == ([], None)
    assert ex.backfill_kv_shared_tensors_safe(str(ckpt), None) == ([], None)
    keys, warn = ex.backfill_kv_shared_tensors_safe(str(ckpt), str(tmp_path / "missing-dir-x"))
    assert keys == [] and warn is not None and "failed" in warn
    assert not (ckpt / ex.KV_SHARED_SHARD).exists()
