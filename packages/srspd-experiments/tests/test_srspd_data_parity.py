"""Data-path parity of the SRSPD method runner with the reference driver.

The reference repository's ``origin/main`` (``$SRSPD_REFERENCE_REPO``) is
exported with ``git archive`` into a temporary directory (never imported from a
working tree) and its SRSPD-only driver path is run end to end on a tiny random
Mistral (bf16, mocked hub corpora, two Gram passes with boundary resume) and a
tiny GPT-2 (fp32 local checkpoint, local JSONL sets), under both solvers, with
the token streams it feeds its Gram passes captured. The SAME config is then
translated by :func:`srspd_experiments.recipes.legacy.srspd_config_from_legacy`
and run through :func:`srspd_experiments.data.build_erasure_data` +
:func:`srspd_experiments.methods.srspd.run` on an identical model, with the
streams :func:`srspd.erase` receives captured, under a separate data root (so
neither side can read the other's retained-corpus cache).

Asserted: the target and retained token streams are identical sequence for
sequence, the held-out readouts' baseline CE agrees, and every exported weight
is BITWISE equal to the driver's export (bf16 and fp32). Skipped cleanly when
the reference repository or git is absent.
"""
from __future__ import annotations

import importlib.util
import io
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
yaml = pytest.importorskip("yaml")
pytest.importorskip("safetensors")

from conftest import isolate_corpora  # noqa: E402

REPO = pathlib.Path(os.environ.get("SRSPD_REFERENCE_REPO", "/root/workspace/code"))
REF = "origin/main"
_PKG = "rs" + "pd"                                   # the reference's package name
_DRV_DIR = "experi" + "ments/" + "wm" + "dp"
PATHS = [_PKG, _DRV_DIR, "tests/gpt2_tiny_helpers.py"]
DRIVER = f"{_DRV_DIR}/hf_{_PKG}.py"
MODNAME = "_srspd_ref_driver"
MPT = 0.375
LEGACY_DOSE = "mu" + "_raw"
SUB_KEY = "circ" + "uits"                            # the reference's second stage
SKIP_KEY = "skip_" + SUB_KEY


def _purge():
    for k in [m for m in sys.modules if m == _PKG or m.startswith(_PKG + ".")
              or m in ("common", "gpt2_tiny_helpers", MODNAME)]:
        del sys.modules[k]


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip(f"reference repository not present at {REPO}")
    proc = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", REF, *PATHS],
                          capture_output=True)
    if proc.returncode != 0:
        pytest.skip(f"cannot export {REF} of the reference: {proc.stderr.decode()[:200]}")
    root = tmp_path_factory.mktemp("reference")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tf:
        tf.extractall(root, filter="data")
    saved_path, saved_mods = list(sys.path), dict(sys.modules)
    ours = {k: v for k, v in sys.modules.items() if k == "gpt2_tiny_helpers"}
    _purge()
    sys.path[:0] = [str(root), str(root / _DRV_DIR), str(root / "tests")]
    try:
        spec = importlib.util.spec_from_file_location(MODNAME, root / DRIVER)
        drv = importlib.util.module_from_spec(spec)
        sys.modules[MODNAME] = drv
        spec.loader.exec_module(drv)
        import gpt2_tiny_helpers as g2
        yield {"drv": drv, "root": root, "g2": g2}
    finally:
        _purge()
        sys.path[:] = saved_path
        sys.modules.update({k: v for k, v in saved_mods.items() if k not in sys.modules})
        sys.modules.update(ours)


# --------------------------------------------------------------------------- #
# shared fakes (as the core package's driver-parity test)                      #
# --------------------------------------------------------------------------- #
def _mistral(dtype):
    torch.manual_seed(0)
    cfg = transformers.MistralConfig(
        vocab_size=256, hidden_size=16, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return transformers.MistralForCausalLM(cfg).to(dtype).eval()


class _WordTok:
    """Deterministic word-level tokenizer: BOS + crc32(word) per word."""
    bos_token_id = 1
    name_or_path = "fake/wordtok"

    def __call__(self, text, add_special_tokens=True, **_kw):
        import zlib
        ids = [(zlib.crc32(w.encode()) % 240) + 10 for w in text.split()]
        return {"input_ids": ([self.bos_token_id] if add_special_tokens else []) + ids}


def _fake_docs(bucket):
    for i in range(int(bucket.get("skip_docs", 0)), 10 ** 6):
        yield i, " ".join(f"{bucket['name']}d{i}w{j}" for j in range(50))


def _texts(corpus, n, words):
    return [" ".join(f"{corpus[:4]}{i}w{(j * 7 + i) % 97}" for j in range(words))
            for i in range(n)]


def _fake_corpus_texts(corpus, *, min_len=50, max_rows=None):
    return _texts(corpus, {"wikitext": 40}.get(corpus, 12), 60)[:max_rows]


def _capture_driver_streams(monkeypatch, drv):
    ep = drv.ep
    real = ep.run_gram_passes
    seen = {}

    def spy(*args, **kw):
        it = kw["chunk_iter"] if "chunk_iter" in kw else args[2]
        seen["retained"] = [x.detach().cpu().clone() for x in it()]
        seen["target"] = [x.detach().cpu().clone() for x in kw["target_chunk_iter"]()]
        return real(*args, **kw)

    monkeypatch.setattr(ep, "run_gram_passes", spy)
    return seen


def _capture_package_streams(monkeypatch):
    import srspd
    real = srspd.erase
    seen = []

    def spy(model, *a, target, retained, **kw):
        seen.append({"target": [x.detach().cpu().clone() for x in target],
                     "retained": [x.detach().cpu().reshape(1, -1).clone() for x in retained],
                     "kw": dict(kw)})
        return real(model, *a, target=target, retained=retained, **kw)

    monkeypatch.setattr(srspd, "erase", spy)
    return seen


def _mpt_cfg(r, solver):
    r.pop(LEGACY_DOSE, None)
    r["mu_per_token"] = [MPT]
    r.update({SKIP_KEY: True, "solver": solver})
    if SUB_KEY in r:
        r[SUB_KEY]["export_ks"] = []


def _exported(model_dir):
    from safetensors.torch import load_file
    (sub,) = [p for p in pathlib.Path(model_dir).iterdir() if p.is_dir() and p.name.startswith("srspd_mu")]
    w = {}
    for f in sub.glob("*.safetensors"):
        w.update(load_file(str(f)))
    return w


def _ours_exported(out_root):
    from safetensors.torch import load_file
    w = {}
    for f in (pathlib.Path(out_root) / f"srspd_mpt{MPT:g}").glob("*.safetensors"):
        w.update(load_file(str(f)))
    return w


def _assert_streams(ref_seen, ours_seen):
    (call,) = ours_seen
    assert len(call["target"]) == len(ref_seen["target"])
    for a, b in zip(ref_seen["target"], call["target"]):
        assert torch.equal(a.reshape(1, -1).long(), b.reshape(1, -1).long())
    assert len(call["retained"]) == len(ref_seen["retained"])
    for a, b in zip(ref_seen["retained"], call["retained"]):
        assert torch.equal(a.reshape(1, -1).long(), b.long())


def _assert_weights(ref_w, ours_w, pristine):
    assert set(ref_w) == set(ours_w)
    n_edited = 0
    for k, a in ref_w.items():
        b = ours_w[k]
        assert a.dtype == b.dtype, k
        assert torch.equal(a, b), (k, float((a.float() - b.float()).abs().max()))
        if k in pristine and not torch.equal(a, pristine[k]):
            n_edited += 1
    return n_edited


def _ours_cfg(drv_cfg, name):
    from srspd_experiments.recipes.legacy import srspd_config_from_legacy
    cfg = srspd_config_from_legacy(drv_cfg, name=name)
    assert cfg["srspd"]["mu_per_token"] == [MPT]
    return cfg


@pytest.mark.parametrize("solver", ["readout", "grams"])
def test_tiny_mistral_data_path_and_edit_match_the_reference(ref, tmp_path, monkeypatch, solver):
    drv = ref["drv"]
    C, rcorp = drv.C, drv.rcorp
    monkeypatch.setenv(_PKG.upper() + "_DATA_DIR", str(tmp_path / "ref_data"))
    monkeypatch.setattr(C, "load_corpus_texts", _fake_corpus_texts)
    monkeypatch.setattr(C, "load_causal_lm",
                        lambda model_id, **kw: (_mistral(torch.bfloat16), _WordTok()))
    monkeypatch.setattr(rcorp, "iter_documents", _fake_docs)
    seen = _capture_driver_streams(monkeypatch, drv)
    cfg = yaml.safe_load((ref["root"] / _DRV_DIR / "configs" /
                          "hf_rspd_zephyr_7b_mix1m_smoke.yaml").read_text())
    name = f"parity_mistral_{solver}"
    cfg.update(name=name, device_map="cpu", wandb={"enabled": False})
    r = cfg[_PKG]
    r.update(max_forget_rows=3, heldout_rows=2, max_length=48, retain_max_length=64,
             heldout_mix_tokens=0, rho_retain_tokens=512, scratch_dir=str(tmp_path / "scratch"))
    r["retained_corpus"]["chunk_tokens"] = 128
    r["retained_corpus"]["buckets"] = [dict(b, tokens=640) for b in
                                       r["retained_corpus"]["buckets"]]
    r["gram"] = dict(r.get("gram") or {}, budget_gib_per_device=(20000 if solver == "readout"
                                                                  else 40000) / 2 ** 30)
    _mpt_cfg(r, solver)
    cp = tmp_path / f"{name}.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    metrics = drv.run(str(cp))
    ref_w = _exported(tmp_path / "ref_data" / "models" / "wmdp" / name)

    # ours: separate data root, the package's own loaders mocked the same way
    from srspd_experiments.data import corpora as PC
    from srspd_experiments.data import retained_corpus as PR
    from srspd_experiments.methods import srspd as runner
    isolate_corpora(monkeypatch, drop_local=True)
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path / "our_data"))
    monkeypatch.setattr(PC, "load_corpus_texts", _fake_corpus_texts)
    monkeypatch.setattr(PR, "iter_documents", _fake_docs)
    ours_seen = _capture_package_streams(monkeypatch)
    model = _mistral(torch.bfloat16)
    pristine = {n: p.detach().clone() for n, p in model.named_parameters()}
    out_root = tmp_path / "ours_models"
    m = runner.run(_ours_cfg(cfg, name), model=model, tokenizer=_WordTok(), out_root=out_root)

    _assert_streams(seen, ours_seen)
    assert m["n_x"] == metrics["n_X_target_tokens"] and m["n_y"] == metrics["n_Y_retain_tokens"]
    for k, v in metrics["baseline_ce_heldout"].items():
        assert m["baseline_ce_heldout"][k] == pytest.approx(v, rel=1e-12), k
    (ent,) = m["exports"].values()
    assert ent["dose"]["mu"] == pytest.approx(next(iter(metrics["srspd"].values()))["mu"],
                                              rel=0, abs=0)
    assert _assert_weights(ref_w, _ours_exported(out_root), pristine) == 14
    assert all(torch.equal(p.detach(), pristine[n]) for n, p in model.named_parameters())


@pytest.mark.parametrize("solver", ["readout", "grams"])
def test_tiny_gpt2_data_path_and_edit_match_the_reference(ref, tmp_path, monkeypatch, solver):
    drv, g2 = ref["drv"], ref["g2"]
    data = tmp_path / "ref_data"
    monkeypatch.setenv(_PKG.upper() + "_DATA_DIR", str(data))
    _rel, ckpt = g2.save_checkpoint(data)
    g2.write_sets(data, n_target=4, n_contrastive=30, n_heldout=12, words=40,
                  sets_name="gpt2_unlearning")
    seen = _capture_driver_streams(monkeypatch, drv)
    cfg = yaml.safe_load((ref["root"] / _DRV_DIR / "configs" / "gpt2_unlearn_srspd.yaml").read_text())
    name = f"parity_gpt2_{solver}"
    cfg.update(name=name, device_map="cpu", wandb={"enabled": False})
    r = cfg[_PKG]
    r.update(max_forget_tokens=60, max_length=48, retain_max_length=64, heldout_rows=2,
             rho_retain_tokens=128, scratch_dir=str(tmp_path / "scratch"))
    r["retained_corpus"]["chunk_tokens"] = 64
    r["retained_corpus"]["buckets"] = [dict(b, tokens=640) for b in
                                       r["retained_corpus"]["buckets"]]
    r["gram"]["budget_gib_per_device"] = 2 ** 20 / 2 ** 30
    _mpt_cfg(r, solver)
    cp = tmp_path / f"{name}.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    metrics = drv.run(str(cp))
    ref_w = _exported(data / "models" / "wmdp" / name)

    # ours: the same local sets under a fresh data root (no shared caches)
    from srspd_experiments.data.corpora import load_tokenizer
    from srspd_experiments.methods import srspd as runner
    isolate_corpora(monkeypatch, drop_local=True)
    ours = tmp_path / "our_data"
    shutil.copytree(data / "datasets" / "gpt2_unlearning", ours / "datasets" / "gpt2_unlearning")
    monkeypatch.setenv("SRSPD_DATA_DIR", str(ours))
    ours_seen = _capture_package_streams(monkeypatch)
    model = transformers.GPT2LMHeadModel.from_pretrained(ckpt).eval()
    pristine = {n: p.detach().clone() for n, p in model.named_parameters()}
    out_root = tmp_path / "ours_models"
    m = runner.run(_ours_cfg(cfg, name), model=model, tokenizer=load_tokenizer(ckpt),
                   out_root=out_root)

    _assert_streams(seen, ours_seen)
    assert m["n_x"] == metrics["n_X_target_tokens"] == 60
    for k, v in metrics["baseline_ce_heldout"].items():
        assert m["baseline_ce_heldout"][k] == pytest.approx(v, rel=1e-12), k
    assert _assert_weights(ref_w, _ours_exported(out_root), pristine) == 12
