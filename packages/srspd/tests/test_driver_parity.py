"""Driver-level parity with the reference implementation.

The reference repository's ``origin/main`` (``$SRSPD_REFERENCE_REPO``, default
``/root/workspace/code``) is exported with ``git archive`` into a temporary
directory -- never imported from a working tree, which may sit on another
branch -- and its SRSPD-only driver path (``skip_circuits: true``) is run end
to end on a tiny random Mistral (bf16, mocked corpora, two Gram passes with
boundary resume) and a tiny GPT-2 (fp32 local checkpoint, local JSONL sets),
under both solvers. The token ids the driver streamed (its target sequences
and retained chunks, captured at its Gram-pass entry point) are then handed to
:func:`srspd.erase` on an identical model at the same ``mu_per_token`` (the
driver is configured with ``mu_per_token`` too, so both sides derive
``mu = n_x / (mu_per_token * n_y)`` by the same arithmetic), and the edited
weights are compared with the driver's exported checkpoint.

Tolerance: both sides harvest, accumulate the float64 Grams, solve in float64,
spill the readout (or the dense grams-solver component) in fp32 and form
``W - M`` in fp32 from those spills -- the same operations in the same order --
so the edited weights are asserted BITWISE equal (bf16 and fp32). Any
difference would come from the float64 solve, which the package matches to
<= 1e-12 relative (``test_parity_reference.py``); that cannot survive the
fp32 spill except by a rounding-boundary flip, which this test would report.
The leaks are compared to 1e-10 relative. Skipped cleanly when the reference
repository or git is absent.
"""
from __future__ import annotations

import importlib.util
import io
import json
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

import srspd  # noqa: E402

REPO = pathlib.Path(os.environ.get("SRSPD_REFERENCE_REPO", "/root/workspace/code"))
REF = "origin/main"
PATHS = ["rspd", "experiments/wmdp", "tests/gpt2_tiny_helpers.py",
         "experiments/scaling/configs/hf_dutch_erasure_gemma4_31b_it_1m.yaml"]
DRIVER = "experiments/wmdp/hf_rspd.py"
MODNAME = "_srspd_ref_hf_rspd"
LEAK_TOL = 1e-10


def _purge():
    for k in [m for m in sys.modules if m == "rspd" or m.startswith("rspd.")
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
    saved_path, saved_mods = list(sys.path), {k: v for k, v in sys.modules.items()}
    _purge()
    sys.path[:0] = [str(root), str(root / "experiments" / "wmdp"), str(root / "tests")]
    try:
        spec = importlib.util.spec_from_file_location(MODNAME, root / DRIVER)
        drv = importlib.util.module_from_spec(spec)
        sys.modules[MODNAME] = drv
        spec.loader.exec_module(drv)
        import gpt2_tiny_helpers as g2
        import rspd
        assert pathlib.Path(rspd.__file__).resolve().is_relative_to(root.resolve()), \
            f"imported the wrong reference package: {rspd.__file__}"
        yield {"drv": drv, "root": root, "g2": g2}
    finally:
        _purge()
        sys.path[:] = saved_path
        sys.modules.update({k: v for k, v in saved_mods.items() if k not in sys.modules})


# --------------------------------------------------------------------------- #
# the reference run, with the driver's token streams captured                  #
# --------------------------------------------------------------------------- #
def _capture_streams(monkeypatch, drv):
    ep = drv.ep
    real = ep.run_gram_passes
    seen = {}

    def spy(*args, **kw):
        seen["retained"] = [x.detach().cpu().clone() for x in kw["chunk_iter"]()] \
            if "chunk_iter" in kw else [x.detach().cpu().clone() for x in args[2]()]
        seen["target"] = [x.detach().cpu().clone() for x in kw["target_chunk_iter"]()]
        seen["n_target"] = int(kw["n_nl"])
        return real(*args, **kw)

    monkeypatch.setattr(ep, "run_gram_passes", spy)
    return seen


def _mpt_cfg(r, mpt, solver):
    r.pop("mu_raw", None)
    r["mu_per_token"] = [mpt]
    r.update(skip_circuits=True, solver=solver)
    if "circuits" in r:
        r["circuits"]["export_ks"] = []


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


def _run_mistral(ref, tmp_path, monkeypatch, solver, mpt):
    drv = ref["drv"]
    C, rcorp = drv.C, drv.rcorp
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(C, "load_corpus_texts", lambda corpus, *, min_len=50, max_rows=None: (
        _texts(corpus, {"wikitext": 40}.get(corpus, 12), 60)[:max_rows]))
    monkeypatch.setattr(C, "load_causal_lm",
                        lambda model_id, **kw: (_mistral(torch.bfloat16), _WordTok()))
    monkeypatch.setattr(rcorp, "iter_documents", _fake_docs)
    seen = _capture_streams(monkeypatch, drv)
    cfg = yaml.safe_load((ref["root"] / "experiments/wmdp/configs/"
                          "hf_rspd_zephyr_7b_mix1m_smoke.yaml").read_text())
    name = f"parity_mistral_{solver}"
    cfg.update(name=name, device_map="cpu", wandb={"enabled": False})
    r = cfg["rspd"]
    r.update(max_forget_rows=3, heldout_rows=2, max_length=48, retain_max_length=64,
             heldout_mix_tokens=0, rho_retain_tokens=512, scratch_dir=str(tmp_path / "scratch"))
    r["retained_corpus"]["chunk_tokens"] = 128
    r["retained_corpus"]["buckets"] = [dict(b, tokens=640) for b in
                                       r["retained_corpus"]["buckets"]]
    # small enough for two passes (exercises the boundary resume on both sides)
    r["gram"] = dict(r.get("gram") or {}, budget_gib_per_device=(20000 if solver == "readout"
                                                                  else 40000) / 2 ** 30)
    _mpt_cfg(r, mpt, solver)
    cp = tmp_path / f"{name}.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    metrics = drv.run(str(cp))
    return metrics, seen, tmp_path / "data" / "models" / "wmdp" / name


def _run_gpt2(ref, tmp_path, monkeypatch, solver, mpt):
    drv, g2 = ref["drv"], ref["g2"]
    data = tmp_path / "data"
    monkeypatch.setenv("RSPD_DATA_DIR", str(data))
    _rel, ckpt = g2.save_checkpoint(data)
    g2.write_sets(data, n_target=4, n_contrastive=30, n_heldout=12, words=40,
                  sets_name="gpt2_unlearning")
    seen = _capture_streams(monkeypatch, drv)
    cfg = yaml.safe_load((ref["root"] / "experiments/wmdp/configs/"
                          "gpt2_unlearn_srspd.yaml").read_text())
    name = f"parity_gpt2_{solver}"
    cfg.update(name=name, device_map="cpu", wandb={"enabled": False})
    r = cfg["rspd"]
    r.update(max_forget_tokens=60, max_length=48, retain_max_length=64, heldout_rows=2,
             rho_retain_tokens=128, scratch_dir=str(tmp_path / "scratch"))
    r["retained_corpus"]["chunk_tokens"] = 64
    r["retained_corpus"]["buckets"] = [dict(b, tokens=640) for b in
                                       r["retained_corpus"]["buckets"]]
    r["gram"]["budget_gib_per_device"] = 2 ** 20 / 2 ** 30
    _mpt_cfg(r, mpt, solver)
    cp = tmp_path / f"{name}.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    metrics = drv.run(str(cp))
    return metrics, seen, data / "models" / "wmdp" / name, ckpt


def _exported(model_dir):
    from safetensors.torch import load_file
    (sub,) = [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("srspd_mu")]
    w = {}
    for f in sub.glob("*.safetensors"):
        w.update(load_file(str(f)))
    return w


def _compare(model, exported, metrics, rep, dose_key, pristine):
    params = dict(model.named_parameters())
    h = rep._session.h
    n_cmp = 0
    for name in h.all_names:
        path = h.path_by_name[name] + ".weight"
        a = params[path].detach()
        b = exported[path]
        assert a.dtype == b.dtype, path
        assert torch.equal(a, b), (path, float((a.float() - b.float()).abs().max()))
        assert not torch.equal(a, pristine[path]), f"{path}: no edit was applied"
        n_cmp += 1
    # everything else untouched on both sides
    for path, p in params.items():
        if path in exported and not path.endswith(("_proj.weight", "c_attn.weight",
                                                   "c_proj.weight", "c_fc.weight")):
            assert torch.equal(p.detach(), exported[path]), path
    leaks_ref = metrics["srspd"][dose_key]["leaks_per_matrix"]
    assert set(leaks_ref) == set(rep.matrices)
    for n, lk in leaks_ref.items():
        for f in ("L_X_rel", "L_Y_rel"):
            assert rep.matrices[n][f] == pytest.approx(lk[f], rel=LEAK_TOL, abs=1e-300), (n, f)
    return n_cmp


MPT = 0.375


@pytest.mark.parametrize("solver", ["readout", "grams"])
def test_tiny_mistral_matches_the_reference_driver(ref, tmp_path, monkeypatch, solver):
    metrics, seen, mdir = _run_mistral(ref, tmp_path, monkeypatch, solver, MPT)
    (dose_key,) = metrics["srspd"]
    assert metrics["solver"] == solver and len(seen["retained"]) == 20
    assert len(metrics["srspd"]) == 1
    exported = _exported(mdir)
    model = _mistral(torch.bfloat16)
    pristine = {n: p.detach().clone() for n, p in model.named_parameters()}
    rep = srspd.erase(model, target=seen["target"], retained=seen["retained"],
                      mu_per_token=MPT, solver=solver, ridge_rel=1e-10,
                      gram_budget_gib=(20000 if solver == "readout" else 40000) / 2 ** 30)
    assert rep.n_x == seen["n_target"] == metrics["n_X_target_tokens"]
    assert rep.n_y == metrics["n_Y_retain_tokens"]
    assert rep.dose["mu"] == pytest.approx(metrics["srspd"][dose_key]["mu"], rel=0, abs=0)
    assert len(rep.pass_plan) > 1 and rep.stats["resume_from_cache"]
    assert _compare(model, exported, metrics, rep, dose_key, pristine) == 14
    assert rep.edit["survival"] == pytest.approx(metrics["srspd"][dose_key]["bf16_survival"],
                                                 rel=1e-12)
    rep.close()
    assert all(torch.equal(p.detach(), pristine[n]) for n, p in model.named_parameters())


@pytest.mark.parametrize("solver", ["readout", "grams"])
def test_tiny_gpt2_matches_the_reference_driver(ref, tmp_path, monkeypatch, solver):
    metrics, seen, mdir, ckpt = _run_gpt2(ref, tmp_path, monkeypatch, solver, MPT)
    (dose_key,) = metrics["srspd"]
    exported = _exported(mdir)
    model = transformers.GPT2LMHeadModel.from_pretrained(ckpt).eval()
    pristine = {n: p.detach().clone() for n, p in model.named_parameters()}
    rep = srspd.erase(model, target=seen["target"], retained=seen["retained"],
                      mu_per_token=MPT, solver=solver, ridge_rel=1e-10)
    assert rep.n_x == 60 and rep.selector["preset"] == "gpt2"
    assert rep.dose["mu"] == pytest.approx(metrics["srspd"][dose_key]["mu"], rel=0, abs=0)
    assert _compare(model, exported, metrics, rep, dose_key, pristine) == 12
    rep.close()


def test_reference_metrics_record_the_same_dose(ref, tmp_path, monkeypatch):
    """The driver's ``mu_per_token`` spec and srspd's Dose agree on every derived
    number (the conversion is the same arithmetic)."""
    metrics, seen, _ = _run_mistral(ref, tmp_path, monkeypatch, "readout", MPT)
    (dose_key,) = metrics["srspd"]
    e = metrics["srspd"][dose_key]
    d = srspd.dose_from_mu_per_token(MPT, metrics["n_X_target_tokens"],
                                     metrics["n_Y_retain_tokens"]).as_dict()
    assert d["mu"] == e["mu"] and json.dumps(d)
