"""The authors' own KUDA code on MUSE News / Books, run in the authors' own
software stack -- the reference rows the KUDA port is compared against on MUSE.

**Credit.** Nothing here reimplements KUDA: the runner checks out
https://github.com/AceNagi/KUDA (MIT licence) at :data:`UPSTREAM_SHA` and runs
its ``MUSE/baselines/unlearn.py`` verbatim with the hyperparameters of its
``MUSE/scripts/muse_unlearn.sh`` (:data:`HPARAMS`). Paper: Fang et al.,
arXiv:2602.19275. MUSE: Weijia Shi et al., *MUSE: Machine Unlearning Six-Way
Evaluation for Language Models*, arXiv:2407.06460 (data CC-BY-4.0).

The authors' MUSE code needs ITS stack (an 8-bit paged optimizer and the
transformers / torch releases of its ``requirements.txt``, :data:`STACK`), not
this package's, so it is run as a subprocess with ``python`` pointing at an
interpreter that has that stack installed; this module never imports it. The
Trainer writes one checkpoint per epoch; :func:`collect_epochs` copies each one
(without optimizer / scheduler / RNG state) to
``<out_root>/<name>_ep<N>/``. The published News row is epoch 2 of the 3-epoch
schedule (the paper's epoch) and the Books row epoch 5. Score each copy with
:mod:`srspd_experiments.eval.muse` in fp32.

    from srspd_experiments.methods import upstream_kuda_muse as U
    U.argv("news")                       # the exact command, no side effects
    U.run("news", python="/opt/kuda-stack/bin/python", out_root="...")
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time

UPSTREAM_URL = "https://github.com/AceNagi/KUDA.git"
UPSTREAM_SHA = "3aae1819982348d98942a8a2720aaacca508ef47"
CLONE_DIR = os.path.join(tempfile.gettempdir(), "KUDA-muse")

#: The authors' requirements at :data:`UPSTREAM_SHA` (flash-attn and the editable
#: lm-eval harness omitted; MUSE/baselines imports neither). torch 2.5.1 (cu121).
STACK = (
    "torch==2.5.1", "transformers==4.57.1", "tokenizers==0.22.1", "accelerate==1.11.0",
    "bitsandbytes==0.48.2", "datasets==4.2.0", "huggingface-hub==0.35.3", "numpy==2.3.3",
    "peft==0.17.1", "trl==0.24.0", "safetensors==0.6.2", "sentencepiece==0.2.1",
    "scipy==1.16.2", "scikit-learn==1.7.2", "pandas==2.3.3",
)

#: ``MUSE/scripts/muse_unlearn.sh`` at :data:`UPSTREAM_SHA`.
HPARAMS = {
    "news": ["--beta", "2.0", "--tau", "4e-3", "--lr", "5e-5", "--epochs", "3"],
    "books": ["--beta", "0.1", "--tau", "0.08", "--lr", "1e-5", "--epochs", "5"],
}
LAYERS = ["--layer_id", "7", "--layer_ids", "4", "5", "6", "7"]


def argv(corpus: str, *, python: str = "python3") -> list:
    """The command run from ``<clone>/MUSE`` for ``corpus`` (``news`` | ``books``)."""
    if corpus not in HPARAMS:
        raise ValueError(f"corpus must be one of {sorted(HPARAMS)}, got {corpus!r}")
    return ([python, "-u", "./baselines/unlearn.py", "--algo", "kuda", "--corpus", corpus]
            + HPARAMS[corpus] + LAYERS + ["--out_dir", f"./outputs/muse/{corpus}_kuda"])


def ensure_clone(path: str = CLONE_DIR, sha: str = UPSTREAM_SHA, url: str = UPSTREAM_URL) -> str:
    """Clone (once) and check out ``sha``; returns HEAD."""
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(["git", "clone", url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "--quiet", sha], check=True)
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()
    if head != sha:
        raise RuntimeError(f"{path} is at {head}, expected the pinned {sha}")
    return head


def _epoch_of(ckpt: str):
    try:
        with open(os.path.join(ckpt, "trainer_state.json")) as f:
            return int(round(float(json.load(f).get("epoch", 0))))
    except Exception:
        return None


def collect_epochs(outputs_dir: str, out_root: str, name: str) -> list:
    """Copy every ``checkpoint-<step>`` under ``outputs_dir`` to
    ``<out_root>/<name>_ep<epoch>/`` (``_step<step>`` when the epoch is
    unreadable), without optimizer / scheduler / RNG state. Returns the records."""
    out = []
    for d in sorted(os.listdir(outputs_dir)) if os.path.isdir(outputs_dir) else []:
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if not m:
            continue
        src = os.path.join(outputs_dir, d)
        ep = _epoch_of(src)
        dst = os.path.join(out_root, f"{name}_ep{ep}" if ep is not None
                           else f"{name}_step{m.group(1)}")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt",
                                                                "rng_state*"))
        out.append({"checkpoint": src, "copy": dst, "epoch": ep, "step": int(m.group(1))})
    return out


def run(corpus: str, *, python: str, out_root: str, clone_dir: str = CLONE_DIR,
        sha: str = UPSTREAM_SHA, name: str | None = None) -> dict:
    """Run the authors' MUSE KUDA for ``corpus`` with interpreter ``python``
    (which must carry :data:`STACK`), then collect the epoch checkpoints into
    ``out_root``. Returns ``{"argv", "upstream_sha", "returncode", "epochs",
    "wall_seconds"}``; raises ``CalledProcessError`` on a failed training run."""
    head = ensure_clone(clone_dir, sha)
    muse = os.path.join(clone_dir, "MUSE")
    cmd = argv(corpus, python=python)
    name = name or f"kuda_authors_code_muse_{corpus}"
    t0 = time.time()
    subprocess.run(cmd, cwd=muse, check=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    os.makedirs(out_root, exist_ok=True)
    epochs = collect_epochs(os.path.join(muse, "outputs", "muse", f"{corpus}_kuda"),
                            out_root, name)
    rec = {"argv": cmd, "upstream_url": UPSTREAM_URL, "upstream_sha": head, "returncode": 0,
           "epochs": epochs, "wall_seconds": round(time.time() - t0, 1)}
    with open(os.path.join(out_root, f"{name}.run.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec
