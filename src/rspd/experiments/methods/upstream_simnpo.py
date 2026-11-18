"""The authors' own SimNPO code (WMDP track), run end to end -- the reference
row a port is compared against.

**Credit.** Nothing here reimplements SimNPO. The runner clones the authors'
repository, https://github.com/OPTML-Group/Unlearn-Simple (MIT licence), at the
pinned commit :data:`UPSTREAM_SHA`, stages the bio forget corpus as
``WMDP/files/data/bio_remove_dataset.jsonl`` (the file upstream's ``WMDPBio``
reads), and drives upstream's OWN classes -- ``src/model/unlearn.py``
``Unlearn`` (model load, dataset build, ``TrainingArguments``, save) and
``src/unlearn/GA.py`` ``SimNPO_FT`` + ``src/unlearn/base.py`` ``BaseTrainer``
(loss and the custom ``_inner_training_loop``) -- with the values of upstream's
``WMDP/configs/unlearn/wmdp/SimNPO.json`` (SimNPO+FT, beta 0.1, gamma 1.0 [= the
retain weight in the WMDP track], lr 9.5e-6, max_steps 500, grad_accum 4,
batch_size 1, weight_decay 0, seed 0, forget WMDPBio, retain wikitext-2-raw-v1
test, dataset_seed 42, forget_ratio 1.0). Paper: Fan et al., arXiv:2410.07163.

    from rspd.experiments.methods import upstream_simnpo
    upstream_simnpo.run(out_dir="...")    # needs network (git clone, hub downloads)

Bypassed, and why:

* ``src/exec/unlearn_model.py`` -- it only parses SimNPO.json through
  ``fastargs``; the identical kwargs dict is built by hand (:func:`build_kwargs`)
  and its ``setup_seed`` replayed verbatim;
* upstream's eval tail (``Unlearn.eval``, its own lm-eval call) -- the saved
  checkpoint is scored with :mod:`rspd.experiments.eval.lm_eval` like the port.

Every other deviation is an ENVIRONMENT SHIM for transformers 5.x / torch 2.11,
documented at its definition with the reason it cannot change the trained weights.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import types

from . import _common as C
from ..paths import models_dir

UPSTREAM_URL = "https://github.com/OPTML-Group/Unlearn-Simple.git"
UPSTREAM_SHA = "3083017cb317753725f35cc404c2d5aede1242ef"   # "neurips 2025 accept"
UPSTREAM_DIR = os.path.join(tempfile.gettempdir(), "Unlearn-Simple")
OUT_NAME = "simnpo_authors_code_wmdp"
#: The model upstream's SimNPO.json trains.
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
LOG_EVERY = 10          # optimizer steps between loss-trajectory prints (observation only)


def stage_bio_forget(wmdp_root: str) -> None:
    """Write ``files/data/bio_remove_dataset.jsonl`` in the format upstream's
    ``WMDPBio.get_dataset`` reads: ``load_dataset("json", data_files=...)`` and
    only the ``text`` column is ever consumed (``filter len(text) > 50`` then
    tokenise; ``set_format`` keeps only the tokenised columns). The source is
    ``cais/wmdp-bio-forget-corpus`` (the gated HF copy of that file), fetched
    through :func:`rspd.experiments.data.corpora.load_corpus_texts_all` in its
    unfiltered, order-preserving form
    (``load_corpus_texts_all``) so the row set and ORDER match the jsonl
    one-for-one and upstream's own ``> 50`` filter does the filtering.
    """
    path = os.path.join(wmdp_root, "files", "data", "bio_remove_dataset.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bio = C.load_corpus_texts_all("bio-forget-corpus")
    with open(path, "w") as f:
        for t in bio:
            f.write(json.dumps({"text": t}) + "\n")
    n50 = sum(len(t) > 50 for t in bio)
    print(f"[upstream-simnpo] staged {len(bio)} bio-forget rows ({n50} with len>50) "
          f"-> {path}", flush=True)


def build_kwargs(model_id: str = MODEL_ID) -> dict:
    """Exactly the kwargs ``unlearn_model.Main.init_model`` hands to
    ``model.unlearn.get(**kwargs)`` for SimNPO.json: sections overall + unlearn +
    dataset + ``unlearn.SimNPO+FT`` merged, plus ``dataset_names``. Undeclared JSON
    keys (mask_path, optim, p, q, mu) are not fastargs params and never reach it;
    ``swa`` takes its declared default False; ``perturb_dataset_name`` default None.

    ``cache_dir`` is the one value changed: SimNPO.json says ``./.cache``, i.e. a
    fresh 14.5 GB download into the clone. ``None`` uses the image's HF cache.
    It only decides WHERE the same hub files are stored — result-neutral.
    """
    return {
        # overall
        "model_name": model_id, "logger": "json", "cache_dir": None, "seed": 0,
        # unlearn
        "unlearn_method": "SimNPO+FT", "num_epochs": 5, "lr": 9.5e-06,
        "weight_decay": 0.0, "gradient_accumulation_steps": 4, "task_name": "wmdp",
        "resume_path": None, "max_steps": 500, "use_lora": False, "swa": False,
        # dataset
        "forget_dataset_name": "WMDPBio", "retain_dataset_name": "WMDPBio",
        "perturb_dataset_name": None, "dataset_seed": 42, "forget_ratio": 1.0,
        "self_retain": False, "batch_size": 1,
        # unlearn.SimNPO+FT
        "gamma": 1.0, "beta": 0.1,
        "dataset_names": {"forget": "WMDPBio", "retain": "WMDPBio", "perturb": None},
    }


def setup_seed(seed: int) -> None:
    """Verbatim ``unlearn_model.Main.setup_seed``."""
    import numpy as np
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class VolumeLogger:
    """Stand-in for upstream's ``loggers.json_`` (whose package imports fastargs).
    ``Unlearn`` uses a logger for exactly two things on the training path:
    ``get_root()`` (where TrainingArguments.output_dir points) and ``save_ckpt``,
    which upstream implements as ``model.save_pretrained(ckpt_root)`` — replicated
    here with ``ckpt_root`` = the runner's output directory."""

    def __init__(self, root: str, ckpt_root: str):
        self.root, self.ckpt_root = root, ckpt_root
        os.makedirs(root, exist_ok=True)
        os.makedirs(ckpt_root, exist_ok=True)

    def get_root(self):
        return self.root

    def log(self, data):
        print("logging:", data, flush=True)

    def save_ckpt(self, name, model, use_lora):
        if "tokenizer" not in name and use_lora:
            model = model.merge_and_unload()
        model.save_pretrained(self.ckpt_root)


def install_import_shims() -> None:
    """Import-time environment shims — none touches the training computation."""
    import datasets as _datasets
    import torch

    # (1) Bare `wikitext` id -> `Salesforce/wikitext`. The pinned huggingface_hub
    # rejects bare repo ids ("Repository id must be 'namespace/name'"); it is the
    # same dataset/config/split (wikitext-2-raw-v1 test). Patched BEFORE importing
    # upstream because src/dataset/wmdp.py binds `from datasets import load_dataset`.
    _orig_load = _datasets.load_dataset

    def _load_dataset(path, *a, **k):
        if path == "wikitext":
            path = "Salesforce/wikitext"
        return _orig_load(path, *a, **k)

    _datasets.load_dataset = _load_dataset
    print("[upstream-simnpo] shim: load_dataset('wikitext') -> 'Salesforce/wikitext'",
          flush=True)

    # (2) `metrics` stub. src/model/unlearn.py imports six eval functions from
    # src/metrics at module top; they are only called by `Unlearn.eval`, the eval
    # tail we skip (the checkpoint is evaluated separately). The real package
    # pulls eval-only deps (rouge_score, sacrebleu, sklearn …); a stub keeps the
    # import from depending on them. Never called on the training path.
    stub = types.ModuleType("metrics")
    for fn in ("eval_copyright", "eval_few_shots", "eval_PII", "eval_ppl",
               "eval_toxic", "eval_wmdp", "eval_MIA", "eval_toxic_forget"):
        setattr(stub, fn, lambda *a, _fn=fn, **k: (_ for _ in ()).throw(
            RuntimeError(f"upstream eval {_fn} is skipped by the authors'-code runner")))
    sys.modules["metrics"] = stub
    print("[upstream-simnpo] shim: src/metrics stubbed (eval tail skipped)", flush=True)

    # (3) peft: see `import_upstream_model_module` — stubbed only around that
    # import, because transformers probes find_spec("peft") and a stub left in
    # sys.modules breaks it.

    # (4) `torch.backends.cuda.sdp_kernel` (removed from recent torch) is imported
    # by src/unlearn/GA.py and never used. Provide a no-op only if it is missing.
    if not hasattr(torch.backends.cuda, "sdp_kernel"):
        torch.backends.cuda.sdp_kernel = lambda *a, **k: contextlib.nullcontext()
        print("[upstream-simnpo] shim: torch.backends.cuda.sdp_kernel no-op "
              "(imported, unused by GA.py)", flush=True)

    # (5) `transformers.utils.is_torch_tpu_available` — imported by base.py; only
    # gates TPU branches. Provide False if the pinned transformers dropped it.
    import transformers.utils as _tu
    if not hasattr(_tu, "is_torch_tpu_available"):
        _tu.is_torch_tpu_available = lambda *a, **k: False
        print("[upstream-simnpo] shim: is_torch_tpu_available -> False", flush=True)

    # (6) Trainer(tokenizer=...) -> processing_class=. transformers 5 renamed the
    # kwarg; upstream passes the tokenizer only so Trainer can save it with
    # checkpoints. The custom `unlearncollector` does all batching — no padding
    # or tokenisation happens in Trainer. Pure rename.
    import transformers
    _orig_init = transformers.Trainer.__init__

    def _trainer_init(self, *a, **k):
        if "tokenizer" in k:
            k["processing_class"] = k.pop("tokenizer")
        return _orig_init(self, *a, **k)

    transformers.Trainer.__init__ = _trainer_init
    print("[upstream-simnpo] shim: Trainer(tokenizer=) -> processing_class=", flush=True)


def import_upstream_model_module():
    """Import upstream ``src/model/unlearn.py``.

    (3) peft is imported at its module top but used only when use_lora=True
    (SimNPO.json: false); the runtime image has no peft. A stub is placed in
    sys.modules ONLY for the duration of this import and removed right after,
    because transformers probes ``importlib.util.find_spec("peft")`` and a stub
    without a spec raises there ("peft.__spec__ is None"). transformers' Trainer
    is fully imported first for the same reason. Never called on the SimNPO path.
    """
    import importlib.util
    import transformers
    import transformers.trainer  # noqa: F401  (resolve lazy import before stubbing)
    stubbed = importlib.util.find_spec("peft") is None
    if stubbed:
        p = types.ModuleType("peft")
        for n in ("get_peft_model", "LoraConfig", "AdaLoraConfig", "TaskType"):
            setattr(p, n, None)
        sys.modules["peft"] = p
    try:
        import model.unlearn as MU
    finally:
        if stubbed:
            del sys.modules["peft"]
            print("[upstream-simnpo] shim: peft stubbed for the import only "
                  "(use_lora=False; not in image)", flush=True)
    return MU


def install_trainer_shims(trainer, model) -> dict:
    """Instance-level shims for running upstream's transformers-4.3x-era
    ``BaseTrainer._inner_training_loop`` against transformers 5.x. Returns the
    loss-trajectory record (observation only)."""
    import torch
    gas = trainer.args.gradient_accumulation_steps

    # (7) Unused frozen reference copy. BaseTrainer.__init__(if_kl=True) deep-copies
    # the whole 7B model to GPU as `infer_model`; only NPO_FT reads it
    # (GA.py:89) — SimNPO_FT.compute_loss never does. Keeping it (+14.5 GB) OOMs an
    # 80 GB H100 with a full-parameter AdamW update. Dropping an object nothing
    # reads cannot change the update.
    if hasattr(trainer, "infer_model"):
        del trainer.infer_model
        torch.cuda.empty_cache()
        print("[upstream-simnpo] shim: dropped unused infer_model deepcopy "
              "(read only by NPO_FT)", flush=True)

    # (8) compute_loss(num_items_in_batch=...). transformers 5's training_step calls
    # compute_loss(model, inputs, num_items_in_batch=N); upstream's signature
    # predates the kwarg. N is None here (the batch dict has no top-level
    # "labels"), so dropping it discards nothing.
    _orig_cl = trainer.compute_loss

    def _compute_loss(model, inputs, return_outputs=False, num_items_in_batch=None):
        return _orig_cl(model, inputs, return_outputs=return_outputs)

    trainer.compute_loss = _compute_loss

    # (9) Grad-accumulation loss scaling. In the transformers 4.3x upstream was
    # written against, the accelerator was built with num_steps=gas and
    # `accelerator.backward` divided each micro-batch loss by gas. transformers 5
    # builds the accelerator with num_steps=1 and instead divides by
    # `self.current_gradient_accumulation_steps` inside training_step — an
    # attribute only ITS OWN _inner_training_loop sets, which upstream's override
    # replaces. Setting it to gas reproduces upstream's scaling exactly
    # (every micro-loss / 4), not transformers 5's per-window count.
    trainer.current_gradient_accumulation_steps = gas

    # (10) TrainingArguments fields upstream's loop reads that transformers 5
    # removed/changed; restored to the 4.3x defaults. All three are
    # logging/RNN-memory bookkeeping: include_tokens_per_second (throughput metric),
    # past_index=-1 (no RNN mems), include_num_input_tokens_seen=False (5.x
    # default is the truthy string "no", which would take a warning-only branch
    # every micro-step).
    a = trainer.args
    for k, v in (("include_tokens_per_second", False), ("past_index", -1),
                 ("include_num_input_tokens_seen", False)):
        object.__setattr__(a, k, v)

    # (10b) `_initial_num_input_tokens_seen`: transformers 5's `Trainer.log` reads
    # it whenever include_num_input_tokens_seen != "no" (False included); only
    # 5.x's own loop sets it. 0 = tokens-seen counter baseline for a throughput
    # log field. Bookkeeping only.
    trainer._initial_num_input_tokens_seen = 0

    # (11) `self.use_apex`: removed in transformers 5; upstream reads it to pick
    # the non-apex path. Apex is not used (bf16 via accelerate) -> False.
    trainer.use_apex = False

    # (12) `_maybe_log_save_evaluate` signature changed (5.x adds grad_norm and
    # start_time). Upstream calls it once per epoch end with the 4.3x signature.
    # Adapted, and with should_save/should_evaluate cleared: upstream's checkpoint
    # is written by `Unlearn.save` after train() (the Trainer-side
    # `unlearn_checkpoint/` is `rm -rf`'d by upstream right after) and there is no
    # eval set. Logging still happens. Nothing here touches the weights.
    _orig_mlse = trainer._maybe_log_save_evaluate
    t_start = time.time()

    def _mlse(tr_loss, model_, trial, epoch, ignore_keys_for_eval, *rest, **kw):
        trainer.control.should_save = False
        trainer.control.should_evaluate = False
        return _orig_mlse(tr_loss, None, model_, trial, epoch, ignore_keys_for_eval,
                          t_start)

    trainer._maybe_log_save_evaluate = _mlse

    # (13) `_sorted_checkpoints` was removed in transformers 5. Upstream calls it
    # after training only to prune checkpoints when best_model_checkpoint is set
    # (never, here). Returns [] -> no pruning; weights untouched.
    trainer._sorted_checkpoints = lambda *a, **k: []

    # (14) OBSERVATION ONLY: a forward hook records outputs.loss of each model
    # call (SimNPO_FT does forget then retain per micro-step), and a training_step
    # wrapper prints the running means every LOG_EVERY optimizer steps. Upstream's
    # loop has per-step logging commented out, so this is the only trajectory.
    rec = {"calls": [], "micro": [], "traj": []}

    def _hook(mod, args, output):
        loss = getattr(output, "loss", None)
        if loss is not None:
            rec["calls"].append(float(loss.detach().float()))

    model.register_forward_hook(_hook)
    _orig_ts = trainer.training_step

    def _training_step(model_, inputs, num_items_in_batch=None):
        rec["calls"].clear()
        out = _orig_ts(model_, inputs, num_items_in_batch)
        c = rec["calls"]
        f_nll, r_nll = (c[0], c[1]) if len(c) >= 2 else (float("nan"), float("nan"))
        rec["micro"].append((f_nll, r_nll, float(out) * gas))
        n = len(rec["micro"])
        if n % (gas * LOG_EVERY) == 0:
            w = rec["micro"][-gas * LOG_EVERY:]
            m = [sum(x[i] for x in w) / len(w) for i in range(3)]
            step = n // gas
            lr = trainer.lr_scheduler.get_last_lr()[0] if trainer.lr_scheduler else float("nan")
            rec["traj"].append({"step": step, "forget_nll": m[0], "retain_nll": m[1],
                                "loss": m[2], "lr": lr})
            print(f"[upstream-simnpo] step {step:4d}  forget_nll {m[0]:.4f}  "
                  f"retain_nll {m[1]:.4f}  loss {m[2]:.4f}  lr {lr:.3e}", flush=True)
        return out

    trainer.training_step = _training_step
    print("[upstream-simnpo] trainer shims (7)-(14) installed", flush=True)
    return rec


def ensure_clone(path: str = UPSTREAM_DIR, sha: str = UPSTREAM_SHA,
                 url: str = UPSTREAM_URL) -> str:
    """Clone ``url`` into ``path`` (once), check out ``sha``; returns HEAD."""
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(["git", "clone", url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "-q", sha], check=True)
    head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    if head != sha:
        raise RuntimeError(f"{path} is at {head}, expected the pinned {sha}")
    return head


def run(*, out_dir=None, model_id: str = MODEL_ID, clone_dir: str = UPSTREAM_DIR,
        sha: str = UPSTREAM_SHA) -> dict:
    """Clone the authors' code at ``sha``, train, save to ``out_dir`` (default
    ``<data root>/models/methods/simnpo_authors_code_wmdp``); returns
    ``{"out_dir", "upstream_sha", "global_step", "train_loss", "files"}``. Changes
    the working directory into the clone for the run (restored afterwards)."""
    out_dir = str(out_dir or models_dir("methods", OUT_NAME))
    os.makedirs(out_dir, exist_ok=True)
    cwd0, path0 = os.getcwd(), list(sys.path)
    try:
        return _run(out_dir, model_id, clone_dir, sha)
    finally:
        os.chdir(cwd0)
        sys.path[:] = path0


def _run(out_dir, model_id, clone_dir, sha_pin):
    print(f"[upstream-simnpo] checkpoint -> {out_dir}", flush=True)
    t0 = time.time()

    # -- clone upstream at the pinned commit ------------------------------------
    sha = ensure_clone(clone_dir, sha_pin)
    print(f"[upstream-simnpo] upstream @ {sha}", flush=True)
    wmdp = os.path.join(clone_dir, "WMDP")

    # -- stage the gated forget corpus BEFORE chdir ------------------------------
    stage_bio_forget(wmdp)

    os.chdir(wmdp)
    sys.path.insert(0, os.path.join(wmdp, "src"))
    print(f"[upstream-simnpo] cwd={os.getcwd()} (upstream WMDP root)", flush=True)

    install_import_shims()

    import torch
    MU = import_upstream_model_module()  # upstream src/model/unlearn.py

    kwargs = build_kwargs(model_id)
    print("[upstream-simnpo] kwargs (= SimNPO.json via fastargs): "
          + json.dumps(kwargs), flush=True)

    # Replay unlearn_model.Main: setup_seed -> init_model -> run().
    setup_seed(kwargs["seed"])
    u = MU.get(**kwargs)
    logger = VolumeLogger(root=os.path.join(wmdp, "files", "results",
                                            "unlearn_wmdp_bio", "SimNPO", "run"),
                          ckpt_root=out_dir)

    # --- Unlearn.run(logger), resume_path None branch, eval tail skipped -------
    u.init_model()
    MU.print_trainable_parameters(u.model)
    u.init_dataset()
    print(f"[upstream-simnpo] unlearn_dataset rows={len(u.unlearn_dataset)} "
          f"max_steps={u.max_steps}", flush=True)
    u.init_unlearner(logger)
    rec = install_trainer_shims(u.unlearner, u.model)
    print(f"[upstream-simnpo] trainer={type(u.unlearner).__name__} "
          f"beta={u.unlearner.beta} gamma={u.unlearner.gamma} "
          f"lr={u.unlearner.args.learning_rate} gas={u.unlearner.args.gradient_accumulation_steps} "
          f"bs={u.unlearner.args.per_device_train_batch_size} "
          f"max_steps={u.unlearner.args.max_steps} warmup={u.unlearner.args.warmup_steps}",
          flush=True)
    torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    try:
        out = u.unlearner.train()
    except Exception:
        # If upstream's post-loop bookkeeping fails AFTER all 500 optimizer steps,
        # still persist the trained weights for inspection — the run is reported
        # as failed regardless (the exception is re-raised).
        if u.unlearner.state.global_step == u.max_steps:
            print("[upstream-simnpo] train() raised after the final step; saving the "
                  "weights before re-raising", flush=True)
            u.save(logger)
        raise
    print(f"[upstream-simnpo] train() -> {out}  ({time.time() - t_train:.1f}s, "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
    u.save(logger)                       # upstream: model + tokenizer save_pretrained
    subprocess.run(["rm", "-rf", f"{logger.get_root()}/unlearn_checkpoint"])

    with open(os.path.join(out_dir, "loss_trajectory.json"), "w") as f:
        json.dump({"upstream_sha": sha, "kwargs": kwargs, "trajectory": rec["traj"],
                   "global_step": out.global_step, "train_loss": out.training_loss}, f,
                  indent=1)
    files = sorted(os.listdir(out_dir))
    print(f"[upstream-simnpo] DONE in {time.time() - t0:.1f}s — {out_dir}: {files}",
          flush=True)
    assert any(f.endswith(".safetensors") for f in files), "no weights written"
    assert out.global_step == 500, f"expected 500 optimizer steps, got {out.global_step}"
    return {"out_dir": out_dir, "upstream_sha": sha, "upstream_url": UPSTREAM_URL,
            "global_step": int(out.global_step), "train_loss": float(out.training_loss),
            "wall_seconds": round(time.time() - t0, 1), "files": files, "kwargs": kwargs}
