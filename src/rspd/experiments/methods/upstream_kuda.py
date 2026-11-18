"""The authors' own KUDA code, run end to end on the WMDP corpora -- the
reference row a port is compared against.

**Credit.** Nothing here reimplements KUDA. The runner clones the authors'
repository, https://github.com/AceNagi/KUDA (MIT licence), at a pinned commit,
stages the WMDP forget corpora in the layout its ``get_data`` expects, and calls
its own ``unlearning.unlearn.run_unlearn`` and its ROME-based projector
generation verbatim, at the hyperparameters of its ``scripts/wmdp_unlearn.sh``
for the 7B instruct model (beta 2.0, tau 2e-3, alpha 1,1, lr 1e-5, layer_id 9,
layer_ids 6,7,8,9, 150 batches, batch size 4, retain wikitext,wikitext). Paper:
Fang et al., arXiv:2602.19275. The only part skipped is upstream's
``auto_eval_*`` tail (it needs the authors' MCQ files and harness); the saved
checkpoint is scored with :mod:`rspd.experiments.eval.lm_eval` like every
other row.

    from rspd.experiments.methods import upstream_kuda
    upstream_kuda.run(out_dir="...")      # needs network (git clone, hub downloads)

Three result-neutral environment shims are applied (each documented where it
is installed): the bare ``wikitext`` dataset id is rewritten to its namespaced
hub id; forget inputs are moved to the model's device inside
``forward_with_cache`` (upstream relied on accelerate hooks that do not fire on
one GPU); and every parameter but the target ``mlp.down_proj.weight`` tensors
is frozen (the optimizer only ever holds those, so the checkpoint is
bit-identical while backward stops storing gradients for the other layers).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import types

from . import _common as C
from ..paths import models_dir

KUDA_URL = "https://github.com/AceNagi/KUDA.git"
#: The authors' commit this runner checks out. The published WMDP row was made
#: from a shallow clone of the default branch; this is the commit the authors'
#: MUSE run recorded, pinned here so the runner is reproducible.
KUDA_SHA = "3aae1819982348d98942a8a2720aaacca508ef47"
KUDA_DIR = os.path.join(tempfile.gettempdir(), "KUDA")
OUT_NAME = "kuda_authors_code_wmdp"
#: The model upstream's WMDP script trains (its own default; overridable).
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"


def stage_forget_corpora(corpora_dir: str) -> None:
    """Write the forget corpora in upstream's exact ``get_data`` format:

    * ``bio-forget-corpus.jsonl`` — one JSON object per line, read as
      ``json.loads(line)['text']`` (unlearning/utils.py:124);
    * ``cyber-forget-corpus.jsonl`` — one RAW line per document, read as
      ``raw_text = line`` (utils.py:127). Internal newlines are flattened so one
      document is one line.

    Fetched through :func:`rspd.experiments.data.corpora.load_corpus_texts`
    (which carries the HF token for the gated bio corpus), at ``min_len=0`` to match upstream's
    default (unlearn.py ``--min_len`` default 0).
    """
    os.makedirs(corpora_dir, exist_ok=True)
    bio = C.load_corpus_texts("bio-forget-corpus", min_len=0)
    with open(os.path.join(corpora_dir, "bio-forget-corpus.jsonl"), "w") as f:
        for t in bio:
            f.write(json.dumps({"text": t}) + "\n")
    print(f"[upstream-kuda] staged {len(bio)} bio-forget docs (jsonl 'text')", flush=True)
    cyber = C.load_corpus_texts("cyber-forget-corpus", min_len=0)
    with open(os.path.join(corpora_dir, "cyber-forget-corpus.jsonl"), "w") as f:
        for t in cyber:
            f.write(" ".join(str(t).split()) + "\n")     # one doc per raw line
    print(f"[upstream-kuda] staged {len(cyber)} cyber-forget docs (raw lines)", flush=True)


def build_args(output_dir: str, model_id: str = MODEL_ID) -> argparse.Namespace:
    """Exactly the argparse.Namespace upstream's ``get_args`` would produce for
    the 7B-model KUDA command in ``scripts/wmdp_unlearn.sh`` (post-processed:
    lists split, steering/alpha floated)."""
    return argparse.Namespace(
        model_name_or_path=model_id,
        module_str="{model_name}.model.layers[{layer_id}]",
        output_dir=output_dir,
        retain_corpora=["wikitext", "wikitext"],
        forget_corpora=["bio-forget-corpus", "cyber-forget-corpus"],
        alpha=[1.0, 1.0],
        steering_coeffs="6.5,6.5",
        steering_coeff_list=[6.5, 6.5],          # unused by kuda loss, but referenced
        lr=1e-5,
        min_len=0,
        max_len=2000,
        batch_size=4,
        max_num_batches=150,
        layer_id=9,
        layer_ids=[6, 7, 8, 9],
        module_name="mlp.down_proj",
        seed=42,
        verbose=False,
        eval_dir_prex="./eval_res",
        beta=2.0,
        tau=2e-3,
        loss_type="kuda",
        ns_hparams_path="./nsproject/hyparams/Zephyr-7B-Beta-Wikitext.json",
        epoch=1,
        exp_type=None,
        sample_p=100,
    )


def ensure_clone(path: str = KUDA_DIR, sha: str = KUDA_SHA, url: str = KUDA_URL) -> str:
    """Clone ``url`` into ``path`` (once) and check out ``sha``; returns the
    resolved HEAD. Raises when the checkout is at another commit."""
    if not os.path.isdir(os.path.join(path, ".git")):
        subprocess.run(["git", "clone", url, path], check=True)
    subprocess.run(["git", "-C", path, "checkout", "--quiet", sha], check=True)
    head = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"]).decode().strip()
    if head != sha:
        raise RuntimeError(f"{path} is at {head}, expected the pinned {sha}")
    return head


def run(*, out_dir=None, model_id: str = MODEL_ID, clone_dir: str = KUDA_DIR,
        sha: str = KUDA_SHA) -> dict:
    """Clone the authors' code at ``sha``, train, save to ``out_dir`` (default
    ``<data root>/models/methods/kuda_authors_code_wmdp``); returns
    ``{"out_dir", "upstream_sha", "wall_seconds", "files"}``. Changes the
    working directory into the clone (upstream reads relative paths)."""
    out_dir = str(out_dir or models_dir("methods", OUT_NAME))
    os.makedirs(out_dir, exist_ok=True)
    cwd0 = os.getcwd()
    try:
        return _run(out_dir, model_id, clone_dir, sha)
    finally:
        os.chdir(cwd0)


def _run(out_dir, model_id, clone_dir, sha):
    print(f"[upstream-kuda] checkpoint -> {out_dir}", flush=True)

    # -- fetch forget texts through our loader BEFORE we chdir into the clone ---
    t0 = time.time()

    # -- clone upstream verbatim, at the pinned commit ----------------------------
    head = ensure_clone(clone_dir, sha)
    wmdp = os.path.join(clone_dir, "WMDP")
    stage_forget_corpora(os.path.join(wmdp, "data", "corpora"))

    os.chdir(wmdp)
    sys.path.insert(0, wmdp)
    print(f"[upstream-kuda] cwd={os.getcwd()} (upstream WMDP root)", flush=True)

    # -- environment shim (NOT a KUDA change): the pinned huggingface_hub rejects
    # the bare `wikitext` repo id upstream's layer_stats/get_data pass to
    # load_dataset ("Repository id must be 'namespace/name'"); rewrite it to the
    # canonical `Salesforce/wikitext` (same dataset/config/split, the id the
    # rspd.experiments corpus registry uses). Patch BEFORE importing upstream, because
    # nsproject/layer_stats.py binds `from datasets import load_dataset` at import.
    import torch
    import datasets as _datasets
    _orig_load = _datasets.load_dataset

    def _load_dataset(path, *a, **k):
        if path == "wikitext":
            path = "Salesforce/wikitext"
        return _orig_load(path, *a, **k)

    _datasets.load_dataset = _load_dataset
    print("[upstream-kuda] shim: load_dataset('wikitext') -> 'Salesforce/wikitext'",
          flush=True)

    from unlearning.unlearn import run_unlearn, MODEL_LIST
    from unlearning.utils import load_model, get_data, match_model_FormalName
    from nsproject.nullspace import is_project_exist, generate_project
    from nsproject.hparams import HyperParams
    import unlearning.unlearn as U
    # Neutralise upstream's auto_eval tail (needs their MCQ files + harness); we
    # eval the saved checkpoint with our lm-eval config instead.
    U.auto_eval_retain = lambda *a, **k: None  # noqa: E731
    U.auto_eval_forget = lambda *a, **k: None

    # Environment shim (NOT a KUDA change): upstream's run_unlearn moves
    # retain_inputs to the model device but leaves unlearn_inputs on CPU
    # (unlearn.py:112-118), relying on accelerate's device_map="auto" hooks to move
    # them — which don't fire on this single-GPU box, so embed_tokens sees CPU
    # input_ids against cuda weights. Wrap the forget/retain forward so inputs land
    # on the model's device, exactly what accelerate would do. Same computation.
    _orig_fwc = U.forward_with_cache

    def _fwc(model, inputs, module, no_grad=True):
        dev = next(model.parameters()).device
        inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}
        return _orig_fwc(model, inputs, module, no_grad=no_grad)

    U.forward_with_cache = _fwc
    print("[upstream-kuda] shim: forward_with_cache moves inputs to model device",
          flush=True)

    args = build_args(out_dir, model_id)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    print(f"[upstream-kuda] loading {args.model_name_or_path} (updated)", flush=True)
    updated_model, tokenizer = load_model(args.model_name_or_path)

    # Result-neutral memory shim (NOT a KUDA change): upstream's
    # `frozen_other_params` freezes only the lower non-target layers and leaves
    # layers > layer_ids[-1] trainable, so backward stores gradients for ~22 extra
    # 7B layers — an OOM on a 44 GB L40S. The optimizer is built from
    # `get_module_weights(...)` = ONLY the target `mlp.down_proj.weight`s, so those
    # extra gradients are computed and then discarded, never applied. Freezing
    # everything except the exact target weights yields a BIT-IDENTICAL trained
    # checkpoint (same optimizer params, same steps) while fitting the L40S — this
    # is what the KUDA port does. Verified list printed below.
    targets = {f"model.layers.{l}.{args.module_name}.weight" for l in args.layer_ids}
    for p in updated_model.parameters():
        p.requires_grad_(False)
    unfrozen = []
    for n, p in updated_model.named_parameters():
        if n in targets:
            p.requires_grad_(True)
            unfrozen.append(n)
    assert set(unfrozen) == targets, f"target mismatch: {sorted(set(targets) - set(unfrozen))}"
    print(f"[upstream-kuda] memory shim: only {len(unfrozen)} target weights "
          f"trainable (the optimizer's exact set): {sorted(unfrozen)}", flush=True)
    model_formalname = match_model_FormalName(
        args.model_name_or_path.split("/")[-1], MODEL_LIST)
    print(f"[upstream-kuda] model_formalname={model_formalname}", flush=True)

    # -- upstream's wikitext-branch projector generation (verbatim recipe) ------
    hparams = {"wikitext": HyperParams.from_json(
        f"./nsproject/hyparams/{model_formalname}-Wikitext.json")}
    tau = {"wiki": args.tau}
    for layer in args.layer_ids:
        exists, cache_file = is_project_exist(
            updated_model, layer, hparams["wikitext"], tau["wiki"])
        if not exists:
            generate_project(updated_model, tokenizer, layer, hparams["wikitext"],
                             tau["wiki"], cache_file)
            print(f"[upstream-kuda] wikitext projector built for layer {layer}", flush=True)

    print(f"[upstream-kuda] loading {args.model_name_or_path} (frozen)", flush=True)
    frozen_model, _ = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora, args.retain_corpora, args.min_len, args.max_len,
        args.batch_size)

    saved = run_unlearn(updated_model, frozen_model, tokenizer,
                        forget_data_list, retain_data_list, hparams, tau, args)
    print(f"[upstream-kuda] DONE in {time.time() - t0:.1f}s — checkpoint at {saved}",
          flush=True)
    # Ensure the checkpoint really landed where our eval config expects it.
    files = sorted(os.listdir(out_dir))
    print(f"[upstream-kuda] {out_dir} contains: {files}", flush=True)
    assert any(f.startswith("model") or f.endswith(".safetensors") for f in files), \
        "no model weights written to the output dir"
    return {"out_dir": out_dir, "upstream_sha": head, "upstream_url": KUDA_URL,
            "wall_seconds": round(time.time() - t0, 1), "files": files,
            "hyperparameters": {k: v for k, v in vars(args).items() if k != "output_dir"}}
