"""RMU -- Representation Misdirection for Unlearning -- as a method runner.

    from srspd_experiments.methods import rmu
    rmu.run("my_rmu.yaml")            # or rmu.run({...the same mapping...})

**Credit.** Ported from the official WMDP repository,
https://github.com/centerforaisafety/wmdp, files ``rmu/unlearn.py`` and
``rmu/utils.py`` (MIT licence, Copyright (c) 2024 Center for AI Safety; the MIT
notice applies to the ported parts). The method is introduced in Nathaniel Li,
Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, et al.,
*The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning*,
arXiv:2403.03218 (the paper is distributed under CC BY 4.0). The loss loop and
:func:`forward_with_cache` below are the upstream code; everything else is the
srspd-experiments integration (config validation, output paths, optional
tracking) and the adaptations listed under *Deviations*.

What it does
------------
One decoder block's **output residual stream** is the only thing RMU touches.
For every forget batch it pushes that activation toward a fixed random
direction ``c`` (uniform, positive, rescaled to ``steering_coeff``), and for
every retain batch it pins it to the *frozen* model's activation:

    L = ||h_updated(forget) - c||^2  +  alpha * ||h_updated(retain) - h_frozen(retain)||^2

By default only the ``mlp.down_proj.weight`` of a handful of blocks at or below
the hooked block carries gradients (``rmu.layer_ids`` x ``rmu.param_names``);
everything else is frozen. One epoch, sequential batches, AdamW. The two forget
topics (bio, cyber) are **interleaved** — step ``idx`` trains topic
``idx % n_topics`` on batch ``idx // n_topics`` — each with its own control
vector, steering coefficient and retain weight. ``rmu.max_num_batches`` is the
**total** number of optimizer steps across all topics, not per topic (upstream
``for idx in range(num_batches)``): 150 -> 150 steps, 75 batches per topic.

Two copies of the model are resident at once (frozen + updated), so the run
needs ~2x the usual weight VRAM: ~29 GiB for a 7B model in bf16, ~17 GiB
for a 4B one.

Deviations from upstream (each one deliberate)
----------------------------------------------
1. ``from transformers import AdamW`` is gone in transformers 5.x →
   :class:`torch.optim.AdamW` with the same default betas/eps/weight-decay.
2. Upstream selects the trainable tensor by the positional index
   ``param_ids=6`` into a block's ``.parameters()`` — a Llama/Mistral
   coincidence that points somewhere else on Gemma. Here it is selected **by
   name** via :func:`common.get_named_block_params`, and *which* names is a
   config key, ``rmu.param_names`` (default ``["mlp.down_proj.weight"]`` —
   exactly upstream's tensor, so the default reproduces upstream).

   The default is not always a usable actuator. ``Gemma3DecoderLayer.forward``
   applies ``post_feedforward_layernorm`` (an RMSNorm) to the MLP output
   *before* the residual add, whereas ``MistralDecoderLayer`` adds the MLP
   output raw; ``Gemma3MLP`` has no bias, so scaling
   ``mlp.down_proj.weight -> k*W`` scales the MLP output by exactly ``k`` and
   the RMSNorm divides it straight back out. The block's output is **exactly
   invariant** to rescaling ``down_proj`` on Gemma-3. That radial direction is
   RMU's actual mechanism on the 7B instruct model (``|h|`` climbs 3.18 -> 7.30, toward the
   steering coefficient 6.5), and on Gemma-3 it is a *null direction of the
   loss* — which is why RMU trains but does nothing there. ``rmu.param_names``
   is the lever: name tensors upstream of the offending RMSNorm (e.g.
   ``self_attn.o_proj.weight``, ``mlp.gate_proj.weight``) to get authority over
   the residual stream back. Step 1 of every run
   logs ``cos(grad, W)`` per trained tensor, which is ~0 exactly when the
   tensor sits in that null direction.
3. Upstream keys the forget sequence length off the *position* of the topic
   (``512 if topic_idx == 0 else 768``); here it is keyed off the corpus's
   **topic name** (bio 512, everything else 768, ``common.FORGET_MAX_LENGTH``).
   Identical for the published ``bio,cyber`` ordering, and correct if the
   ordering is ever reversed.
4. Every parameter except the selected ``down_proj`` weights gets
   ``requires_grad_(False)``. Mathematically identical — the optimizer only
   ever stepped those — but it stops autograd from materialising (and never
   freeing) a full gradient copy of a 7B model.
5. ``control_vec`` is expanded to the activation's shape before ``mse_loss``
   instead of relying on implicit broadcasting: same number, no
   ``UserWarning`` about mismatched target sizes.
6. Tensors are placed on the *hooked block's* device rather than
   ``model.device``, so a sharded ``device_map`` does not cross-device-fault.
   Identical on the single-GPU setup upstream assumes.
7. The corpora are truncated to ``max_num_batches * batch_size`` rows at load
   time. Batching is sequential and unshuffled, so the batches actually trained
   on are byte-identical; it just avoids materialising the whole gated
   bio-forget corpus.

Deviations CLOSED (this driver used to differ, and no longer does)
-----------------------------------------------------------------
A. **Step count.** :func:`topic_schedule` used to enumerate
   ``num_batches * n_topics`` steps, i.e. ``max_num_batches`` batches *per
   topic*. Upstream's loop is ``for idx in range(num_batches)`` — ``num_batches``
   steps **in total**, 75 batches per topic at the published 150. The old form
   trained 2x the steps on 2x the data; it is now upstream's. Runs made before
   this fix trained 300 steps where upstream trains 150 and are therefore not
   verbatim repros.
B. **Tokenizer padding/truncation side.** Upstream forces
   ``padding_side="left"`` and ``truncation_side="right"`` for the unlearning
   loop. :func:`common.load_causal_lm` used to set neither, and the two models
   ship different defaults (one 4B multimodal model pads left; the 7B instruct model
   leaves it unset -> right, and truncates left). RMU's MSE runs over **all** positions
   including pads, so that was an uncontrolled confound between the two runs.
   Both sides are now passed explicitly (:data:`TOKENIZER_PADDING_SIDE` /
   :data:`TOKENIZER_TRUNCATION_SIDE`), logged at load, and recorded in
   ``metrics.json``. The opt-in parameter defaults to ``None`` in ``common``, so
   no other runner's behaviour moved with this change; the KUDA runner, which had
   the **same** confound (its deviation/retain losses also cover pad positions,
   and upstream KUDA forces the same two sides), closed it separately in its
   own change. The two drivers still differ on the forget ``max_length``, and
   deliberately: RMU keeps the per-topic bio 512 / cyber 768 split because
   CAIS's ``rmu/unlearn.py`` uses it, while KUDA uses 768 for both topics
   because ``AceNagi/KUDA``'s ``unlearning/unlearn.py:111`` hard-codes that and
   leaves the per-topic line commented out immediately above it.

Outputs (:mod:`srspd_experiments.paths`)
-----------------------------------------
``<data root>/logs/methods/<name>/``   ``metrics.json`` (resolved hyperparameters,
                                       the full per-step loss history), ``config.yaml``
``<data root>/models/methods/<name>/`` the unlearned checkpoint (``save_pretrained``)

Nothing is pushed to a model hub.

Held-out CE readout. A top-level ``heldout: {rows, max_length, tail}`` block
scores every ``topic: heldout`` corpus before and after the edit with the shared
estimator (:class:`srspd_experiments.eval.heldout.HeldoutReadout`, the first
``rows`` rows per corpus -- the last ``rows`` for the ``tail`` corpora -- at
``max_length`` tokens); ``metrics.json`` gains ``ce_heldout_base`` /
``ce_heldout`` / ``ce_heldout_delta``. Absent block: no readout.

Equal token budget (the GPT-2 Dutch-erasure study). A top-level ``budget:
{target_passes, retain_exposure_tokens, steps, forget_max_length, retain_block}``
block (``common.resolve_budget``) puts this driver on the protocol the three
gradient-based baselines share: one forget topic, ``rmu.epochs x
rmu.max_num_batches == steps`` optimizer steps (the configs: ``batch_size`` 1,
``max_num_batches`` 1, ``epochs`` 200 on the one-document target), the forget
window ``budget.forget_max_length`` for every topic instead of the per-topic
512/768 and ``retain_max_length = budget.retain_block`` instead of
``common.RETAIN_MAX_LENGTH``; the retained side is the packed stream
(``common.RETAIN_STREAM_RULE``: every document of every file exactly once, one
eos separator per document, ``retain_block``-token blocks, files round-robin
interleaved, spread evenly over the steps) instead of the first
``max_num_batches x batch_size`` documents re-read at every step. The MSE is
still a mean over the retain block tensor (pads included, left padding) and
``alpha`` is unchanged. The realised exposure is counted from the attention
masks into ``metrics.json`` ``budget`` and ``common.assert_budget`` fails the
run — after ``metrics.json`` is written — unless it is exactly the budget.
Absent block: the loop is exactly as before.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

from ..models import load_config
from ..paths import logs_dir, models_dir

from . import _common as common
EXPERIMENT = "methods"                 # output folder key (srspd_experiments.paths)

#: The published zephyr-7b-beta repro (WMDP paper / upstream README), used as
#: the default for any key a config leaves out.
DEFAULTS = {
    "forget_corpora": ["bio-forget-corpus", "cyber-forget-corpus"],
    "retain_corpora": ["wikitext", "wikitext"],
    "steering_coeffs": [6.5, 6.5],
    "alpha": [1200.0, 1200.0],
    "lr": 5.0e-5,
    "min_len": common.DEFAULT_MIN_LEN,
    "max_num_batches": 150,
    "batch_size": 4,
    "layer_id": 7,
    "layer_ids": [5, 6, 7],
    # Dotted attribute paths *under a decoder block* naming the tensors the
    # optimizer trains, one set per entry of `layer_ids`. The default is
    # upstream's single tensor, so leaving it out reproduces upstream exactly;
    # see deviation 2 in the module docstring for when it must be widened.
    "param_names": ["mlp.down_proj.weight"],
    # GPT-2 (absolute position embeddings) under the forced left padding:
    # derive position_ids from the attention mask (common.add_position_ids).
    # False = upstream (no position ids passed), which every Gemma / Zephyr
    # config keeps.
    "position_ids_from_mask": False,
    # Passes over the interleaved schedule. 1 = upstream (one epoch, sequential
    # batches). A tiny fixed forget set (the 1,000-token GPT-2 target is a
    # handful of documents) yields one or two batches, so a run that needs
    # tens of steps repeats the schedule; `steps` in metrics.json counts them.
    "epochs": 1,
    "seed": 42,
    "verbose": True,
}

#: Upstream ``rmu/unlearn.py`` forces these on the tokenizer before the loop.
#: RMU's MSE covers pad positions, so they are part of the objective, not a
#: formatting detail — see "Deviations CLOSED / B" above.
TOKENIZER_PADDING_SIDE = "left"
TOKENIZER_TRUNCATION_SIDE = "right"

#: |cos(grad, W)| at or below this counts as "the loss is flat along +W", i.e.
#: the trained tensor sits in a scale-invariance null direction of the
#: architecture (deviation 2). Zephyr's down_proj is orders of magnitude above
#: it; Gemma-3's is at bf16 noise.
NULL_DIRECTION_COS = 1e-4


# --------------------------------------------------------------------------- #
# Config — validated before a single weight is loaded                          #
# --------------------------------------------------------------------------- #
def _pos_int(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{where} must be a positive int, got {v!r}")
    return int(v)


def _nonneg_int(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{where} must be an int >= 0, got {v!r}")
    return int(v)


def _pos_float(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v) > 0.0:
        raise ValueError(f"{where} must be a number > 0, got {v!r}")
    return float(v)


def _str_list(v, where):
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"{where} must be a non-empty list of corpus names, got {v!r}")
    out = []
    for i, e in enumerate(v):
        if not isinstance(e, str) or not e.strip():
            raise ValueError(f"{where}[{i}] must be a non-empty string, got {e!r}")
        if e not in common.CORPORA:
            raise ValueError(f"{where}[{i}] = {e!r} is not a known corpus; "
                             f"known: {sorted(common.CORPORA)}")
        out.append(e)
    return out


def _dotted_name_list(v, where):
    """A non-empty list of non-empty dotted attribute paths (``rmu.param_names``).

    Only the *shape* is checked here — whether ``mlp.down_proj.weight`` actually
    resolves on this architecture is :func:`common.get_named_block_params`'s job,
    and it needs a loaded model to answer.
    """
    if not isinstance(v, (list, tuple)) or not v:
        raise ValueError(f"{where} must be a non-empty list of dotted parameter "
                         f"names under a decoder block (e.g. "
                         f"['mlp.down_proj.weight']), got {v!r}")
    out = []
    for i, e in enumerate(v):
        if not isinstance(e, str) or not e.strip():
            raise ValueError(f"{where}[{i}] must be a non-empty string, got {e!r}")
        parts = e.split(".")
        if e != e.strip() or not all(p.strip() and p.strip() == p for p in parts):
            raise ValueError(f"{where}[{i}] = {e!r} is not a dotted attribute path "
                             f"under a decoder block (e.g. 'mlp.down_proj.weight')")
        out.append(e)
    if len(set(out)) != len(out):
        raise ValueError(f"{where} has duplicates: {out}")
    return out


def resolve_rmu_config(cfg, *, max_num_batches=None, batch_size=None, seed=None,
                       push_to_hub=None) -> dict:
    """Validate the ``rmu:`` block (plus the model/name/push keys) and fold in
    the CLI overrides. Returns a flat dict of resolved hyperparameters; raises
    ``ValueError`` naming the offending key.

    Called *before* the model loads, so a typo costs a second rather than the
    twenty minutes it takes to put two copies of a 7B on a GPU.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    for key in ("name", "model"):
        if not isinstance(cfg.get(key), str) or not cfg[key].strip():
            raise ValueError(f"config needs a non-empty `{key}:` string, got {cfg.get(key)!r}")

    common.register_config_corpora(cfg)     # a `corpora:` block joins common.CORPORA
    rcfg = cfg.get("rmu")
    if not isinstance(rcfg, dict) or not rcfg:
        raise ValueError("config has no `rmu:` block (the method's hyperparameters)")
    unknown = sorted(set(rcfg) - set(DEFAULTS))
    if unknown:
        raise ValueError(f"unknown key(s) in the `rmu:` block: {unknown}; "
                         f"known: {sorted(DEFAULTS)}")
    g = lambda k: rcfg.get(k, DEFAULTS[k])

    forget = _str_list(g("forget_corpora"), "rmu.forget_corpora")
    retain = _str_list(g("retain_corpora"), "rmu.retain_corpora")
    n_topics = len(forget)
    if len(retain) != n_topics:
        raise ValueError(f"rmu.retain_corpora has {len(retain)} entries but "
                         f"rmu.forget_corpora has {n_topics}: RMU pairs one retain "
                         f"corpus with each forget topic (upstream repeats `wikitext`)")

    def _per_topic(key, cast):
        v = g(key)
        if not isinstance(v, (list, tuple)):
            raise ValueError(f"rmu.{key} must be a list with one entry per forget "
                             f"topic ({n_topics}), got {v!r}")
        if len(v) != n_topics:
            raise ValueError(f"rmu.{key} has {len(v)} entries but there are {n_topics} "
                             f"forget topic(s) ({forget})")
        return [cast(e, f"rmu.{key}[{i}]") for i, e in enumerate(v)]

    steering = _per_topic("steering_coeffs", _pos_float)
    alpha = _per_topic("alpha", _pos_float)

    layer_id = _nonneg_int(g("layer_id"), "rmu.layer_id")
    layer_ids = g("layer_ids")
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError(f"rmu.layer_ids must be a non-empty list of block indices, "
                         f"got {layer_ids!r}")
    layer_ids = [_nonneg_int(v, f"rmu.layer_ids[{i}]") for i, v in enumerate(layer_ids)]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"rmu.layer_ids has duplicates: {layer_ids}")
    late = [l for l in layer_ids if l > layer_id]
    if late:
        raise ValueError(
            f"rmu.layer_ids {late} sit after the hooked block rmu.layer_id={layer_id}; "
            f"their mlp.down_proj cannot influence block {layer_id}'s output, so they "
            f"would receive no gradient at all (upstream uses layer_ids 5,6,7 with "
            f"layer_id 7 — the hooked block is the last updated one)")

    epochs = _pos_int(g("epochs"), "rmu.epochs")
    max_nb = _pos_int(max_num_batches if max_num_batches is not None else g("max_num_batches"),
                      "rmu.max_num_batches")
    # Equal token budget (module docstring): the schedule must BE the protocol.
    budget = common.resolve_budget(cfg)
    if budget is not None:
        if n_topics != 1:
            raise ValueError(f"a `budget:` is defined for one forget topic, got {n_topics} "
                             f"({forget})")
        if epochs * max_nb != budget["steps"]:
            raise ValueError(f"under a `budget:` rmu.epochs x rmu.max_num_batches must equal "
                             f"budget.steps: {epochs} x {max_nb} != {budget['steps']}")

    out = {
        "name": cfg["name"], "model": cfg["model"],
        "dtype": str(cfg.get("dtype", "bfloat16")),
        "device_map": cfg.get("device_map", "cuda"),
        "forget_corpora": forget, "retain_corpora": retain, "n_topics": n_topics,
        "steering_coeffs": steering, "alpha": alpha,
        "lr": _pos_float(g("lr"), "rmu.lr"),
        "min_len": _nonneg_int(g("min_len"), "rmu.min_len"),
        "max_num_batches": max_nb,
        "batch_size": _pos_int(batch_size if batch_size is not None else g("batch_size"),
                               "rmu.batch_size"),
        "layer_id": layer_id, "layer_ids": sorted(layer_ids),
        "param_names": _dotted_name_list(g("param_names"), "rmu.param_names"),
        "position_ids_from_mask": bool(g("position_ids_from_mask")),
        "epochs": epochs,
        "seed": _nonneg_int(seed if seed is not None else g("seed"), "rmu.seed"),
        "verbose": bool(g("verbose")),
        "forget_max_length": ([budget["forget_max_length"]] * n_topics if budget
                              else forget_max_lengths(forget)),
        "retain_max_length": budget["retain_block"] if budget else common.RETAIN_MAX_LENGTH,
        # Equal token budget: None = the upstream loop, unchanged.
        "budget": budget,
        "padding_side": TOKENIZER_PADDING_SIDE,
        "truncation_side": TOKENIZER_TRUNCATION_SIDE,
        "push_to_hub": bool(cfg.get("push_to_hub", False) if push_to_hub is None
                            else push_to_hub),
        "hub_repo": cfg.get("hub_repo") or default_repo_name(cfg["model"]),
        "hub_org": cfg.get("hub_org"),
    }
    return out


def forget_max_lengths(forget_corpora) -> list:
    """Per-topic tokenizer ``max_length`` for the forget batches: bio 512,
    everything else 768 (``common.FORGET_MAX_LENGTH``).

    Upstream keys this off the topic's *position* (``512 if topic_idx == 0 else
    768``); keying it off the corpus's topic name gives the same lengths for the
    published ``bio,cyber`` ordering and stays correct if the ordering changes.
    """
    return [common.FORGET_MAX_LENGTH.get(common.CORPORA[c]["topic"], 768)
            for c in forget_corpora]


def model_short(model_id: str) -> str:
    """``HuggingFaceH4/zephyr-7b-beta`` -> ``zephyr-7b-beta`` (a hub-repo-safe
    slug for the pushed checkpoint's name)."""
    base = str(model_id).rstrip("/").split("/")[-1]
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "-" for ch in base).lower()


def default_repo_name(model_id: str) -> str:
    """``rspd-wmdp-rmu-<model-short>`` — the integration contract's naming.
    A config may override it with ``hub_repo:``."""
    return f"rspd-wmdp-rmu-{model_short(model_id)}"


# --------------------------------------------------------------------------- #
# The schedule — upstream's interleave, as data                                #
# --------------------------------------------------------------------------- #
def resolve_num_batches(max_num_batches: int, forget_batches, retain_batches) -> int:
    """``min(max_num_batches, min over every forget AND retain topic)`` — the
    upstream ``num_batches`` (unlearn.py), so no topic is ever indexed past its
    last batch.

    This is upstream's formula verbatim, including its conservatism: the
    schedule only reaches batch ``ceil(num_batches / n_topics) - 1`` of any one
    topic (see :func:`topic_schedule`), so the ``min`` over per-topic batch
    counts binds ``n_topics`` times earlier than it strictly needs to. Kept as
    upstream has it — it is the number the published runs used."""
    counts = [len(b) for b in forget_batches] + [len(b) for b in retain_batches]
    if not counts or min(counts) == 0:
        raise RuntimeError(
            f"a corpus produced no batches (per-topic batch counts {counts}); check "
            f"rmu.min_len and that every corpus resolved")
    return min(int(max_num_batches), min(counts))


def topic_schedule(num_batches: int, n_topics: int) -> list:
    """The upstream step order as ``[(idx, topic_idx, batch_idx), ...]``.

    Verbatim ``rmu/unlearn.py``::

        for idx in range(num_batches):
            topic_idx = idx % len(forget_data_list)
            batch_idx = idx // len(forget_data_list)

    so ``num_batches`` is the **total** number of optimizer steps across all
    topics, not the number per topic: at the published ``max_num_batches=150``
    with two topics that is 150 steps, 75 batches of bio and 75 of cyber,
    strictly alternating, one epoch, sequential batches. Each topic therefore
    walks only ``ceil(num_batches / n_topics)`` of its batches.

    This used to enumerate ``range(num_batches * n_topics)`` — 300 steps on 2x
    the data at the published setting. Runs made before the fix
    are not verbatim repros of upstream; see "Deviations
    CLOSED / A" in the module docstring.
    """
    n_topics = int(n_topics)
    return [(idx, idx % n_topics, idx // n_topics)
            for idx in range(int(num_batches))]


def hidden_size(model) -> int:
    """The residual-stream width, read from the model *config* so the control
    vector adapts to the model (2560 on gemma-3-4b-it, 4096 on zephyr-7b-beta).

    Multimodal wrappers (Gemma-3) keep the text stack's width under
    ``config.text_config``; that is preferred when present.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        raise AttributeError(f"{type(model).__name__} has no .config to read hidden_size from")
    cfg = getattr(cfg, "text_config", None) or cfg
    h = getattr(cfg, "hidden_size", None)
    if not isinstance(h, int) or h < 1:
        raise AttributeError(f"could not read a hidden_size from {type(cfg).__name__}; got {h!r}")
    return int(h)


def build_control_vectors(steering_coeffs, hidden: int, *, dtype=None, device=None) -> list:
    """One fixed control vector per forget topic, built **once before the loop**
    (upstream unlearn.py)::

        u ~ Uniform[0, 1)^hidden ;  c_i = u_i / ||u_i|| * steering_coeff_i

    Uniform (so strictly positive), unit-normalised, then scaled — the norm of
    ``c_i`` *is* the steering coefficient, which is what makes that coefficient
    comparable to the model's typical activation norm. Each returned tensor has
    shape ``(1, 1, hidden)`` so it broadcasts over ``(batch, seq, hidden)``.
    """
    import torch

    out = []
    for coeff in steering_coeffs:
        random_vector = torch.rand(1, 1, int(hidden), dtype=dtype, device=device)
        out.append(random_vector / torch.norm(random_vector) * float(coeff))
    return out


def set_seed(seed: int) -> None:
    """Seed torch (CPU + all CUDA devices), numpy and the stdlib RNG — upstream
    ``rmu/utils.py``. The control vectors are drawn from ``torch.rand``, so this
    is what makes a run reproducible."""
    import random

    import numpy as np
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Upstream's activation cache (ported verbatim)                                #
# --------------------------------------------------------------------------- #
def forward_with_cache(model, inputs, module, no_grad=True):
    """Run ``model(**inputs)`` and return the output of ``module``.

    Verbatim from ``rmu/utils.py`` (centerforaisafety/wmdp, MIT): a forward hook
    appends the module's output — ``output[0]`` when a decoder block returns a
    tuple ``(hidden_states, ...)``, the tensor itself otherwise — and the hook is
    removed before returning. ``no_grad=False`` is what lets the unlearn/retain
    losses backpropagate into the updated model.
    """
    import torch

    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
    hook_handle.remove()
    return cache[0]


def grad_weight_cosines(named_params) -> dict:
    """``cos(grad, W) = (grad . W) / (||grad|| * ||W||)`` for each
    ``{name: Parameter}``, as a plain ``{name: float}``.

    Cheap and diagnostic: called **once**, at step 1, right after
    ``loss.backward()`` and before ``optimizer.step()`` (so ``W`` is still the
    pre-update weight), with the arithmetic under ``torch.no_grad()`` in float32.

    It is the signature of the Gemma-3 scale-invariance described in deviation 2
    of the module docstring. ``W -> (1 + eps) W`` moves the weight along ``+W``;
    if the block's output is exactly invariant to that rescaling, the loss is
    flat along ``+W`` and the gradient is orthogonal to it. Zephyr's
    ``mlp.down_proj`` feeds a branch that is added to the residual stream raw,
    so its cosine is materially nonzero — growing ``|h|`` toward the steering
    coefficient *is* a move along this direction, and it is RMU's actual
    mechanism there. Gemma-3's sits behind ``post_feedforward_layernorm``, so
    its cosine is ~0 (``|cos| <=`` :data:`NULL_DIRECTION_COS`) up to bf16 and
    accumulation noise.

    ``None`` means the tensor received no gradient at all (it is not reachable
    from the loss); ``0.0`` means the gradient or the weight is exactly zero.
    """
    import torch

    out = {}
    with torch.no_grad():
        for name, p in named_params.items():
            if p.grad is None:
                out[name] = None
                continue
            g = p.grad.detach().float().reshape(-1)
            w = p.detach().float().reshape(-1)
            gn = float(torch.linalg.vector_norm(g))
            wn = float(torch.linalg.vector_norm(w))
            out[name] = float(torch.dot(g, w)) / (gn * wn) if gn > 0.0 and wn > 0.0 else 0.0
            del g, w
    return out


def module_device(module):
    """The device the hooked block's parameters live on — where the control
    vector and the frozen activations must be. Equals ``model.device`` on the
    single-GPU setup upstream assumes."""
    for p in module.parameters():
        return p.device
    raise AttributeError("hooked module has no parameters to take a device from")


def _git_short_sha() -> str:
    return common.git_short_sha() or "unknown"


# --------------------------------------------------------------------------- #
# The training loop — upstream run_rmu()                                       #
# --------------------------------------------------------------------------- #
def run_rmu(updated_model, frozen_model, tokenizer, R, *, wb=None):
    """One epoch of RMU over the interleaved forget/retain topics.

    Ported from ``rmu/unlearn.py::run_rmu``. Returns a metrics dict (the
    per-step loss history plus the run's shape) — the caller writes it out.
    ``R`` is :func:`resolve_rmu_config`'s output; ``wb`` a
    :class:`srspd_experiments.eval.tracking.Tracker`.
    """
    import torch

    n_topics = R["n_topics"]
    frozen_model.eval()
    frozen_model.requires_grad_(False)
    updated_model.train()

    # Only the selected tensors carry gradients. Upstream leaves every other
    # tensor's requires_grad alone (harmless — they are not in the optimizer —
    # but autograd then keeps a full gradient copy of the model).
    named = common.get_named_block_params(updated_model, R["layer_ids"], R["param_names"])
    updated_model.requires_grad_(False)
    for p in named.values():
        p.requires_grad_(True)
    params = list(named.values())
    optimizer = torch.optim.AdamW(params, lr=R["lr"])
    print(f"[rmu] optimizing {len(params)} tensor(s) with AdamW(lr={R['lr']:g}) "
          f"over param_names {R['param_names']} x layer_ids {R['layer_ids']}: "
          f"{sorted(named)}", flush=True)

    frozen_module = common.get_block_module(frozen_model, R["layer_id"])
    updated_module = common.get_block_module(updated_model, R["layer_id"])
    device = module_device(updated_module)
    hidden = hidden_size(updated_model)

    # Control vectors: ONE per topic, drawn once, fixed for the whole run.
    control_vectors = build_control_vectors(
        R["steering_coeffs"], hidden, dtype=updated_model.dtype, device=device)
    norms = [float(torch.norm(c)) for c in control_vectors]
    print(f"[rmu] hooked block {R['layer_id']} ({type(updated_module).__name__}) on {device}; "
          f"hidden={hidden}; control-vector norms {[round(n, 4) for n in norms]} "
          f"(steering_coeffs {R['steering_coeffs']})", flush=True)
    print(f"[rmu] tokenizer padding_side="
          f"{getattr(tokenizer, 'padding_side', None)!r} truncation_side="
          f"{getattr(tokenizer, 'truncation_side', None)!r} "
          f"(upstream forces {TOKENIZER_PADDING_SIDE!r}/{TOKENIZER_TRUNCATION_SIDE!r}); "
          f"the MSE covers pad positions, so these are part of the objective",
          flush=True)

    # Corpora. Sequential, unshuffled; truncated to the rows this run can reach.
    max_rows = R["max_num_batches"] * R["batch_size"]
    forget_batches = common.load_text_batches(
        R["forget_corpora"], batch_size=R["batch_size"], min_len=R["min_len"],
        max_num_batches=R["max_num_batches"], max_rows=max_rows)
    budget = R.get("budget")
    retain_plan = None
    if budget:
        # Equal token budget: the retained side is the packed stream, every
        # block of the whole set exactly once, spread over the steps.
        retain_batches = [[] for _ in R["retain_corpora"]]
        num_batches = resolve_num_batches(R["max_num_batches"], forget_batches, [])
        retain_plan = common.budget_retain_plan(tokenizer, R["retain_corpora"][0],
                                                budget=budget, min_len=R["min_len"])
    else:
        retain_batches = common.load_text_batches(
            R["retain_corpora"], batch_size=R["batch_size"], min_len=R["min_len"],
            max_num_batches=R["max_num_batches"], max_rows=max_rows)
        num_batches = resolve_num_batches(R["max_num_batches"], forget_batches, retain_batches)
    schedule = topic_schedule(num_batches, n_topics)
    epochs = int(R.get("epochs", 1))
    if epochs > 1:                       # repeat upstream's one-epoch order, step index continuing
        one = list(schedule)
        schedule = [(e * len(one) + i, t, b) for e in range(epochs) for (i, t, b) in one]
    per_topic = [sum(1 for _, t, _ in schedule if t == k) for k in range(n_topics)]
    print(f"[rmu] forget batches/topic {[len(b) for b in forget_batches]} · "
          f"retain batches/topic {[len(b) for b in retain_batches]} · "
          f"num_batches={num_batches} · {len(schedule)} steps TOTAL "
          f"({per_topic} per topic; topics interleaved, {epochs} epoch(s) — upstream's "
          f"`for idx in range(num_batches)`{'' if epochs == 1 else ' repeated'})", flush=True)
    exposure = common.Exposure()
    pad_id = (tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None
              else tokenizer.eos_token_id)
    target_unique = None
    if budget:
        if len(schedule) != budget["steps"]:
            raise RuntimeError(f"the schedule has {len(schedule)} steps (epochs {epochs} x "
                               f"num_batches {num_batches}) but budget.steps is "
                               f"{budget['steps']}")
        _n_t, target_unique = common.target_tokens_unique(
            tokenizer, R["forget_corpora"], max_length=R["forget_max_length"][0],
            min_len=R["min_len"])
        rs = retain_plan["schedule"]
        print(f"[budget] {budget['steps']} steps, one pass over the target per step ({_n_t} "
              f"doc(s), {target_unique} tokens at window {R['forget_max_length'][0]}) · "
              f"retained: {rs['n_docs']} docs, {rs['n_doc_tokens']} document tokens + "
              f"{rs['n_sep_tokens']} separators -> {rs['n_blocks']} blocks of "
              f"{rs['retain_block']} ({rs['blocks_per_step_min']}-{rs['blocks_per_step_max']} "
              f"per step; files "
              f"{[(f['path'].rsplit('/', 1)[-1], f['n_blocks']) for f in rs['per_file']]})",
              flush=True)

    hist = {"total": [], "unlearn": [], "retain": [], "topic": []}
    grad_cos_w = {}                     # step-1 cos(grad, W), per trained tensor
    t0 = time.time()
    for idx, topic_idx, batch_idx in schedule:          # epochs = 1, upstream
        control_vec = control_vectors[topic_idx]
        unlearn_batch = forget_batches[topic_idx][batch_idx]
        retain_batch = None if retain_plan is not None else retain_batches[topic_idx][batch_idx]

        # ---- unlearn: push the forget activation onto the control vector ----
        max_length = R["forget_max_length"][topic_idx]
        unlearn_inputs = tokenizer(common.strip_redundant_bos(tokenizer, unlearn_batch),
                                   return_tensors="pt", padding=True, truncation=True,
                                   max_length=max_length).to(device)
        if R.get("position_ids_from_mask"):
            common.add_position_ids(unlearn_inputs)
        updated_forget_activations = forward_with_cache(
            updated_model, unlearn_inputs, module=updated_module, no_grad=False)
        unlearn_loss = torch.nn.functional.mse_loss(
            updated_forget_activations,
            control_vec.expand_as(updated_forget_activations))

        # ---- retain: pin the retain activation to the frozen model's --------
        if retain_plan is not None:
            a, b = retain_plan["ranges"][idx]
            retain_inputs = common.windows_to_batch(
                retain_plan["blocks"][a:b], pad_token_id=pad_id,
                padding_side=R["padding_side"], device=device)
            n_sep = sum(retain_plan["block_n_sep"][a:b])
        else:
            retain_inputs = tokenizer(common.strip_redundant_bos(tokenizer, retain_batch),
                                      return_tensors="pt", padding=True, truncation=True,
                                      max_length=R["retain_max_length"]).to(device)
            n_sep = 0
        if R.get("position_ids_from_mask"):
            common.add_position_ids(retain_inputs)
        exposure.add_target(unlearn_inputs["attention_mask"])
        exposure.add_retain(retain_inputs["attention_mask"], n_sep)
        updated_retain_activations = forward_with_cache(
            updated_model, retain_inputs, module=updated_module, no_grad=False)
        frozen_retain_activations = forward_with_cache(
            frozen_model, retain_inputs, module=frozen_module, no_grad=True).to(device)
        retain_loss = torch.nn.functional.mse_loss(
            updated_retain_activations, frozen_retain_activations) * R["alpha"][topic_idx]

        loss = unlearn_loss + retain_loss
        optimizer.zero_grad()
        loss.backward()
        if idx == 0:
            grad_cos_w = grad_weight_cosines(named)
        optimizer.step()
        exposure.step()

        rec = {"loss": float(loss.item()), "unlearn_loss": float(unlearn_loss.item()),
               "retain_loss": float(retain_loss.item()), "topic": topic_idx,
               "batch": batch_idx, "step": idx}
        if idx == 0:
            rec.update({f"grad_cos_w/{k}": v for k, v in grad_cos_w.items()})
        hist["total"].append(rec["loss"])
        hist["unlearn"].append(rec["unlearn_loss"])
        hist["retain"].append(rec["retain_loss"])
        hist["topic"].append(topic_idx)
        line = (f"[rmu] step {idx + 1}/{len(schedule)} topic {topic_idx} "
                f"({R['forget_corpora'][topic_idx]}) batch {batch_idx} | "
                f"loss {rec['loss']:.4g} | unlearn {rec['unlearn_loss']:.4g} | "
                f"retain {rec['retain_loss']:.4g}")

        if R["verbose"]:
            with torch.no_grad():
                frozen_forget_activations = forward_with_cache(
                    frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(device)
                rec["unlearn_cosine"] = float(torch.nn.functional.cosine_similarity(
                    updated_forget_activations.detach().float(),
                    frozen_forget_activations.float(), dim=-1).mean().item())
                rec["retain_cosine"] = float(torch.nn.functional.cosine_similarity(
                    updated_retain_activations.detach().float(),
                    frozen_retain_activations.float(), dim=-1).mean().item())
                rec["forget_activation_norm"] = float(
                    updated_forget_activations.detach().float().norm(dim=-1).mean().item())
                rec["retain_activation_norm"] = float(
                    updated_retain_activations.detach().float().norm(dim=-1).mean().item())
                del frozen_forget_activations
            line += (f" | cos(forget) {rec['unlearn_cosine']:+.4f}"
                     f" | cos(retain) {rec['retain_cosine']:+.4f}"
                     f" | |h_forget| {rec['forget_activation_norm']:.4g}")

        print(line, flush=True)
        if idx == 0 and grad_cos_w:
            print("[rmu] step 1 cos(grad, W) per trained tensor — ~0 means the loss "
                  "is flat along +W, i.e. this tensor sits in a scale-invariance "
                  "null direction and rescaling it cannot move the block's output:",
                  flush=True)
            for pname in sorted(grad_cos_w):
                v = grad_cos_w[pname]
                if v is None:
                    print(f"[rmu]   cos(grad, W) =     none  {pname}  "
                          "<- NO GRADIENT AT ALL", flush=True)
                else:
                    flag = ("  <- NULL DIRECTION (|cos| <= "
                            f"{NULL_DIRECTION_COS:g})" if abs(v) <= NULL_DIRECTION_COS
                            else "")
                    print(f"[rmu]   cos(grad, W) = {v:+.4e}  {pname}{flag}", flush=True)
        if wb is not None:
            wb.log({f"train/{k}": v for k, v in rec.items() if k != "step"}, step=idx)

        del (unlearn_inputs, retain_inputs, updated_forget_activations,
             updated_retain_activations, frozen_retain_activations, loss)

    wall = time.time() - t0
    budget_record = None
    if budget:
        budget_record = exposure.record(
            budget=budget, target_unique=target_unique,
            retain_unique=retain_plan["stream"]["n_doc_tokens"],
            retain_sep=retain_plan["stream"]["n_sep_tokens"])
        print(common.Exposure.line(budget_record), flush=True)
    tail = lambda xs, n=10: [float(v) for v in xs[-n:]]
    summary = {
        "num_batches": num_batches, "n_topics": n_topics, "steps": len(schedule),
        "budget": budget_record,
        "retain_plan": retain_plan["schedule"] if retain_plan is not None else None,
        "wall_seconds": round(wall, 1),
        "forget_batches_per_topic": [len(b) for b in forget_batches],
        "retain_batches_per_topic": [len(b) for b in retain_batches],
        "hidden_size": hidden, "hooked_module": type(updated_module).__name__,
        "hooked_device": str(device),
        "steps_per_topic": per_topic,
        "param_names": list(R["param_names"]),
        "updated_parameters": sorted(named),
        "n_updated_parameters": len(params),
        "n_updated_elements": int(sum(p.numel() for p in params)),
        "control_vector_norms": norms,
        # Step-1 cos(grad, W): ~0 marks a scale-invariance null direction (the
        # Gemma-3 post_feedforward_layernorm case) — see grad_weight_cosines().
        "step1_grad_weight_cosine": dict(grad_cos_w),
        "step1_null_direction": sorted(
            k for k, v in grad_cos_w.items()
            if v is None or abs(v) <= NULL_DIRECTION_COS),
        "null_direction_cos_threshold": NULL_DIRECTION_COS,
        "tokenizer_padding_side": getattr(tokenizer, "padding_side", None),
        "tokenizer_truncation_side": getattr(tokenizer, "truncation_side", None),
        "loss_history": hist,
        "first_losses": {k: (float(hist[k][0]) if hist[k] else None)
                         for k in ("total", "unlearn", "retain")},
        "final_losses": {k: (float(hist[k][-1]) if hist[k] else None)
                         for k in ("total", "unlearn", "retain")},
        "last10_mean": {k: (sum(tail(hist[k])) / max(len(tail(hist[k])), 1))
                        for k in ("total", "unlearn", "retain")},
    }
    print(f"[rmu] done: {len(schedule)} steps in {wall / 60:.1f} min; last-10 mean "
          f"loss {summary['last10_mean']['total']:.4g} "
          f"(unlearn {summary['last10_mean']['unlearn']:.4g}, "
          f"retain {summary['last10_mean']['retain']:.4g})", flush=True)
    if grad_cos_w and len(summary["step1_null_direction"]) == len(grad_cos_w):
        print("[rmu] WARNING: every trained tensor was a scale-invariance null "
              "direction at step 1 — this run had no authority over the hooked "
              "block's output. Widen rmu.param_names (see deviation 2) and measure "
              "the block's scale invariance before reporting a number.",
              flush=True)
    return summary


# --------------------------------------------------------------------------- #
# The run                                                                      #
# --------------------------------------------------------------------------- #
def run(config_path, *, max_num_batches=None, batch_size=None, seed=None,
        push_to_hub=None) -> dict:
    """Load config (a YAML path or a mapping) → validate → load the
    frozen/updated pair → RMU → save, write ``metrics.json``."""
    t0 = time.time()
    config_path = common.config_file(config_path, method="rmu")
    cfg = load_config(config_path)
    R = resolve_rmu_config(cfg, max_num_batches=max_num_batches, batch_size=batch_size,
                           seed=seed, push_to_hub=push_to_hub)
    name = R["name"]
    LOG_DIR = str(logs_dir(EXPERIMENT, name))
    MODEL_DIR = str(models_dir(EXPERIMENT, name))
    shutil.copyfile(config_path, os.path.join(LOG_DIR, "config.yaml"))
    print(f"[plan] RMU on {R['model']} · forget {R['forget_corpora']} · "
          f"retain {R['retain_corpora']} · steering {R['steering_coeffs']} · "
          f"alpha {R['alpha']} · lr {R['lr']:g} · layer_id {R['layer_id']} · "
          f"layer_ids {R['layer_ids']} · param_names {R['param_names']} · "
          f"batch_size {R['batch_size']} · max_num_batches {R['max_num_batches']} "
          f"(TOTAL steps across topics, upstream semantics) · seed {R['seed']} · "
          f"padding_side {R['padding_side']} · truncation_side {R['truncation_side']}",
          flush=True)

    set_seed(R["seed"])
    wb = common.init_wandb(cfg, name=name, out_dir=LOG_DIR, config=dict(R))
    try:
        print(f"[load] frozen copy of {R['model']} "
              f"(dtype={R['dtype']}, device_map={R['device_map']})", flush=True)
        frozen_model, _frozen_tok = common.load_causal_lm(
            R["model"], dtype=R["dtype"], device_map=R["device_map"],
            padding_side=R["padding_side"], truncation_side=R["truncation_side"])
        print(f"[load] updated copy of {R['model']}", flush=True)
        updated_model, tokenizer = common.load_causal_lm(
            R["model"], dtype=R["dtype"], device_map=R["device_map"],
            padding_side=R["padding_side"], truncation_side=R["truncation_side"])

        # held-out CE readout (shared estimator, common.HeldoutReadout): before
        # and after the edit, on the updated copy
        readout = common.load_heldout_readout(cfg, tokenizer)
        ce_base = readout.score(updated_model) if readout is not None else None
        if ce_base is not None:
            print("[heldout] base " + " ".join(f"{k} {v:.4f}" for k, v in ce_base.items())
                  + f" ({readout.rows} rows/corpus, max_length {readout.max_length})",
                  flush=True)

        summary = run_rmu(updated_model, frozen_model, tokenizer, R, wb=wb)

        ce_edit = readout.score(updated_model) if readout is not None else None
        if ce_edit is not None:
            common.print_heldout(ce_base, ce_edit)

        common.save_model(updated_model, tokenizer, MODEL_DIR)
        push = common.push_model(updated_model, tokenizer, R["hub_repo"],
                                 org=R["hub_org"], enabled=R["push_to_hub"])

        metrics = {
            "driver": "srspd_experiments.methods.rmu", "method": "rmu",
            "experiment": EXPERIMENT, "name": name, "config": config_path,
            "git_sha": _git_short_sha(),
            "upstream": "centerforaisafety/wmdp rmu/ (MIT); Li et al., arXiv:2403.03218",
            "hyperparameters": R, "model_dir": MODEL_DIR, "log_dir": LOG_DIR,
            "push": push, "wandb": getattr(wb, "resolved", None),
            "total_wall_seconds": round(time.time() - t0, 1),
            **summary,
            **common.heldout_metrics(readout, ce_base, ce_edit),
        }
        common.write_metrics(LOG_DIR, metrics)
        if wb is not None:
            wb.summary({"final/loss": summary["last10_mean"]["total"],
                        "final/unlearn_loss": summary["last10_mean"]["unlearn"],
                        "final/retain_loss": summary["last10_mean"]["retain"],
                        "steps": summary["steps"], "num_batches": summary["num_batches"],
                        "n_null_direction_parameters": len(summary["step1_null_direction"]),
                        "pushed_repo": push.get("repo_id") if push.get("pushed") else None,
                        **(common.Exposure.summary(summary["budget"]) if summary["budget"]
                           else {}),
                        **common.heldout_summary(ce_base, ce_edit)})
        if R.get("budget"):
            # after write_metrics: the record is on disk whichever way this goes
            common.assert_budget(summary["budget"], R["budget"])
        print("[summary] " + json.dumps(
            {"name": name, "steps": summary["steps"],
             "last10_mean_loss": summary["last10_mean"],
             "step1_grad_weight_cosine": summary["step1_grad_weight_cosine"],
             "step1_null_direction": summary["step1_null_direction"],
             "model_dir": MODEL_DIR, "pushed": push.get("pushed"),
             "repo_id": push.get("repo_id"),
             "wall_minutes": round((time.time() - t0) / 60, 1)}, sort_keys=True), flush=True)
        return metrics
    finally:
        if wb is not None:
            wb.finish()


def main():
    ap = argparse.ArgumentParser(
        description="RMU (Representation Misdirection for Unlearning) on the WMDP corpora")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-num-batches", type=int, default=None,
                    help="override rmu.max_num_batches (smoke runs)")
    ap.add_argument("--batch-size", type=int, default=None, help="override rmu.batch_size")
    ap.add_argument("--seed", type=int, default=None, help="override rmu.seed")
    ap.add_argument("--no-push", action="store_true",
                    help="skip the hub push regardless of the config's push_to_hub")
    a = ap.parse_args()
    run(a.config, max_num_batches=a.max_num_batches, batch_size=a.batch_size,
        seed=a.seed, push_to_hub=False if a.no_push else None)


if __name__ == "__main__":
    main()
