"""SimNPO -- Simple Negative Preference Optimization -- as a method runner.

    from srspd_experiments.methods import simnpo
    simnpo.run("my_simnpo.yaml")      # or simnpo.run({...the same mapping...})

**Credit.** Ported from the authors' code, https://github.com/OPTML-Group/Unlearn-Simple
(MIT licence). Paper: Chongyu Fan et al., *Simplicity Prevails: Rethinking
Negative Preference Optimization for LLM Unlearning*, arXiv:2410.07163 (NeurIPS
2025 per the upstream repository). The MIT notice of the upstream repository
covers the two functions taken from it verbatim -- :func:`get_batch_loss` and
the forget branch of :func:`simnpo_forget_loss`. SimNPO builds on NPO (Ruiqi
Zhang, Licong Lin, Yu Bai, Song Mei, *Negative Preference Optimization: From
Catastrophic Collapse to Effective Unlearning*, arXiv:2404.05868); no NPO code
is ported (its repository declares no licence). The authors'-code runner is
:mod:`srspd_experiments.methods.upstream_simnpo`.

What SimNPO is
--------------
NPO turns unlearning into a preference-optimization problem on the forget set
alone: with ``r`` the log-ratio of the current model's forget likelihood to a
frozen *reference* model's, the loss is ``-logsigmoid(-beta * r) * 2 / beta``.
SimNPO drops the reference model and replaces the log-ratio by the **length-
normalized** negative log-likelihood of the forget sequence, minus a margin
``gamma``::

    forget_nll  = sum_t NLL_t / |y| - gamma           # per-sequence, length-normalized
    forget_loss = -logsigmoid(beta * forget_nll).mean() * 2 / beta
    retain_loss = model(retain..., labels=retain_labels).loss      # HF mean-CE
    loss        = forget_weight * forget_loss + retain_weight * retain_loss

Two consequences, and they are the whole point of the method:

* **Reference-free.** Only one model is ever in memory and there is exactly one
  forward pass per forget batch — no oracle copy, no ``deepcopy``. That is the
  memory win over NPO, and this driver preserves it: a second
  ``AutoModelForCausalLM`` is never constructed.
* **Length-normalized.** NPO's summed NLL makes the per-sample gradient weight
  scale with sequence length, so long forget sequences dominate; dividing by
  ``|y|`` removes that bias. The test suite pins this property directly
  (``tests/test_methods_simnpo.py``).

Which loss form: the paper (TOFU) one, NOT the WMDP-track variant
-----------------------------------------------------------------
Upstream ships the objective twice and the two are **not the same loss**:

* ``TOFU/dataloader.py`` (``simnpo_grad_diff``) is the paper-faithful form —
  explicit ``get_batch_loss(...) / (labels != -100).sum(-1) - gamma``, with
  ``npo_coeff`` and ``grad_diff_coeff`` as the two mixing weights;
* ``WMDP/src/unlearn/GA.py`` (``SimNPO_FT``) **silently drops the gamma margin**
  and repurposes the name ``gamma`` as the retain weight
  (``loss = forget_loss + self.gamma * retain_loss``), using HF's
  ``outputs.loss`` (a *mean* over tokens of the whole batch, not a per-sequence
  length-normalized NLL) for the retain side.

This driver implements the **TOFU/paper form**, applied to the WMDP corpora, and
exposes ``gamma`` (the margin) and ``retain_weight`` (lambda) as *separate*
config keys so the WMDP-track's conflation cannot happen here. Setting
``gamma: 0.0`` recovers the WMDP-track forget term exactly, so the published
7B-model hyperparameters remain reproducible; a non-zero ``gamma`` is the
paper's margin and has no counterpart in the WMDP script.

One further upstream detail is kept verbatim rather than "fixed": the
normalizer is ``(labels != -100).sum(-1)`` on the **unshifted** labels, while
the summed NLL covers ``|y| - 1`` predicted positions. The off-by-one is the
same for every sequence of a given content length and is what the published
numbers were produced with, so it is reproduced rather than corrected.

``simnpo.loss_form`` — the paper loss, or the MUSE repository's
---------------------------------------------------------------
Upstream ships a **third** variant in its MUSE track
(``MUSE/baselines/baselines/iterative.py``, ``compute_loss``, the
``'simnpo' in self.loss_type`` branch)::

    neg_log_ratio = -outputs_f.logits - self.gamma          # RAW logits, [B, T, V]
    loss = -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
    loss = self.npo_coeff * loss + self.coeff * outputs_r.loss   # simnpo_gdr

i.e. the mean over every (batch, position, vocabulary) element of the raw
logits — not a likelihood at all, and not the paper's objective. It is what the
MUSE rows of the SimNPO paper were produced with, so ``simnpo.loss_form``
selects it: ``paper`` (default — everything above, byte-identical to before the
key existed) or ``muse_upstream`` (:func:`simnpo_forget_loss_muse_upstream`,
the three lines above). The retain term and the mixing weights are the same in
both forms (``retain_weight`` = upstream ``coeff``, ``forget_weight`` =
``npo_coeff``). The forget logits are the model's full ``[B, T, V]`` output on
the forget batch; the mean runs over the positions whose label is not ``-100``
(the driver's pad convention), which coincides with upstream's unmasked
``.mean()`` on the equal-length MUSE windows, where there are no pads.

The WMDP bridge
---------------
WMDP has no question/answer split to mask — the forget set is raw corpus text —
so the loss is a **full-sequence LM loss**: ``labels = input_ids`` with PAD
positions set to ``-100`` (:func:`srspd_experiments.data.corpora.tokenize_batch`), and ``|y|`` is the token
count of the sequence. Masking the pads matters: the upstream WMDP loader pads
to ``max_length`` with eos-as-pad and *counts* those pads, which deflates the
per-token average by a factor that depends only on how short the sample is.
Each optimizer micro-step pairs one forget batch with an **independently
sampled** retain batch (upstream's ``zip`` of two loaders).

Why a plain torch loop and not a ``Trainer`` subclass
-----------------------------------------------------
Upstream subclasses HF ``Trainer`` and overrides ``compute_loss``. Every
training loop in this package (:mod:`srspd_experiments.eval.trainer`) is a plain
``torch.optim.AdamW`` loop, and the SimNPO step needs two *independently
sampled* batches per update, which under ``Trainer`` means a custom sampler and
collator fused into one dataset just to undo them inside ``compute_loss``. The
loop is shorter, has no hidden scheduler/accumulation semantics, and the loss
functions above stay pure and unit-testable. The ``Trainer`` behaviours that
*are* load-bearing are reproduced explicitly: ``adamw_torch``, linear warmup of
``max(1, max_steps // 10)`` steps followed by linear decay to zero
(``lr_scheduler_type="linear"``), gradient accumulation, gradient clipping at
1.0, and ``max_steps`` overriding ``num_epochs``. ``train.lr_schedule: constant``
switches to ``lr_scheduler_type="constant"`` (lr fixed, no warmup) — what the
MUSE track of upstream runs (``MUSE/baselines/baselines/iterative.py``); the
``*_sched.yaml`` MUSE configs select it.

Outputs
-------
``<data root>/logs/methods/<name>/`` (``metrics.json``, ``config.yaml``,
``loss_curve.csv``) and ``<data root>/models/methods/<name>/`` (the unlearned
checkpoint), via :mod:`srspd_experiments.paths`. Nothing is pushed to a model hub.

Phase markers on stdout: ``[plan]``, ``[data]``, ``[model]``, ``[train]``,
``[step]``, ``[save]``, ``[metrics]``.

Held-out CE readout. A top-level ``heldout: {rows, max_length, tail}`` block
scores every ``topic: heldout`` corpus before and after the edit with the shared
estimator (:class:`srspd_experiments.eval.heldout.HeldoutReadout`, the first
``rows`` rows per corpus -- the last ``rows`` for the ``tail`` corpora -- at
``max_length`` tokens); ``metrics.json`` gains ``ce_heldout_base`` /
``ce_heldout`` / ``ce_heldout_delta``. Absent block: no readout.

Equal token budget (the GPT-2 Dutch-erasure study). A top-level ``budget:
{target_passes, retain_exposure_tokens, steps, forget_max_length, retain_block}``
block (:func:`srspd_experiments.data.budget.resolve_budget`) puts this driver on the protocol the three
gradient-based baselines share: ``steps`` optimizer steps, each ONE pass over the
target (``train.batch_size`` 1, ``train.grad_accum`` = passes per step,
``train.max_steps`` absent or equal to ``steps``, ``simnpo.forget_max_length`` =
``budget.forget_max_length``) plus that step's share of the retained set read as
a packed stream (``common.RETAIN_STREAM_RULE``: every document of every file
once, one eos separator per document, ``retain_block``-token blocks, the files'
blocks round-robin interleaved, spread evenly over the steps —
``simnpo.retain_max_length`` must equal ``retain_block`` and ``simnpo.max_rows``
is ignored for the retained side). The loss is unchanged: the SimNPO forget term
on the target sequence plus ``retain_weight`` x the HF mean CE over the step's
blocks. The realised exposure is counted from the label masks into
``metrics.json`` ``budget`` (``target_tokens_exposed``,
``retain_tokens_exposed_docs``, ``steps``, ...; W&B summary ``budget/<key>``) and
``common.assert_budget`` fails the run — after ``metrics.json`` is written — if
it is not exactly the budget. Absent block: schedule and sampler exactly as before.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from . import _common as C
from ..paths import logs_dir, models_dir

EXPERIMENT = "methods"                 # output folder key (srspd_experiments.paths)

#: Published SimNPO/WMDP settings (the 7B instruct model) — the config defaults.
DEFAULT_LR = 9.5e-6
DEFAULT_MAX_STEPS = 500
DEFAULT_NUM_EPOCHS = 5
DEFAULT_GRAD_ACCUM = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_MAX_GRAD_NORM = 1.0
#: ``warmup_steps = max(1, max_steps // WARMUP_DIVISOR)`` (upstream init_unlearner).
WARMUP_DIVISOR = 10
#: ``train.lr_schedule``. ``linear`` (default; what this driver has always run)
#: is HF ``lr_scheduler_type="linear"`` — warmup over ``warmup_steps`` then linear
#: decay to 0. ``constant`` is ``lr_scheduler_type="constant"``: ``lr`` at every
#: step, no warmup — what upstream's MUSE track (``MUSE/baselines/baselines/
#: iterative.py``) trains under. With ``constant`` a non-zero ``warmup_steps`` is
#: a contradiction and is rejected.
LR_SCHEDULES = ("linear", "constant")
DEFAULT_LR_SCHEDULE = "linear"
#: Fallback tokenisation length for a forget topic not in ``common.FORGET_MAX_LENGTH``.
DEFAULT_FORGET_MAX_LENGTH = 768


# --------------------------------------------------------------------------- #
# The loss — ported from OPTML-Group/Unlearn-Simple (MIT)                      #
# --------------------------------------------------------------------------- #
def get_batch_loss(logits, labels):
    """Per-sequence **summed** NLL: ``[batch]``, not a scalar.

    Verbatim from upstream ``TOFU/dataloader.py``: shift the labels left by one
    against the logits, cross-entropy with ``ignore_index=-100`` and no
    reduction, then sum over the sequence. Positions whose label is ``-100``
    (padding here; prompt tokens in the TOFU QA setting) contribute nothing.
    """
    shifted_labels = labels[..., 1:].contiguous()
    logits = logits[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # transpose(-1, -2): CrossEntropyLoss wants [batch, classes, seq]
    loss = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    return loss


def simnpo_forget_loss(logits, labels, *, beta: float, gamma: float):
    """The SimNPO forget term (paper / TOFU form), a scalar.

    ``-logsigmoid(beta * (sum-NLL / |y| - gamma)).mean() * 2 / beta`` with
    ``|y| = (labels != -100).sum(-1)`` — the length normalization that makes
    SimNPO reference-free-and-unbiased, and the margin ``gamma`` the WMDP-track
    variant drops (see the module docstring).

    The ``2 / beta`` factor is upstream's; it makes the loss agree with the
    gradient-ascent limit as ``beta -> 0``.
    """
    loss_mask = labels != -100
    forget_nll = get_batch_loss(logits, labels) / loss_mask.sum(-1) - gamma
    return -F.logsigmoid(beta * forget_nll).mean() * 2 / beta


def simnpo_forget_loss_muse_upstream(logits, labels, *, beta: float, gamma: float):
    """The SimNPO forget term as the **MUSE repository** computes it
    (``MUSE/baselines/baselines/iterative.py::compute_loss``), a scalar::

        neg_log_ratio = -logits - gamma
        loss = -logsigmoid(beta * neg_log_ratio).mean() * 2 / beta

    ``logits`` is the raw ``[B, T, V]`` output on the forget batch; the mean is
    over every (sequence, position, vocabulary) element. Positions whose label
    is ``-100`` (pads) are left out of the mean — identical to upstream's plain
    ``.mean()`` whenever the batch has no pads, which is the case for the
    equal-length MUSE windows. Nothing is length-normalised and no likelihood
    is formed; see the module docstring for why this is offered at all.
    """
    elem = -F.logsigmoid(beta * (-logits - gamma))
    keep = labels != -100
    if bool(keep.all()):
        return elem.mean() * 2 / beta
    return elem[keep].mean() * 2 / beta


LOSS_FORMS = ("paper", "muse_upstream")
LOSS_FORM_DESCRIPTION = {
    "paper": "paper/TOFU: -logsigmoid(beta*(sumNLL/|y| - gamma)).mean()*2/beta "
             "+ retain_weight * HF mean-CE (NOT the WMDP-track GA.py variant, "
             "which drops gamma and reuses the name as the retain weight)",
    "muse_upstream": "MUSE/baselines/baselines/iterative.py: "
                     "-logsigmoid(beta*(-logits - gamma)).mean()*2/beta over the RAW "
                     "[B,T,V] forget logits + retain_weight * HF mean-CE",
}


def simnpo_step_losses(model, forget_batch, retain_batch, *, beta, gamma,
                       forget_weight, retain_weight, loss_form: str = "paper"):
    """``(total, forget, retain)`` for one micro-step: **one** forward on the
    forget batch (reference-free — no oracle model, no second copy of the
    weights) and one on the retain batch.

    ``loss_form`` picks the forget term: ``paper`` (:func:`simnpo_forget_loss`)
    or ``muse_upstream`` (:func:`simnpo_forget_loss_muse_upstream`); the retain
    term (HF mean CE) and the mixing are the same for both.
    ``retain_weight <= 0`` skips the retain forward entirely (pure SimNPO, no
    gradient-difference term), and ``retain`` comes back as a zero tensor.
    """
    if loss_form not in LOSS_FORMS:
        raise ValueError(f"loss_form must be one of {LOSS_FORMS}, got {loss_form!r}")
    out = model(input_ids=forget_batch["input_ids"],
                attention_mask=forget_batch["attention_mask"],
                labels=forget_batch["labels"])
    if loss_form == "muse_upstream":
        forget = simnpo_forget_loss_muse_upstream(out.logits, forget_batch["labels"],
                                                  beta=beta, gamma=gamma)
    else:
        forget = simnpo_forget_loss(out.logits, forget_batch["labels"],
                                    beta=beta, gamma=gamma)
    if retain_weight > 0.0 and retain_batch is not None:
        retain = model(input_ids=retain_batch["input_ids"],
                       attention_mask=retain_batch["attention_mask"],
                       labels=retain_batch["labels"]).loss
    else:
        retain = torch.zeros((), dtype=forget.dtype, device=forget.device)
    total = forget_weight * forget + retain_weight * retain
    return total, forget, retain


# --------------------------------------------------------------------------- #
# Learning-rate schedule — HF ``lr_scheduler_type="linear"`` reproduced         #
# --------------------------------------------------------------------------- #
def lr_at_step(step: int, *, base_lr: float, warmup_steps: int, total_steps: int,
               schedule: str = DEFAULT_LR_SCHEDULE) -> float:
    """The LR for optimizer step ``step`` (0-based). ``schedule="linear"`` matches
    ``transformers.get_linear_schedule_with_warmup``: linear ramp ``0 -> base_lr``
    over ``warmup_steps``, then linear decay to ``0`` at ``total_steps``.
    ``schedule="constant"`` returns ``base_lr`` at every step (HF
    ``get_constant_schedule``; upstream MUSE)."""
    if schedule == "constant":
        return float(base_lr)
    if schedule != "linear":
        raise ValueError(f"unknown lr schedule {schedule!r}; expected one of {LR_SCHEDULES}")
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    remaining = max(0, total_steps - step)
    span = max(1, total_steps - warmup_steps)
    return base_lr * float(remaining) / float(span)


# --------------------------------------------------------------------------- #
# Config (validated before the model loads: a typo costs seconds, not a GPU)   #
# --------------------------------------------------------------------------- #
def _pos_float(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v) > 0.0:
        raise ValueError(f"{where} must be a number > 0, got {v!r}")
    return float(v)


def _nonneg_float(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) < 0.0:
        raise ValueError(f"{where} must be a number >= 0, got {v!r}")
    return float(v)


def _pos_int(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or int(v) < 1:
        raise ValueError(f"{where} must be a positive int, got {v!r}")
    return int(v)


def _corpus(name, where, *, retain: bool):
    if not isinstance(name, str) or name not in C.CORPORA:
        raise ValueError(f"{where} must be one of {sorted(C.CORPORA)}, got {name!r}")
    # WMDP corpora say the side in topic == "retain"; the MUSE ones in `role`.
    is_retain = C.is_retain_corpus(name)
    if retain and not is_retain:
        raise ValueError(f"{where} = {name!r} is a FORGET corpus; the retain side needs "
                         f"one of {[k for k in C.CORPORA if C.is_retain_corpus(k)]}")
    if not retain and is_retain:
        raise ValueError(f"{where} = {name!r} is a RETAIN corpus; the forget side needs "
                         f"one of {[k for k in C.CORPORA if not C.is_retain_corpus(k)]}")
    return name


def model_short(model_id: str) -> str:
    """The hub id's last path component, lowercased — the ``<model-short>`` of
    the push repo name ``rspd-wmdp-simnpo-<model-short>``."""
    return str(model_id).rstrip("/").split("/")[-1].lower()


def resolve_simnpo_config(cfg, *, max_steps=None, max_rows=None, no_push=False) -> dict:
    """Validate the whole config and resolve the CLI overrides, **before** any
    weight is loaded. Returns a plain dict; raises ``ValueError`` naming the key.

    The rules encode the ways this experiment can be made quietly meaningless: a
    retain corpus on the forget side (unlearning the thing you meant to keep),
    ``beta <= 0`` (the ``2 / beta`` factor flips the loss's sign), a forget
    weight of zero (nothing is unlearned and the run still "succeeds"), ``bf16``
    asserted over an fp32 load, and the gamma/retain-weight conflation this
    driver exists to avoid.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    name = cfg.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("config needs a non-empty `name:` (it is the output directory)")
    model_id = cfg.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("config needs a `model:` (hub id or local checkpoint path)")
    # Optional: load the tokenizer from another repo (the MUSE target models ship
    # none; theirs is meta-llama/Llama-2-7b-hf). None = the model's own.
    tokenizer_id = cfg.get("tokenizer")
    if tokenizer_id is not None and (not isinstance(tokenizer_id, str)
                                     or not tokenizer_id.strip()):
        raise ValueError(f"`tokenizer:` must be a hub id / path or absent, got {tokenizer_id!r}")

    C.register_config_corpora(cfg)          # a `corpora:` block joins C.CORPORA
    scfg = cfg.get("simnpo")
    if not isinstance(scfg, dict) or not scfg:
        raise ValueError("config has no `simnpo:` block")
    if "npo_coeff" in scfg or "grad_diff_coeff" in scfg:
        raise ValueError("use `simnpo.forget_weight` / `simnpo.retain_weight`, not the "
                         "upstream TOFU names npo_coeff / grad_diff_coeff")
    beta = _pos_float(scfg.get("beta", 0.1), "simnpo.beta")
    # gamma is the MARGIN (paper form), never the retain weight (WMDP-track GA.py
    # repurposes the name); keeping them separate is the point of this driver.
    gamma = _nonneg_float(scfg.get("gamma", 0.0), "simnpo.gamma")
    forget_weight = _pos_float(scfg.get("forget_weight", 1.0), "simnpo.forget_weight")
    retain_weight = _nonneg_float(scfg.get("retain_weight", 1.0), "simnpo.retain_weight")
    loss_form = scfg.get("loss_form", "paper")
    if loss_form not in LOSS_FORMS:
        raise ValueError(f"simnpo.loss_form must be one of {LOSS_FORMS} (paper = the "
                         f"length-normalised NLL objective; muse_upstream = the raw-logit "
                         f"mean of MUSE/baselines/baselines/iterative.py), got {loss_form!r}")

    forget_corpora = scfg.get("forget_corpora")
    if not isinstance(forget_corpora, (list, tuple)) or not forget_corpora:
        raise ValueError("simnpo.forget_corpora must be a non-empty list of corpus names, "
                         f"got {forget_corpora!r}")
    forget_corpora = [_corpus(c, f"simnpo.forget_corpora[{i}]", retain=False)
                      for i, c in enumerate(forget_corpora)]
    if len(set(forget_corpora)) != len(forget_corpora):
        raise ValueError(f"simnpo.forget_corpora has duplicates: {forget_corpora}")
    retain_corpus = _corpus(scfg.get("retain_corpus", "wikitext"),
                            "simnpo.retain_corpus", retain=True)
    min_len = int(scfg.get("min_len", C.DEFAULT_MIN_LEN))
    if min_len < 0:
        raise ValueError(f"simnpo.min_len must be >= 0, got {min_len!r}")
    forget_max_length = scfg.get("forget_max_length")
    if forget_max_length is not None:
        forget_max_length = _pos_int(forget_max_length, "simnpo.forget_max_length")
    retain_max_length = _pos_int(scfg.get("retain_max_length", C.RETAIN_MAX_LENGTH),
                                 "simnpo.retain_max_length")
    # A chunked (MUSE) forget corpus has no per-topic length table: the window
    # length IS the tokenisation length, so it must be said explicitly.
    if forget_max_length is None and any(C.is_chunked_corpus(c) for c in forget_corpora):
        raise ValueError("simnpo.forget_max_length must be set explicitly when a forget "
                         "corpus is chunked (MUSE windows are cut at that length; "
                         "upstream MUSE/SimNPO uses 2048)")

    tcfg = cfg.get("train") or {}
    if not isinstance(tcfg, dict):
        raise ValueError(f"`train:` must be a mapping, got {type(tcfg).__name__}")
    optim = str(tcfg.get("optim", "adamw_torch"))
    if optim != "adamw_torch":
        raise ValueError(f"train.optim must be 'adamw_torch' (the published setting; this "
                         f"driver builds torch.optim.AdamW), got {optim!r}")
    lr = _pos_float(tcfg.get("lr", DEFAULT_LR), "train.lr")
    num_epochs = _pos_int(tcfg.get("num_epochs", DEFAULT_NUM_EPOCHS), "train.num_epochs")
    cfg_max_steps = tcfg.get("max_steps", DEFAULT_MAX_STEPS)
    steps = max_steps if max_steps is not None else cfg_max_steps
    if steps is not None:
        steps = _pos_int(steps, "train.max_steps")
    batch_size = _pos_int(tcfg.get("batch_size", DEFAULT_BATCH_SIZE), "train.batch_size")
    grad_accum = _pos_int(tcfg.get("grad_accum", DEFAULT_GRAD_ACCUM), "train.grad_accum")
    weight_decay = _nonneg_float(tcfg.get("weight_decay", DEFAULT_WEIGHT_DECAY),
                                 "train.weight_decay")
    max_grad_norm = _pos_float(tcfg.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM),
                               "train.max_grad_norm")
    lr_schedule = tcfg.get("lr_schedule", DEFAULT_LR_SCHEDULE)
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"train.lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule!r}")
    warmup_steps = tcfg.get("warmup_steps")
    if lr_schedule == "constant":
        # No warmup under a constant schedule: null or 0 both mean "none"; a
        # positive count contradicts the schedule and must not be silently dropped.
        if warmup_steps is not None:
            if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) \
                    or warmup_steps != 0:
                raise ValueError(f"train.warmup_steps must be null or 0 under "
                                 f"train.lr_schedule 'constant' (no warmup), got "
                                 f"{warmup_steps!r}")
        warmup_steps = 0
    elif warmup_steps is not None:
        warmup_steps = _pos_int(warmup_steps, "train.warmup_steps")

    dtype = str(cfg.get("dtype", "bfloat16"))
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be bfloat16 / float16 / float32, got {dtype!r}")
    bf16 = bool(tcfg.get("bf16", True))
    if bf16 and dtype != "bfloat16":
        raise ValueError(f"train.bf16 is true but dtype is {dtype!r}; this driver trains the "
                         f"weights in the dtype they are loaded in (as upstream does), so the "
                         f"two must agree — set dtype: bfloat16 or train.bf16: false")

    rows = max_rows if max_rows is not None else scfg.get("max_rows")
    if rows is not None:
        rows = _pos_int(rows, "simnpo.max_rows")

    # Equal token budget (see the module docstring): the train block must BE the
    # protocol, not approximate it.
    budget = C.resolve_budget(cfg)
    if budget is not None:
        if batch_size != 1:
            raise ValueError(f"under a `budget:` train.batch_size must be 1 (one pass = the "
                             f"one target sequence per micro-step), got {batch_size}")
        if grad_accum != budget["passes_per_step"]:
            raise ValueError(f"under a `budget:` train.grad_accum must equal the passes per "
                             f"step (target_passes / steps = {budget['passes_per_step']}), "
                             f"got {grad_accum}")
        if steps not in (None, budget["steps"]):
            raise ValueError(f"train.max_steps ({steps}) disagrees with budget.steps "
                             f"({budget['steps']}); drop it or make them equal")
        steps = budget["steps"]
        if forget_max_length != budget["forget_max_length"]:
            raise ValueError(f"simnpo.forget_max_length ({forget_max_length}) must equal "
                             f"budget.forget_max_length ({budget['forget_max_length']})")
        if retain_max_length != budget["retain_block"]:
            raise ValueError(f"simnpo.retain_max_length ({retain_max_length}) must equal "
                             f"budget.retain_block ({budget['retain_block']})")

    push = bool(cfg.get("push_to_hub", False)) and not no_push
    hub_repo = cfg.get("hub_repo") or f"rspd-wmdp-simnpo-{model_short(model_id)}"
    if not isinstance(hub_repo, str) or "/" in hub_repo:
        raise ValueError(f"hub_repo must be a bare repo name (the org is added by "
                         f"common.push_model), got {hub_repo!r}")

    return {
        "name": name, "model": model_id, "tokenizer": tokenizer_id, "dtype": dtype,
        "device_map": cfg.get("device_map", "auto"),
        "seed": int(cfg.get("seed", 0)),
        "beta": beta, "gamma": gamma, "loss_form": loss_form,
        "forget_weight": forget_weight, "retain_weight": retain_weight,
        "forget_corpora": forget_corpora, "retain_corpus": retain_corpus,
        "min_len": min_len, "max_rows": rows,
        "forget_max_length": forget_max_length, "retain_max_length": retain_max_length,
        "lr": lr, "num_epochs": num_epochs, "max_steps": steps,
        "batch_size": batch_size, "grad_accum": grad_accum,
        "weight_decay": weight_decay, "max_grad_norm": max_grad_norm,
        "warmup_steps": warmup_steps, "lr_schedule": lr_schedule, "optim": optim, "bf16": bf16,
        "gradient_checkpointing": bool(tcfg.get("gradient_checkpointing", True)),
        "log_every": _pos_int(tcfg.get("log_every", 1), "train.log_every"),
        "push_to_hub": push, "hub_repo": hub_repo,
        "hub_private": bool(cfg.get("hub_private", True)),
        # Equal token budget: None = the schedule and sampler above, unchanged.
        "budget": budget,
    }


# --------------------------------------------------------------------------- #
# Batch plumbing                                                               #
# --------------------------------------------------------------------------- #
def forget_max_length_for(corpus: str, override=None) -> int:
    """Tokenisation length for one forget corpus: the config override if given,
    else the per-topic length the published repros use (bio 512, cyber 768)."""
    if override is not None:
        return int(override)
    topic = C.CORPORA[corpus]["topic"]
    return int(C.FORGET_MAX_LENGTH.get(topic, DEFAULT_FORGET_MAX_LENGTH))


def build_forget_plan(corpora, batches_per_corpus, max_length_override=None):
    """Flatten ``[[batch, ...], ...]`` (aligned with ``corpora``) into one list of
    ``(corpus, max_length, texts)`` items, taking one batch from each corpus in
    turn.

    The round-robin is only how the list is *built*; it is **not** the order the
    training loop sees. :func:`_cycle` reshuffles the whole flattened plan every
    epoch, which destroys the interleaving, so forget batches are drawn
    **uniformly over the union of the corpora, in proportion to their sizes** —
    a 25432-batch bio corpus against a 1000-batch cyber one makes ~3.9% of the
    draws cyber, not the 50/50 an "interleaved" reading would suggest. That is
    harmless, and arguably preferable (it is an unbiased sample of the forget
    set), but it is what the loop does, so do not describe the schedule itself as
    interleaved.
    """
    plan = []
    longest = max((len(b) for b in batches_per_corpus), default=0)
    for i in range(longest):
        for corpus, batches in zip(corpora, batches_per_corpus):
            if i < len(batches):
                plan.append((corpus, forget_max_length_for(corpus, max_length_override),
                             batches[i]))
    return plan


def _cycle(indices, rng):
    """Yield ``indices`` in a fresh shuffled order forever (a seeded, epoch-wise
    reshuffling sampler — ``Trainer``'s default behaviour, without ``Trainer``)."""
    order = list(indices)
    while True:
        rng.shuffle(order)
        for i in order:
            yield i


def resolve_schedule(n_forget_batches: int, *, max_steps, num_epochs: int,
                     grad_accum: int, warmup_steps=None) -> dict:
    """``{total_steps, warmup_steps, micro_steps}``.

    ``max_steps`` overrides ``num_epochs`` when set (upstream's
    ``TrainingArguments`` semantics); otherwise one epoch is one pass over the
    forget batches and ``total_steps = ceil(num_epochs * n / grad_accum)``.
    Warmup defaults to ``max(1, total_steps // 10)``.
    """
    if n_forget_batches < 1:
        raise ValueError("the forget corpora yielded no batches; check simnpo.min_len / "
                         "max_rows and the corpus names")
    if max_steps is not None:
        total = int(max_steps)
    else:
        total = max(1, math.ceil(num_epochs * n_forget_batches / grad_accum))
    warm = int(warmup_steps) if warmup_steps is not None else max(1, total // WARMUP_DIVISOR)
    warm = min(warm, total)
    return {"total_steps": total, "warmup_steps": warm, "micro_steps": total * grad_accum}


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #
def _git_short_sha():
    return C.git_short_sha()


def _sha256(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:                                   # pragma: no cover - env-only
        return None


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():                       # pragma: no cover - GPU only
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Run                                                                          #
# --------------------------------------------------------------------------- #
def run(config_path, *, max_steps=None, max_rows=None, no_push=False) -> dict:
    """Load the config (a YAML path or a mapping), train, save and write
    ``metrics.json``. Returns the metrics dict."""
    t0 = time.time()
    config_path = C.config_file(config_path, method="simnpo")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    S = resolve_simnpo_config(cfg, max_steps=max_steps, max_rows=max_rows, no_push=no_push)

    name = S["name"]
    LOG_DIR = str(logs_dir(EXPERIMENT, name))
    MODEL_DIR = str(models_dir(EXPERIMENT, name))
    git_sha = _git_short_sha()
    config_sha = _sha256(config_path)
    print(f"[plan] SimNPO ({LOSS_FORM_DESCRIPTION[S['loss_form']]}) · "
          f"loss_form={S['loss_form']} "
          f"model={S['model']} beta={S['beta']:g} gamma={S['gamma']:g} "
          f"forget_weight={S['forget_weight']:g} retain_weight={S['retain_weight']:g} · "
          f"forget={S['forget_corpora']} retain={S['retain_corpus']} · "
          f"lr={S['lr']:g} lr_schedule={S['lr_schedule']} max_steps={S['max_steps']} "
          f"epochs={S['num_epochs']} "
          f"bs={S['batch_size']}x{S['grad_accum']} · logs={LOG_DIR}", flush=True)

    set_seeds(S["seed"])

    # ----- data (before the weights: a corpus failure must not cost a load) ----
    # A chunked (MUSE) corpus is windowed with the tokenizer, so the tokenizer
    # alone is loaded first (cheap); the windows are then exactly
    # forget_max_length / retain_max_length long and tokenize_batch stacks them
    # without re-truncating or re-prefixing BOS.
    chunked = any(C.is_chunked_corpus(c) for c in S["forget_corpora"] + [S["retain_corpus"]])
    tok_for_windows = (C.load_tokenizer(S["tokenizer"] or S["model"]) if chunked else None)
    forget_batches = C.load_text_batches(S["forget_corpora"], batch_size=S["batch_size"],
                                         min_len=S["min_len"], max_rows=S["max_rows"],
                                         tokenizer=tok_for_windows,
                                         chunk_max_len=S["forget_max_length"])
    budget = S["budget"]
    # Under a budget the retained side is the packed stream (built once the
    # tokenizer is loaded); the sampled document batches are the no-budget path.
    retain_batches = [] if budget else C.load_text_batches(
        [S["retain_corpus"]], batch_size=S["batch_size"], min_len=S["min_len"],
        max_rows=S["max_rows"], tokenizer=tok_for_windows,
        chunk_max_len=S["retain_max_length"])[0]
    forget_plan = build_forget_plan(S["forget_corpora"], forget_batches,
                                    S["forget_max_length"])
    if budget and len(forget_plan) != 1:
        raise ValueError(f"under a `budget:` one pass over the target is one forget batch, "
                         f"so the target must fit one batch of train.batch_size "
                         f"{S['batch_size']}; it gave {len(forget_plan)} batches")
    if not budget and not retain_batches and S["retain_weight"] > 0.0:
        raise ValueError(f"retain corpus {S['retain_corpus']!r} yielded no batches but "
                         f"retain_weight = {S['retain_weight']:g} > 0")
    sched = resolve_schedule(len(forget_plan), max_steps=S["max_steps"],
                             num_epochs=S["num_epochs"], grad_accum=S["grad_accum"],
                             warmup_steps=S["warmup_steps"])
    print(f"[data] forget: {len(forget_plan)} batches over {S['forget_corpora']} · retain: "
          + (f"the packed stream of {S['retain_corpus']} (budget)" if budget
             else f"{len(retain_batches)} batches of {S['retain_corpus']}")
          + f" · {sched['total_steps']} optimizer steps ({sched['micro_steps']} micro), "
          f"warmup {sched['warmup_steps']}", flush=True)

    # ----- model: exactly ONE (reference-free) --------------------------------
    model, tokenizer = C.load_causal_lm(S["model"], dtype=S["dtype"],
                                        device_map=S["device_map"],
                                        tokenizer_id=S["tokenizer"])
    device = getattr(model, "device", None) or next(model.parameters()).device
    model.train()
    if S["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        except Exception as exc:                        # pragma: no cover - env-only
            print(f"[model] gradient checkpointing unavailable ({type(exc).__name__}: {exc})",
                  flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {S['model']} · {n_params / 1e9:.2f}B params · dtype={S['dtype']} · "
          f"device={device} · reference-free (1 model in memory)", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=S["lr"], weight_decay=S["weight_decay"])

    # ----- equal token budget: the packed retained stream + the unique counts --
    retain_plan, target_unique, exposure = None, None, C.Exposure()
    pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
              else tokenizer.eos_token_id)
    if budget:
        retain_plan = C.budget_retain_plan(tokenizer, S["retain_corpus"], budget=budget,
                                           min_len=S["min_len"])
        sched["retain_plan"] = retain_plan["schedule"]
        _n_t, target_unique = C.target_tokens_unique(
            tokenizer, S["forget_corpora"], max_length=S["forget_max_length"],
            min_len=S["min_len"])
        rs = retain_plan["schedule"]
        print(f"[budget] {budget['steps']} steps x {budget['passes_per_step']} pass(es) over "
              f"the target ({_n_t} doc(s), {target_unique} tokens) · retained: {rs['n_docs']} "
              f"docs, {rs['n_doc_tokens']} document tokens + {rs['n_sep_tokens']} separators "
              f"-> {rs['n_blocks']} blocks of {rs['retain_block']} "
              f"({rs['blocks_per_step_min']}-{rs['blocks_per_step_max']} per step; files "
              f"{[(f['path'].rsplit('/', 1)[-1], f['n_blocks']) for f in rs['per_file']]})",
              flush=True)

    # ----- held-out CE readout (shared estimator, common.HeldoutReadout) ------
    readout = C.load_heldout_readout(cfg, tokenizer)
    ce_base = readout.score(model) if readout is not None else None
    if ce_base is not None:
        print("[heldout] base " + " ".join(f"{k} {v:.4f}" for k, v in ce_base.items())
              + f" ({readout.rows} rows/corpus, max_length {readout.max_length})", flush=True)

    wb = C.init_wandb(cfg, name=name, out_dir=LOG_DIR, config={
        "method": "simnpo", "loss_form": S["loss_form"],
        "loss_form_description": LOSS_FORM_DESCRIPTION[S["loss_form"]],
        "model": S["model"], "beta": S["beta"], "gamma": S["gamma"],
        "forget_weight": S["forget_weight"], "retain_weight": S["retain_weight"],
        "forget_corpora": S["forget_corpora"], "retain_corpus": S["retain_corpus"],
        "lr": S["lr"], "batch_size": S["batch_size"], "grad_accum": S["grad_accum"],
        "total_steps": sched["total_steps"], "warmup_steps": sched["warmup_steps"],
        "lr_schedule": S["lr_schedule"],
        "weight_decay": S["weight_decay"], "seed": S["seed"],
        "git_sha": git_sha, "config_sha256": config_sha})

    curve_path = os.path.join(LOG_DIR, "loss_curve.csv")
    with open(curve_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "lr", "loss", "forget_loss", "retain_loss",
                                "grad_norm", "seconds"])

    rng = random.Random(S["seed"])
    forget_iter = _cycle(range(len(forget_plan)), rng)
    retain_iter = _cycle(range(len(retain_batches)), rng) if retain_batches else None
    history, failure = [], None
    try:
        for step in range(sched["total_steps"]):
            lr_now = lr_at_step(step, base_lr=S["lr"], warmup_steps=sched["warmup_steps"],
                                total_steps=sched["total_steps"], schedule=S["lr_schedule"])
            for g in opt.param_groups:
                g["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            acc = {"loss": 0.0, "forget": 0.0, "retain": 0.0}
            if retain_plan is not None:
                # the step's block slice, split over its micro-steps (one, in
                # the GPT-2 configs: grad_accum == passes_per_step == 1)
                a0, b0 = retain_plan["ranges"][step]
                sub = [(a0 + x, a0 + y) for x, y in C.spread_ranges(b0 - a0, S["grad_accum"])]
            for micro in range(S["grad_accum"]):
                corpus, flen, texts = forget_plan[next(forget_iter)]
                fb = C.tokenize_batch(tokenizer, texts, max_length=flen, device=device)
                rb, n_sep = None, 0
                if retain_plan is not None:
                    a, b = sub[micro]
                    if b > a:
                        rb = C.windows_to_batch(
                            retain_plan["blocks"][a:b], pad_token_id=pad_id,
                            padding_side=getattr(tokenizer, "padding_side", "right"),
                            device=device, with_labels=True)
                        n_sep = sum(retain_plan["block_n_sep"][a:b])
                elif retain_iter is not None:
                    # independently sampled: the retain stream is NOT aligned with
                    # the forget stream (upstream zips two separate loaders)
                    rb = C.tokenize_batch(tokenizer, retain_batches[next(retain_iter)],
                                          max_length=S["retain_max_length"], device=device)
                exposure.add_target(fb["labels"] != -100)
                if rb is not None and S["retain_weight"] > 0.0:
                    exposure.add_retain(rb["labels"] != -100, n_sep)
                total, forget, retain = simnpo_step_losses(
                    model, fb, rb, beta=S["beta"], gamma=S["gamma"],
                    forget_weight=S["forget_weight"], retain_weight=S["retain_weight"],
                    loss_form=S["loss_form"])
                (total / S["grad_accum"]).backward()
                acc["loss"] += float(total) / S["grad_accum"]
                acc["forget"] += float(forget) / S["grad_accum"]
                acc["retain"] += float(retain) / S["grad_accum"]
            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                         S["max_grad_norm"]))
            opt.step()
            exposure.step()

            row = {"step": step + 1, "lr": lr_now, "loss": acc["loss"],
                   "forget_loss": acc["forget"], "retain_loss": acc["retain"],
                   "grad_norm": gnorm, "seconds": round(time.time() - t0, 1)}
            history.append(row)
            with open(curve_path, "a", newline="") as f:
                csv.writer(f).writerow([row["step"], f"{lr_now:.6e}", f"{acc['loss']:.6f}",
                                        f"{acc['forget']:.6f}", f"{acc['retain']:.6f}",
                                        f"{gnorm:.4f}", row["seconds"]])
            wb.log({"train/loss": acc["loss"], "train/forget_loss": acc["forget"],
                    "train/retain_loss": acc["retain"], "train/lr": lr_now,
                    "train/grad_norm": gnorm}, step=step + 1)
            if (step + 1) % S["log_every"] == 0 or step + 1 == sched["total_steps"]:
                print(f"[step] {step + 1}/{sched['total_steps']} lr={lr_now:.3e} "
                      f"loss={acc['loss']:.4f} forget={acc['forget']:.4f} "
                      f"retain={acc['retain']:.4f} |g|={gnorm:.3f} "
                      f"({(time.time() - t0) / 60:.1f}m)", flush=True)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        wb.finish()
        raise
    print(f"[train] {len(history)} steps in {(time.time() - t0) / 60:.1f}m", flush=True)
    budget_record = None
    if budget:
        budget_record = exposure.record(
            budget=budget, target_unique=target_unique,
            retain_unique=retain_plan["stream"]["n_doc_tokens"],
            retain_sep=retain_plan["stream"]["n_sep_tokens"])
        print(C.Exposure.line(budget_record), flush=True)

    # ----- save, push, record -------------------------------------------------
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    ce_edit = readout.score(model) if readout is not None else None
    if ce_edit is not None:
        C.print_heldout(ce_base, ce_edit)
    C.save_model(model, tokenizer, MODEL_DIR)
    push = C.push_model(model, tokenizer, S["hub_repo"], private=S["hub_private"],
                        enabled=S["push_to_hub"])

    first, last = (history[0] if history else {}), (history[-1] if history else {})
    tail = history[-10:]
    metrics = {
        "method": "simnpo",
        "loss_form": S["loss_form"],
        "loss_form_description": LOSS_FORM_DESCRIPTION[S["loss_form"]],
        "upstream": "OPTML-Group/Unlearn-Simple (MIT), arXiv:2410.07163",
        "reference_free": True, "models_in_memory": 1,
        "config": config_path, "config_sha256": config_sha, "git_sha": git_sha,
        "name": name, "model": S["model"], "tokenizer": S["tokenizer"] or S["model"],
        "n_params": n_params,
        "hyperparameters": {k: S[k] for k in (
            "beta", "gamma", "loss_form", "forget_weight", "retain_weight", "lr", "num_epochs",
            "max_steps", "batch_size", "grad_accum", "weight_decay", "max_grad_norm",
            "dtype", "seed", "optim", "lr_schedule", "gradient_checkpointing")},
        "lr_schedule": S["lr_schedule"],
        "schedule": {**sched, "lr_schedule": S["lr_schedule"]},
        "budget": budget_record,
        "data": {"forget_corpora": S["forget_corpora"], "retain_corpus": S["retain_corpus"],
                 "forget_batches": len(forget_plan), "retain_batches": len(retain_batches),
                 "forget_max_length": {c: forget_max_length_for(c, S["forget_max_length"])
                                       for c in S["forget_corpora"]},
                 "retain_max_length": S["retain_max_length"], "min_len": S["min_len"],
                 "max_rows": S["max_rows"],
                 "chunked": {c: C.is_chunked_corpus(c)
                             for c in S["forget_corpora"] + [S["retain_corpus"]]},
                 "labels": "full-sequence LM (labels = input_ids, PAD -> -100)"},
        "losses": {"first": first, "final": last,
                   "final10_mean": {k: (sum(r[k] for r in tail) / len(tail) if tail else None)
                                    for k in ("loss", "forget_loss", "retain_loss")}},
        "artifacts": {"log_dir": LOG_DIR, "model_dir": MODEL_DIR,
                      "loss_curve": curve_path},
        "push": push, "wandb": wb.resolved,
        "wall_seconds": round(time.time() - t0, 1),
        "failure": failure,
        **C.heldout_metrics(readout, ce_base, ce_edit),
    }
    C.write_metrics(LOG_DIR, metrics)
    with open(os.path.join(LOG_DIR, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    wb.summary({"final/loss": last.get("loss"), "final/forget_loss": last.get("forget_loss"),
                "final/retain_loss": last.get("retain_loss"),
                "wall_minutes": round((time.time() - t0) / 60, 2),
                "hub_repo": push.get("repo_id") if push.get("pushed") else None,
                **(C.Exposure.summary(budget_record) if budget_record else {}),
                **C.heldout_summary(ce_base, ce_edit)})
    wb.finish()
    if budget:
        # after write_metrics: the record is on disk whichever way this goes
        C.assert_budget(budget_record, budget)
    print(f"[done] {name}: loss {first.get('loss', float('nan')):.4f} -> "
          f"{last.get('loss', float('nan')):.4f} (forget "
          f"{first.get('forget_loss', float('nan')):.4f} -> "
          f"{last.get('forget_loss', float('nan')):.4f}) in "
          f"{(time.time() - t0) / 60:.1f}m · model -> {MODEL_DIR}"
          + (f" · pushed {push['repo_id']}@{push.get('commit')}" if push.get("pushed") else ""),
          flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="SimNPO unlearning on the WMDP forget corpora")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="override train.max_steps (smoke runs)")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="cap each corpus at N rows (smoke runs)")
    ap.add_argument("--no-push", action="store_true",
                    help="skip the hub push even if push_to_hub is true")
    a = ap.parse_args()
    run(a.config, max_steps=a.max_steps, max_rows=a.max_rows, no_push=a.no_push)


if __name__ == "__main__":
    main()
