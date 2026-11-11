"""KUDA -- Knowledge Unlearning by Deviating representAtion -- as a method runner.

    from srspd_experiments.methods import kuda
    kuda.run("my_kuda.yaml")          # or kuda.run({...the same mapping...})

**Credit.** Port of the ``WMDP/`` track of the authors' code,
https://github.com/AceNagi/KUDA (MIT licence; the MIT notice of the upstream
repository applies to the ported parts). Paper: Ce Fang, Zhikun Zhang, Min Chen,
Qing Liu, Lu Zhou, Zhe Liu, Yunjun Gao, *KUDA: Knowledge Unlearning by Deviating
Representation for Large Language Models*, arXiv:2602.19275 (the upstream README
gives NDSS 2027 as the venue; not independently verified). Per its abstract the
method locates the layers storing the target knowledge by causal tracing, then
trains with an objective that makes their representations deviate from the
original, under a relaxed null-space projection of the gradients. The
authors'-code runner is :mod:`srspd_experiments.methods.upstream_kuda`.

KUDA is a fork of CAIS's RMU driver with two substitutions, and this driver
keeps exactly those two and drops the rest of the vestigial RMU scaffolding
(control vectors, ``steering_coeffs``):

1. **Representation deviation** replaces RMU's MSE-to-a-random-vector forget
   loss. On a forget batch, the *updated* model's layer-``layer_id`` output is
   pushed **away** from the *frozen* model's output at the same layer — cosine
   similarity driven towards ``-1`` through a log-sigmoid margin
   (``unlearn.py:137-143``, ported verbatim in :func:`deviation_loss`)::

       cos          = F.cosine_similarity(updated_act, frozen_act, dim=-1).mean(-1)
       unlearn_loss = -F.logsigmoid(beta * (1.0 - cos)).mean() * 2 / beta
       retain_loss  = F.mse_loss(updated_retain_act, frozen_retain_act) * alpha[topic]
       loss         = unlearn_loss + retain_loss

   The retain term is RMU's, unchanged: MSE back to the frozen model's own
   activations on a retain batch.

2. **Relaxed null-space projection** replaces the plain SGD/AdamW step. For each
   target layer we precompute, over the **retain** corpus, the uncentered second
   moment ``cov = E[x xᵀ]`` of the *input features* of that layer's
   ``mlp.down_proj``, and take the subspace its SVD calls low-energy
   (``nsproject/nullspace.py::generate_project``)::

       U, S, _ = torch.linalg.svd(cov, full_matrices=False)
       small   = (S < tau).nonzero(as_tuple=True)[0]      # argmin(S) if empty
       P       = (U[:, small] @ U[:, small].T)            # stored bf16

   ``P`` projects onto the directions the retain distribution barely occupies.
   After ``loss.backward()`` and **before** ``optimizer.step()`` every target
   weight's gradient is right-multiplied by its own ``P`` (:func:`project_gradients`),
   so the update can only move the layer in retain-null input directions —
   "relaxed" because the threshold ``tau`` admits small-but-nonzero energy
   rather than the exact kernel. ``P`` is ``d_intermediate²`` (14336² for
   a 7B model — 410 MB bf16 per layer), so it is **cached to disk** under
   ``<data root>/models/methods/kuda_nullspace_cache/``, in **two tiers** mirroring
   upstream's own split (``layer_stats.py:127`` caches the second moment without
   ``tau``; ``nsproject/nullspace.py:98`` caches the projector with it):

   * a **spectrum** entry keyed by :func:`_cov_key` — everything that changes
     ``cov`` or its SVD, but **not** ``tau`` — holding ``U`` and ``S``;
   * a **projector** entry keyed by :func:`_proj_key` (= the cov key plus
     ``tau``) holding ``P``.

   A ``tau`` sweep therefore repays neither the ~8-minute covariance pass nor the
   ``d×d`` SVD after the first build: it re-cuts the cached spectrum. The
   covariance matrix itself is **not** stored — ``U``/``S`` re-cut any ``tau``,
   and everything that would make the SVD come out differently (``cov_dtype``,
   the resolved device pair) is *in* the key, so those cases must rebuild
   anyway. Keeping it would cost 411 MB per layer per entry on a shared volume
   to avoid an SVD that only a key change can invalidate.

   ``kuda.nullspace.sample_p`` is upstream's ``--sample_p``: a **percentage** of
   the retain corpus in ``(0, 100]``, not a row count — upstream's 7B-model scripts
   pass ``100``, i.e. *all* of it, and their hparams record a second moment over
   tens of thousands of rows (``mom2_n_samples: 36718``, the wikitext-2 train
   split; this driver, like the RMU/KUDA training loaders it is forked from,
   reads the **test** split — ``srspd_experiments.data.corpora.CORPORA``).
   ``tau`` is an **absolute** threshold on the
   spectrum of that covariance, so starving the covariance pass does not merely
   add noise: it moves the whole spectrum and silently selects a different,
   far more permissive null subspace. ``kuda.nullspace.max_cov_texts`` is an
   **srspd-experiments addition** (not upstream): an optional hard cap on the number of texts
   the covariance pass consumes, applied *after* the percentage, for cost
   control on very large retain corpora.

**The causal-tracing / layer-selection stage is deliberately skipped.**
Upstream's ``layer_selection/`` runs offline, delegates causal tracing to the
external ROME repository (not vendored, not a dependency of the training run),
and its output never enters ``unlearn.py`` — the layers arrive as the plain
``--layer_id`` / ``--layer_ids`` command-line arguments that upstream's own
7B-model scripts hard-code to ``9`` and ``6,7,8,9``. Hand-picking the layers in
the config is therefore *faithful to upstream practice*, not a shortcut: it
reproduces what the published scripts actually run. Re-running causal tracing
would be a separate offline experiment with its own driver and its own
`DECISION` entry.

**The forget objective starts at an exact stationary point, and this driver can
escape it.** ``cos(u, f)`` is maximised at ``u == f``, so its gradient there is
*identically* zero (``∇_u cos|_{u=f} = f/‖f‖² − f/‖f‖² = 0``); the retain MSE is
simultaneously at its own minimum. The updated model starts bit-identical to the
frozen one, so at step 0 the whole objective sits at a stationary point with
``loss = ln 2`` exactly (at ``beta = 2``) and nothing but bf16 rounding noise
moves the weights — which AdamW, being scale-invariant, then amplifies into
full ``lr``-sized steps in a random direction. That is upstream's behaviour, not
a port defect (upstream applies no jitter of any kind; verified against
``unlearning/unlearn.py``). ``kuda.init_jitter`` (:func:`apply_init_jitter`) is
an **srspd-experiments addition** that breaks the symmetry deterministically before the
optimizer is built; its default ``0.0`` is an exact no-op and reproduces
published KUDA. The driver warns at step 0 when the raw gradient norm is below
``STATIONARY_GRAD_NORM`` and logs ``grad_norm`` / ``grad_norm_projected`` /
``rel_weight_delta`` every step so "did anything move, and was it signal?" is
answerable from ``metrics.json`` alone.

**Optimizer and learning-rate schedule.** The optimizer is ``torch.optim.AdamW``
at ``kuda.weight_decay`` (default 0.01 — torch's own default, which this driver
always ran at implicitly, and also upstream's ``PagedAdam8bit`` value). The lr
follows ``kuda.lr_schedule``: ``constant`` (default; the driver's historical
behaviour) or ``linear_decay`` (:func:`lr_at_step`), which reproduces the HF
``Trainer`` default upstream's MUSE track trains under — a linear decay from
``lr`` to 0 over ``kuda.schedule_epochs`` epochs with no warmup.
``schedule_epochs`` defaults to ``epoch`` and may exceed it, so ``epoch: 2,
schedule_epochs: 3`` is "the 2nd checkpoint of a 3-epoch run" (the lr ends at
``lr/3``). The per-step ``lr`` is in every ``[step ...]`` line and in
``history``; ``lr_schedule`` / ``schedule_epochs`` / ``weight_decay`` /
``lr_last`` are top-level ``metrics.json`` fields.

**Fidelity choices worth naming** (each is a deliberate match to what upstream
*runs*, not to what its comments say):

* the covariance accumulates in **bfloat16** (``kuda.nullspace.cov_dtype``),
  because upstream's ``mom2_dtype`` is ``bfloat16`` and its published
  ``tau = 2e-3`` was tuned against that spectrum — see
  :func:`accumulate_down_proj_cov`;
* ``padding_side="left"`` / ``truncation_side="right"``
  (:data:`TOKENIZER_PADDING_SIDE` / :data:`TOKENIZER_TRUNCATION_SIDE`), as
  upstream forces (``unlearning/utils.py:102`` and ``unlearn.py:72``). Both
  losses run over *every* position including pads, so this is result-affecting;
* the forget ``max_length`` is **768 for both topics**
  (:data:`FORGET_MAX_LENGTH`). Upstream hard-codes ``max_length = 768`` at
  ``unlearn.py:111`` with the per-topic ``512 if topic_idx == 0 else 768`` line
  commented out immediately above it. This driver used to follow the
  commented-out line. the RMU runner keeps the per-topic 512/768 split, because
  CAIS's RMU genuinely does use it — the two drivers differ here on purpose.

Hardware: two full models resident, plus a ``d×d`` covariance per target layer
during the (first) projector build -- an 80 GB-class GPU for a 7B model.
Subsequent runs reuse the cached projectors and only need the two models.

Outputs (:mod:`srspd_experiments.paths`): ``<data root>/logs/methods/<name>/``
(``metrics.json``, the config copy) and ``<data root>/models/methods/<name>/``
(the unlearned checkpoint), plus the shared projector cache above. Nothing is
pushed to a model hub.

Held-out CE readout. A top-level ``heldout: {rows, max_length, tail}`` block
scores every ``topic: heldout`` corpus before and after the edit with the
estimator every runner shares (:class:`srspd_experiments.eval.heldout.HeldoutReadout`:
teacher-forced mean next-token CE over the first ``rows`` rows per corpus -- the
last ``rows`` for the ``tail`` corpora, the SRSPD runner's retain-side
convention -- at ``max_length`` tokens); ``metrics.json`` gains
``ce_heldout_base`` / ``ce_heldout`` / ``ce_heldout_delta``.
Absent block: no readout.

Equal token budget (the GPT-2 Dutch-erasure study). A top-level ``budget:
{target_passes, retain_exposure_tokens, steps, forget_max_length, retain_block}``
block (:func:`srspd_experiments.data.budget.resolve_budget`) puts this driver on the protocol the three
gradient-based baselines share: one forget topic, ``kuda.epoch x
kuda.max_num_batches == steps`` updates (the configs: ``max_num_batches`` 1,
``epoch`` 200 on the one-document target), the forget window
``budget.forget_max_length`` instead of :data:`FORGET_MAX_LENGTH` and
``retain_max_length = budget.retain_block``; the retained side of the LOSS is the
packed stream (:data:`srspd_experiments.data.budget.RETAIN_STREAM_RULE`: every document of every file
exactly once, one eos separator per document, ``retain_block``-token blocks,
files round-robin interleaved, spread evenly over the steps) instead of the
first retained document re-read at every step. The covariance / null-space pass
over the retained set is unchanged (and keyed as before, so the spectrum cache
is shared with the un-budgeted configs); its token count is reported separately
as ``retain_tokens_covariance`` (``n_tokens`` is persisted in the cache, so a
cache hit still reports it). The realised exposure is counted from the attention
masks into ``metrics.json`` ``budget`` and :func:`~srspd_experiments.data.budget.assert_budget` fails the
run — after ``metrics.json`` is written — unless it is exactly the budget
(``--skip-train`` records and does not assert). Absent block: the loop is
exactly as before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time

import torch
import torch.nn.functional as F
import yaml

from . import _common as C
from ..paths import logs_dir, models_dir

EXPERIMENT = "methods"                 # output folder key (srspd_experiments.paths)

#: Where the (expensive) null-space projectors are cached, shared across configs
#: — the key already pins everything that changes one.
NULLSPACE_CACHE = "kuda_nullspace_cache"
#: The default target module (upstream's ``input_feature_module =
#: model.layers.{}.mlp.down_proj``). ``kuda.module_name`` may name any of
#: ``common.MLP_OUT_MODULE_NAMES``: ``mlp.c_proj`` is GPT-2's, a ``Conv1D`` whose
#: weight is stored ``[d_in, d_out]`` -- the covariance hook sees the same
#: INPUT features either way, and :func:`project_gradients` applies ``P`` on the
#: ``d_in`` side of whichever layout the weight has.
MODULE_NAME = C.DEFAULT_MLP_OUT_MODULE
MODULE_NAMES = C.MLP_OUT_MODULE_NAMES
#: Forget-batch ``max_length`` — **768 for every topic**, which is what upstream
#: actually runs::
#:
#:     # 512 for bio, 768 for cyber
#:     # max_length = 512 if topic_idx == 0 else 768
#:     max_length = 768                       # unlearning/unlearn.py:109-111
#:
#: The per-topic line is commented out upstream. This driver followed the
#: commented-out line until now and no longer does; ``common.FORGET_MAX_LENGTH``
#: (bio 512 / cyber 768) still describes **RMU**, whose upstream does use the
#: split, and is deliberately not consulted here.
FORGET_MAX_LENGTH = 768
#: Tokenizer sides upstream forces for the unlearning loop:
#: ``tokenizer.padding_side = "left"`` (``unlearning/utils.py:102``) and
#: ``tokenizer.truncation_side = "right"`` (``unlearning/unlearn.py:72``).
#: Both :func:`deviation_loss` and :func:`retain_loss` average over **every**
#: position, pads included, so which end the pads sit on and which end a long
#: document is cut from are part of the objective — not cosmetics. The two
#: models ship different defaults (one 4B multimodal model pads left; the 7B instruct model
#: leaves ``padding_side`` unset → right, and truncates *left*, i.e. it kept the
#: document's tail where upstream keeps its head), so leaving them unset was an
#: uncontrolled confound between the two runs, the same one the RMU runner closed.
TOKENIZER_PADDING_SIDE = "left"
TOKENIZER_TRUNCATION_SIDE = "right"
#: ``kuda.init_jitter`` default. ``0.0`` is an exact no-op — no RNG is drawn and
#: no parameter is touched — so the published KUDA behaviour stays reachable.
#: See :func:`apply_init_jitter` and the module docstring for why anything else
#: is ever wanted.
DEFAULT_INIT_JITTER = 0.0
#: Learning-rate schedules. ``constant`` (the default) holds ``kuda.lr`` for the
#: whole run — what this driver has always done. ``linear_decay`` reproduces the
#: HF ``Trainer`` default that upstream's MUSE track runs under
#: (``TrainingArguments(lr_scheduler_type="linear", warmup_steps=0)`` →
#: ``get_linear_schedule_with_warmup(0, T)``): ``lr_t = lr * (T - t) / T`` for
#: optimizer step ``t`` in ``[0, T)``, where ``T = schedule_epochs *
#: steps_per_epoch``. ``kuda.schedule_epochs`` (default ``kuda.epoch``) is the
#: horizon the decay is laid over, so a 2-epoch run with ``schedule_epochs: 3``
#: stops at the lr upstream's 2nd-of-3 checkpoint saw (``lr/3``).
LR_SCHEDULES = ("constant", "linear_decay")
DEFAULT_LR_SCHEDULE = "constant"
#: ``torch.optim.AdamW``'s own default, which is what this driver has always
#: passed implicitly (it never set ``weight_decay``). Upstream's bitsandbytes
#: ``PagedAdam8bit`` also runs at 0.01 (``kuda.py``'s
#: ``kwargs.pop("module_optim", 0.01)``), so the two already agree; the key
#: exists so the value is explicit in ``metrics.json`` and overridable.
DEFAULT_WEIGHT_DECAY = 0.01
#: Below this raw gradient norm at step 0 the run is sitting on the forget
#: objective's stationary point and is warned about, loudly, once.
STATIONARY_GRAD_NORM = 1e-6
#: Defaults for the ``kuda.nullspace`` block. ``sample_p`` is upstream's
#: ``--sample_p``: a **percentage** of the retain corpus in ``(0, 100]``, never a
#: row count — the upstream 7B-model scripts pass ``100``, i.e. *all* of wikitext.
DEFAULT_SAMPLE_P = 100.0
#: srspd-experiments addition (not upstream): an optional hard cap on the number of retain
#: texts the covariance pass consumes, applied AFTER ``sample_p``. ``None`` =
#: uncapped, which is what reproduces upstream.
DEFAULT_MAX_COV_TEXTS = None
DEFAULT_NS_MAX_LENGTH = 512
DEFAULT_NS_BATCH_SIZE = 4
#: ``kuda.nullspace.cov_dtype`` — the dtype the second moment ACCUMULATES in.
#: Default ``bfloat16``, matching upstream's ``mom2_dtype`` (see
#: :func:`accumulate_down_proj_cov` for why the numerically worse choice is the
#: right one here). ``float64`` remains reachable for comparison, and is keyed,
#: so the two never share a cache entry.
DEFAULT_COV_DTYPE = "bfloat16"
COV_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32,
              "float64": torch.float64}
#: Cache-key version. Bumped 2 → 3 when the covariance moved to bfloat16 and the
#: key gained ``batch_size`` / the resolved devices / the tokenizer sides and
#: lost ``tau`` to the separate projector key: every v2 entry on disk was built
#: from a float64 accumulation and must never be served to a v3 run.
CACHE_VERSION = 3


# --------------------------------------------------------------------------- #
# Config — validated before a model loads (a typo costs seconds, not a GPU)    #
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
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError(f"{where} must be a positive int, got {v!r}")
    return int(v)


def _nonneg_int(v, where):
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"{where} must be an int >= 0, got {v!r}")
    return int(v)


def _opt_pos_int(v, where):
    """A positive int, or ``None`` for "no limit"."""
    return None if v is None else _pos_int(v, where)


def _percent(v, where):
    """A **percentage** in ``(0, 100]`` — upstream KUDA's ``--sample_p`` units.

    Deliberately strict at both ends: ``0`` would sample nothing and ``>100``
    would silently mean "all of it", and both readings used to be plausible back
    when this key was (wrongly) an absolute row count.
    """
    bad = (f"{where} must be a number in (0, 100] — it is a PERCENTAGE of the "
           f"retain corpus (upstream's --sample_p), not a row count; use "
           f"kuda.nullspace.max_cov_texts to cap the row count. Got {v!r}")
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(bad)
    v = float(v)
    if not 0.0 < v <= 100.0:
        raise ValueError(bad)
    return v


def sample_p_divergence_warning(sample_p: float) -> str | None:
    """The warning text for a sub-100% ``sample_p``, or ``None`` at 100%.

    Below 100% this driver's :func:`select_cov_texts` diverges from upstream in
    **two** ways at once, and neither is visible in the result — only in which
    retain rows the covariance saw:

    * *the sampling rule.* Upstream takes a **seeded random sample**
      (``unlearning/utils.py:265-269``: ``random.seed(seed)`` then
      ``random.sample(data, n_sample)``); this takes a deterministic **prefix**;
    * *the rounding.* Upstream sizes it ``max(1, int(n * sample_p / 100))``
      (floor); this uses ``ceil``, so it can take one row more.

    At ``sample_p = 100`` upstream skips that branch entirely and the two agree
    **exactly**, which is what the upstream 7B-model scripts and both shipped
    configs use — so today the deviation is inert. It stops being inert the
    moment someone edits one config value or passes ``--sample-p``, and because
    ``tau`` is an absolute threshold on the spectrum of ``E[x xᵀ]``, a different
    *sample* is a different subspace, not merely a noisier one.

    **A warning, not a hard error**, deliberately. The knob has a legitimate
    non-replication use the driver advertises — ``--sample-p`` is documented as
    a smoke-run flag, and a cheap covariance is how one sanity-checks the
    pipeline before booking an H100 — so refusing outright would break a
    supported workflow to prevent a mistake that only matters for a *published*
    number. What is not acceptable is running one silently, so this fires on
    stdout before either model loads, and ``metrics.json`` carries
    ``sample_p_is_upstream_faithful`` for the run record. Returning the text (rather
    than printing here) keeps it testable without capturing stdout.
    """
    if float(sample_p) >= 100.0:
        return None
    return (f"[kuda] WARNING: kuda.nullspace.sample_p is {float(sample_p):g}%, not 100%. "
            f"Below 100% this driver's retain selection DIVERGES FROM UPSTREAM in two "
            f"ways: upstream takes a seeded RANDOM sample (utils.py:265-269, "
            f"random.seed(seed) + random.sample) where this takes a deterministic "
            f"PREFIX, and upstream sizes it with int() (floor) where this uses ceil(). "
            f"tau is an absolute threshold on the spectrum of E[x x^T], so a different "
            f"sample selects a different null subspace, not merely a noisier one. "
            f"A sub-100% run is therefore NOT a faithful replication of published KUDA "
            f"until that is reconciled — treat it as a smoke run, not as a result. "
            f"Use sample_p: 100 (the default, and what upstream's 7B-model scripts pass) "
            f"and cap the cost with kuda.nullspace.max_cov_texts instead.")


def _corpora(value, where):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{where} must be a non-empty list of corpus names, got {value!r}")
    out = []
    for i, name in enumerate(value):
        if not isinstance(name, str) or name not in C.CORPORA:
            raise ValueError(f"{where}[{i}] = {name!r} is not a known corpus; "
                             f"known: {sorted(C.CORPORA)}")
        out.append(name)
    return out


def resolve_kuda_config(cfg, *, max_num_batches=None, sample_p=None,
                        push_to_hub=None) -> dict:
    """Validate the ``kuda:`` block (plus ``name`` / ``model`` / ``dtype``) and
    resolve the CLI overrides. Returns a plain dict; every failure raises a
    ``ValueError`` naming the offending key.

    This runs **before** either model is loaded: a bad config must cost seconds,
    not a GPU-hour of checkpoint download.

    ``sample_p`` (config key or CLI override) is a **percentage** of the retain
    corpus in ``(0, 100]`` — upstream's units — and is returned as a float.
    """
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    name = cfg.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("config has no `name:` key (it is the output directory)")
    model_id = cfg.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("config has no `model:` key (hub id or checkpoint dir)")
    dtype = cfg.get("dtype", "bfloat16")
    if dtype not in ("bfloat16", "float16", "float32"):
        raise ValueError(f"dtype must be one of bfloat16/float16/float32, got {dtype!r}")
    C.register_config_corpora(cfg)          # a `corpora:` block joins C.CORPORA
    # Optional: load the tokenizer from another repo (the MUSE target models ship
    # none; theirs is meta-llama/Llama-2-7b-hf). None = the model's own.
    tokenizer_id = cfg.get("tokenizer")
    if tokenizer_id is not None and (not isinstance(tokenizer_id, str)
                                     or not tokenizer_id.strip()):
        raise ValueError(f"`tokenizer:` must be a hub id / path or absent, got {tokenizer_id!r}")

    k = cfg.get("kuda")
    if not isinstance(k, dict) or not k:
        raise ValueError("config has no `kuda:` block")

    module_name = k.get("module_name", MODULE_NAME)
    if module_name not in MODULE_NAMES:
        raise ValueError(f"kuda.module_name must be one of {MODULE_NAMES} (the null-space "
                         f"projector is built from that module's input features: "
                         f"mlp.down_proj on an nn.Linear decoder, mlp.c_proj on GPT-2), "
                         f"got {module_name!r}")
    pos_ids = k.get("position_ids_from_mask", False)
    if not isinstance(pos_ids, bool):
        raise ValueError(f"kuda.position_ids_from_mask must be true/false, got {pos_ids!r}")

    forget = _corpora(k.get("forget_corpora"), "kuda.forget_corpora")
    retain = _corpora(k.get("retain_corpora"), "kuda.retain_corpora")
    if len(forget) != len(retain):
        raise ValueError(f"kuda.forget_corpora ({len(forget)}) and kuda.retain_corpora "
                         f"({len(retain)}) must have the same length: each step pairs "
                         f"topic i's forget batch with topic i's retain batch")

    alpha = k.get("alpha")
    if not isinstance(alpha, (list, tuple)) or len(alpha) != len(forget):
        raise ValueError(f"kuda.alpha must be a list with one retain weight per forget "
                         f"corpus ({len(forget)}), got {alpha!r}")
    alpha_r = []
    for i, a in enumerate(alpha):
        if isinstance(a, bool) or not isinstance(a, (int, float)) or float(a) < 0.0:
            raise ValueError(f"kuda.alpha[{i}] must be a number >= 0, got {a!r}")
        alpha_r.append(float(a))

    layer_ids = k.get("layer_ids")
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError(f"kuda.layer_ids must be a non-empty list of layer indices, "
                         f"got {layer_ids!r}")
    layer_ids = [_nonneg_int(v, f"kuda.layer_ids[{i}]") for i, v in enumerate(layer_ids)]
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError(f"kuda.layer_ids has duplicates: {layer_ids}")
    layer_id = _nonneg_int(k.get("layer_id"), "kuda.layer_id")
    if layer_id not in layer_ids:
        raise ValueError(f"kuda.layer_id ({layer_id}) must be one of kuda.layer_ids "
                         f"({layer_ids}): the hooked layer is the deepest layer whose "
                         f"{module_name} is being updated (upstream: 9 of 6,7,8,9)")

    ns = k.get("nullspace") or {}
    if not isinstance(ns, dict):
        raise ValueError(f"kuda.nullspace must be a mapping, got {ns!r}")
    svd_device = ns.get("svd_device", "auto")
    if svd_device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"kuda.nullspace.svd_device must be auto/cpu/cuda, got {svd_device!r}")
    cov_device = ns.get("cov_device", "auto")
    if cov_device not in ("auto", "cpu", "cuda"):
        raise ValueError(f"kuda.nullspace.cov_device must be auto/cpu/cuda, got {cov_device!r}")
    cov_dtype = ns.get("cov_dtype", DEFAULT_COV_DTYPE)
    if cov_dtype not in COV_DTYPES:
        raise ValueError(f"kuda.nullspace.cov_dtype must be one of "
                         f"{sorted(COV_DTYPES)}, got {cov_dtype!r}")

    epoch = _pos_int(k.get("epoch", 1), "kuda.epoch")
    max_nb = _pos_int(k.get("max_num_batches") if max_num_batches is None else max_num_batches,
                      "kuda.max_num_batches")
    retain_max_length = _pos_int(k.get("retain_max_length", C.RETAIN_MAX_LENGTH),
                                 "kuda.retain_max_length")
    # Forget tokenisation length: upstream WMDP's hard-coded 768 unless the
    # config says otherwise. The MUSE track needs the override because its
    # windows are cut at max_len 1024 (unlearn_config.yaml) and a chunked
    # corpus has no other source for the length.
    forget_max_length = k.get("forget_max_length")
    if forget_max_length is None:
        if any(C.is_chunked_corpus(c) for c in forget):
            raise ValueError("kuda.forget_max_length must be set explicitly when a forget "
                             "corpus is chunked (MUSE windows are cut at that length; "
                             "upstream MUSE/KUDA uses 1024)")
        forget_max_length = FORGET_MAX_LENGTH
    forget_max_length = _pos_int(forget_max_length, "kuda.forget_max_length")
    # Equal token budget (module docstring): the schedule must BE the protocol.
    budget = C.resolve_budget(cfg)
    if budget is not None:
        if len(forget) != 1:
            raise ValueError(f"a `budget:` is defined for one forget topic, got {len(forget)} "
                             f"({forget})")
        if epoch * max_nb != budget["steps"]:
            raise ValueError(f"under a `budget:` kuda.epoch x kuda.max_num_batches must equal "
                             f"budget.steps: {epoch} x {max_nb} != {budget['steps']}")
        if "retain_max_length" in k and retain_max_length != budget["retain_block"]:
            raise ValueError(f"kuda.retain_max_length ({retain_max_length}) must equal "
                             f"budget.retain_block ({budget['retain_block']}) or be left out")
        if "forget_max_length" in k and forget_max_length != budget["forget_max_length"]:
            raise ValueError(f"kuda.forget_max_length ({forget_max_length}) must equal "
                             f"budget.forget_max_length ({budget['forget_max_length']}) "
                             f"or be left out")
        retain_max_length = budget["retain_block"]
        forget_max_length = budget["forget_max_length"]

    # Learning-rate schedule (see LR_SCHEDULES). schedule_epochs defaults to the
    # run length; a longer horizon means the run stops part-way down the decay
    # (upstream MUSE-News: 2nd checkpoint of a 3-epoch linear decay).
    lr_schedule = k.get("lr_schedule", DEFAULT_LR_SCHEDULE)
    if lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"kuda.lr_schedule must be one of {LR_SCHEDULES}, got {lr_schedule!r}")
    schedule_epochs = _pos_int(k.get("schedule_epochs", epoch), "kuda.schedule_epochs")
    if schedule_epochs < epoch:
        raise ValueError(f"kuda.schedule_epochs ({schedule_epochs}) must be >= kuda.epoch "
                         f"({epoch}): the decay horizon cannot be shorter than the run")

    resolved = {
        "name": name,
        "model": model_id,
        "tokenizer": tokenizer_id,
        "dtype": dtype,
        "device_map": cfg.get("device_map", "auto"),
        "seed": _nonneg_int(cfg.get("seed", 42), "seed"),
        "push_to_hub": bool(cfg.get("push_to_hub", False)) if push_to_hub is None
                       else bool(push_to_hub),
        "hub_repo": cfg.get("hub_repo") or None,
        "hub_org": cfg.get("hub_org"),
        "hub_private": bool(cfg.get("hub_private", True)),
        "module_name": module_name,
        # GPT-2 (absolute position embeddings) + forced left padding: derive
        # position_ids from the mask (common.add_position_ids). Off = upstream.
        "position_ids_from_mask": pos_ids,
        "beta": _pos_float(k.get("beta"), "kuda.beta"),
        "tau": _pos_float(k.get("tau"), "kuda.tau"),
        "alpha": alpha_r,
        "lr": _pos_float(k.get("lr"), "kuda.lr"),
        "batch_size": _pos_int(k.get("batch_size"), "kuda.batch_size"),
        "max_num_batches": max_nb,
        "epoch": epoch,
        "lr_schedule": lr_schedule,
        "schedule_epochs": schedule_epochs,
        "weight_decay": _nonneg_float(k.get("weight_decay", DEFAULT_WEIGHT_DECAY),
                                      "kuda.weight_decay"),
        # srspd-experiments addition; 0.0 (the default) is an exact no-op and is upstream.
        "init_jitter": _nonneg_float(k.get("init_jitter", DEFAULT_INIT_JITTER),
                                     "kuda.init_jitter"),
        "layer_id": layer_id,
        "layer_ids": layer_ids,
        "forget_corpora": forget,
        "retain_corpora": retain,
        # Non-negative: upstream's training-data --min_len default is 0. The
        # covariance uses the ROME all-rows loader and ignores this (see
        # select_cov_texts); this filters the forget/retain TRAINING batches.
        "min_len": _nonneg_int(k.get("min_len", C.DEFAULT_MIN_LEN), "kuda.min_len"),
        # 768 for EVERY topic — upstream's hard-coded value, not the per-topic
        # 512/768 line it leaves commented out (see :data:`FORGET_MAX_LENGTH`) —
        # unless `kuda.forget_max_length` overrides it (MUSE: 1024) or a
        # `budget:` fixes it.
        "forget_max_length": [forget_max_length] * len(forget),
        "retain_max_length": retain_max_length,
        # Equal token budget: None = the upstream loop, unchanged.
        "budget": budget,
        # Upstream MUSE pairs forget window i with retain window i % len(retain)
        # (ForgetRetainDataset.__getitem__), so a retain set shorter than the
        # forget set does NOT shorten the run. The WMDP loop's `min(...)` over
        # both lengths is kept as the default; `true` opts into the modulo.
        "cycle_retain": bool(k.get("cycle_retain", False)),
        # Not config keys: upstream forces both, and both are result-affecting
        # because the two losses cover pad positions.
        "padding_side": TOKENIZER_PADDING_SIDE,
        "truncation_side": TOKENIZER_TRUNCATION_SIDE,
        "nullspace": {
            # A PERCENTAGE of the retain corpus, upstream's --sample_p units.
            "sample_p": _percent(ns.get("sample_p", DEFAULT_SAMPLE_P) if sample_p is None
                                 else sample_p, "kuda.nullspace.sample_p"),
            # srspd-experiments addition: a hard row cap applied AFTER the percentage.
            "max_cov_texts": _opt_pos_int(ns.get("max_cov_texts", DEFAULT_MAX_COV_TEXTS),
                                          "kuda.nullspace.max_cov_texts"),
            "max_length": _pos_int(ns.get("max_length", DEFAULT_NS_MAX_LENGTH),
                                   "kuda.nullspace.max_length"),
            "batch_size": _pos_int(ns.get("batch_size", DEFAULT_NS_BATCH_SIZE),
                                   "kuda.nullspace.batch_size"),
            "svd_device": svd_device,
            "cov_device": cov_device,
            # Upstream's mom2_dtype. See accumulate_down_proj_cov's docstring.
            "cov_dtype": str(cov_dtype),
            "use_cache": bool(ns.get("use_cache", True)),
        },
    }
    # Sub-100% sampling diverges from upstream (sampling rule AND rounding) and
    # is inert only while every config says 100. Warn here — before either model
    # loads, and covering the CLI override as well as the config key — and carry
    # the verdict into metrics.json so the run record carries it either way.
    faithful = float(resolved["nullspace"]["sample_p"]) >= 100.0
    resolved["nullspace"]["sample_p_is_upstream_faithful"] = faithful
    note = sample_p_divergence_warning(resolved["nullspace"]["sample_p"])
    if note:
        print(note, flush=True)
    return resolved


def default_repo_name(model_id: str) -> str:
    """``rspd-wmdp-kuda-<model-short>`` from a hub id, when the config does not
    set ``hub_repo:`` explicitly."""
    short = str(model_id).rstrip("/").split("/")[-1].lower()
    return f"rspd-wmdp-kuda-{short}"


# --------------------------------------------------------------------------- #
# The objective (ported verbatim from unlearn.py:137-143)                      #
# --------------------------------------------------------------------------- #
def deviation_loss(updated_act, frozen_act, beta: float):
    """KUDA's representation-deviation forget loss and the mean cosine it is
    built from.

    ``updated_act`` / ``frozen_act`` are the layer-``layer_id`` outputs of the
    updated and frozen models on the *same* forget batch, shaped
    ``(batch, tokens, d_model)``. The loss falls monotonically as the cosine
    similarity goes to ``-1``:

        cos          = cosine_similarity(updated, frozen, dim=-1).mean(-1)
        unlearn_loss = -logsigmoid(beta * (1 - cos)).mean() * 2 / beta

    Returns ``(loss, cos_mean)`` with ``cos_mean`` a detached float for logging.
    """
    cos = F.cosine_similarity(updated_act, frozen_act, dim=-1).mean(-1)
    complement = 1.0 - cos
    loss = -F.logsigmoid(beta * complement).mean() * 2 / beta
    return loss, float(cos.detach().float().mean())


def retain_loss(updated_act, frozen_act, alpha: float):
    """RMU's retain term, unchanged by KUDA: MSE back to the frozen model's own
    activations on a retain batch, weighted by this topic's ``alpha``."""
    return F.mse_loss(updated_act, frozen_act) * alpha


def lr_at_step(step: int, *, base_lr: float, total_steps: int, schedule: str) -> float:
    """The learning rate applied at optimizer step ``step`` (0-based).

    ``constant`` returns ``base_lr`` for every step. ``linear_decay`` is HF's
    ``get_linear_schedule_with_warmup(num_warmup_steps=0,
    num_training_steps=total_steps)`` evaluated once per optimizer step:
    ``base_lr * max(0, (total_steps - step) / total_steps)`` — ``base_lr`` at
    step 0, ``0`` at (and beyond) ``total_steps``. ``total_steps`` is the
    *schedule* horizon (``schedule_epochs * steps_per_epoch``), which may be
    longer than the run.
    """
    if schedule == "constant":
        return float(base_lr)
    if schedule != "linear_decay":
        raise ValueError(f"unknown lr schedule {schedule!r}; expected one of {LR_SCHEDULES}")
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1 for linear_decay, got {total_steps}")
    return float(base_lr) * max(0.0, float(total_steps - step) / float(total_steps))


def forward_with_cache(model, inputs, module, *, no_grad: bool = True):
    """Run ``model(**inputs)`` and return ``module``'s output activations
    (upstream ``utils.forward_with_cache``).

    A decoder block returns a tuple ``(hidden_states, ...)`` on transformers 4.x
    and (for several architectures) a bare tensor on 5.x, so both shapes are
    accepted — the runtime image pins transformers 5.x and the tuple assumption
    is exactly the kind of thing that breaks silently across that boundary.
    """
    cache = []

    def hook(_module, _inp, out):
        cache.append(out[0] if isinstance(out, (tuple, list)) else out)
        return None

    handle = module.register_forward_hook(hook)
    try:
        if no_grad:
            with torch.no_grad():
                model(**inputs)
        else:
            model(**inputs)
    finally:
        handle.remove()
    if not cache:
        raise RuntimeError("the hooked module produced no output — the forward pass "
                           "never reached it (wrong module or an early exit)")
    return cache[0]


# --------------------------------------------------------------------------- #
# Relaxed null-space projection (nsproject/nullspace.py::generate_project)     #
# --------------------------------------------------------------------------- #
def _svd_with_fallback(cov, *, prefer: str = "auto"):
    """``(U, S, path)`` of ``svd(cov)``, falling back off the GPU when cuSOLVER
    refuses.

    ``cov`` is the accumulated second moment (bfloat16 by default — see
    :func:`accumulate_down_proj_cov`). The GPU attempt is float32 (cuSOLVER's
    ``gesvdj``/``gesvd`` has a size ceiling and no float64 path on every device —
    a ``d`` of 14336, let alone Gemma-3's wider MLP, can trip it); the CPU
    fallbacks are float64 first, then float32 if LAPACK also gives up. ``path``
    is a string like ``"cuda:0/float32"`` and is logged and stored in the cache
    metadata, because *which* path ran is part of the experimental record.
    """
    dev = cov.device if prefer == "auto" else torch.device(prefer)
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")
    attempts = []
    if dev.type == "cuda":
        attempts.append((dev, torch.float32))
    attempts += [(torch.device("cpu"), torch.float64), (torch.device("cpu"), torch.float32)]

    errors = []
    for device, dtype in attempts:
        path = f"{device}/{str(dtype).replace('torch.', '')}"
        try:
            work = cov.to(device=device, dtype=dtype)
            U, S, _ = torch.linalg.svd(work, full_matrices=False)
            del work
            return U, S, path
        except Exception as exc:                       # cuSOLVER ceiling / OOM / LAPACK
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            print(f"[nullspace] SVD on {path} failed ({type(exc).__name__}: {exc}); "
                  f"falling back", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    raise RuntimeError("every SVD path failed for the null-space projector: "
                       + " | ".join(errors))


def _resolved_device(setting: str, *, default=None) -> str:
    """``"cpu"`` or ``"cuda"`` — what ``setting`` (``auto``/``cpu``/``cuda``)
    will actually resolve to on *this* box.

    Part of the cache key, because the SVD's precision follows the device
    (:func:`_svd_with_fallback`: float32 on CUDA, float64 on the CPU) and an
    fp32-vs-fp64 spectrum moves membership of a ``tau = 2e-3`` cut. ``auto``
    follows the tensors (``default`` when given, else the GPU if there is one);
    asking for ``cuda`` on a CPU-only box resolves to ``cpu``, exactly as
    :func:`_svd_with_fallback` does.

    This records the device that will be *attempted*. If cuSOLVER then refuses
    and the CPU fallback runs, ``info["svd_path"]`` (recorded in the cache
    metadata and in ``metrics.json``) is the record of what actually ran.
    """
    if setting == "cpu":
        return "cpu"
    if setting == "auto" and default is not None:
        return default
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolved_ns_devices(ns: dict) -> tuple:
    """``(cov_device, svd_device)`` as they will actually resolve.

    ``svd_device: auto`` means "wherever the covariance ended up" —
    :func:`_svd_with_fallback` starts from ``cov.device`` — so a config that
    pins ``cov_device: cpu`` and leaves the SVD on ``auto`` takes the SVD on the
    CPU too, even on a GPU box. Resolving the pair together keeps the cache key
    honest about that.
    """
    cov = _resolved_device(ns["cov_device"])
    return cov, _resolved_device(ns["svd_device"], default=cov)


def projector_from_spectrum(U, S, tau: float, *, dtype=torch.bfloat16):
    """``(P, info)`` — cut an already-computed spectrum at ``tau``.

    Split out of :func:`build_projector` so a ``tau`` sweep can re-cut a cached
    ``U``/``S`` without repaying the covariance pass or the ``d×d`` SVD (this is
    upstream's own split: the second moment is cached without ``tau``, the
    projector with it).
    """
    small = (S < float(tau)).nonzero(as_tuple=True)[0]
    if small.numel() == 0:
        small = torch.argmin(S).reshape(1)
    basis = U[:, small]
    P = (basis @ basis.T).to(dtype)
    info = {
        "n_dirs": int(small.numel()),
        "d": int(S.numel()),
        "tau": float(tau),
        "s_max": float(S.max()),
        "s_min": float(S.min()),
        "frac_kept": float(small.numel()) / float(S.numel()),
    }
    del basis
    return P, info


def build_projector(cov, tau: float, *, svd_device: str = "auto",
                    dtype=torch.bfloat16):
    """The relaxed null-space projector ``P`` of one layer, and its metadata.

    ``cov`` is the uncentered second moment ``E[x xᵀ]`` of that layer's
    ``mlp.down_proj`` **input** features over the retain corpus. Directions whose
    singular value is below ``tau`` carry almost no retain energy, so moving the
    weight along them barely disturbs retained behavior::

        U, S, _ = svd(cov);  small = (S < tau);  P = U[:, small] @ U[:, small].T

    If nothing clears the threshold the single smallest direction is kept (never
    an empty projector — that would zero every gradient). ``P`` is symmetric and
    idempotent by construction.

    Returns ``(P, info)`` where ``info`` records ``n_dirs`` (how many directions
    the update is confined to), ``d``, the SVD path that actually ran, and the
    spectrum's extremes.
    """
    U, S, path = _svd_with_fallback(cov, prefer=svd_device)
    P, info = projector_from_spectrum(U, S, tau, dtype=dtype)
    info["svd_path"] = path
    del U, S
    return P, info


def cov_corpora(resolved: dict) -> list:
    """The retain corpora the covariance pass reads, deduplicated in
    **insertion** order.

    The one definition, used by both :func:`select_cov_texts` and
    :func:`_cov_key`. It used to be ``dict.fromkeys(...)`` in the first and
    ``sorted(set(...))`` in the second, which is a silent collision: with two
    *distinct* retain corpora and ``max_cov_texts`` set, ``[A, B]`` and
    ``[B, A]`` select different texts (the cap truncates a differently-ordered
    concatenation) but hashed to the same key, so the second run would have been
    served the first run's projector.
    """
    return list(dict.fromkeys(resolved["retain_corpora"]))


def _local_corpora_fingerprint(names) -> dict:
    """``{"retain_corpora_files": {name: {path: sha256}}}`` for the
    ``local_jsonl`` corpora among ``names`` — or ``{}`` when there is none, so
    the hub-corpus keys (hub corpora) are unchanged. A config-level corpus is
    keyed by its NAME in ``retain_corpora``, and the same name (``contrastive``)
    can point at different files across set builds (the 2M and the 1M GPT-2
    contrastive sets); without the content in the key the second build would
    be served the first build's covariance."""
    files = {}
    for n in names:
        spec = C.CORPORA.get(n) or {}
        if C.corpus_kind(spec) != "local_jsonl":
            continue
        paths = spec["path"] if isinstance(spec["path"], list) else [spec["path"]]
        files[n] = {p: _sha256_file(C.resolve_local_path(p)) for p in paths}
    return {"retain_corpora_files": files} if files else {}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except FileNotFoundError:
        return f"missing:{path}"
    return h.hexdigest()


def _cov_key(resolved: dict, layer_id: int) -> str:
    """Cache key for one layer's **covariance and its SVD** — everything that
    changes ``cov``, ``U`` or ``S``, and deliberately **not** ``tau``.

    Upstream splits the cache the same way (``nsproject/layer_stats.py:127``
    keys the second-moment file without ``tau``; ``nsproject/nullspace.py:98``
    keys the projector file with it), and the reason is cost: ``tau`` is a cut
    of an already-computed spectrum, so a ``tau`` sweep that keys both together
    repays a ~8-minute covariance pass and a ``d³`` SVD per grid point for
    nothing. :func:`_proj_key` is this key plus ``tau``.

    What is in here and why:

    * ``retain_corpora`` in **insertion order** (:func:`cov_corpora`) — see that
      docstring for the ordering collision this closes;
    * ``batch_size`` — result-affecting since ``padding_side="left"``: left
      padding shifts every sequence's RoPE position ids by the batch's own
      longest text, so a different batching produces different features;
    * the **resolved** ``cov_device`` / ``svd_device``
      (:func:`resolved_ns_devices`)
      and ``cov_dtype`` — the accumulation dtype and the SVD's precision both
      move membership of a ``tau = 2e-3`` cut, and a driver that records
      ``svd_path`` in ``info`` precisely because "which path ran is part of the
      record" cannot then key the cache without it;
    * ``padding_side`` / ``truncation_side`` — the covariance is taken over
      non-pad positions of *these* tokenizations.
    """
    ns = resolved["nullspace"]
    cov_device, svd_device = resolved_ns_devices(ns)
    payload = {
        "model": resolved["model"],
        "retain_corpora": cov_corpora(resolved),
        **_local_corpora_fingerprint(cov_corpora(resolved)),
        "layer_id": int(layer_id),
        "module_name": resolved["module_name"],
        "sample_p_percent": float(ns["sample_p"]),
        "max_cov_texts": (None if ns.get("max_cov_texts") is None
                          else int(ns["max_cov_texts"])),
        "max_length": int(ns["max_length"]),
        "batch_size": int(ns["batch_size"]),
        "min_len": int(resolved["min_len"]),
        "dtype": resolved["dtype"],
        "cov_dtype": str(ns["cov_dtype"]),
        "cov_device": cov_device,
        "svd_device": svd_device,
        "padding_side": resolved["padding_side"],
        "truncation_side": resolved["truncation_side"],
        "version": CACHE_VERSION,
    }
    # A tokenizer other than the model's own changes every tokenisation the
    # covariance sees. Added only when set, so every existing key (tokenizer
    # from the model) hashes exactly as before.
    if resolved.get("tokenizer"):
        payload["tokenizer"] = resolved["tokenizer"]
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _proj_key(resolved: dict, layer_id: int) -> str:
    """Cache key for one layer's **projector** ``P``: :func:`_cov_key` plus
    ``tau``, which is the only thing that turns a spectrum into a projector."""
    payload = {
        "cov_key": _cov_key(resolved, layer_id),
        "tau": float(resolved["tau"]),
        "version": CACHE_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def select_cov_texts(resolved: dict, tokenizer=None) -> list:
    """The retain texts the covariance pass consumes, in corpus order.

    A ``chunk: True`` (MUSE) retain corpus contributes its official windows
    (``common.load_corpus_windows`` at ``kuda.nullspace.max_length`` — upstream
    MUSE builds the second moment from ``DefaultDataset(retain1, max_len=1024)``,
    ``kuda.py``) instead of rows, and needs ``tokenizer``; ``sample_p`` and
    ``max_cov_texts`` then count windows. Upstream hands the *decoded strings* of
    those windows to ROME's ``layer_stats``, which re-tokenizes them (BOS
    re-added); passing the token windows directly skips that decode/re-encode
    round trip and is otherwise the same sample.

    Two knobs, applied in this order:

    1. ``kuda.nullspace.sample_p`` — upstream's ``--sample_p``, a **percentage**
       in ``(0, 100]``. Each retain corpus is loaded in full (after the
       ``min_len`` filter) and its first ``ceil(n * sample_p / 100)`` texts are
       taken — deterministic and sequential, a prefix rather than a draw.

       *Known deviation, immaterial at the shipped setting:* below 100%
       upstream takes a **seeded random sample**, not a prefix
       (``unlearning/utils.py:265-269``: ``random.seed(seed)`` then
       ``random.sample(data, max(1, int(n * sample_p / 100)))``), and sizes it
       with ``int`` where this rounds up with ``ceil``. At ``sample_p = 100`` —
       what upstream's 7B-model scripts pass and what both shipped configs use —
       upstream skips the branch entirely and the two agree exactly; below it
       they differ in *which* rows, and by at most one in how many.

       The percentage is applied *per corpus*
       rather than to the concatenation, so a config with two distinct retain
       corpora samples both — concatenating first would take 100% of the first
       and nothing from the second. Upstream (and both shipped configs) use a
       single retain corpus, where the two readings coincide.
    2. ``kuda.nullspace.max_cov_texts`` — an **srspd-experiments addition**, not upstream: an
       optional hard cap on the total, for cost control on very large corpora.
       ``None`` (the default) reproduces upstream.
    """
    ns = resolved["nullspace"]
    pct = float(ns["sample_p"])
    cap = ns.get("max_cov_texts")
    # At 100% "the first `cap` of all of them" == "all of them, then cap", so the
    # loader may stop early; below 100% the denominator is the full corpus size.
    max_rows = cap if (cap is not None and pct >= 100.0) else None

    texts: list[str] = []
    for corpus in cov_corpora(resolved):      # dedup, keep INSERTION order
        # ROME fidelity: the covariance sees EVERY row (empties -> BOS), NOT the
        # min_len-filtered training set. Upstream's layer_stats tokenises the raw
        # wikitext test split verbatim; matching that is worth a measurable slice
        # of the MMLU gap between the port and the authors' code on the 7B model.
        if C.is_chunked_corpus(corpus):
            rows = C.load_corpus_windows(corpus, tokenizer, ns["max_length"],
                                         max_rows=max_rows)
        else:
            rows = C.load_corpus_texts_all(corpus, max_rows=max_rows)
        take = len(rows) if pct >= 100.0 else math.ceil(len(rows) * pct / 100.0)
        texts += rows[:take]
    if cap is not None:
        texts = texts[:cap]
    return texts


def resolve_cov_dtype(cov_dtype):
    """``kuda.nullspace.cov_dtype`` (a name, or a ``torch.dtype``) as a
    ``torch.dtype``. Raises ``ValueError`` naming the key for anything else."""
    if isinstance(cov_dtype, torch.dtype):
        return cov_dtype
    try:
        return COV_DTYPES[str(cov_dtype)]
    except KeyError:
        raise ValueError(f"kuda.nullspace.cov_dtype must be one of "
                         f"{sorted(COV_DTYPES)}, got {cov_dtype!r}") from None


def accumulate_down_proj_cov(model, tokenizer, layer_ids, texts, *, batch_size: int,
                             max_length: int, cov_device: str = "auto",
                             cov_dtype=DEFAULT_COV_DTYPE, module_name: str = MODULE_NAME):
    """Uncentered second moment ``E[x xᵀ]`` of the ``mlp.down_proj`` **input**
    features of every layer in ``layer_ids``, over ``texts``, accumulated in
    ``cov_dtype`` (**bfloat16** by default). ``module_name`` picks the module
    (``mlp.c_proj`` on GPT-2); a forward PRE-hook sees its input either way.

    One forward pass feeds every layer: a *forward pre-hook* on each target
    ``mlp.down_proj`` sees that module's input (the post-activation MLP hidden
    state, width = the MLP intermediate size), and PAD positions are dropped via
    the batch's attention mask. Returns ``({layer_id: cov}, n_tokens)``.

    **Why bfloat16, deliberately, when float64 is right there.** This is a
    fidelity-over-precision choice, and it is the *worse* numerics on purpose.
    Upstream accumulates the second moment in bf16 — ``"mom2_dtype":
    "bfloat16"`` in every ``nsproject/hyparams/*.json``, read at
    ``layer_stats.py:117`` and applied to the features at ``:163``, after which
    ``runningstats.SecondMoment.add`` allocates ``mom2`` in the features' own
    dtype and does ``self.mom2 += a.t().mm(a)`` in it
    (``runningstats.py:256-260``), dividing by the count in bf16 too
    (``moment()``) before a final ``.float()`` for the SVD. bf16 carries ~8 bits
    of mantissa, so the small end of that spectrum — exactly the end ``tau``
    cuts at — is substantially quantisation noise rather than retain energy. The
    published ``tau = 2e-3`` was tuned against *that* spectrum. A float64
    accumulation produces a genuinely more accurate spectrum whose small end is
    orders of magnitude lower, and the same constant then selects a different
    (much smaller) subspace: the repro would not be a repro. ``cov_dtype:
    float64`` stays available for a deliberate comparison, participates in the
    cache key, and is recorded in ``metrics.json``.

    Memory: one ``d²`` matrix per layer — 411 MB at ``d`` = 14336 in bf16, 1.6 GB
    in float64 — which is why ``cov_device: cpu`` exists and why this is cached.
    """
    acc_dtype = resolve_cov_dtype(cov_dtype)
    mods = {lid: C.get_mlp_out_module(model, lid, module_name) for lid in layer_ids}
    state = {lid: None for lid in layer_ids}
    counts = {lid: 0 for lid in layer_ids}
    live_mask = {}

    def make_hook(lid):
        def pre_hook(_module, args):
            x = args[0]
            if x.dim() == 2:                       # (tokens, d) — no batch dim
                flat = x.reshape(-1, x.shape[-1])
            else:
                mask = live_mask.get("mask")
                flat = x.reshape(-1, x.shape[-1])
                if mask is not None:
                    flat = flat[mask.reshape(-1).to(flat.device).bool()]
            if flat.numel() == 0:
                return None
            target = (flat.device if cov_device == "auto"
                      else torch.device(cov_device))
            feats = flat.to(device=target, dtype=acc_dtype)
            gram = feats.T @ feats
            state[lid] = gram if state[lid] is None else state[lid] + gram
            counts[lid] += int(feats.shape[0])
            return None
        return pre_hook

    # The covariance MUST be tokenised with RIGHT padding, regardless of the
    # training loop's ``padding_side``. Upstream builds this second moment through
    # ROME's ``nsproject/layer_stats.py`` + ``tok_dataset.TokenizedDataset``, which
    # supplies explicit natural ``position_ids`` (``range(len)``) and right-pads via
    # ``pad_sequence`` — so every real token sees its true position. This driver
    # used the training tokenizer's ``padding_side="left"`` here and passed no
    # ``position_ids``; transformers-5.x then assigns RoPE positions ``arange(L)``,
    # so left-padded real tokens are shifted by the batch's pad count. That corrupts
    # the ``mlp.down_proj`` input activations, collapsing the spectrum (trace ≈ 30,
    # s_max ≈ 1.8) so that ~80% of singular values fall below tau=2e-3 — a
    # near-identity projector that gives KUDA no retain protection and destroys the
    # model (MMLU 0.589 -> 0.293). Right padding puts real tokens at indices
    # 0..len-1, where the default ``arange`` positions are correct; pads sit at the
    # end and are dropped by the attention mask. This matches upstream and is not a
    # tunable — a corrupted covariance is not a "different tau", it is wrong.
    saved_padding_side = getattr(tokenizer, "padding_side", None)
    tokenizer.padding_side = "right"
    handles = [mods[lid].register_forward_pre_hook(make_hook(lid)) for lid in layer_ids]
    try:
        device = next(model.parameters()).device
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i + batch_size]
                enc = C.tokenize_batch(tokenizer, chunk, max_length=max_length,
                                       device=device)
                enc = {k: v for k, v in enc.items() if k != "labels"}
                live_mask["mask"] = enc.get("attention_mask")
                model(**enc, use_cache=False)
                live_mask.pop("mask", None)
    finally:
        for h in handles:
            h.remove()
        if saved_padding_side is not None:
            tokenizer.padding_side = saved_padding_side

    n_tokens = max(counts.values()) if counts else 0
    covs = {}
    for lid in layer_ids:
        if state[lid] is None or counts[lid] == 0:
            raise RuntimeError(f"no {module_name} input features captured for layer {lid}; "
                               f"the forward never reached it")
        covs[lid] = state[lid] / float(counts[lid])
        state[lid] = None
    return covs, n_tokens


def cov_cache_path(cache_dir: str, resolved: dict, layer_id: int) -> str:
    """Path of one layer's **covariance + SVD** cache entry (:func:`_cov_key`)."""
    return os.path.join(cache_dir,
                        f"kuda_cov_L{layer_id:03d}_{_cov_key(resolved, layer_id)}.pt")


def projector_cache_path(cache_dir: str, resolved: dict, layer_id: int) -> str:
    """Path of one layer's **projector** cache entry (:func:`_proj_key`)."""
    return os.path.join(cache_dir,
                        f"kuda_nullspace_L{layer_id:03d}_{_proj_key(resolved, layer_id)}.pt")


def _read_cache(path: str, what: str, layer_id: int):
    """``torch.load`` that degrades to ``None`` with a log line, never an
    exception: an unreadable cache entry is a rebuild, not a dead run."""
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        print(f"[nullspace] layer {layer_id}: {what} cache at {path} unreadable "
              f"({type(exc).__name__}: {exc}); rebuilding", flush=True)
        return None


def _write_cache(path: str, blob: dict, what: str, layer_id: int) -> None:
    """Best-effort ``torch.save``: a full disk must not sink a finished build."""
    try:
        torch.save(blob, path)
        print(f"[nullspace] layer {layer_id}: {what} cached -> {path}", flush=True)
    except Exception as exc:
        print(f"[nullspace] layer {layer_id}: caching the {what} failed "
              f"({type(exc).__name__}: {exc}); continuing", flush=True)


def load_or_build_projectors(model, tokenizer, resolved: dict, *, cache_dir: str):
    """``({layer_id: P}, {layer_id: info})`` — the relaxed null-space projectors,
    read from the disk cache where the keys match and built (one shared forward
    pass over the retain sample) for whatever is missing.

    **Two tiers, as upstream has.** A layer is resolved in this order:

    1. the **projector** entry (:func:`_proj_key` = the cov key *plus* ``tau``) —
       a hit costs one ``torch.load``;
    2. the **spectrum** entry (:func:`_cov_key`, no ``tau``) holding ``U`` and
       ``S`` — a hit re-cuts it at this run's ``tau``
       (:func:`projector_from_spectrum`), skipping both the covariance pass and
       the ``d×d`` SVD. This is what makes a ``tau`` sweep nearly free;
    3. otherwise the layer joins the shared covariance pass.

    The cache is shared across configs, so a rerun at a different learning rate,
    batch count or ``tau`` never repays work it does not change. ``info`` records
    ``cached`` (the projector came from disk) and ``cov_cached`` (the covariance
    and its SVD did) separately, plus both keys, so ``metrics.json`` says which
    tier each layer came from.

    Tier 2 is not free on disk: at a 7B model's ``d`` = 14336 one entry is ``U`` at
    the SVD's own precision — 822 MB float32 from the CUDA path, 1.6 GB float64
    from the CPU one — plus a negligible ``S``, against 411 MB for the bf16
    projector. The covariance matrix is deliberately **not** kept: it would add
    411 MB per layer per entry and buys nothing, because ``U``/``S`` already
    re-cut any ``tau`` and every input that would change the SVD is in the key.

    The retain sample is :func:`select_cov_texts` — ``sample_p`` percent of each
    retain corpus, then the optional ``max_cov_texts`` cap.
    """
    ns = resolved["nullspace"]
    layer_ids = resolved["layer_ids"]
    use_cache = ns["use_cache"]
    tau = resolved["tau"]
    os.makedirs(cache_dir, exist_ok=True)

    cov_device, svd_device = resolved_ns_devices(ns)

    def _base_info(lid):
        """The provenance every ``info`` carries whichever tier answered."""
        return {"cov_key": _cov_key(resolved, lid), "proj_key": _proj_key(resolved, lid),
                "cov_dtype": str(ns["cov_dtype"]),
                "cov_device_resolved": cov_device,
                "svd_device_resolved": svd_device,
                "cov_cache_path": cov_cache_path(cache_dir, resolved, lid),
                "cache_path": projector_cache_path(cache_dir, resolved, lid)}

    def _log_projector(lid, info):
        print(f"[nullspace] layer {lid}: {info['n_dirs']}/{info['d']} directions "
              f"below tau={tau:g} ({info['frac_kept']:.4%}), SVD on "
              f"{info.get('svd_path')}, S in "
              f"[{info['s_min']:.3e}, {info['s_max']:.3e}]", flush=True)

    projectors, infos = {}, {}
    from_spectrum, needs_cov = [], []

    # -- tier 1: the projector itself -----------------------------------------
    for lid in layer_ids:
        info = _base_info(lid)
        blob = _read_cache(info["cache_path"], "projector", lid) if use_cache else None
        if blob is not None and "P" in blob:
            projectors[lid] = blob["P"]
            info.update(dict(blob.get("info") or {}))
            info.update({"cached": True, "cov_cached": True})
            infos[lid] = info
            print(f"[nullspace] layer {lid}: reused cached projector "
                  f"({info.get('n_dirs')} directions) <- {info['cache_path']}",
                  flush=True)
            continue
        # -- tier 2: the covariance and its spectrum --------------------------
        blob = _read_cache(info["cov_cache_path"], "covariance", lid) if use_cache else None
        if blob is not None and "U" in blob and "S" in blob:
            from_spectrum.append((lid, info, blob))
        else:
            needs_cov.append((lid, info))

    for lid, info, blob in from_spectrum:
        P, cut = projector_from_spectrum(blob["U"], blob["S"], tau)
        info.update(dict(blob.get("info") or {}))
        info.update(cut)
        info.update({"cached": False, "cov_cached": True})
        projectors[lid], infos[lid] = P, info
        print(f"[nullspace] layer {lid}: re-cut the CACHED spectrum at tau={tau:g} "
              f"(no covariance pass, no SVD) <- {info['cov_cache_path']}", flush=True)
        _log_projector(lid, info)
        if use_cache:
            _write_cache(info["cache_path"], {"P": P.to("cpu"), "info": info},
                         "projector", lid)

    # -- tier 3: one shared covariance pass over whatever is left -------------
    if needs_cov:
        t0 = time.time()
        texts = select_cov_texts(resolved, tokenizer)
        if not texts:
            raise RuntimeError("the retain corpora yielded no texts for the covariance "
                               "pass; nothing to build a null-space projector from")
        print(f"[nullspace] covariance pass: {len(texts)} retain samples "
              f"(sample_p {ns['sample_p']:g}% of {cov_corpora(resolved)}, max_cov_texts "
              f"{ns['max_cov_texts']}), layers {[lid for lid, _ in needs_cov]}, "
              f"max_length {ns['max_length']}, batch_size {ns['batch_size']}, "
              f"accumulating in {ns['cov_dtype']} (upstream's mom2_dtype)", flush=True)
        covs, n_tokens = accumulate_down_proj_cov(
            model, tokenizer, [lid for lid, _ in needs_cov], texts,
            batch_size=ns["batch_size"], max_length=ns["max_length"],
            cov_device=ns["cov_device"], cov_dtype=ns["cov_dtype"],
            module_name=resolved["module_name"])
        print(f"[nullspace] covariance over {n_tokens} non-pad tokens "
              f"({time.time() - t0:.1f}s)", flush=True)

        cov_meta = {"n_tokens": n_tokens, "n_samples": len(texts),
                    "sample_p_percent": float(ns["sample_p"]),
                    "max_cov_texts": ns["max_cov_texts"],
                    "cov_corpora": cov_corpora(resolved)}
        for lid, info in needs_cov:
            cov = covs.pop(lid)
            U, S, svd_path = _svd_with_fallback(cov, prefer=ns["svd_device"])
            info.update(cov_meta)
            info["svd_path"] = svd_path
            del cov                      # the SPECTRUM is what gets cached
            if use_cache:
                _write_cache(info["cov_cache_path"],
                             {"U": U.to("cpu"), "S": S.to("cpu"),
                              "info": dict(cov_meta, svd_path=svd_path,
                                           cov_dtype=str(ns["cov_dtype"]))},
                             "spectrum", lid)
            P, cut = projector_from_spectrum(U, S, tau)
            del U, S
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            info.update(cut)
            info.update({"cached": False, "cov_cached": False})
            projectors[lid], infos[lid] = P, info
            _log_projector(lid, info)
            if use_cache:
                _write_cache(info["cache_path"], {"P": P.to("cpu"), "info": info},
                             "projector", lid)
    return projectors, infos


def project_gradients(params: dict, projectors: dict, layouts: dict | None = None) -> int:
    """Multiply every target weight's gradient by its own projector on the
    ``d_in`` side, **after** ``loss.backward()`` and **before**
    ``optimizer.step()``.

    ``params`` maps parameter name -> parameter; ``projectors`` maps the *same*
    names -> the layer's ``P`` (``(d_in, d_in)``); ``layouts`` maps names ->
    ``common.weight_layout`` of the owning module, default ``"out_in"``:

    * ``"out_in"`` (``nn.Linear``: ``down_proj.weight`` is ``(d_model,
      d_intermediate)``) -- ``grad <- grad @ P``;
    * ``"in_out"`` (GPT-2's ``Conv1D``: ``c_proj.weight`` is ``(d_intermediate,
      d_model)``) -- ``grad <- P @ grad``.

    Either way the update's *input* directions are confined to the retain-null
    subspace and the output side is untouched (the projected gradient's rows /
    columns are orthogonal to the retained-input subspace; pinned for both
    layouts in ``tests/test_wmdp_kuda_gpt2.py``). Returns the number of
    gradients projected.

    Note what the guarantee is: KUDA projects the **gradient**, and the
    optimizer is AdamW (upstream's), whose elementwise second-moment
    preconditioning does not commute with a projection — so the realised weight
    *step* is not exactly confined to the subspace, only the gradient that feeds
    it. That is upstream's arrangement, kept as is; with a linear step (SGD) the
    weight change itself stays in the subspace
    (``tests/test_wmdp_kuda.py::test_a_projected_sgd_step_...``).
    """
    n = 0
    for name, p in params.items():
        if p.grad is None:
            continue
        P = projectors.get(name)
        if P is None:
            raise KeyError(f"no null-space projector for target parameter {name!r}")
        P = P.to(device=p.grad.device, dtype=p.grad.dtype)
        layout = (layouts or {}).get(name, "out_in")
        if layout == "out_in":
            p.grad = torch.mm(p.grad, P)
        elif layout == "in_out":
            p.grad = torch.mm(P, p.grad)
        else:
            raise ValueError(f"unknown weight layout {layout!r} for {name!r}; "
                             f"expected 'out_in' (nn.Linear) or 'in_out' (Conv1D)")
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def _norm(x) -> float:
    """``‖x‖₂`` accumulated in **at least** float32, without materialising an
    upcast copy of a 5·10⁷-element tensor.

    A bf16 parameter's squared norm is not representable to anything like full
    precision in bf16 (~8-bit mantissa, and these are exactly the near-zero
    quantities this driver is trying to tell apart from zero), so the
    accumulation is promoted. ``promote_types`` rather than a flat float32
    because ``vector_norm`` refuses a narrowing ``dtype=``, and the toy tests
    (and a ``cov_dtype: float64`` comparison) run in float64.
    """
    return float(torch.linalg.vector_norm(
        x, dtype=torch.promote_types(x.dtype, torch.float32)))


def apply_init_jitter(params: dict, init_jitter: float, *, seed: int) -> dict:
    """Break the forget objective's step-0 symmetry, deterministically, in place.

    Applied **once**, under ``torch.no_grad()``, after the target parameters are
    unfrozen and **before** the optimizer is constructed (so no stale Adam state
    describes the pre-jitter weights)::

        p += randn_like(p) * init_jitter * p.std()

    Why it exists: ``cos(u, f)`` is maximised at ``u == f``, so
    ``∇_u cos|_{u=f} = f/‖f‖² − f/‖f‖² = 0`` *exactly*, and the retain MSE is at
    its own minimum at the same point. The updated model starts bit-identical to
    the frozen one, so the whole objective is at a stationary point at step 0 and
    the only thing that moves the weights is bf16 rounding noise — which AdamW,
    normalising by its own second moment, amplifies into a full ``lr``-sized step
    in an arbitrary direction. Upstream has this property too (it applies no
    jitter anywhere; ``unlearning/unlearn.py`` loads two plain copies of the same
    checkpoint), which is why ``init_jitter = 0.0`` is the default and is an
    **exact** no-op: it returns immediately, touches no parameter and draws no
    random number, so the published behaviour is bit-for-bit reachable.

    Determinism: the noise comes from a **dedicated CPU generator seeded from the
    config's ``seed``**, not from the global stream ``set_seeds`` primes. Two runs
    of the same config therefore get the same jitter even though model loading,
    W&B init and corpus loading sit between ``set_seeds`` and here and may consume
    the global stream by differing amounts (and the draw is CPU-side, so it does
    not vary with the device placement either). Parameters are visited in sorted
    name order for the same reason.

    ``p.std()`` is taken in float32 over the whole tensor; the noise is drawn in
    float32 and cast to the parameter's dtype, so a bf16 weight is jittered by a
    correctly-scaled bf16 delta rather than by a rounded-to-zero one.

    Returns ``{param_name: {"weight_std", "delta_norm", "rel_delta"}}`` — empty
    when ``init_jitter`` is 0.0 — for ``metrics.json``.
    """
    if not init_jitter:
        return {}
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    info = {}
    with torch.no_grad():
        for pname in sorted(params):
            p = params[pname]
            std = float(p.detach().float().std())
            before = _norm(p.detach())
            noise = torch.randn(tuple(p.shape), generator=gen, dtype=torch.float32)
            delta = (noise * (init_jitter * std)).to(device=p.device, dtype=p.dtype)
            p.add_(delta)
            dn = _norm(delta)
            info[pname] = {"weight_std": std, "delta_norm": dn,
                           "rel_delta": (dn / before) if before > 0.0 else None}
            del noise, delta
    return info


def grad_norm(params: dict) -> float:
    """The L2 norm of the concatenated gradients of ``params``, accumulated in
    at least float32 (see :func:`_norm`). Parameters with no gradient contribute
    nothing."""
    total = 0.0
    for p in params.values():
        if p.grad is None:
            continue
        total += _norm(p.grad.detach()) ** 2
    return math.sqrt(total)


def rel_weight_delta(params: dict, reference: dict) -> float:
    """``max over params of ‖p − p_ref‖ / ‖p_ref‖`` — how far training has
    actually moved the weights from where the loop started, in float32.

    The max (rather than a mean) so one moving tensor is visible even when the
    others are frozen solid, which is exactly the diagnosis this driver needs.
    """
    worst = 0.0
    for pname, p in params.items():
        ref = reference.get(pname)
        if ref is None:
            continue
        den = _norm(ref)
        if den <= 0.0:
            continue
        worst = max(worst, _norm(p.detach() - ref) / den)
    return worst


def _topic_of(corpus: str) -> str:
    return C.CORPORA.get(corpus, {}).get("topic", "other")


def _encode(tokenizer, texts, *, max_length, device, position_ids_from_mask=False):
    enc = C.tokenize_batch(tokenizer, texts, max_length=max_length, device=device)
    enc = {k: v for k, v in enc.items() if k != "labels"}
    return C.add_position_ids(enc) if position_ids_from_mask else enc


def run(config_path, *, max_num_batches=None, sample_p=None, no_push=False,
        skip_train=False) -> dict:
    """Load the config (a YAML path or a mapping), build the projectors, run the
    KUDA loop, save and write ``metrics.json``. Returns the metrics dict."""
    config_path = C.config_file(config_path, method="kuda")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    R = resolve_kuda_config(cfg, max_num_batches=max_num_batches, sample_p=sample_p,
                            push_to_hub=(False if no_push else None))
    name = R["name"]
    log_dir = str(logs_dir(EXPERIMENT, name))
    model_dir = str(models_dir(EXPERIMENT, name))
    cache_dir = str(models_dir(EXPERIMENT, NULLSPACE_CACHE))
    print(f"[kuda] config {config_path} -> logs {log_dir}, models {model_dir}", flush=True)
    ns_cfg = R["nullspace"]
    jitter_note = ("" if R["init_jitter"]
                   else " (0 = upstream; the forget objective starts stationary)")
    print(f"[kuda] plan: {R['model']} · beta {R['beta']:g} · tau {R['tau']:g} · "
          f"alpha {R['alpha']} · lr {R['lr']:g} · batch_size {R['batch_size']} · "
          f"max_num_batches {R['max_num_batches']} · epoch {R['epoch']} · "
          f"lr_schedule {R['lr_schedule']} (schedule_epochs {R['schedule_epochs']}) · "
          f"weight_decay {R['weight_decay']:g} · "
          f"layer_id {R['layer_id']} · layer_ids {R['layer_ids']} · seed {R['seed']} · "
          f"init_jitter {R['init_jitter']:g}{jitter_note} · "
          f"cov_dtype {ns_cfg['cov_dtype']} · padding_side {R['padding_side']} · "
          f"truncation_side {R['truncation_side']} · forget_max_length "
          f"{R['forget_max_length']} "
          + ("(budget.forget_max_length, the equal token budget) · " if R.get("budget")
             else ("(768 for BOTH topics, as upstream WMDP runs) · "
                   if R['forget_max_length'][0] == FORGET_MAX_LENGTH
                   else "(kuda.forget_max_length override) · "))
          + f"retain_max_length {R['retain_max_length']} · "
          f"cycle_retain {R['cycle_retain']} · "
          f"sample_p {ns_cfg['sample_p']:g}%"
          f"{'' if ns_cfg['sample_p_is_upstream_faithful'] else ' (NOT upstream-faithful)'}",
          flush=True)
    # Repeated here, after the plan, so it is the last thing on screen before the
    # two multi-GB model loads rather than scrolled off by them.
    note = sample_p_divergence_warning(ns_cfg["sample_p"])
    if note:
        print(note, flush=True)
    try:
        shutil.copyfile(config_path, os.path.join(log_dir, "config.yaml"))
    except Exception as exc:
        print(f"[kuda] could not copy the config ({type(exc).__name__}: {exc})", flush=True)

    set_seeds(R["seed"])
    wb = C.init_wandb(cfg, name=name, out_dir=log_dir, config=R)
    t_start = time.time()

    # -- corpora first: a gated-dataset failure must not cost two 7B downloads --
    n_topics = len(R["forget_corpora"])
    cap_rows = R["batch_size"] * R["max_num_batches"]
    # A chunked (MUSE) corpus is windowed with the tokenizer before the models
    # load; the windows are exactly forget/retain_max_length long, so the
    # per-step tokenize_batch stacks them without truncation or a second BOS.
    chunked = any(C.is_chunked_corpus(c)
                  for c in R["forget_corpora"] + R["retain_corpora"])
    tok_for_windows = (C.load_tokenizer(R["tokenizer"] or R["model"],
                                        padding_side=R["padding_side"],
                                        truncation_side=R["truncation_side"])
                       if chunked else None)
    if chunked and len(set(R["forget_max_length"])) != 1:
        raise RuntimeError("chunked forget corpora need one forget_max_length")
    forget_batches = C.load_text_batches(R["forget_corpora"], batch_size=R["batch_size"],
                                         min_len=R["min_len"],
                                         max_num_batches=R["max_num_batches"],
                                         max_rows=cap_rows, tokenizer=tok_for_windows,
                                         chunk_max_len=R["forget_max_length"][0])
    budget = R.get("budget")
    # Under a budget the retained side of the loss is the packed stream (built
    # once the tokenizer is loaded); the document batches are the no-budget path.
    retain_batches = ([[] for _ in R["retain_corpora"]] if budget else
                      C.load_text_batches(R["retain_corpora"], batch_size=R["batch_size"],
                                          min_len=R["min_len"],
                                          max_num_batches=(None if R["cycle_retain"]
                                                           else R["max_num_batches"]),
                                          max_rows=(None if R["cycle_retain"] else cap_rows),
                                          tokenizer=tok_for_windows,
                                          chunk_max_len=R["retain_max_length"]))
    num_batches = min([R["max_num_batches"]] + [len(b) for b in forget_batches]
                      + ([] if (budget or R["cycle_retain"]) else [len(b) for b in retain_batches]))
    if R["cycle_retain"] and not budget and any(len(b) == 0 for b in retain_batches):
        raise RuntimeError("cycle_retain is set but a retain corpus yielded no batches")
    if num_batches < 1:
        raise RuntimeError("no batches: at least one corpus yielded fewer texts than "
                           "one batch_size after the min_len filter")
    if budget and num_batches * R["epoch"] != budget["steps"]:
        raise RuntimeError(f"the schedule has {num_batches} batches x epoch {R['epoch']} = "
                           f"{num_batches * R['epoch']} steps but budget.steps is "
                           f"{budget['steps']}")
    print(f"[kuda] {n_topics} topic(s), {num_batches} steps "
          f"(max_num_batches {R['max_num_batches']}, batch_size {R['batch_size']})",
          flush=True)

    # -- the frozen/updated pair ------------------------------------------------
    print(f"[kuda] loading {R['model']} (updated + frozen, {R['dtype']})", flush=True)
    updated, tokenizer = C.load_causal_lm(R["model"], dtype=R["dtype"],
                                          device_map=R["device_map"],
                                          padding_side=R["padding_side"],
                                          truncation_side=R["truncation_side"],
                                          tokenizer_id=R["tokenizer"])
    frozen, _ = C.load_causal_lm(R["model"], dtype=R["dtype"],
                                 device_map=R["device_map"],
                                 padding_side=R["padding_side"],
                                 truncation_side=R["truncation_side"],
                                 tokenizer_id=R["tokenizer"])
    print(f"[kuda] tokenizer padding_side="
          f"{getattr(tokenizer, 'padding_side', None)!r} truncation_side="
          f"{getattr(tokenizer, 'truncation_side', None)!r} "
          f"(upstream forces {TOKENIZER_PADDING_SIDE!r}/{TOKENIZER_TRUNCATION_SIDE!r}); "
          f"the deviation cosine and the retain MSE both average over pad "
          f"positions, so these are part of the objective", flush=True)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    for p in updated.parameters():
        p.requires_grad_(False)

    # -- equal token budget: the packed retained stream + the unique counts -----
    retain_plan, target_unique, exposure = None, None, C.Exposure()
    pad_id = (tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None
              else tokenizer.eos_token_id)
    if budget:
        retain_plan = C.budget_retain_plan(tokenizer, R["retain_corpora"][0], budget=budget,
                                           min_len=R["min_len"])
        _n_t, target_unique = C.target_tokens_unique(
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

    # -- held-out CE readout (shared estimator, common.HeldoutReadout) ---------
    # Scored BEFORE init_jitter: the base is the checkpoint as loaded.
    readout = C.load_heldout_readout(cfg, tokenizer)
    ce_base = readout.score(updated) if readout is not None else None
    if ce_base is not None:
        print("[heldout] base " + " ".join(f"{k} {v:.4f}" for k, v in ce_base.items())
              + f" ({readout.rows} rows/corpus, max_length {readout.max_length})", flush=True)

    params, name_to_layer, layouts = {}, {}, {}
    for lid in R["layer_ids"]:
        named = C.get_named_block_params(updated, [lid], [f"{R['module_name']}.weight"])
        params.update(named)
        layout = C.weight_layout(C.get_mlp_out_module(updated, lid, R["module_name"]))
        for pname in named:
            name_to_layer[pname] = lid
            layouts[pname] = layout
    for p in params.values():
        p.requires_grad_(True)
    print(f"[kuda] target parameters ({len(params)}, {R['module_name']}.weight, layout "
          f"{sorted(set(layouts.values()))}): {sorted(params)}", flush=True)

    # -- escape the stationary point (opt-in; 0.0 is an exact no-op) ------------
    # Before the optimizer exists, so AdamW never carries state describing the
    # pre-jitter weights.
    jitter_info = apply_init_jitter(params, R["init_jitter"], seed=R["seed"])
    if jitter_info:
        worst = max((i["rel_delta"] or 0.0) for i in jitter_info.values())
        print(f"[kuda] init_jitter {R['init_jitter']:g} applied to {len(jitter_info)} "
              f"target parameter(s) (seed {R['seed']}); max relative weight "
              f"perturbation {worst:.3e}", flush=True)

    # -- relaxed null-space projectors (cached) ---------------------------------
    projectors_by_layer, ns_info = load_or_build_projectors(
        frozen, tokenizer, R, cache_dir=cache_dir)
    # Park each P beside its parameter once (a cached one arrives on the CPU, and
    # a d^2 host->device copy per step per layer would dwarf the step itself).
    projectors = {}
    for pname, lid in name_to_layer.items():
        w = params[pname]
        projectors[pname] = projectors_by_layer[lid].to(device=w.device, dtype=w.dtype)
    projectors_by_layer.clear()

    upd_module = C.get_block_module(updated, R["layer_id"])
    frz_module = C.get_block_module(frozen, R["layer_id"])
    optimizer = torch.optim.AdamW(list(params.values()), lr=R["lr"],
                                  weight_decay=R["weight_decay"])
    device = next(updated.parameters()).device
    # The schedule horizon: schedule_epochs passes over the per-epoch step count,
    # which is >= the run's own length (the resolver enforces it).
    lr_total_steps = R["schedule_epochs"] * num_batches
    lr_last = None

    history = []
    stationary_at_step0 = None
    if not skip_train:
        updated.train()
        # The yardstick for rel_weight_delta: where the loop starts, i.e. AFTER
        # any init_jitter (whose own displacement is reported separately), so
        # this measures what TRAINING moved.
        reference = {pname: p.detach().clone() for pname, p in params.items()}
        step = 0
        for epoch in range(R["epoch"]):
            for idx in range(num_batches):
                topic_idx = idx % n_topics
                batch_idx = idx // n_topics
                f_corpus = R["forget_corpora"][topic_idx]
                r_corpus = R["retain_corpora"][topic_idx]
                f_texts = forget_batches[topic_idx][batch_idx]
                if retain_plan is not None:
                    r_texts = None
                else:
                    r_list = retain_batches[topic_idx]
                    # cycle_retain: upstream MUSE's `retain[index % len(retain)]`.
                    r_texts = r_list[batch_idx % len(r_list) if R["cycle_retain"] else batch_idx]

                enc_f = _encode(tokenizer, f_texts,
                                max_length=R["forget_max_length"][topic_idx],
                                device=device,
                                position_ids_from_mask=R["position_ids_from_mask"])
                if retain_plan is not None:
                    a, b = retain_plan["ranges"][step]
                    enc_r = C.windows_to_batch(retain_plan["blocks"][a:b], pad_token_id=pad_id,
                                               padding_side=R["padding_side"], device=device)
                    if R["position_ids_from_mask"]:
                        C.add_position_ids(enc_r)
                    n_sep = sum(retain_plan["block_n_sep"][a:b])
                else:
                    enc_r = _encode(tokenizer, r_texts,
                                    max_length=R["retain_max_length"], device=device,
                                    position_ids_from_mask=R["position_ids_from_mask"])
                    n_sep = 0
                exposure.add_target(enc_f["attention_mask"])
                exposure.add_retain(enc_r["attention_mask"], n_sep)

                f_upd = forward_with_cache(updated, enc_f, upd_module, no_grad=False)
                f_frz = forward_with_cache(frozen, enc_f, frz_module, no_grad=True)
                u_loss, cos_mean = deviation_loss(f_upd, f_frz.to(f_upd.device), R["beta"])

                r_upd = forward_with_cache(updated, enc_r, upd_module, no_grad=False)
                r_frz = forward_with_cache(frozen, enc_r, frz_module, no_grad=True)
                rt_loss = retain_loss(r_upd, r_frz.to(r_upd.device), R["alpha"][topic_idx])

                loss = u_loss + rt_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                g_raw = grad_norm(params)                 # BEFORE the projection
                n_proj = project_gradients(params, projectors, layouts)
                g_proj = grad_norm(params)                # AFTER it
                if step == 0:
                    stationary_at_step0 = bool(g_raw < STATIONARY_GRAD_NORM)
                    if stationary_at_step0:
                        print("[kuda] WARNING: the forget objective is at its "
                              "stationary point (updated == frozen => grad cos = 0). "
                              "Any motion from here is bf16 rounding noise, not "
                              "signal. Set kuda.init_jitter > 0 to escape "
                              "deterministically.", flush=True)
                lr_now = lr_at_step(step, base_lr=R["lr"], total_steps=lr_total_steps,
                                    schedule=R["lr_schedule"])
                for g in optimizer.param_groups:
                    g["lr"] = lr_now
                lr_last = lr_now
                optimizer.step()
                exposure.step()
                d_w = rel_weight_delta(params, reference)  # AFTER the step

                rec = {"step": step, "epoch": epoch, "topic": _topic_of(f_corpus),
                       "topic_idx": topic_idx,
                       "loss": float(loss.detach().float()),
                       "unlearn_loss": float(u_loss.detach().float()),
                       "retain_loss": float(rt_loss.detach().float()),
                       "cos": cos_mean, "n_projected": n_proj,
                       "grad_norm": g_raw, "grad_norm_projected": g_proj,
                       "rel_weight_delta": d_w, "lr": lr_now}
                history.append(rec)
                print(f"[step {step + 1}/{num_batches * R['epoch']}] "
                      f"topic={rec['topic']} lr={lr_now:.3e} loss={rec['loss']:.4f} "
                      f"unlearn={rec['unlearn_loss']:.4f} retain={rec['retain_loss']:.4f} "
                      f"cos={rec['cos']:+.4f} grad_norm={g_raw:.3e} "
                      f"grad_norm_projected={g_proj:.3e} rel_weight_delta={d_w:.3e}",
                      flush=True)
                wb.log({"train/loss": rec["loss"], "train/unlearn_loss": rec["unlearn_loss"],
                        "train/retain_loss": rec["retain_loss"], "train/cos": rec["cos"],
                        "train/grad_norm": g_raw, "train/grad_norm_projected": g_proj,
                        "train/rel_weight_delta": d_w, "train/lr": lr_now,
                        "train/topic_idx": topic_idx, "train/epoch": epoch}, step=step)
                step += 1
        reference.clear()
        updated.eval()

    ce_edit = readout.score(updated) if readout is not None else None
    if ce_edit is not None:
        C.print_heldout(ce_base, ce_edit)

    budget_record = None
    if budget:
        # the covariance pass reads the retained set outside the loss; n_tokens
        # is in every tier of the cache, so a hit still reports it
        exposure.add_retain_covariance(
            max([int(info.get("n_tokens") or 0) for info in ns_info.values()], default=0))
        budget_record = exposure.record(
            budget=budget, target_unique=target_unique,
            retain_unique=retain_plan["stream"]["n_doc_tokens"],
            retain_sep=retain_plan["stream"]["n_sep_tokens"])
        print(C.Exposure.line(budget_record), flush=True)

    # -- save, push, record -----------------------------------------------------
    C.save_model(updated, tokenizer, model_dir)
    repo_name = R["hub_repo"] or default_repo_name(R["model"])
    push = C.push_model(updated, tokenizer, repo_name, private=R["hub_private"],
                        org=R["hub_org"], enabled=R["push_to_hub"])

    with open(config_path, "rb") as f:
        config_sha = hashlib.sha256(f.read()).hexdigest()
    first = history[0] if history else {}
    last = history[-1] if history else {}
    metrics = {
        "name": name,
        "method": "kuda",
        "config_path": os.path.abspath(config_path),
        "config_sha256": config_sha,
        "resolved": R,
        "target_parameters": sorted(params),
        "weight_layouts": dict(sorted(layouts.items())),
        "num_batches": num_batches,
        "steps": len(history),
        "budget": budget_record,
        "retain_plan": retain_plan["schedule"] if retain_plan is not None else None,
        "wall_seconds": round(time.time() - t_start, 1),
        "nullspace": {str(lid): info for lid, info in ns_info.items()},
        "nullspace_dirs": {str(lid): info.get("n_dirs") for lid, info in ns_info.items()},
        "svd_paths": {str(lid): info.get("svd_path") for lid, info in ns_info.items()},
        "cov_keys": {str(lid): info.get("cov_key") for lid, info in ns_info.items()},
        "proj_keys": {str(lid): info.get("proj_key") for lid, info in ns_info.items()},
        # The resolved fidelity knobs, inline, so the run record does not have
        # to dig them out of `resolved`.
        "cov_dtype": R["nullspace"]["cov_dtype"],
        "cov_device_resolved": resolved_ns_devices(R["nullspace"])[0],
        "svd_device_resolved": resolved_ns_devices(R["nullspace"])[1],
        "sample_p_percent": R["nullspace"]["sample_p"],
        # False => the retain selection diverged from upstream's (prefix vs
        # seeded random sample, ceil vs int) and this is not a faithful repro.
        "sample_p_is_upstream_faithful": R["nullspace"]["sample_p_is_upstream_faithful"],
        "init_jitter": R["init_jitter"],
        "init_jitter_applied": jitter_info,
        # Optimizer / schedule, inline: which schedule ran, over what horizon,
        # and where the lr ended (constant => lr; linear_decay => lr * (T - steps) / T).
        "lr": R["lr"],
        "lr_schedule": R["lr_schedule"],
        "schedule_epochs": R["schedule_epochs"],
        "epoch": R["epoch"],
        "lr_total_steps": lr_total_steps,
        "lr_last": lr_last,
        "weight_decay": R["weight_decay"],
        "tokenizer_padding_side": getattr(tokenizer, "padding_side", None),
        "tokenizer_truncation_side": getattr(tokenizer, "truncation_side", None),
        "forget_max_length": R["forget_max_length"],
        "retain_max_length": R["retain_max_length"],
        "cycle_retain": R["cycle_retain"],
        "tokenizer": R["tokenizer"] or R["model"],
        "chunked": {c: C.is_chunked_corpus(c)
                    for c in R["forget_corpora"] + R["retain_corpora"]},
        "loss_first": first.get("loss"), "loss_last": last.get("loss"),
        "unlearn_first": first.get("unlearn_loss"), "unlearn_last": last.get("unlearn_loss"),
        "retain_first": first.get("retain_loss"), "retain_last": last.get("retain_loss"),
        "cos_first": first.get("cos"), "cos_last": last.get("cos"),
        "cos_mean": (sum(h["cos"] for h in history) / len(history)) if history else None,
        # Dynamics: was there a gradient at all, did the projection keep any of
        # it, and did the weights actually move?
        "grad_norm_first": first.get("grad_norm"),
        "grad_norm_last": last.get("grad_norm"),
        "grad_norm_projected_first": first.get("grad_norm_projected"),
        "grad_norm_projected_last": last.get("grad_norm_projected"),
        "rel_weight_delta_last": last.get("rel_weight_delta"),
        "stationary_point_at_step0": stationary_at_step0,
        "history": history,
        "model_dir": model_dir,
        "log_dir": log_dir,
        "push": push,
        "wandb": getattr(wb, "resolved", None),
        "torch": torch.__version__,
        **C.heldout_metrics(readout, ce_base, ce_edit),
    }
    C.write_metrics(log_dir, metrics)
    wb.summary({**{k: metrics[k] for k in
                   ("steps", "wall_seconds", "loss_last", "unlearn_last", "retain_last",
                    "cos_last", "cos_mean", "grad_norm_first", "grad_norm_projected_first",
                    "rel_weight_delta_last") if metrics.get(k) is not None},
                **(C.Exposure.summary(budget_record) if budget_record else {}),
                **C.heldout_summary(ce_base, ce_edit)})
    wb.finish()
    if budget and not skip_train:
        # after write_metrics: the record is on disk whichever way this goes
        C.assert_budget(budget_record, budget)
    print(f"[kuda] done in {metrics['wall_seconds']}s — cos {metrics['cos_first']} -> "
          f"{metrics['cos_last']}, loss {metrics['loss_first']} -> {metrics['loss_last']}, "
          f"null-space dirs {metrics['nullspace_dirs']}, grad_norm "
          f"{metrics['grad_norm_first']} -> {metrics['grad_norm_last']}, "
          f"rel_weight_delta {metrics['rel_weight_delta_last']}", flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="KUDA unlearning on the WMDP corpora")
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-num-batches", type=int, default=None,
                    help="override kuda.max_num_batches (smoke runs)")
    ap.add_argument("--sample-p", type=float, default=None, metavar="PERCENT",
                    help="override kuda.nullspace.sample_p — the PERCENTAGE "
                         "(0 < p <= 100, upstream's --sample_p units) of each retain "
                         "corpus the covariance pass consumes; 100 = all of it. Not a "
                         "row count — cap rows with kuda.nullspace.max_cov_texts")
    ap.add_argument("--no-push", action="store_true",
                    help="do not push the checkpoint to the hub (config: push_to_hub)")
    ap.add_argument("--skip-train", action="store_true",
                    help="build/refresh the null-space projectors and exit without "
                         "training (cache warming)")
    a = ap.parse_args()
    run(a.config, max_num_batches=a.max_num_batches, sample_p=a.sample_p,
        no_push=a.no_push, skip_train=a.skip_train)


if __name__ == "__main__":
    main()
