"""The GPT-2 training loop -- nanoGPT's recipe, driving HF ``GPT2LMHeadModel``.

AdamW with decay only on 2-D parameters, cosine schedule with linear warmup,
gradient accumulation to a fixed tokens-per-step budget, bf16 autocast on
CUDA, ``torch.compile``, gradient clipping. The loop needs only
``model(input_ids).logits``, scored by :func:`lm_loss`, so the checkpoint that
is trained is bit for bit the HF object the erasure later edits.

One loop serves pretraining and the retrain-to-recover runs
(:mod:`srspd_experiments.eval.recovery`): they differ in what they are handed
(a fresh model vs. an edited checkpoint, a mixture sampler, a token budget vs. a
budget plus an early-stop predicate), not in how the optimizer is driven --
which is what makes "recovery is measured on the same instrument as
pretraining" true by construction.

Datasets
--------
Anything with ``batch(n, rng, device) -> (x, y)`` is a dataset to the loop and
to :func:`evaluate`. :class:`ShardDataset` samples uniform windows from a span;
:class:`MixtureDataset` (pretraining) and :class:`MixedShardDataset`
(recovery) cycle one micro-batch per language; :class:`FixedSpanDataset` walks
a small span end to end in fixed windows, for monitor sets smaller than
``eval_batches x micro_batch`` windows -- or shorter than one block.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch


def lm_loss(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Next-token CE of ``model`` on ``(x, y)``, where ``y`` is ``x`` shifted by one.

    Do **not** pass ``y`` as HF ``labels=``: HF's causal-LM loss shifts labels
    itself, so handing it already-shifted targets trains and scores the model on
    predicting the token *two* ahead. The loss is computed here from the logits
    so the alignment is explicit and not version-dependent.
    """
    logits = model(input_ids=x).logits
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))


@dataclass
class TrainConfig:
    """Everything the loop needs that is not the data or the model."""
    total_tokens: int
    batch_tokens: int = 2 ** 19          # tokens per optimizer step (~524k)
    micro_batch: int = 16                # sequences per forward
    block_size: int = 1024
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    warmup_tokens: int = 100_000_000
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    seed: int = 1337
    eval_every_tokens: int = 50_000_000
    eval_batches: int = 40
    compile: bool = True
    dtype: str = "bfloat16"
    log_every_steps: int = 10
    extra: dict = field(default_factory=dict)


class ShardDataset:
    """Flat ``uint16`` token stream over a span's shards, memory-mapped.

    Memory-mapped rather than loaded: the pretraining train span is ~5 GB across
    three languages, and concurrent recovery runs may read one shared copy.

    Sampling is uniform over valid start offsets with a per-call generator, so a
    given ``seed`` reproduces a given token stream exactly.
    """

    def __init__(self, shard_paths: Sequence[str], block_size: int):
        if not shard_paths:
            raise ValueError("no shards — check the corpus manifest")
        self.arrays = [np.memmap(p, dtype=np.uint16, mode="r")
                       for p in shard_paths]
        self.block_size = int(block_size)
        self.lengths = [len(a) for a in self.arrays]
        self.total = int(sum(self.lengths))
        usable = [max(0, n - block_size - 1) for n in self.lengths]
        if sum(usable) <= 0:
            raise ValueError(f"shards hold {self.total:,} tokens, too few for "
                             f"block_size {block_size}")
        w = np.array(usable, dtype=np.float64)
        self.weights = w / w.sum()
        self.usable = usable

    def batch(self, n: int, rng: np.random.Generator, device: str):
        """One (x, y) pair; ``y`` is ``x`` shifted by one, the LM objective."""
        which = rng.choice(len(self.arrays), size=n, p=self.weights)
        xs = np.empty((n, self.block_size), dtype=np.int64)
        ys = np.empty((n, self.block_size), dtype=np.int64)
        for i, k in enumerate(which):
            start = int(rng.integers(0, self.usable[k]))
            chunk = self.arrays[k][start:start + self.block_size + 1]
            xs[i] = chunk[:-1].astype(np.int64)
            ys[i] = chunk[1:].astype(np.int64)
        x = torch.from_numpy(xs).to(device, non_blocking=True)
        y = torch.from_numpy(ys).to(device, non_blocking=True)
        return x, y


class MixtureDataset:
    """Round-robin over per-language datasets -- the balanced mixture.

    The mixture is balanced by **token count**, and every language's block is
    the same size, so cycling one micro-batch per language in turn delivers
    exactly 1/3 each. Doing it deterministically rather than by weighted
    sampling means the realised mixture is the intended one at *every* prefix of
    training, not merely in expectation over the whole run — which matters
    because ``N_nld`` is read off the actual count.
    """

    def __init__(self, datasets: dict[str, ShardDataset]):
        if not datasets:
            raise ValueError("mixture needs at least one language")
        self.langs = sorted(datasets)
        self.datasets = datasets
        self._i = 0

    def batch(self, n: int, rng: np.random.Generator, device: str):
        lang = self.langs[self._i % len(self.langs)]
        self._i += 1
        return self.datasets[lang].batch(n, rng, device)

    @property
    def total(self) -> int:
        return sum(d.total for d in self.datasets.values())


class MixedShardDataset:
    """Round-robin over ``{lang: ShardDataset}`` in a **fixed, explicit order**,
    with its own sampling generator -- the retrain-to-recover stream.

    Same mixture rule as :class:`MixtureDataset` (one micro-batch per language
    in turn, so every optimizer step of ``accum`` micro-batches sees the
    languages equally when ``accum`` is a multiple of the language count; the
    recovery driver asserts that), with two differences that matter for a
    *second* pass over the pretraining span:

    * the order is the one given (``order``; default sorted), recorded so the
      stream is reproducible from the config alone;
    * the window offsets come from this dataset's own generator seeded with
      ``seed`` (when given) rather than the trainer's, so the retraining stream
      is a fresh sample of the span — not the pretraining sequence replayed —
      and is pinned by ``stream_seed`` independently of the optimizer seed.
      With ``seed=None`` the caller's generator is used, as the other datasets do.
    """

    def __init__(self, datasets: dict[str, ShardDataset], *, order=None, seed=None):
        if not datasets:
            raise ValueError("mixture needs at least one language")
        self.langs = list(order) if order else sorted(datasets)
        missing = [l for l in self.langs if l not in datasets]
        if missing or len(set(self.langs)) != len(self.langs):
            raise ValueError(f"order {self.langs} does not match datasets {sorted(datasets)}")
        self.datasets = datasets
        self.seed = seed
        self._rng = np.random.default_rng(seed) if seed is not None else None
        self._i = 0

    @property
    def n_langs(self) -> int:
        return len(self.langs)

    def next_lang(self) -> str:
        return self.langs[self._i % len(self.langs)]

    def batch(self, n: int, rng: np.random.Generator, device: str):
        lang = self.langs[self._i % len(self.langs)]
        self._i += 1
        return self.datasets[lang].batch(n, self._rng if self._rng is not None else rng,
                                         device)

    @property
    def total(self) -> int:
        return sum(d.total for d in self.datasets.values())


class FixedSpanDataset:
    """Every window of a small span, in order — a monitor set read end to end.

    Built for the unlearning sets (a 1,000-token forget set, 200k-token held-out
    sets, 1M-token contrastive sets): ``evaluate`` should read *all* of such a
    set once, deterministically, rather than ``eval_batches`` random windows of
    it. The span is cut into consecutive non-overlapping windows of
    ``block_size + 1`` tokens (stride ``block_size``); a trailing partial window
    is dropped when at least one full window exists. A span **shorter than one
    block** (the 1,000-token Dutch target against ``block_size`` 1024 is at most
    one block) is a single window of its full length — no padding, no crash.
    ``max_tokens`` restricts the read to a prefix of the concatenated shards
    (the pretraining train-sample monitor: a fixed slice of the first shard).

    ``batch`` hands out the next ``n`` windows cyclically; :attr:`eval_batches`
    is how many calls cover the set once, which :func:`evaluate` honours.
    """

    def __init__(self, shard_paths: Sequence[str], block_size: int, *,
                 max_tokens: int | None = None, micro_batch: int = 16):
        if not shard_paths:
            raise ValueError("no shards — check the set manifest")
        parts = []
        remaining = int(max_tokens) if max_tokens else None
        for p in shard_paths:
            a = np.memmap(p, dtype=np.uint16, mode="r")
            if remaining is not None:
                a = a[:remaining]
                remaining -= len(a)
            parts.append(np.asarray(a))
            if remaining is not None and remaining <= 0:
                break
        self.tokens = np.concatenate(parts) if len(parts) > 1 else parts[0]
        self.total = int(len(self.tokens))
        if self.total < 2:
            raise ValueError(f"span holds {self.total} tokens; need at least 2")
        self.block_size = int(block_size)
        n_full = (self.total - 1) // self.block_size
        if n_full >= 1:
            self.windows = [(k * self.block_size, k * self.block_size + self.block_size + 1)
                            for k in range(n_full)]
        else:
            self.windows = [(0, self.total)]
        self.window_tokens = [hi - lo - 1 for lo, hi in self.windows]
        self.micro_batch = int(micro_batch)
        self.eval_batches = math.ceil(len(self.windows) / self.micro_batch)
        self._i = 0

    def reset(self) -> None:
        self._i = 0

    def batch(self, n: int, rng=None, device: str = "cpu"):
        """The next ``n`` windows (fewer at the end of a pass). Windows in one
        call are equal-length by construction except the single short-span
        case, which is one window."""
        idx = [(self._i + k) % len(self.windows) for k in range(n)]
        idx = idx[:len(self.windows)]
        # Never wrap within a call: the last batch of a pass is simply shorter.
        cut = len(idx)
        for k in range(1, len(idx)):
            if idx[k] < idx[k - 1]:
                cut = k
                break
        idx = idx[:cut]
        self._i = (self._i + len(idx)) % len(self.windows)
        lo0, hi0 = self.windows[idx[0]]
        width = hi0 - lo0
        xs = np.empty((len(idx), width - 1), dtype=np.int64)
        ys = np.empty((len(idx), width - 1), dtype=np.int64)
        for row, k in enumerate(idx):
            lo, hi = self.windows[k]
            chunk = self.tokens[lo:hi]
            xs[row] = chunk[:-1].astype(np.int64)
            ys[row] = chunk[1:].astype(np.int64)
        x = torch.from_numpy(xs).to(device, non_blocking=True)
        y = torch.from_numpy(ys).to(device, non_blocking=True)
        return x, y


def configure_optimizer(model, weight_decay: float, lr: float, betas):
    """Decay 2-D parameters only — nanoGPT's rule.

    Weights that participate in matmuls and embeddings get decay; biases and
    LayerNorm gains do not. Decaying a LayerNorm gain pulls it toward zero,
    which is a scale change rather than a regularizer.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    return torch.optim.AdamW(groups, lr=lr, betas=betas,
                             **({"fused": True} if fused else {}))


def lr_at(tokens: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay to ``min_lr_frac * lr``."""
    if tokens < cfg.warmup_tokens:
        return cfg.lr * (tokens + 1) / max(1, cfg.warmup_tokens)
    span = max(1, cfg.total_tokens - cfg.warmup_tokens)
    frac = min(1.0, (tokens - cfg.warmup_tokens) / span)
    coeff = 0.5 * (1.0 + math.cos(math.pi * frac))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * coeff)


@torch.no_grad()
def evaluate(model, datasets: dict, cfg: TrainConfig,
             device: str, seed: int = 0) -> dict[str, float]:
    """Mean per-token CE on a fixed slice of each named dataset: ``{name: ce}``.

    The names are the caller's (``eng`` / ``deu`` / ``nld`` for the val spans;
    ``heldout_eng``, ``forget_nld``, ... for the recovery monitors) — nothing
    here assumes they are languages. ``seed`` is fixed and independent of
    training so every evaluation reads the *same* token windows — the
    comparison across steps, and later across conditions, is only meaningful if
    the measuring stick does not move. A dataset that knows
    how many batches cover it (:class:`FixedSpanDataset`'s ``eval_batches``)
    is read that many times from its start; the others get
    ``cfg.eval_batches`` windows of ``cfg.micro_batch``. The mean is
    token-weighted, which equals the plain batch mean when every batch has the
    same shape (the val spans) and is the honest number when the last batch of
    a fixed span is shorter.
    """
    was_training = model.training
    model.eval()
    out = {}
    for name, ds in sorted(datasets.items()):
        rng = np.random.default_rng(seed)
        n_batches = getattr(ds, "eval_batches", None) or cfg.eval_batches
        if hasattr(ds, "reset"):
            ds.reset()
        loss_sum = tok_sum = 0.0
        for _ in range(n_batches):
            x, y = ds.batch(cfg.micro_batch, rng, device)
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                dtype=getattr(torch, cfg.dtype),
                                enabled=("cuda" in device)):
                loss = lm_loss(model, x, y).item()
            loss_sum += loss * y.numel()
            tok_sum += y.numel()
        out[name] = float(loss_sum / tok_sum)
    if was_training:
        model.train()
    return out


def train(model, train_data, val_datasets: dict,
          cfg: TrainConfig, device: str, *, csv_path: str,
          csv_extra_cols: Sequence[str] = (),
          on_eval: Callable[[int, dict], bool] | None = None,
          log_hook: Callable[[int, int, float, float], None] | None = None,
          log=print) -> dict:
    """Run the loop. Returns a summary dict; writes a CSV row per evaluation.

    ``train_data`` is any dataset with ``batch(n, rng, device)`` (a
    :class:`ShardDataset`, a :class:`MixtureDataset`, a
    :class:`MixedShardDataset`); ``val_datasets`` is ``{name: dataset}`` and
    the CSV columns ``ce_val_<name>`` / ``ppl_val_<name>`` follow the names given.

    ``on_eval(tokens, ce_by_name) -> stop?`` is the early-exit hook. Recovery
    uses it to stop the moment the target-language CE comes back to the
    criterion, which keeps the expected spend far below the worst-case cap.
    Pretraining passes ``None`` and runs its full budget.

    ``log_hook(step, tokens, lr, train_ce)`` is called every
    ``cfg.log_every_steps`` optimizer steps (the same cadence as the printed
    step line), for a caller that streams the training CE to a tracker. It must
    not raise; the loop does not guard it.
    """
    import csv

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed % (2 ** 32))
    rng = np.random.default_rng(cfg.seed)

    model.to(device)
    model.train()
    # use_cache is an inference feature; leaving it on wastes memory per step.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    optimizer = configure_optimizer(model, cfg.weight_decay, cfg.lr, cfg.betas)
    step_model = model
    if cfg.compile and hasattr(torch, "compile") and "cuda" in device:
        log("  [train] torch.compile ...")
        step_model = torch.compile(model)

    accum = max(1, cfg.batch_tokens // (cfg.micro_batch * cfg.block_size))
    tokens_per_step = accum * cfg.micro_batch * cfg.block_size
    log(f"  [train] {cfg.total_tokens:,} tokens, {tokens_per_step:,}/step, "
        f"accum {accum}, micro {cfg.micro_batch}x{cfg.block_size}")

    fields = (["step", "tokens", "lr", "ce_train", "wall_clock_s"]
              + [f"ce_val_{l}" for l in sorted(val_datasets)]
              + [f"ppl_val_{l}" for l in sorted(val_datasets)]
              + list(csv_extra_cols))
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    writer.writeheader()

    t0 = time.time()
    tokens = step = 0
    next_eval = 0
    last_train_ce = float("nan")
    stopped_early = False
    history: list[dict] = []

    while tokens < cfg.total_tokens:
        if tokens >= next_eval:
            ce = evaluate(model, val_datasets, cfg, device)
            row = {"step": step, "tokens": tokens, "lr": lr_at(tokens, cfg),
                   "ce_train": last_train_ce,
                   "wall_clock_s": round(time.time() - t0, 2)}
            for l, v in ce.items():
                row[f"ce_val_{l}"] = round(v, 6)
                # exp() overflows to inf for a CE above ~709; an untrained model
                # is nowhere near that, but a diverged one could be.
                row[f"ppl_val_{l}"] = round(math.exp(min(v, 700)), 4)
            writer.writerow(row)
            fh.flush()
            history.append(row)
            log(f"  [eval] {tokens:,} tok  " +
                "  ".join(f"{l}={v:.4f}" for l, v in sorted(ce.items())))
            if on_eval is not None and on_eval(tokens, ce):
                log(f"  [train] early-stop criterion met at {tokens:,} tokens")
                stopped_early = True
                break
            next_eval = tokens + cfg.eval_every_tokens

        lr = lr_at(tokens, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accum):
            x, y = train_data.batch(cfg.micro_batch, rng, device)
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                                dtype=getattr(torch, cfg.dtype),
                                enabled=("cuda" in device)):
                loss = lm_loss(step_model, x, y) / accum
            loss.backward()
            total_loss += loss.item()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        last_train_ce = total_loss
        tokens += tokens_per_step
        step += 1
        if step % cfg.log_every_steps == 0:
            el = time.time() - t0
            log(f"  [step {step}] {tokens:,} tok  ce={total_loss:.4f}  "
                f"lr={lr:.2e}  {tokens/max(el,1e-9):,.0f} tok/s")
            if log_hook is not None:
                log_hook(step, tokens, lr, total_loss)

    # Final evaluation, so the last row always reflects the shipped weights.
    ce = evaluate(model, val_datasets, cfg, device)
    row = {"step": step, "tokens": tokens, "lr": lr_at(tokens, cfg),
           "ce_train": last_train_ce, "wall_clock_s": round(time.time() - t0, 2)}
    for l, v in ce.items():
        row[f"ce_val_{l}"] = round(v, 6)
        row[f"ppl_val_{l}"] = round(math.exp(min(v, 700)), 4)
    writer.writerow(row)
    fh.close()
    history.append(row)

    elapsed = time.time() - t0
    log(f"  [train] {tokens:,} tokens in {elapsed/60:.1f} min "
        f"({tokens/max(elapsed,1e-9):,.0f} tok/s)")
    return {"tokens": tokens, "steps": step, "elapsed_s": elapsed,
            "final_ce": ce, "stopped_early": stopped_early,
            "tokens_per_step": tokens_per_step, "history": history}


def save_run_json(path: str, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
