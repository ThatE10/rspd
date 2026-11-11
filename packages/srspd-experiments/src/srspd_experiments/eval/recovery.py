"""Retrain-to-recover: how many tokens of continued pretraining an edited GPT-2
needs before the erased language comes back.

    from srspd_experiments.eval import recovery
    manifest = recovery.run({"name": "recover_srspd_mu10", "corpus_name": "gpt2_corpus",
                             "checkpoint": "models/.../srspd_mu10",
                             "base_checkpoint": "models/.../checkpoint",
                             "recovery": {"ce_nld_original": 3.5437, ...}})

One edited checkpoint (one ``condition``) per call; the un-edited model is the
``base`` control. Nothing is read from the erasure except the checkpoint.

* **The retraining stream is the pretraining mixture.** ``stream: train``
  reads the corpus ``train`` span of every language in ``stream_languages``
  (default all), one micro-batch per language in a fixed round-robin
  (:class:`trainer.MixedShardDataset`), window offsets drawn from
  ``stream_seed`` (default ``seed + 1``) -- a fresh sample of the span, not the
  pretraining order replayed. ``stream: recovery`` is the Dutch-only protocol:
  fresh Dutch from the ``recovery`` span, disjoint from train and val.
* **The cap may exceed ``N_nld``** (the original Dutch exposure, read from the
  corpus manifest, never from the config). ``cap_tokens`` defaults to
  ``N_nld``; the manifest carries ``ratio_can_exceed_one`` / ``cap_over_n_nld``.
  A run that never meets the criterion is ``censored`` with ``M = cap``, and its
  ratio ``M / N_nld`` is a **lower bound**, never an extrapolation.
* **Two tolerances, one stop rule.** The criterion is target val CE
  ``<= ce_nld_original * (1 + tolerance)`` (default 1%; the published configs use
  0.0). The curve records the first crossing at tolerance 0 **and** 1%
  (``recovered_tol0`` / ``recovered_tol1pct``, ``M_tokens_tol0`` /
  ``M_tokens_tol1pct``); training stops at the first evaluation meeting the
  *configured* tolerance.
* **Monitors at every evaluation** (``eval_every_tokens``, default 25M): val CE
  per language and ``ce_retained_mean``; a fixed slice of the train span per
  language (``ce_train_<lang>``); and, from the unlearning sets
  (``monitors.unlearning_sets``: a set name under the data root or a path),
  held-out text per language, the target itself (``ce_forget_nld``) and the
  contrastive sets (``ce_contrastive_<lang>``). The sets' shards must exist
  (:func:`srspd_experiments.data.gpt2_unlearning.materialize_shards`).
* **Weight delta at t = 0.** With ``base_checkpoint`` set,
  :func:`srspd.report.weight_delta_report` records how much every 2-D matrix
  moved (headline ``grad_mass = sum_i ||W_i^edit - W_i^base||_F^2``);
  ``weight_delta.{json,csv}`` land beside the curve.

The loop is :func:`trainer.train`, the one pretraining used, so the recovery
curve and the pretraining curve come from the same optimizer and estimator.
Tracking (:mod:`.tracking`) is off unless ``tracking.enabled``.

Outputs (``out_dir``): ``recovery_curve.csv``, ``recovery_summary.csv``,
``recovery_manifest.json``, ``train_ce.csv``, ``weight_delta.{json,csv}``,
``results.md`` and ``results.json`` (:func:`result_rows`, the common schema).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time

from ..results import ResultRow, write_results

#: Success is target-language CE within this relative margin of the original
#: model's. 1% is the default; the published recovery configs set 0.0 ("same
#: or better").
DEFAULT_TOLERANCE = 0.01
#: The second tolerance every curve records beside the configured one.
REPORTED_TOLERANCES = {"tol0": 0.0, "tol1pct": 0.01}

DEFAULT_CAP_TOKENS = None                 # None -> N_nld (the original Dutch exposure)
DEFAULT_EVAL_EVERY = 25_000_000
DEFAULT_TRAIN_SAMPLE_TOKENS = 200_000
DEFAULT_TARGET_LANGUAGE = "nld"
#: ``shards/<role>_<lang>`` roles of an unlearning-set directory -> the CE
#: column stem. The target set is the forget set.
ROLE_COLUMN = {"target": "forget", "contrastive": "contrastive", "heldout": "heldout"}
STREAMS = ("train", "recovery")


CONFIG_KEYS = ("name", "condition", "corpus_name", "corpus_root", "checkpoint",
               "base_checkpoint", "recovery", "tracking")


def resolve_config(cfg: dict) -> dict:
    """Validate a recovery config (``name``, ``corpus_name`` or ``corpus_root``,
    ``checkpoint``, ``recovery: {ce_nld_original, ...}``; optional
    ``condition``, ``base_checkpoint``, ``tracking``). Returns a copy with
    ``condition`` defaulted to ``name``."""
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping, got {type(cfg).__name__}")
    unknown = set(cfg) - set(CONFIG_KEYS)
    if unknown:
        raise ValueError(f"recovery config has unknown keys {sorted(unknown)}; known: "
                         f"{CONFIG_KEYS}")
    for key in ("name", "checkpoint", "recovery"):
        if key not in cfg:
            raise ValueError(f"recovery config needs '{key}'")
    if not cfg.get("corpus_name") and not cfg.get("corpus_root"):
        raise ValueError("recovery config needs 'corpus_name' (a gpt2_corpus build under the "
                         "data root) or 'corpus_root' (its corpus directory)")
    cfg = dict(cfg)
    rec = dict(cfg["recovery"] or {})
    if "ce_nld_original" not in rec:
        raise ValueError("recovery needs 'ce_nld_original' (the unedited model's target-language "
                         "val CE, measured by the same evaluate())")
    stream = str(rec.get("stream", "train"))
    if stream not in STREAMS:
        raise ValueError(f"recovery.stream must be one of {STREAMS}, got {stream!r}")
    cfg["recovery"] = rec
    cfg.setdefault("condition", cfg["name"])
    return cfg


def resolve_plan(cfg: dict, man: dict) -> dict:
    """The run's scalar facts from config + corpus manifest (pure; tested)."""
    rec = cfg["recovery"]
    target_lang = str(rec.get("target_language", DEFAULT_TARGET_LANGUAGE))
    n_nld = int(man["n_tokens_by_language"][target_lang])
    cap = int(rec.get("cap_tokens") or n_nld)
    langs = sorted(man["languages"])
    stream = str(rec.get("stream", "train"))
    stream_langs = list(rec.get("stream_languages") or langs) if stream == "train" \
        else [target_lang]
    unknown = [l for l in stream_langs if l not in langs]
    if unknown:
        raise ValueError(f"stream_languages {unknown} not in the corpus {langs}")
    tol = float(rec.get("tolerance", DEFAULT_TOLERANCE))
    ce_original = float(rec["ce_nld_original"])
    mon = rec.get("monitors") or {}
    return {"target_language": target_lang, "N_nld": n_nld, "cap_tokens": cap,
            "ratio_can_exceed_one": cap > n_nld, "cap_over_n_nld": cap / n_nld,
            "languages": langs, "retained_languages": [l for l in langs if l != target_lang],
            "stream": stream, "stream_languages": stream_langs,
            # Default: the optimizer seed + 1, so the stream is never the
            # pretraining sampler's sequence replayed.
            "stream_seed": int(rec["stream_seed"]) if rec.get("stream_seed") is not None
            else int(rec.get("seed", 1337)) + 1,
            "tolerance": tol, "ce_nld_original": ce_original,
            "ce_target": ce_original * (1.0 + tol),
            "ce_targets": {k: ce_original * (1.0 + v) for k, v in REPORTED_TOLERANCES.items()},
            "eval_every_tokens": int(rec.get("eval_every_tokens", DEFAULT_EVAL_EVERY)),
            "condition": str(cfg.get("condition", cfg["name"])),
            "unlearning_sets": mon.get("unlearning_sets"),
            "train_sample_tokens": int(mon.get("train_sample_tokens", DEFAULT_TRAIN_SAMPLE_TOKENS))}


def curve_fields(langs, retained, monitor_names) -> list[str]:
    """Column order of ``recovery_curve.csv`` from the sets that exist."""
    cols = ["tokens", "condition"]
    cols += [f"ce_val_{l}" for l in langs] + [f"ppl_val_{l}" for l in langs]
    cols += ["ce_retained_mean"]
    for stem in ("train", "heldout"):
        names = [n for n in monitor_names if n.startswith(stem + "_")]
        if names:
            cols += [f"ce_{n}" for n in names] + [f"ce_{stem}_mean"]
    cols += [f"ce_{n}" for n in monitor_names
             if n.startswith("forget_") or n.startswith("contrastive_")]
    cols += ["ce_nld_target", "recovered", "recovered_tol0", "recovered_tol1pct"]
    return cols


def curve_row(tokens, ce_val, ce_mon, plan) -> dict:
    """One ``recovery_curve.csv`` row (pure; tested)."""
    tl = plan["target_language"]
    row = {"tokens": int(tokens), "condition": plan["condition"],
           "ce_nld_target": round(plan["ce_target"], 6)}
    for l, v in ce_val.items():
        row[f"ce_val_{l}"] = round(v, 6)
        # exp() overflows past ~709; a diverged run should not take the writer down.
        row[f"ppl_val_{l}"] = round(math.exp(min(v, 700)), 4)
    ret = [ce_val[l] for l in plan["retained_languages"] if l in ce_val]
    row["ce_retained_mean"] = round(sum(ret) / len(ret), 6) if ret else ""
    for stem in ("train", "heldout"):
        vals = [v for n, v in ce_mon.items() if n.startswith(stem + "_")]
        if vals:
            row[f"ce_{stem}_mean"] = round(sum(vals) / len(vals), 6)
    for n, v in ce_mon.items():
        row[f"ce_{n}"] = round(v, 6)
    ce_t = ce_val[tl]
    row["recovered"] = int(ce_t <= plan["ce_target"])
    for k, t in plan["ce_targets"].items():
        row[f"recovered_{k}"] = int(ce_t <= t)
    return row


def _tracking_payload(row: dict) -> dict:
    return {k: v for k, v in row.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "tokens"}


def run(cfg: dict, *, root=None, out_dir=None, config_path: str | None = None,
        log=print) -> dict:
    """Retrain ``cfg["checkpoint"]`` until the target-language CE recovers (see
    the module docstring) and write the outputs under ``out_dir`` (default
    ``<data root>/logs/recovery/<name>``). Returns the recovery manifest."""
    import torch
    from transformers import AutoTokenizer, GPT2LMHeadModel

    from srspd import report as WD

    from ..data import gpt2_corpus as pc
    from ..data import gpt2_unlearning as gu
    from ..paths import logs_dir, resolve_data_ref
    from .tracking import Tracker
    from .trainer import (FixedSpanDataset, MixedShardDataset, ShardDataset, TrainConfig,
                          evaluate, train)

    cfg = resolve_config(cfg)
    rec = cfg["recovery"]
    # Recovery writes only logs/metrics (no model is saved).
    out_dir = str(out_dir or logs_dir("recovery", cfg["name"], root=root))
    os.makedirs(out_dir, exist_ok=True)

    corpus_root = (resolve_data_ref(cfg["corpus_root"], root=root) if cfg.get("corpus_root")
                   else pc.corpus_root(cfg["corpus_name"], root=root))
    man = pc.load_manifest(corpus_root)
    # The denominator is read from the corpus manifest, never from the config:
    # a typo'd budget here would silently rescale every permanence ratio.
    plan = resolve_plan(cfg, man)
    n_nld, cap, tl = plan["N_nld"], plan["cap_tokens"], plan["target_language"]
    langs = plan["languages"]

    ckpt = resolve_data_ref(cfg["checkpoint"], root=root)
    if not os.path.isdir(ckpt):
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    base_ckpt = resolve_data_ref(cfg["base_checkpoint"], root=root) \
        if cfg.get("base_checkpoint") else None
    if base_ckpt and not os.path.isdir(base_ckpt):
        raise FileNotFoundError(f"base checkpoint not found: {base_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ckpt)   # noqa: F841 — asserts the export is complete
    model = GPT2LMHeadModel.from_pretrained(ckpt)

    print(f"[recover] {cfg['name']}  condition: {plan['condition']}")
    print(f"[recover] checkpoint: {ckpt}")
    print(f"[recover] stream: {plan['stream']} {plan['stream_languages']}  "
          f"seed {plan['stream_seed']}")
    print(f"[recover] N_nld = {n_nld:,}  cap = {cap:,}  (cap/N_nld = "
          f"{plan['cap_over_n_nld']:.3f}{', ratio may exceed 1' if plan['ratio_can_exceed_one'] else ''})")
    print(f"[recover] original {tl} CE {plan['ce_nld_original']:.4f}; "
          f"target <= {plan['ce_target']:.4f} ({plan['tolerance']:.1%} relative)")

    # ── weight delta at t = 0 ────────────────────────────────────────────────
    wd_totals = None
    if base_ckpt:
        base = GPT2LMHeadModel.from_pretrained(base_ckpt)
        wd_cfg = rec.get("weight_delta") or {}
        report = WD.weight_delta_report(base, model, erank=bool(wd_cfg.get("erank", True)))
        report["checkpoint"] = cfg["checkpoint"]
        report["base_checkpoint"] = cfg["base_checkpoint"]
        WD.write_weight_delta(report, out_dir)
        wd_totals = report["totals"]
        print(f"[recover] weight delta vs base: grad_mass={wd_totals['grad_mass']:.4e}  "
              f"global_rel={wd_totals['global_rel']:.4e}  "
              f"sum_rel={wd_totals['sum_rel']:.4e}  changed "
              f"{wd_totals['n_changed']}/{wd_totals['n_tensors']} tensors "
              f"({wd_totals['changed_param_fraction']:.3f} of params)")
        del base, report

    # ── datasets ─────────────────────────────────────────────────────────────
    block = int(rec.get("block_size", 1024))
    micro = int(rec.get("micro_batch", 16))
    val_sets = {l: ShardDataset(pc.span_shards(corpus_root, l, "val"), block)
                for l in langs}
    if plan["stream"] == "train":
        stream = MixedShardDataset(
            {l: ShardDataset(pc.span_shards(corpus_root, l, "train"), block)
             for l in plan["stream_languages"]},
            order=plan["stream_languages"], seed=plan["stream_seed"])
    else:
        stream = ShardDataset(pc.span_shards(corpus_root, tl, "recovery"), block)

    monitors: dict = {}
    if plan["train_sample_tokens"] > 0:
        for l in langs:
            monitors[f"train_{l}"] = FixedSpanDataset(
                pc.span_shards(corpus_root, l, "train")[:1], block,
                max_tokens=plan["train_sample_tokens"], micro_batch=micro)
    sets_manifest = None
    if plan["unlearning_sets"]:
        us = str(plan["unlearning_sets"])
        sets_root = (resolve_data_ref(us, root=root) if ("/" in us or os.path.isabs(us))
                     else gu.sets_root(us, root=root))
        with open(os.path.join(sets_root, "manifest.json")) as f:
            sets_manifest = json.load(f)
        for role, stem in ROLE_COLUMN.items():
            for l in langs:
                shards = gu.unlearning_shards(sets_root, role, l)
                if shards:
                    # The 1,000-token target is at most one 1024-block; FixedSpanDataset
                    # evaluates as many windows as fit and a short span as one window.
                    monitors[f"{stem}_{l}"] = FixedSpanDataset(shards, block, micro_batch=micro)
        print(f"[recover] unlearning sets {plan['unlearning_sets']}: "
              + ", ".join(f"{k}={v:,}" for k, v in sorted(
                  (sets_manifest.get("n_tokens") or {}).items())))
    print("[recover] monitors: " + ", ".join(sorted(monitors)))

    tcfg = TrainConfig(
        total_tokens=cap,
        batch_tokens=int(rec.get("batch_tokens", 2 ** 19)),
        micro_batch=micro, block_size=block,
        lr=float(rec.get("lr", 1e-4)),
        min_lr_frac=float(rec.get("min_lr_frac", 0.1)),
        warmup_tokens=int(rec.get("warmup_tokens", 5_000_000)),
        weight_decay=float(rec.get("weight_decay", 0.1)),
        grad_clip=float(rec.get("grad_clip", 1.0)),
        seed=int(rec.get("seed", 1337)),
        eval_every_tokens=plan["eval_every_tokens"],
        eval_batches=int(rec.get("eval_batches", 40)),
        compile=bool(rec.get("compile", True)),
        dtype=str(rec.get("dtype", "bfloat16")),
    )
    accum = max(1, tcfg.batch_tokens // (tcfg.micro_batch * tcfg.block_size))
    if plan["stream"] == "train" and accum % len(plan["stream_languages"]) != 0:
        raise ValueError(
            f"accum {accum} (= batch_tokens / (micro_batch * block_size)) must be a "
            f"multiple of the {len(plan['stream_languages'])} stream languages so "
            f"every optimizer step sees them equally")

    # ── tracking (off unless enabled) ─────────────────────────────────────────
    wb = Tracker(cfg.get("tracking"), out_dir=out_dir, run_name=cfg["name"],
                  config={"config": cfg, "plan": {k: v for k, v in plan.items()},
                          "corpus_spec_sha256": man.get("spec_sha256"),
                          "unlearning_sets_spec_sha256":
                              (sets_manifest or {}).get("spec_sha256"),
                          "unlearning_sets_n_tokens": (sets_manifest or {}).get("n_tokens"),
                          "checkpoint": cfg["checkpoint"],
                          "base_checkpoint": cfg.get("base_checkpoint"),
                          "device": device})
    if wd_totals is not None:
        wb.summary(WD.totals_summary({"totals": wd_totals}))

    # ── curve ────────────────────────────────────────────────────────────────
    fields = curve_fields(langs, plan["retained_languages"], sorted(monitors))
    rows: list[dict] = []
    hit: dict = {"configured": None, **{k: None for k in REPORTED_TOLERANCES}}

    def record(tokens: int, ce_val: dict) -> bool:
        ce_mon = evaluate(model, monitors, tcfg, device) if monitors else {}
        row = curve_row(tokens, ce_val, ce_mon, plan)
        rows.append(row)
        wb.log(_tracking_payload(row), step=int(tokens))
        extra = "  ".join(f"{n}={v:.4f}" for n, v in sorted(ce_mon.items()))
        if extra:
            print(f"  [monitor] {tokens:,} tok  {extra}")
        for k in REPORTED_TOLERANCES:
            if row[f"recovered_{k}"] and hit[k] is None:
                hit[k] = tokens
        if row["recovered"] and hit["configured"] is None:
            hit["configured"] = tokens
            return True          # stop: the configured criterion is met
        return False

    def log_hook(step, tokens, lr, train_ce):
        wb.log({"train/ce": train_ce, "train/lr": lr, "train/step": step}, step=int(tokens))

    t0 = time.time()
    summary = train(model, stream, val_sets, tcfg, device,
                    csv_path=os.path.join(out_dir, "train_ce.csv"),
                    on_eval=record, log_hook=log_hook)
    elapsed = time.time() - t0

    ce_final = summary["final_ce"]
    # The loop evaluates once more after the budget is spent; when that is a
    # new token count (not the early-stop repeat) it is a curve row too, and a
    # final-step crossing counts.
    if not rows or rows[-1]["tokens"] != summary["tokens"]:
        record(summary["tokens"], ce_final)
    ce_start = {l: rows[0][f"ce_val_{l}"] for l in langs}

    censored = hit["configured"] is None
    m_tokens = int(hit["configured"]) if not censored else cap
    ratio = m_tokens / n_nld
    m_by_tol = {f"M_tokens_{k}": (int(v) if v is not None else None) for k, v in hit.items()
                if k != "configured"}

    with open(os.path.join(out_dir, "recovery_curve.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

    summary_cols = ["condition", "name", "M_tokens", "M_tokens_tol0",
                    "M_tokens_tol1pct", "N_nld", "permanence_ratio",
                    "ratio_is_lower_bound", "ratio_can_exceed_one", "cap_tokens",
                    "cap_over_n_nld", "censored", "tolerance", "stream",
                    "ce_nld_original", "ce_nld_target"]
    summary_row = {"condition": plan["condition"], "name": cfg["name"],
                   "M_tokens": m_tokens,
                   **m_by_tol, "N_nld": n_nld, "permanence_ratio": round(ratio, 6),
                   "ratio_is_lower_bound": int(censored),
                   "ratio_can_exceed_one": int(plan["ratio_can_exceed_one"]),
                   "cap_tokens": cap, "cap_over_n_nld": round(plan["cap_over_n_nld"], 6),
                   "censored": int(censored), "tolerance": plan["tolerance"],
                   "stream": plan["stream"],
                   "ce_nld_original": round(plan["ce_nld_original"], 6),
                   "ce_nld_target": round(plan["ce_target"], 6)}
    for l in langs:
        summary_cols += [f"ce_{l}_start", f"ce_{l}_final"]
        summary_row[f"ce_{l}_start"] = round(ce_start[l], 6)
        summary_row[f"ce_{l}_final"] = round(ce_final[l], 6)
    for col in fields:
        if col.startswith("ce_") and col not in summary_row and not col.startswith("ce_val_") \
                and col != "ce_nld_target":
            summary_cols += [f"{col}_start", f"{col}_final"]
            summary_row[f"{col}_start"] = rows[0].get(col, "")
            summary_row[f"{col}_final"] = rows[-1].get(col, "")
    if wd_totals is not None:
        for k in ("grad_mass", "grad_mass_proj", "global_rel", "sum_rel",
                  "mean_rel_weighted", "changed_param_fraction", "n_changed",
                  "erank_delta_mean"):
            summary_cols.append(f"weight_delta_{k}")
            summary_row[f"weight_delta_{k}"] = wd_totals[k]
    summary_cols.append("elapsed_s")
    summary_row["elapsed_s"] = round(elapsed, 2)
    with open(os.path.join(out_dir, "recovery_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        w.writerow(summary_row)

    wb.summary({"M_tokens": m_tokens, "permanence_ratio": ratio, "censored": censored,
                **m_by_tol, "elapsed_s": elapsed,
                **{f"final/{k}": v for k, v in rows[-1].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}})
    wb.finish()

    cfg_hash = None
    if config_path is not None:
        with open(config_path, "rb") as f:
            cfg_hash = hashlib.sha256(f.read()).hexdigest()
    manifest = {"name": cfg["name"], "condition": plan["condition"],
                "checkpoint": ckpt, "base_checkpoint": base_ckpt,
                "config_sha256": cfg_hash, "corpus_spec_sha256": man.get("spec_sha256"),
                "stream": plan["stream"], "stream_languages": plan["stream_languages"],
                "stream_seed": plan["stream_seed"],
                "N_nld": n_nld, "cap_tokens": cap, "cap_over_n_nld": plan["cap_over_n_nld"],
                "ratio_can_exceed_one": plan["ratio_can_exceed_one"],
                "M_tokens": m_tokens, **m_by_tol,
                "permanence_ratio": ratio, "censored": censored,
                "tolerance": plan["tolerance"], "ce_nld_original": plan["ce_nld_original"],
                "ce_target": plan["ce_target"], "ce_targets": plan["ce_targets"],
                "ce_start": ce_start, "ce_final": ce_final,
                "monitors": sorted(monitors), "monitor_start": rows[0],
                "monitor_final": rows[-1],
                "unlearning_sets": plan["unlearning_sets"],
                "unlearning_sets_manifest": ({k: sets_manifest.get(k) for k in
                                             ("n_tokens", "tokenizer_dir", "spec_sha256")}
                                            if sets_manifest else None),
                "train_sample_tokens": plan["train_sample_tokens"],
                "weight_delta_totals": wd_totals,
                "tracking": wb.resolved,
                "hyperparameters": {k: getattr(tcfg, k) for k in
                                    ("total_tokens", "batch_tokens", "micro_batch",
                                     "block_size", "lr", "warmup_tokens", "seed", "dtype",
                                     "eval_every_tokens", "eval_batches")},
                "tokens_trained": summary["tokens"], "steps": summary["steps"],
                "elapsed_s": elapsed}
    with open(os.path.join(out_dir, "recovery_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    write_results(os.path.join(out_dir, "results.json"),
                  result_rows(manifest, rows[-1] if rows else {}), name=cfg["name"])

    _write_results(out_dir, cfg, plan, rows, fields, m_tokens, ratio, censored, hit,
                   wd_totals, elapsed, wb.resolved)
    verdict = ("CENSORED" if censored else f"recovered at {m_tokens:,} tokens")
    log(f"[recover] {verdict}")
    log(f"[recover] M/N_nld = {ratio:.4f}"
        + ("  (LOWER BOUND -- censored)" if censored else "  (measured)"))
    return manifest


def _write_results(out_dir, cfg, plan, rows, fields, m_tokens, ratio, censored, hit,
                   wd_totals, elapsed, wb_resolved) -> None:
    n_nld, cap = plan["N_nld"], plan["cap_tokens"]
    # A censored ratio is a LOWER BOUND, and saying so precisely matters: with
    # cap == N_nld the bound is exactly 1.0, but any other cap makes it
    # cap/N_nld, which is why the artifacts label it rather than asserting 1.0.
    if censored:
        verdict = (f"CENSORED — did not reach the criterion within the "
                   f"{cap:,}-token cap")
        ratio_note = (f"  (**lower bound** — censored at the cap, not a measured "
                      f"crossing{'' if cap == n_nld else f'; cap is {cap/n_nld:.3f} of N_nld'})")
    else:
        verdict = f"recovered at {m_tokens:,} tokens"
        ratio_note = ""
    tol_lines = []
    for k, t in plan["ce_targets"].items():
        v = hit.get(k)
        tol_lines.append(f"- first crossing at tolerance `{k}` (nld CE <= {t:.4f}): "
                         + (f"{int(v):,} tokens (M/N_nld = {v / n_nld:.4f})" if v is not None
                            else "never within the cap (censored)"))
    lines = [f"# Recovery -- `{cfg['name']}` (condition `{plan['condition']}`)", "",
             f"- **{verdict}**",
             f"- permanence ratio M/N_nld = **{ratio:.4f}**" + ratio_note,
             f"- N_nld = {n_nld:,}, cap = {cap:,} (cap/N_nld = {plan['cap_over_n_nld']:.3f}"
             f"{'; the ratio may exceed 1' if plan['ratio_can_exceed_one'] else ''})",
             f"- configured criterion: nld CE <= {plan['ce_target']:.4f} "
             f"({plan['tolerance']:.1%} above the original {plan['ce_nld_original']:.4f})",
             *tol_lines,
             f"- stream: `{plan['stream']}` {plan['stream_languages']}, seed {plan['stream_seed']}",
             f"- wall-clock {elapsed/60:.1f} min",
             f"- tracking: {wb_resolved.get('mode') if wb_resolved.get('active') else 'off'}"
             + (f", {wb_resolved['url']}" if wb_resolved.get("url") else "")
             + (f" ({wb_resolved['reason']})" if wb_resolved.get("reason") else ""),
             "", "## Monitors at t = 0 and at the end", "",
             "| metric | t = 0 | end | delta |", "| --- | ---: | ---: | ---: |"]
    first, last = rows[0], rows[-1]
    for col in fields:
        if not col.startswith("ce_") or col == "ce_nld_target":
            continue
        a, b = first.get(col, ""), last.get(col, "")
        if a == "" or b == "":
            continue
        lines.append(f"| {col} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    lines += ["", f"(end = {last['tokens']:,} tokens; {len(rows)} evaluations)"]
    if wd_totals is not None:
        er = wd_totals.get("erank_delta_mean")
        lines += ["", "## Weight delta vs base (t = 0)", "",
                  "| grad mass sum ||dW||_F^2 | grad mass (proj only) | global ||dW||/||W|| | "
                  "sum rel | weighted mean rel | changed tensors | "
                  "changed-param fraction | mean erank(dW) |",
                  "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                  f"| {wd_totals['grad_mass']:.4e} | {wd_totals['grad_mass_proj']:.4e} | "
                  f"{wd_totals['global_rel']:.4e} | {wd_totals['sum_rel']:.4e} | "
                  f"{wd_totals['mean_rel_weighted']:.4e} | "
                  f"{wd_totals['n_changed']}/{wd_totals['n_tensors']} | "
                  f"{wd_totals['changed_param_fraction']:.3f} | "
                  f"{'-' if er is None else f'{er:.1f}'} |",
                  "", "Per-matrix rows: `weight_delta.csv`; per block / projection: "
                  "`weight_delta.json`."]
    with open(os.path.join(out_dir, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")



def result_rows(manifest: dict, final_row: dict) -> list:
    """The recovery outcome as :class:`srspd_experiments.results.ResultRow` rows
    (task ``recovery/<condition>``): ``M_tokens``, ``permanence_ratio`` (meta
    ``lower_bound`` when censored), ``M_tokens_tol0`` / ``M_tokens_tol1pct``,
    ``censored``, and every final ``ce_*`` monitor. ``tp`` is ``None``."""
    task = f"recovery/{manifest['condition']}"
    model = manifest.get("checkpoint")
    lb = bool(manifest["censored"])
    rows = [ResultRow(task, "M_tokens", manifest["M_tokens"], model=model,
                      meta={"lower_bound": lb, "cap_tokens": manifest["cap_tokens"]}),
            ResultRow(task, "permanence_ratio", manifest["permanence_ratio"], model=model,
                      meta={"lower_bound": lb, "N_nld": manifest["N_nld"]}),
            ResultRow(task, "censored", float(lb), model=model)]
    for k in ("M_tokens_tol0", "M_tokens_tol1pct"):
        rows.append(ResultRow(task, k, manifest.get(k), model=model))
    for k, v in sorted(final_row.items()):
        if k.startswith("ce_") and isinstance(v, (int, float)) and not isinstance(v, bool):
            rows.append(ResultRow(task, f"final_{k}", v, model=model,
                                  n=final_row.get("tokens")))
    return rows
