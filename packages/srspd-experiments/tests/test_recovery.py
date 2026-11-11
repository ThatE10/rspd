"""The retrain-to-recover loop (``srspd_experiments.eval.recovery``) end to end
on a tiny GPT-2 over synthetic shards: config contract, the plan (cap above
N_nld, both tolerances, stream seed), curve rows, an early stop at t = 0, a
censored run, the Dutch-only stream, the accumulation guard, the weight delta
and the common result rows. CPU, no network."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from srspd_experiments.data import gpt2_unlearning as gu  # noqa: E402
from srspd_experiments.eval import recovery as drv  # noqa: E402
from srspd_experiments.paths import logs_dir  # noqa: E402
from srspd_experiments.results import read_results  # noqa: E402

BLOCK = 8
LANGS = ("deu", "eng", "nld")


def _write(path: Path, tokens):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(tokens, dtype=np.uint16).tofile(path)


def _tiny_tokenizer():
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(
        {f"t{i}": i for i in range(64)}, unk_token="t0"))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Corpus (train/val/recovery per language), unlearning sets, two checkpoints."""
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path))
    rng = np.random.default_rng(0)
    corpus = tmp_path / "datasets" / "gpt2_corpus" / "corpus"
    spans = {"train": 600, "val": 200, "recovery": 400}
    langs = {}
    for l in LANGS:
        langs[l] = {}
        for span, n in spans.items():
            _write(corpus / l / span / "shard_0000.bin", rng.integers(1, 64, n))
            langs[l][span] = {"tokens": n, "shards": ["shard_0000.bin"], "doc_lo": 0, "doc_hi": 1}
    (corpus / "corpus_manifest.json").write_text(json.dumps({
        "spec_sha256": "abc", "languages": langs,
        "n_tokens_by_language": {l: spans["train"] for l in LANGS}}))

    sets = Path(gu.sets_root("gpt2_unlearning"))
    n_tok = {"target_nld": 5, "contrastive_eng": 40, "contrastive_deu": 40,
             "heldout_eng": 30, "heldout_deu": 30, "heldout_nld": 30}
    for key, n in n_tok.items():
        _write(sets / "shards" / key / "shard_0000.bin", rng.integers(1, 64, n))
    sets.mkdir(parents=True, exist_ok=True)
    (sets / "manifest.json").write_text(json.dumps(
        {"n_tokens": n_tok, "tokenizer_dir": "x", "spec_sha256": "def"}))

    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=64, n_positions=BLOCK, n_embd=16,
                                  n_layer=1, n_head=2)
    base = transformers.GPT2LMHeadModel(cfg)
    base_dir = tmp_path / "models" / "gpt2" / "gpt2_trilingual_pretrain" / "checkpoint"
    base.save_pretrained(base_dir)
    _tiny_tokenizer().save_pretrained(base_dir)
    with torch.no_grad():
        base.transformer.h[0].mlp.c_fc.weight.add_(0.3)
    edited_dir = tmp_path / "models" / "wmdp" / "gpt2_unlearn_x" / "k0050"
    base.save_pretrained(edited_dir)
    _tiny_tokenizer().save_pretrained(edited_dir)
    return {"root": tmp_path,
            "base": "models/gpt2/gpt2_trilingual_pretrain/checkpoint",
            "edited": "models/wmdp/gpt2_unlearn_x/k0050"}


def _config(tmp_path, world, name, **over):
    rec = {"ce_nld_original": 100.0, "cap_tokens": 4 * 48, "batch_tokens": 48,
           "micro_batch": 2, "block_size": BLOCK, "eval_every_tokens": 96,
           "eval_batches": 2, "warmup_tokens": 10, "compile": False, "dtype": "float32",
           "tolerance": 0.0, "stream": "train",
           "monitors": {"unlearning_sets": "gpt2_unlearning", "train_sample_tokens": 40}}
    rec.update(over.pop("recovery", {}))
    cfg = {"name": name, "condition": "rspd_k50", "corpus_name": "gpt2_corpus",
           "checkpoint": world["edited"], "base_checkpoint": world["base"],
           "recovery": rec, "tracking": {"enabled": False}}
    cfg.update(over)
    return cfg


# ── pure helpers ─────────────────────────────────────────────────────────────

def test_resolve_config_contract():
    base = {"name": "n", "corpus_name": "c", "checkpoint": "m"}
    cfg = drv.resolve_config({**base, "recovery": {"ce_nld_original": 3.5}})
    assert cfg["condition"] == "n"                       # defaults to the name
    with pytest.raises(ValueError, match="stream"):
        drv.resolve_config({**base, "recovery": {"ce_nld_original": 3.5, "stream": "mixed"}})
    with pytest.raises(ValueError, match="ce_nld_original"):
        drv.resolve_config({**base, "recovery": {}})
    with pytest.raises(ValueError, match="unknown keys"):
        drv.resolve_config({**base, "recovery": {"ce_nld_original": 3.5}, "wandb": {}})
    with pytest.raises(ValueError, match="corpus_name"):
        drv.resolve_config({"name": "n", "checkpoint": "m", "recovery": {"ce_nld_original": 1}})


def test_resolve_plan_cap_may_exceed_n_nld_and_records_it():
    man = {"languages": {"eng": {}, "deu": {}, "nld": {}},
           "n_tokens_by_language": {"eng": 100, "deu": 100, "nld": 100}}
    cfg = {"name": "x", "recovery": {"ce_nld_original": 2.0}}
    plan = drv.resolve_plan(cfg, man)
    assert plan["cap_tokens"] == 100 and not plan["ratio_can_exceed_one"]
    assert plan["stream"] == "train" and plan["stream_languages"] == ["deu", "eng", "nld"]
    assert plan["retained_languages"] == ["deu", "eng"]
    assert plan["tolerance"] == drv.DEFAULT_TOLERANCE
    assert plan["ce_targets"] == {"tol0": 2.0, "tol1pct": pytest.approx(2.02)}
    assert plan["stream_seed"] == 1338
    cfg["recovery"].update({"cap_tokens": 1000, "stream": "recovery", "tolerance": 0.0,
                            "stream_seed": 5, "seed": 9})
    plan = drv.resolve_plan(cfg, man)
    assert plan["cap_tokens"] == 1000 and plan["ratio_can_exceed_one"]
    assert plan["cap_over_n_nld"] == 10.0 and plan["ce_target"] == 2.0
    assert plan["stream_languages"] == ["nld"] and plan["stream_seed"] == 5
    cfg["recovery"].update({"stream": "train", "stream_languages": ["eng", "fra"]})
    with pytest.raises(ValueError, match="fra"):
        drv.resolve_plan(cfg, man)


def test_curve_row_records_both_tolerances():
    man = {"languages": {"eng": {}, "deu": {}, "nld": {}},
           "n_tokens_by_language": {"eng": 100, "deu": 100, "nld": 100}}
    plan = drv.resolve_plan({"name": "x", "condition": "kuda",
                             "recovery": {"ce_nld_original": 2.0, "tolerance": 0.0}}, man)
    row = drv.curve_row(50, {"eng": 1.0, "deu": 3.0, "nld": 2.01},
                        {"train_eng": 1.0, "train_deu": 2.0, "heldout_nld": 4.0,
                         "forget_nld": 9.0}, plan)
    assert row["condition"] == "kuda" and row["ce_retained_mean"] == 2.0
    assert row["ce_train_mean"] == 1.5 and row["ce_heldout_mean"] == 4.0
    assert row["ce_forget_nld"] == 9.0 and row["ppl_val_eng"] == pytest.approx(np.e, rel=1e-4)
    assert (row["recovered"], row["recovered_tol0"], row["recovered_tol1pct"]) == (0, 0, 1)
    fields = drv.curve_fields(["deu", "eng", "nld"], ["deu", "eng"],
                              ["forget_nld", "heldout_nld", "train_deu", "train_eng"])
    assert fields[:3] == ["tokens", "condition", "ce_val_deu"]
    assert "ce_train_mean" in fields and "ce_heldout_mean" in fields
    assert fields[-4:] == ["ce_nld_target", "recovered", "recovered_tol0", "recovered_tol1pct"]
    assert "ce_contrastive_mean" not in fields


# ── end to end ───────────────────────────────────────────────────────────────

def test_end_to_end_stops_at_t0_when_already_recovered(tmp_path, world):
    man_ret = drv.run(_config(tmp_path, world, "rec_hit"), log=lambda *_: None)
    out = Path(logs_dir("recovery", "rec_hit", create=False))
    curve = list(csv.DictReader(open(out / "recovery_curve.csv")))
    assert len(curve) == 1 and curve[0]["tokens"] == "0" and curve[0]["recovered"] == "1"
    for col in ("ce_val_eng", "ce_val_deu", "ce_val_nld", "ce_retained_mean",
                "ce_train_eng", "ce_train_mean", "ce_heldout_nld", "ce_heldout_mean",
                "ce_forget_nld", "ce_contrastive_eng", "ce_contrastive_deu",
                "recovered_tol0", "recovered_tol1pct"):
        assert curve[0][col] != "", col
    assert curve[0]["condition"] == "rspd_k50"
    man = json.loads((out / "recovery_manifest.json").read_text())
    assert man["M_tokens"] == 0 and man["M_tokens_tol0"] == 0 and man["M_tokens_tol1pct"] == 0
    assert man["permanence_ratio"] == 0.0 and not man["censored"]
    assert man["ratio_can_exceed_one"] is False and man["cap_over_n_nld"] == pytest.approx(192 / 600)
    assert man["tracking"]["enabled"] is False and man_ret["M_tokens"] == 0
    rows = {(r.task, r.metric): r for r in read_results(str(out / "results.json"))}
    assert rows[("recovery/rspd_k50", "M_tokens")].value == 0
    assert rows[("recovery/rspd_k50", "permanence_ratio")].meta["lower_bound"] is False
    assert ("recovery/rspd_k50", "final_ce_forget_nld") in rows
    assert man["unlearning_sets_manifest"]["n_tokens"]["target_nld"] == 5
    assert man["weight_delta_totals"]["n_changed"] == 1
    wd = json.loads((out / "weight_delta.json").read_text())
    assert [r["name"] for r in wd["tensors"] if r["changed"]] == ["transformer.h.0.mlp.c_fc.weight"]
    assert (out / "weight_delta.csv").exists() and (out / "train_ce.csv").exists()
    summ = list(csv.DictReader(open(out / "recovery_summary.csv")))[0]
    assert summ["condition"] == "rspd_k50" and summ["M_tokens"] == "0"
    assert summ["weight_delta_n_changed"] == "1" and summ["ce_forget_nld_start"] != ""
    res = (out / "results.md").read_text()
    assert "recovered at 0 tokens" in res and "## Weight delta vs base" in res
    assert "| ce_forget_nld |" in res


def test_end_to_end_censored_with_cap_above_n_nld(tmp_path, world):
    cfg = _config(tmp_path, world, "rec_cens",
                   recovery={"ce_nld_original": 0.001, "cap_tokens": 15 * 48,
                             "tolerance": 0.01, "stream_languages": ["nld", "eng", "deu"]})
    drv.run(cfg, log=lambda *_: None)
    out = Path(logs_dir("recovery", "rec_cens", create=False))
    curve = list(csv.DictReader(open(out / "recovery_curve.csv")))
    # evals at 0, 96, 192, ..., plus the final at the cap (720)
    toks = [int(r["tokens"]) for r in curve]
    assert toks[0] == 0 and toks[-1] == 720 and toks == sorted(set(toks))
    assert all(r["recovered"] == "0" for r in curve)
    man = json.loads((out / "recovery_manifest.json").read_text())
    assert man["censored"] and man["M_tokens"] == 720 and man["M_tokens_tol0"] is None
    assert man["ratio_can_exceed_one"] is True and man["permanence_ratio"] == pytest.approx(1.2)
    assert man["stream_languages"] == ["nld", "eng", "deu"] and man["tokens_trained"] == 720
    summ = list(csv.DictReader(open(out / "recovery_summary.csv")))[0]
    assert summ["ratio_is_lower_bound"] == "1" and summ["ratio_can_exceed_one"] == "1"
    assert summ["M_tokens_tol1pct"] == ""
    res = (out / "results.md").read_text()
    assert "lower bound" in res and "the ratio may exceed 1" in res


def test_recovery_stream_keeps_the_old_protocol(tmp_path, world):
    cfg = _config(tmp_path, world, "rec_old",
                   recovery={"stream": "recovery", "cap_tokens": 2 * 48,
                             "ce_nld_original": 0.001, "monitors": {}},
                   base_checkpoint=None)
    drv.run(cfg, log=lambda *_: None)
    out = Path(logs_dir("recovery", "rec_old", create=False))
    man = json.loads((out / "recovery_manifest.json").read_text())
    assert man["stream"] == "recovery" and man["stream_languages"] == ["nld"]
    assert man["weight_delta_totals"] is None and man["unlearning_sets"] is None
    assert man["censored"] and man["cap_tokens"] == 96
    curve = list(csv.DictReader(open(out / "recovery_curve.csv")))
    assert "ce_train_eng" in curve[0] and "ce_heldout_eng" not in curve[0]


def test_accum_must_be_a_multiple_of_the_language_count(tmp_path, world):
    cfg = _config(tmp_path, world, "rec_bad", recovery={"batch_tokens": 64})   # accum 4, 3 langs
    with pytest.raises(ValueError, match="multiple"):
        drv.run(cfg)


def test_unlearning_set_helpers(world):
    root = gu.sets_root("gpt2_unlearning")
    assert root.endswith(os.path.join("datasets", "gpt2_unlearning", "gpt2_unlearning"))
    assert len(gu.unlearning_shards(root, "target", "nld")) == 1
    assert gu.unlearning_shards(root, "target", "eng") == []
