"""The lm-evaluation-harness wrapper without lm-eval: command construction,
config validation, the TP record, result parsing into the common schema, and a
full ``run`` with the subprocess replaced by a fixture writer. CPU, no network."""
from __future__ import annotations

import json
import os

import pytest

from rspd.experiments.eval import lm_eval as E
from rspd.experiments.results import SCHEMA, read_results

PAYLOAD = {
    "results": {
        "global_mmlu_nl_local": {"alias": "global_mmlu_nl_local", "acc,none": 0.512,
                                 "acc_stderr,none": 0.004},
        "wmdp_bio": {"alias": "wmdp_bio", "acc,none": 0.31, "acc_stderr,none": 0.013},
        "wikitext": {"alias": "wikitext", "word_perplexity,none": 9.25,
                     "word_perplexity_stderr,none": "N/A", "bits_per_byte,none": 0.62},
    },
    "n-samples": {"global_mmlu_nl_local": {"original": 14042, "effective": 14042},
                  "wmdp_bio": {"original": 1273, "effective": 1273}},
    "n-shot": {"global_mmlu_nl_local": 5, "wmdp_bio": 0, "wikitext": 0},
}


def _cfg(**over):
    cfg = {"name": "unit_eval", "model_path": "org/model", "tasks": ["global_mmlu_nl_local"],
           "model_args": {"dtype": "bfloat16", "tensor_parallel_size": 2,
                          "max_model_len": 8192}}
    cfg.update(over)
    return cfg


def test_command_is_vllm_with_local_tasks_and_chat_flags(tmp_path):
    cmd = E.build_command(_cfg(apply_chat_template=True, log_samples=True, limit=10),
                          str(tmp_path), python="py")
    assert cmd[:5] == ["py", "-m", "lm_eval", "--model", "vllm"]
    margs = cmd[cmd.index("--model_args") + 1]
    assert margs == "dtype=bfloat16,tensor_parallel_size=2,max_model_len=8192,pretrained=org/model"
    assert cmd[cmd.index("--tasks") + 1] == "global_mmlu_nl_local"
    assert cmd[cmd.index("--batch_size") + 1] == "auto" and cmd[cmd.index("--seed") + 1] == "1234"
    assert cmd[cmd.index("--include_path") + 1] == E.TASKS_DIR
    assert cmd[cmd.index("--output_path") + 1] == os.path.join(str(tmp_path), "lm_eval")
    assert "--apply_chat_template" in cmd and "--fewshot_as_multiturn" in cmd
    assert cmd[cmd.index("--limit") + 1] == "10" and "--log_samples" in cmd
    plain = E.build_command(_cfg(backend="hf", model_args={}), str(tmp_path))
    assert plain[plain.index("--model") + 1] == "hf"
    assert "--apply_chat_template" not in plain and "--limit" not in plain
    assert "--num_fewshot" not in plain
    nshot = E.build_command(_cfg(num_fewshot=0), str(tmp_path))
    assert nshot[nshot.index("--num_fewshot") + 1] == "0"
    off = E.build_command(_cfg(apply_chat_template=True, fewshot_as_multiturn=False), str(tmp_path))
    assert "--apply_chat_template" in off and "--fewshot_as_multiturn" not in off


def test_local_checkpoint_resolves_under_the_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path))
    (tmp_path / "models" / "r" / "ck").mkdir(parents=True)
    cmd = E.build_command(_cfg(model_path="models/r/ck", model_args={}), str(tmp_path))
    assert cmd[cmd.index("--model_args") + 1] == f"pretrained={tmp_path / 'models' / 'r' / 'ck'}"


@pytest.mark.parametrize("over, match", [
    ({"tasks": []}, "non-empty"),
    ({"backend": "tgi"}, "backend"),
    ({"model_args": "dtype=bf16"}, "model_args"),
    ({"wandb": True}, "unknown keys"),
])
def test_bad_configs_are_refused(over, match):
    with pytest.raises(ValueError, match=match):
        E.resolve_config(_cfg(**over))
    with pytest.raises(ValueError, match="needs 'tasks'"):
        E.resolve_config({"name": "x", "model_path": "y"})


def test_tp_is_recorded_and_hf_has_none():
    assert E.tensor_parallel_size(E.resolve_config(_cfg())) == 2
    assert E.tensor_parallel_size(E.resolve_config(_cfg(model_args={}))) == 1
    assert E.tensor_parallel_size(E.resolve_config(_cfg(backend="hf"))) is None


def test_parse_results_folds_stderr_and_filter_into_meta():
    rows = E.parse_results(PAYLOAD, model="m", tp=2)
    got = {(r.task, r.metric): r for r in rows}
    assert set(got) == {("global_mmlu_nl_local", "acc"), ("wmdp_bio", "acc"),
                        ("wikitext", "word_perplexity"), ("wikitext", "bits_per_byte")}
    nl = got[("global_mmlu_nl_local", "acc")]
    assert nl.value == 0.512 and nl.tp == 2 and nl.n == 14042 and nl.model == "m"
    assert nl.meta == {"filter": "none", "harness_key": "acc,none", "stderr": 0.004,
                       "num_fewshot": 5}
    wt = got[("wikitext", "word_perplexity")]
    assert wt.meta["stderr"] is None and wt.n is None


def test_run_writes_the_common_schema(tmp_path):
    def runner(cmd):
        out = cmd[cmd.index("--output_path") + 1]
        d = os.path.join(out, "org__model")
        os.makedirs(d)
        with open(os.path.join(d, "results_2026-01-01T00-00-00.json"), "w") as f:
            json.dump(PAYLOAD, f)
        return 0
    rows = E.run(_cfg(), out_dir=str(tmp_path), runner=runner)
    assert len(rows) == 4 and all(r.tp == 2 for r in rows)
    disk = json.loads((tmp_path / "results.json").read_text())
    assert disk["schema"] == SCHEMA and disk["tp"] == 2 and disk["backend"] == "vllm"
    assert [r.as_dict() for r in read_results(str(tmp_path / "results.json"))] == \
        [r.as_dict() for r in rows]
    md = (tmp_path / "results.md").read_text()
    assert "| global_mmlu_nl_local | acc,none | 0.5120 | 0.0040 |" in md and "(tp 2)" in md
    assert (tmp_path / "config.yaml").exists()
    with pytest.raises(RuntimeError, match="exited with 3"):
        E.run(_cfg(), out_dir=str(tmp_path / "b"), runner=lambda c: 3)
    with pytest.raises(RuntimeError, match="no results"):
        E.run(_cfg(), out_dir=str(tmp_path / "c"), runner=lambda c: 0)


def test_lm_eval_itself_is_not_imported():
    import sys
    assert "vllm" not in sys.modules


def test_real_harness_parses_the_shipped_tasks_when_installed():
    tm = pytest.importorskip("lm_eval.tasks")
    mgr = tm.TaskManager(include_path=E.TASKS_DIR)
    assert set(E.LOCAL_TASKS) <= set(mgr.all_tasks)
