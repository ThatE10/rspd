"""The local five-shot Global-MMLU tasks (``srspd_experiments/eval/tasks/``)
are one recipe: they differ only in the task name, the Global-MMLU language
config and the answer cue, so the Dutch forgetting score and its English and
German controls are measured on identical footing."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from srspd_experiments.eval.lm_eval import LOCAL_TASKS, TASKS_DIR

TASKS = Path(TASKS_DIR)
CUES = {"nl": "Antwoord:", "de": "Antwort:", "en": "Answer:"}


def _load(lang):
    return yaml.safe_load((TASKS / f"global_mmlu_{lang}_local.yaml").read_text())


@pytest.mark.parametrize("lang", sorted(CUES))
def test_task_is_the_shared_recipe(lang):
    t = _load(lang)
    assert t["task"] == f"global_mmlu_{lang}_local"
    assert t["dataset_path"] == "CohereLabs/Global-MMLU" and t["dataset_name"] == lang
    assert t["num_fewshot"] == 5 and t["fewshot_split"] == "dev" and t["test_split"] == "test"
    assert t["doc_to_text"].endswith("\n" + CUES[lang])


@pytest.mark.parametrize("lang", ["de", "en"])
def test_controls_mirror_the_dutch_task(lang):
    nl, other = _load("nl"), _load(lang)
    strip = lambda t: {k: (v.replace(CUES[t["dataset_name"]], "<cue>") if k == "doc_to_text" else v)
                       for k, v in t.items() if k not in ("task", "dataset_name")}
    assert strip(nl) == strip(other)


def test_four_local_tasks_ship_as_package_data():
    assert LOCAL_TASKS == tuple(f"global_mmlu_{l}_local" for l in ("de", "en", "nl", "zh"))
    zh = _load("zh")
    assert zh["dataset_name"] == "zh" and zh["num_fewshot"] == 5
