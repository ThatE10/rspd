"""The rspd[experiments] documentation stays true to the code.

* every ``python runnable`` block of ``docs/experiments.md``
  executes offline (shared runner: ``tests/doc_blocks.py``);
* every ``rspd-exp`` subcommand and option is documented in ``docs/api.md``;
* every public name of the package-level modules (``data``, ``methods``,
  ``models``, ``results``, ``paths``, the recipe catalogue) has an entry there.
"""
from __future__ import annotations

import argparse
import re

import pytest

doc_blocks = pytest.importorskip("doc_blocks")

ROOT = doc_blocks.repo_root()
API = ROOT / "docs" / "api.md"


def _entry(text, name) -> bool:
    pat = re.compile(r"^(?:\|\s*|#+\s+)`" + re.escape(name) + r"[`(\[ ]", re.MULTILINE)
    return bool(pat.search(text))


def test_runnable_examples_execute(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    doc_blocks.run_file(ROOT / "docs" / "experiments.md", tmp_path,
                        monkeypatch)


def test_every_cli_command_and_option_is_documented():
    from rspd.experiments.cli import build_parser
    text = API.read_text()
    missing = []
    for act in build_parser()._actions:
        if not isinstance(act, argparse._SubParsersAction):
            continue
        for cmd, sub in act.choices.items():
            if f"rspd-exp {cmd}" not in text:
                missing.append(f"rspd-exp {cmd}")
            missing += [f"rspd-exp {cmd} {o}" for a in sub._actions for o in a.option_strings
                        if o not in ("-h", "--help") and f"`{o}" not in text]
    assert not missing, f"undocumented in docs/api.md: {missing}"


def test_package_level_public_names_are_documented():
    import rspd.experiments.data as data
    import rspd.experiments.methods as methods
    import rspd.experiments.models as models
    import rspd.experiments.paths as paths
    import rspd.experiments.recipes.catalog as catalog
    import rspd.experiments.recipes.compare as compare
    import rspd.experiments.results as results
    text = API.read_text()
    names = list(data.__all__) + list(methods.__all__) + list(models.__all__)
    names += ["data_dir", "datasets_dir", "logs_dir", "models_dir", "resolve_data_ref",
              "data_ref", "resolve_model_path", "hf_token"]
    names += ["RecipeError", "list_results", "load_recipe", "iter_exports", "find_export",
              "weight_gaps", "plan", "compare", "format_report", "mu_per_token_from_legacy"]
    names += ["ResultRow", "write_results", "read_results", "index"]
    for mod, n in ((paths, "data_dir"), (catalog, "plan"), (compare, "compare"),
                   (results, "ResultRow")):
        assert hasattr(mod, n), f"{mod.__name__}.{n} is gone; update this test and docs/api.md"
    missing = [n for n in names if not _entry(text, n)]
    assert not missing, f"undocumented in docs/api.md: {missing}"
