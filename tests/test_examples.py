"""The example notebooks run end to end, top to bottom, on a CPU.

Each notebook under ``examples/`` is executed in a fresh kernel against this
checkout. ``RSPD_EXAMPLES_SMOKE=1`` is set for the run: a notebook that needs
a GPU or a large download reads it and swaps in a tiny random model of the
same family, so the same cells are exercised. Skipped when the notebook
tooling (the ``examples`` extra) is not installed.
"""
from __future__ import annotations

import os
import pathlib

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")
pytest.importorskip("matplotlib")

ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").glob("*.ipynb"))


def test_there_are_examples():
    assert {p.name for p in NOTEBOOKS} >= {"quickstart.ipynb"}


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path, monkeypatch, tmp_path):
    monkeypatch.setenv("RSPD_EXAMPLES_SMOKE", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "tests"), os.environ.get("PYTHONPATH", "")]))
    nb = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=900, kernel_name="python3",
                                     resources={"metadata": {"path": str(tmp_path)}})
    client.execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_has_a_colab_badge_for_itself(path):
    first = nbformat.read(path, as_version=4).cells[0].source
    assert f"colab.research.google.com/github/ThatE10/rspd/blob/main/examples/{path.name}" in first
