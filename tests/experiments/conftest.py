"""Shared pytest setup: the tests directory on ``sys.path`` (for the tiny-model
helpers) and data isolation -- an autouse fixture points ``RSPD_DATA_DIR`` at a
per-session temp directory, so no test writes into a real data root."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
PKG = HERE.parent


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("RSPD_DATA_DIR", str(tmp_path_factory.getbasetemp() / "rspd_data"))


def isolate_corpora(monkeypatch, *, drop_local=False):
    """Give the test its own copy of the corpus registry: every
    ``rspd.experiments`` module that holds the registry dict gets the copy
    (restored at teardown), so a ``local_jsonl`` corpus a test registers never
    leaks into another. ``drop_local`` removes the ``local_jsonl`` entries other
    tests registered. Returns the copy."""
    from rspd.experiments.data import corpora as C
    orig = C.CORPORA
    new = {k: v for k, v in orig.items()
           if not (drop_local and C.corpus_kind(v) == "local_jsonl")}
    for m in list(sys.modules.values()):
        if (getattr(m, "__name__", "") or "").startswith("rspd.experiments") \
                and getattr(m, "CORPORA", None) is orig:
            monkeypatch.setattr(m, "CORPORA", new)
    return new
