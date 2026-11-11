"""Shared pytest setup: the tests directory on ``sys.path`` (for the tiny-model
helpers) and data isolation -- an autouse fixture points ``SRSPD_DATA_DIR`` at a
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
    monkeypatch.setenv("SRSPD_DATA_DIR", str(tmp_path_factory.getbasetemp() / "srspd_data"))
