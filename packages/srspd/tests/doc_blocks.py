"""Shared helpers of the documentation tests: find the repository root, pull the
fenced code blocks out of a Markdown file, and run the ones marked runnable.

A block is runnable when its info string is ``python runnable``. The blocks of
one file run in order in ONE namespace (a later block may use names an earlier
one bound), pre-seeded with a tiny random-weight GPT-2 from ``tiny_models``:
``model``, ``tokenizer``, ``target``, ``retained``. Nothing touches the network.
"""
from __future__ import annotations

import os
import pathlib
import re

FENCE = re.compile(r"^```([^\n`]*)\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)
RUNNABLE = "python runnable"


def repo_root() -> pathlib.Path:
    here = pathlib.Path(__file__).resolve()
    for p in here.parents:
        if (p / "AGENTS.md").is_file() and (p / "packages").is_dir():
            return p
    raise FileNotFoundError("repository root (AGENTS.md + packages/) not found above "
                            f"{here}")


def blocks(path) -> list:
    """``[(info string, code, line number)]`` of every fenced block in ``path``."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    out = []
    for m in FENCE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        out.append((m.group(1).strip(), m.group(2), line))
    return out


def runnable_blocks(path) -> list:
    return [(code, line) for info, code, line in blocks(path) if info == RUNNABLE]


def tiny_namespace() -> dict:
    import tiny_models as tm
    target, retained = tm.token_sets()
    return {"__name__": "__doc_example__", "model": tm.gpt2(), "tokenizer": tm.word_tokenizer(),
            "target": target, "retained": retained}


def run_file(path, tmp_path, monkeypatch) -> dict:
    """Execute every runnable block of ``path`` in order; returns the namespace.
    The working directory and ``$SRSPD_DATA_DIR`` point into ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SRSPD_DATA_DIR", os.fspath(tmp_path / "data"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ns = tiny_namespace()
    found = runnable_blocks(path)
    assert found, f"{path} has no `{RUNNABLE}` block"
    for code, line in found:
        try:
            exec(compile(code, f"{path}:{line}", "exec"), ns)
        except Exception as exc:                     # name the block that failed
            raise AssertionError(f"runnable block at {path}:{line} failed: "
                                 f"{type(exc).__name__}: {exc}") from exc
    return ns
