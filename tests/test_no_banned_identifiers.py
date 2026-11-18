"""The package and its tests carry none of the identifiers that belong to the
research code it was distilled from, and the dose is never taken as ``mu_raw``.

* banned words (case-insensitive) anywhere in ``src/rspd`` and ``tests``;
* ``mu_raw`` appears in the package source only in ``dose.py``, and nowhere
  (source or tests) as a function parameter or a dataclass field: it is a
  derived, reported property.
"""
from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "rspd"
TESTS = ROOT / "tests"
SELF = pathlib.Path(__file__).resolve()

# assembled from pieces so this file does not match itself
_WORDS = ["hdb" + "scan", "spectral" + r"\s+" + "clustering", "circuit" + "_isolation",
          "recur" + "sive", "a" + "svd", "a" + "_svd", "pull" + "back",
          "DECI" + "SION-", "EXPERI" + "MENT-"]
BANNED = re.compile("|".join(_WORDS), re.IGNORECASE)
MU_RAW = "mu" + "_raw"


def _files():
    out = sorted(SRC.rglob("*.py")) + sorted(TESTS.rglob("*.py"))
    # rspd.experiments and its tests have their own check (tests/experiments)
    return [p for p in out if p.resolve() != SELF and "experiments" not in p.relative_to(ROOT).parts]


def test_files_found():
    names = {p.name for p in _files()}
    assert {"solve.py", "coala.py", "dose.py", "__init__.py", "api.py", "pipeline.py", "cli.py",
            "selector.py", "quirks.py", "harness.py", "inspect.py", "pristine.py", "export.py",
            "base.py", "hf.py", "grams.py"} <= names
    subpackages = {p.parent.name for p in _files() if p.parent != SRC and SRC in p.parents}
    assert {"models", "capture"} <= subpackages


def test_no_banned_words():
    hits = []
    for p in _files():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            m = BANNED.search(line)
            if m:
                hits.append(f"{p.relative_to(ROOT)}:{i}: {m.group(0)!r}")
    assert not hits, "banned identifiers:\n" + "\n".join(hits)


def test_mu_raw_only_in_dose_module():
    hits = [str(p.relative_to(ROOT)) for p in _files() if SRC in p.parents
            and p.name != "dose.py" and MU_RAW in p.read_text()]
    assert not hits, f"{MU_RAW} outside dose.py: {hits}"


def test_mu_raw_is_never_an_input():
    hits = []
    for p in _files():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs + \
                        [x for x in (a.vararg, a.kwarg) if x is not None]:
                    if arg.arg == MU_RAW:
                        hits.append(f"{p.name}:{node.lineno}: parameter")
            if isinstance(node, ast.ClassDef):
                for st in node.body:
                    if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) \
                            and st.target.id == MU_RAW:
                        hits.append(f"{p.name}:{st.lineno}: field of {node.name}")
            if isinstance(node, ast.keyword) and node.arg == MU_RAW:
                hits.append(f"{p.name}:{node.value.lineno}: keyword argument")
    assert not hits, f"{MU_RAW} taken as an input:\n" + "\n".join(hits)
