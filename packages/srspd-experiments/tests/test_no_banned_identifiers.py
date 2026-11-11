"""The distribution and its tests carry none of the identifiers of the research
code it was distilled from, and the dose is never taken as ``mu_raw``.

* banned words (case-insensitive) anywhere in ``src/srspd_experiments``, its
  shipped YAML and ``tests``: research-ledger IDs, the cloud launcher, the
  clustering / circuit / alternative-solver vocabulary;
* no research-repo paths (``experiments/<folder>``, ``infra/``, the research
  package ``rspd``) -- tests may name a reference checkout only through the
  ``$SRSPD_REFERENCE_REPO`` environment variable;
* ``mu_raw`` never appears in the package source, and nowhere (source or
  tests) as a function parameter, a keyword argument or a dataclass field.
"""
from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "srspd_experiments"
TESTS = ROOT / "tests"
SELF = pathlib.Path(__file__).resolve()

# assembled from pieces so this file does not match itself
_WORDS = ["hdb" + "scan", "clust" + "er", "circ" + "uit", "a" + "svd", "a" + "_svd",
          "pull" + "back", "DECI" + "SION-", "EXPERI" + "MENT-", r"\bmod" + r"al\b",
          "led" + "ger"]
BANNED = re.compile("|".join(_WORDS), re.IGNORECASE)
_PATHS = [r"experi" + r"ments/(wmdp|scaling|muse|shapes|dutch-ablation)",
          r"\binf" + r"ra/", r"(?<![\w-])rs" + r"pd(\.|/)(?!report)", r"(?<![\w-])rs" + r"pd\b(?![-_])",
          r"/root/", r"That" + r"E10"]
PATHS = re.compile("|".join(_PATHS))
MU_RAW = "mu" + "_raw"


def _files(patterns=("*.py",)):
    out = []
    for pat in patterns:
        out += sorted(SRC.rglob(pat)) + sorted(TESTS.rglob(pat))
    return [p for p in out if p.resolve() != SELF]


def test_files_found():
    names = {p.name for p in _files()}
    assert {"__init__.py", "paths.py", "results.py", "retained_corpus.py", "corpora.py",
            "spec.py", "target.py", "budget.py", "gpt2_corpus.py", "gpt2_unlearning.py",
            "dutch_sets.py", "lm_eval.py", "krd.py", "muse.py", "rouge_l.py", "heldout.py",
            "ce.py", "trainer.py", "recovery.py", "tracking.py"} <= names


def test_no_banned_words():
    hits = []
    for p in _files(("*.py", "*.yaml")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            m = BANNED.search(line)
            if m:
                hits.append(f"{p.relative_to(ROOT)}:{i}: {m.group(0)!r}")
    assert not hits, "banned identifiers:\n" + "\n".join(hits)


def test_no_research_repo_paths():
    hits = []
    for p in _files(("*.py", "*.yaml")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "SRSPD_REFERENCE_REPO" in line:
                continue
            # the core package and its submodules are this repository's, not research paths
            scrubbed = re.sub(r"srspd[\w.]*|srspd-experiments", "", line)
            m = PATHS.search(scrubbed)
            if m:
                hits.append(f"{p.relative_to(ROOT)}:{i}: {m.group(0)!r}")
    assert not hits, "research-repo paths:\n" + "\n".join(hits)


def test_mu_raw_not_in_the_source():
    hits = [str(p.relative_to(ROOT)) for p in sorted(SRC.rglob("*.py")) if MU_RAW in p.read_text()]
    assert not hits, f"{MU_RAW} in the source: {hits}"


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


def test_tracking_is_off_by_default():
    from srspd_experiments.eval.tracking import Tracker
    t = Tracker(None, out_dir="/nonexistent", run_name="x")
    assert not t.enabled and not t.active and t.resolved["reason"] == "tracking disabled"
    t.log({"a": 1})
    t.summary({"a": 1})
    t.finish()
