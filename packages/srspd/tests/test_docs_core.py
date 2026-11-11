"""The documentation stays true to the code.

* every ``python runnable`` block of ``AGENTS.md`` and ``packages/srspd/README.md``
  executes, in order, on a tiny random-weight GPT-2 with no network;
* every public symbol of ``srspd`` (``srspd.__all__``, ``srspd.models.__all__``,
  ``srspd.capture.__all__``) has an entry in ``docs/api.md``;
* every ``srspd`` subcommand and option is documented in ``docs/api.md``.
"""
from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import doc_blocks  # noqa: E402
import srspd  # noqa: E402

ROOT = doc_blocks.repo_root()
API = ROOT / "docs" / "api.md"


def _entry(text, name) -> bool:
    """A table row or heading that starts with the symbol: ``| `name(`` /
    ``| `name` `` / ``### `name``."""
    pat = re.compile(r"^(?:\|\s*|#+\s+)`" + re.escape(name) + r"[`(\[ ]", re.MULTILINE)
    return bool(pat.search(text))


@pytest.mark.parametrize("doc", ["AGENTS.md", "packages/srspd/README.md"])
def test_runnable_examples_execute(doc, tmp_path, monkeypatch):
    ns = doc_blocks.run_file(ROOT / doc, tmp_path, monkeypatch)
    m = ns.get("model")
    if m is not None and hasattr(m, "named_parameters"):
        assert srspd.restore(m) == [], f"{doc} leaves an applied erasure on the model"


def test_agents_md_states_the_five_rules():
    text = (ROOT / "AGENTS.md").read_text()
    for needle in ("`X` is the target set", "`W_new = W − M`", "Rank truncation acts on `M`",
                   "measured and reported for every edited matrix", "The dose is `mu_per_token`",
                   "f = γ m / (1 + γ m)", "suggest_mu_per_token", "ErasureConventionError",
                   "#8", "#9", "#10", "#11"):
        assert needle in text, needle


def test_every_public_symbol_is_documented():
    import srspd.capture
    import srspd.models
    text = API.read_text()
    missing = [f"{mod.__name__}.{n}" for mod in (srspd, srspd.models, srspd.capture)
               for n in mod.__all__ if not n.startswith("__") and not _entry(text, n)]
    assert not missing, f"undocumented in docs/api.md: {missing}"


def _cli_surface(parser):
    """``{subcommand: [option strings]}`` of an argparse parser with subcommands."""
    import argparse
    out = {}
    for act in parser._actions:
        if isinstance(act, argparse._SubParsersAction):
            for name, sub in act.choices.items():
                out[name] = [o for a in sub._actions for o in a.option_strings
                             if o not in ("-h", "--help")]
    return out


def test_every_cli_command_and_option_is_documented():
    from srspd.cli import build_parser
    text = API.read_text()
    missing = []
    for cmd, opts in _cli_surface(build_parser()).items():
        if f"srspd {cmd}" not in text:
            missing.append(f"srspd {cmd}")
        missing += [f"srspd {cmd} {o}" for o in opts if f"`{o}" not in text]
    assert not missing, f"undocumented in docs/api.md: {missing}"
