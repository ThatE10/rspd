"""A frozen record of what the solve returns on a tiny Mistral.

``tests/fixtures/golden_tiny_mistral.*`` holds pristine weights, a target and a
retained token set, and the leaks and edit size of every matrix at every
solved setting, under both solvers. It was written by ``tests/make_golden.py``
at a commit where the live parity tests against the reference implementation
passed, so this test carries that agreement to machines without the reference
checkout (and across ``torch`` / ``transformers`` versions: the weights are
loaded, not re-initialised).
"""
from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")

import make_golden  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
REL = 1e-6


@pytest.fixture(scope="module")
def golden():
    tensors = safetensors.load_file(str(FIX / "golden_tiny_mistral.safetensors"))
    meta = json.loads((FIX / "golden_tiny_mistral.json").read_text())
    return tensors, meta


def _model(tensors):
    model = transformers.MistralForCausalLM(make_golden.config()).eval()
    state = {k[len("model."):]: v for k, v in tensors.items() if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    return model


@pytest.mark.parametrize("solver", ["readout", "grams"])
def test_the_solve_matches_the_frozen_record(golden, solver):
    tensors, meta = golden
    got = make_golden.solve(_model(tensors), tensors["target"], tensors["retained"], solver)
    want = meta["expected"][solver]
    assert set(got) == set(want) and len(want) == (6 if solver == "readout" else 2)
    for setting, rows in want.items():
        assert set(got[setting]) == set(rows) and len(rows) == 14
        for name, row in rows.items():
            for field, value in row.items():
                assert got[setting][name][field] == pytest.approx(value, rel=REL, abs=1e-9), \
                    (setting, name, field)


def test_the_record_is_not_trivial(golden):
    _, meta = golden
    full = meta["expected"]["readout"]["1.0|None"]
    assert all(0 < r["L_X_rel"] < 1 and 0 < r["L_Y_rel"] < 1 and r["edit_fro"] > 0
               for r in full.values())
    weak = meta["expected"]["readout"]["0.1|None"]
    assert sum(r["edit_fro"] for r in full.values()) > sum(r["edit_fro"] for r in weak.values())
