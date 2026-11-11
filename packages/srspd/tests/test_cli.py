"""``srspd inspect | erase | report`` on a tiny local checkpoint: every command
prints one JSON document."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")

import tiny_models as tm  # noqa: E402
from srspd.cli import main, parse_layers  # noqa: E402


@pytest.fixture
def ckpt(tmp_path):
    d = tmp_path / "ckpt"
    tm.mistral().save_pretrained(d)
    tm.word_tokenizer().save_pretrained(d)
    return d


def _run(capsys, argv):
    code = main(argv)
    out = capsys.readouterr().out
    return code, json.loads(out)


def test_parse_layers():
    assert parse_layers(None) is None and parse_layers("gpt2") == "gpt2"
    assert parse_layers("0,2") == [0, 2]
    assert parse_layers('{"types": ["q_proj"]}') == {"types": ["q_proj"]}


def test_inspect_by_path_loads_no_weights(capsys, ckpt):
    code, out = _run(capsys, ["inspect", str(ckpt), "--gram-budget-gib", "1"])
    assert code == 0 and out["totals"]["matrices"] == 14
    assert all(v["device"] == "meta" for v in out["matrices"].values())
    assert "readout" in out["pass_plan"]


def test_erase_then_report(capsys, ckpt, tmp_path):
    tgt = tmp_path / "t.txt"
    ret = tmp_path / "r.txt"
    tgt.write_text("\n".join(" ".join(f"w{(i * 5 + j) % 60}" for j in range(20))
                             for i in range(3)))
    ret.write_text("\n".join(" ".join(f"w{60 + (i * 3 + j) % 60}" for j in range(30))
                             for i in range(6)) + "\n\n")
    out_dir = tmp_path / "edited"
    rpath = tmp_path / "report.json"
    code, rep = _run(capsys, ["erase", str(ckpt), "--target", str(tgt), "--retained", str(ret),
                              "--mu-per-token", "1.0", "--dtype", "float32", "--out",
                              str(out_dir), "--report", str(rpath)])
    assert code == 0 and rep["applied"] and rep["n_x"] == 60 and rep["n_y"] == 180
    assert len(rep["matrices"]) == 14 and rep["export"]["dir"] == str(out_dir)
    edited = transformers.AutoModelForCausalLM.from_pretrained(out_dir)
    base = transformers.AutoModelForCausalLM.from_pretrained(ckpt)
    w = "model.layers.0.self_attn.q_proj.weight"
    assert not torch.equal(dict(edited.named_parameters())[w], dict(base.named_parameters())[w])
    code, summ = _run(capsys, ["report", str(rpath), "--top", "2"])
    assert code == 0 and summ["matrices"] == 14 and len(summ["worst_L_X_rel"]) == 2
    assert summ["dose"]["mu_per_token"] == 1.0


def test_errors_are_json(capsys, tmp_path):
    code, out = _run(capsys, ["report", str(tmp_path / "missing.json")])
    assert code == 2 and out["type"] == "FileNotFoundError"
