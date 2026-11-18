"""The common result schema: validation, NaN -> None, round trip, indexing."""
from __future__ import annotations

import json

import pytest

from rspd.experiments.results import SCHEMA, ResultRow, index, read_results, write_results


def test_row_validation_and_coercion():
    r = ResultRow("t", "acc", 1, tp="2", n=5.0)
    assert r.value == 1.0 and r.tp == 2 and r.n == 5 and r.meta == {}
    assert ResultRow("t", "acc", float("nan")).value is None
    for bad in ({"task": ""}, {"metric": ""}, {"value": True}, {"value": "0.5"}):
        kw = {"task": "t", "metric": "m", "value": 0.5, **bad}
        with pytest.raises(ValueError):
            ResultRow(**kw)
    with pytest.raises(ValueError, match="unknown result keys"):
        ResultRow.from_dict({"task": "t", "metric": "m", "value": 1, "extra": 1})


def test_round_trip_and_header(tmp_path):
    p = str(tmp_path / "r.json")
    rows = [ResultRow("wmdp_bio", "acc", 0.3, model="m", tp=1, n=1273, meta={"stderr": 0.01}),
            {"task": "wikitext", "metric": "word_perplexity", "value": 9.2}]
    write_results(p, rows, name="x")
    d = json.loads(open(p).read())
    assert d["schema"] == SCHEMA and d["name"] == "x"
    assert set(d["rows"][0]) == {"task", "metric", "value", "model", "tp", "n", "meta"}
    back = read_results(p)
    assert back[0] == rows[0] and back[1].tp is None
    assert set(index(back)) == {("wmdp_bio", "acc"), ("wikitext", "word_perplexity")}
    with pytest.raises(ValueError, match="duplicate"):
        index(back + back)
    with open(p, "w") as f:
        json.dump({"schema": "other", "rows": []}, f)
    with pytest.raises(ValueError, match="schema"):
        read_results(p)
