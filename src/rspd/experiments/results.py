"""The common result schema every evaluator in this package writes.

A result file is JSON::

    {"schema": "rspd[experiments]/results/v1",
     "rows": [{"task", "metric", "value", "model", "tp", "n", "meta"}, ...]}

One row per number:

* ``task``   -- what was scored (``wmdp_bio``, ``global_mmlu_nl_local``,
  ``muse_news``, ``recovery``, ``heldout_ce``, ...);
* ``metric`` -- the quantity (``acc``, ``word_perplexity``, ``krd``,
  ``verbmem_f``, ``ce``, ``M_tokens``, ...);
* ``value``  -- a float (``None`` when the evaluation could not produce it);
* ``model``  -- the checkpoint or hub id that was scored;
* ``tp``     -- tensor-parallel degree of the evaluation, or ``None`` when the
  evaluator does not shard (HF single-device). TP changes the reduction order
  of sharded matmuls and therefore the logprobs: rows with different ``tp``
  are not directly comparable;
* ``n``      -- the number of items / tokens behind the value (``None`` if unknown);
* ``meta``   -- anything else (stderr, backend, n-shot, config hash, ...).

The recipe layer compares these rows to published numbers by ``(task, metric)``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

SCHEMA = "rspd[experiments]/results/v1"
ROW_KEYS = ("task", "metric", "value", "model", "tp", "n", "meta")


@dataclass
class ResultRow:
    task: str
    metric: str
    value: float | None
    model: str | None = None
    tp: int | None = None
    n: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task:
            raise ValueError(f"task must be a non-empty string, got {self.task!r}")
        if not isinstance(self.metric, str) or not self.metric:
            raise ValueError(f"metric must be a non-empty string, got {self.metric!r}")
        if self.value is not None:
            if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
                raise ValueError(f"value must be a number or None, got {self.value!r}")
            self.value = float(self.value)
            if math.isnan(self.value):
                self.value = None
        if self.tp is not None:
            self.tp = int(self.tp)
        if self.n is not None:
            self.n = int(self.n)
        self.meta = dict(self.meta or {})

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResultRow":
        unknown = set(d) - set(ROW_KEYS)
        if unknown:
            raise ValueError(f"unknown result keys {sorted(unknown)}; expected {ROW_KEYS}")
        return cls(**{k: d.get(k) for k in ROW_KEYS if k in d})


def write_results(path: str, rows: Iterable[ResultRow | dict], **header: Any) -> str:
    """Write ``rows`` (``ResultRow`` or dicts) as a result file; extra keyword
    arguments go into the top-level object beside ``schema`` / ``rows``.
    Atomic (written to ``.tmp`` then renamed). Returns ``path``."""
    rows = [r if isinstance(r, ResultRow) else ResultRow.from_dict(r) for r in rows]
    payload = {"schema": SCHEMA, **header, "rows": [r.as_dict() for r in rows]}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=str)
    os.replace(tmp, path)
    return path


def read_results(path: str) -> list[ResultRow]:
    with open(path) as f:
        payload = json.load(f)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {payload.get('schema')!r}, expected {SCHEMA!r}")
    return [ResultRow.from_dict(r) for r in payload["rows"]]


def index(rows: Iterable[ResultRow]) -> dict:
    """``{(task, metric): row}``; a duplicate key raises (ambiguous comparison)."""
    out = {}
    for r in rows:
        k = (r.task, r.metric)
        if k in out:
            raise ValueError(f"duplicate result row {k}")
        out[k] = r
    return out
