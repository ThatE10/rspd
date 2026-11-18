"""Compare measured result rows with a recipe's published numbers.

A measured row matches a published entry when its ``task`` and ``metric`` agree
and its ``model`` is the export id (what ``rspd-exp reproduce`` writes) -- or,
with ``export=`` given, every row is read as belonging to that export. Each
published number passes when ``|measured - published| <= tol`` (absolute) or
``<= rel_tol * |published|`` (relative); a published number without a measured
row is ``missing`` (a failure).
"""
from __future__ import annotations

from .catalog import iter_exports


def _ok(entry, measured):
    d = abs(measured - entry["value"])
    if "tol" in entry:
        return d <= entry["tol"] + 1e-12, entry["tol"]
    lim = entry["rel_tol"] * abs(entry["value"])
    return d <= lim + 1e-12, lim


def compare(recipe, rows, *, export=None) -> dict:
    """``{"result", "checks": [...], "n_pass", "n_fail", "n_missing", "ok"}``;
    one check per published number of the exports present in ``rows``."""
    by = {}
    for r in rows:
        key = (export or r.model, r.task, r.metric)
        by.setdefault(key, r)
    present = {export} if export else {r.model for r in rows}
    checks = []
    for _run, e in iter_exports(recipe):
        if e["id"] not in present:
            continue
        for p in e.get("published") or []:
            row = by.get((e["id"], p["task"], p["metric"]))
            c = {"export": e["id"], "task": p["task"], "metric": p["metric"],
                 "published": p["value"], "tol": p.get("tol"), "rel_tol": p.get("rel_tol")}
            if row is None or row.value is None:
                c.update(measured=None, status="missing")
            else:
                ok, lim = _ok(p, row.value)
                c.update(measured=row.value, delta=row.value - p["value"], limit=lim,
                         status="pass" if ok else "fail", tp=row.tp)
            checks.append(c)
    n = {s: sum(c["status"] == s for c in checks) for s in ("pass", "fail", "missing")}
    tp_bad = [c for c in checks if c.get("tp") not in (None, recipe["tensor_parallel_size"])]
    return {"result": recipe["result"], "checks": checks, "n_pass": n["pass"],
            "n_fail": n["fail"], "n_missing": n["missing"],
            "tp_mismatch": [f"{c['export']}:{c['task']}:{c['metric']} tp={c['tp']}"
                            for c in tp_bad],
            "ok": bool(checks) and n["fail"] == 0 and n["missing"] == 0 and not tp_bad}


def format_report(rep) -> str:
    lines = [f"# compare {rep['result']}: {rep['n_pass']} pass, {rep['n_fail']} fail, "
             f"{rep['n_missing']} missing" + ("" if not rep["tp_mismatch"] else
                                              f", {len(rep['tp_mismatch'])} TP mismatch")]
    for c in rep["checks"]:
        tol = f"±{c['tol']:g}" if c.get("tol") is not None else f"±{c['rel_tol']:g} rel"
        m = "--" if c["measured"] is None else f"{c['measured']:.4g}"
        lines.append(f"{c['status'].upper():7s} {c['export']:40s} {c['task']}/{c['metric']}: "
                     f"measured {m} vs published {c['published']:.4g} ({tol})")
    for t in rep["tp_mismatch"]:
        lines.append(f"TP      {t} (recipe TP {rep.get('tp', '')})")
    return "\n".join(lines)
