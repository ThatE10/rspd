"""``rspd-exp`` -- the reproduction command line.

    rspd-exp list
    rspd-exp show <result> [--json]
    rspd-exp reproduce <result> --mode cheap|full [--dry-run] [--export ID ...]
    rspd-exp compare <result> <results.json> [--export ID] [--json]

``reproduce --dry-run`` resolves the recipe into its full step list (data
preparation, training or weight download, evaluation) and prints it as JSON
without touching the network or loading a model. ``compare`` exits 0 only when
every published number of the exports present in the results file is matched
within its tolerance (and the tensor-parallel degree matches the recipe's).
"""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_list(a) -> int:
    from .recipes import catalog as C
    for r in C.list_results():
        R = C.load_recipe(r)
        n = sum(len(run["exports"]) for run in R["runs"])
        gaps = len(C.weight_gaps(R))
        print(f"{r:28s} {n:3d} exports ({n - gaps} with published weights), TP "
              f"{R['tensor_parallel_size']}  -- {R['title']}")
    return 0


def _cmd_show(a) -> int:
    from .recipes import catalog as C
    R = C.load_recipe(a.result)
    if a.json:
        print(json.dumps(R, indent=1, default=str))
        return 0
    print(f"# {R['result']}: {R['title']}")
    print(f"model {R['model']} · TP {R['tensor_parallel_size']} · eval {R['eval']['kind']}")
    for k, v in (R.get("hardware") or {}).items():
        print(f"hardware ({k}): {v}")
    for run, e in C.iter_exports(R):
        meth = "baseline" if run is None else run["method"]
        dose = (f" mu_per_token {e['mu_per_token']:.6g}" if e.get("mu_per_token") is not None else "")
        rule = f" rank_rule {e['rank_rule']}" if e.get("rank_rule") is not None else ""
        wts = e.get("weights")
        wl = ("" if run is None else
              f" · weights {wts['repo']}" + (f"/{wts['subfolder']}" if wts and wts.get("subfolder")
                                             else "") if wts else " · weights: not published")
        pub = ", ".join(f"{p['task']}/{p['metric']}={p['value']:g}" for p in e.get("published") or [])
        print(f"- {e['id']} [{meth}]{dose}{rule}{wl}\n    published: {pub or '--'}")
    return 0


def _cmd_reproduce(a) -> int:
    from .recipes import catalog as C
    R = C.load_recipe(a.result)
    p = C.plan(R, a.mode, exports=a.export or None)
    if a.dry_run:
        print(json.dumps(p, indent=1, default=str))
        return 0
    from .recipes.reproduce import execute
    path = execute(p, python_authors_stack=a.authors_python)
    print(f"[reproduce] results -> {path}")
    return 0


def _cmd_compare(a) -> int:
    from .recipes import catalog as C
    from .recipes.compare import compare, format_report
    from .results import read_results
    R = C.load_recipe(a.result)
    rep = compare(R, read_results(a.results), export=a.export)
    print(json.dumps(rep, indent=1, default=str) if a.json else format_report(rep))
    return 0 if rep["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rspd-exp", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="the shipped recipes").set_defaults(fn=_cmd_list)
    s = sub.add_parser("show", help="one recipe: exports, doses, weights, published numbers")
    s.add_argument("result")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_show)
    r = sub.add_parser("reproduce", help="train (full) or download (cheap), then evaluate")
    r.add_argument("result")
    r.add_argument("--mode", choices=("cheap", "full"), required=True)
    r.add_argument("--dry-run", action="store_true", help="print the resolved plan, run nothing")
    r.add_argument("--export", action="append", help="restrict to this export id (repeatable)")
    r.add_argument("--authors-python", default=None,
                   help="interpreter carrying the authors' stack (KUDA's MUSE code)")
    r.set_defaults(fn=_cmd_reproduce)
    c = sub.add_parser("compare", help="check a results file against the published numbers")
    c.add_argument("result")
    c.add_argument("results")
    c.add_argument("--export", default=None,
                   help="read every row as this export (when the file's model field is a path)")
    c.add_argument("--json", action="store_true")
    c.set_defaults(fn=_cmd_compare)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
