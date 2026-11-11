"""``srspd`` command line. Every command prints ONE JSON document on stdout;
on failure it prints ``{"error": ..., "type": ...}`` and exits with status 2.

    srspd inspect MODEL [--layers SPEC] [--subtract-bias] [--gram-budget-gib G]
    srspd erase MODEL --target FILE --retained FILE --mu-per-token M [M ...]
                [--apply-mu-per-token M] [--solver readout|grams] [--rank-rule R ...]
                [--apply-rank-rule R|full]
                [--ridge-rel R] [--layers SPEC] [--dtype bfloat16|float16|float32]
                [--device DEV] [--max-length N] [--out DIR] [--report PATH]
                [--scratch-dir DIR] [--gram-budget-gib G]
    srspd report PATH [--top N]

``MODEL`` is a hub id or a local checkpoint directory. ``--layers`` is a preset
name (``decoder``, ``gpt2``), a regex with groups (block index, matrix tail),
comma-separated block indices (``0,1,5``) or a JSON object
(``{"types": ["down_proj"], "blocks": [3]}``). ``erase`` reads one document per
non-empty line of each text file, tokenises it with the model's tokenizer and
cuts it to ``--max-length`` tokens (each document its own forward pass);
``--out`` writes the edited checkpoint (and the tokenizer) there, with the
KV-shared backfill from ``MODEL``. ``--rank-rule`` may be repeated (every rule
is truncated from the same solve); with more than one, ``--apply-rank-rule``
names the one to apply (``full`` = the full-rank edit), as ``--apply-mu-per-token``
does for a list of doses.
"""
from __future__ import annotations

import argparse
import json
import sys

__all__ = ["main", "parse_layers"]


def parse_layers(spec):
    """The ``--layers`` string -> the ``layers=`` argument of :func:`srspd.erase`."""
    if spec is None:
        return None
    s = spec.strip()
    if s.startswith("{"):
        return json.loads(s)
    parts = [p.strip() for p in s.split(",")]
    if all(p.isdigit() for p in parts):
        return [int(p) for p in parts]
    return s


def _rank_rule(v):
    if v is None or v == "erank":
        return v
    if v == "full":
        return None
    return float(v)


def _rank_rules(vals):
    """Repeated ``--rank-rule`` -> ``rank_rule=`` (``None``, one rule or a list)."""
    if not vals:
        return None
    rules = [_rank_rule(v) for v in vals]
    return rules[0] if len(rules) == 1 else rules


def _read_docs(path):
    with open(path, encoding="utf-8") as f:
        docs = [ln.strip() for ln in f]
    docs = [d for d in docs if d]
    if not docs:
        raise ValueError(f"{path} holds no non-empty lines")
    return docs


def _tokenize(tok, docs, max_length):
    import torch
    out = []
    for d in docs:
        ids = tok(d)["input_ids"][: int(max_length)]
        if ids:
            out.append(torch.as_tensor(ids, dtype=torch.long))
    return out


def _cmd_inspect(a):
    from .models.inspect import inspect_model
    return inspect_model(a.model, parse_layers(a.layers), subtract_bias=a.subtract_bias,
                         gram_budget_gib=a.gram_budget_gib)


def _cmd_erase(a):
    import torch
    import transformers

    from .api import erase
    from .models.export import export_checkpoint
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        a.model, dtype=getattr(torch, a.dtype)).to(a.device).eval()
    target = _tokenize(tok, _read_docs(a.target), a.max_length)
    retained = _tokenize(tok, _read_docs(a.retained), a.max_length)
    mpt = a.mu_per_token if len(a.mu_per_token) > 1 else a.mu_per_token[0]
    extra = ({} if a.apply_rank_rule is None
             else {"apply_rank_rule": _rank_rule(a.apply_rank_rule)})
    rep = erase(model, tok, target=target, retained=retained, mu_per_token=mpt,
                apply_mu_per_token=a.apply_mu_per_token, layers=parse_layers(a.layers),
                solver=a.solver, rank_rule=_rank_rules(a.rank_rule), ridge_rel=a.ridge_rel,
                gram_budget_gib=a.gram_budget_gib, scratch_dir=a.scratch_dir,
                subtract_bias=a.subtract_bias, **extra)
    out = rep.to_dict()
    if a.out:
        out["export"] = export_checkpoint(model, tok, a.out, source=a.model)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(out, f, indent=1)
    rep.close()
    return out


def _cmd_report(a):
    with open(a.path) as f:
        r = json.load(f)
    mats = r.get("matrices") or {}

    def worst(field):
        rows = sorted(((n, v.get(field)) for n, v in mats.items() if v.get(field) is not None),
                      key=lambda kv: -float(kv[1]))
        return [{"matrix": n, field: v} for n, v in rows[: a.top]]
    return {"solver": r.get("solver"), "convention": r.get("convention"),
            "applied": r.get("applied"), "dose": r.get("dose"), "n_x": r.get("n_x"),
            "n_y": r.get("n_y"), "rank_rule": r.get("rank_rule"),
            "rank_rules": r.get("rank_rules"), "matrices": len(mats),
            "leaks_by_type": r.get("leaks_by_type"), "edit": r.get("edit"),
            "worst_L_X_rel": worst("L_X_rel"), "worst_L_Y_rel": worst("L_Y_rel"),
            "doses": [d.get("mu_per_token") for d in r.get("doses") or []],
            "quirks": sorted((r.get("quirks") or {}).keys()), "warnings": r.get("warnings"),
            "export": r.get("export")}


def build_parser():
    ap = argparse.ArgumentParser(prog="srspd", description="Closed-form selective erasure "
                                 "(W_new = W - M); every command prints JSON")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inspect", help="what an erasure would edit: matrices, input keys, "
                       "Gram bytes, quirks")
    p.add_argument("model")
    p.add_argument("--layers")
    p.add_argument("--subtract-bias", action="store_true")
    p.add_argument("--gram-budget-gib", type=float)
    p.set_defaults(fn=_cmd_inspect)

    p = sub.add_parser("erase", help="erase the target text's behaviour, keep the retained "
                       "text's; prints the EditReport")
    p.add_argument("model")
    p.add_argument("--target", required=True, help="text file, one document per line (X)")
    p.add_argument("--retained", required=True, help="text file, one document per line (Y)")
    p.add_argument("--mu-per-token", type=float, nargs="+", required=True)
    p.add_argument("--apply-mu-per-token", type=float)
    p.add_argument("--solver", choices=("readout", "grams"), default="readout")
    p.add_argument("--rank-rule", action="append",
                   help="lam_rel in (0, 1) or 'erank' (readout solver); repeat for several "
                        "rules, all truncated from one solve")
    p.add_argument("--apply-rank-rule",
                   help="which solved rule to apply: a --rank-rule value or 'full' (needed "
                        "with more than one --rank-rule)")
    p.add_argument("--ridge-rel", type=float, default=0.0)
    p.add_argument("--layers")
    p.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--gram-budget-gib", type=float)
    p.add_argument("--scratch-dir")
    p.add_argument("--subtract-bias", action="store_true")
    p.add_argument("--out", help="write the edited checkpoint here")
    p.add_argument("--report", help="also write the report JSON here")
    p.set_defaults(fn=_cmd_erase)

    p = sub.add_parser("report", help="summarise a saved EditReport JSON")
    p.add_argument("path")
    p.add_argument("--top", type=int, default=5)
    p.set_defaults(fn=_cmd_report)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        out = a.fn(a)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}))
        return 2
    print(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
