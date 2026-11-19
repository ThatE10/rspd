"""Regenerate ``tests/fixtures/golden_tiny_mistral.*`` (maintainers only).

    python tests/make_golden.py

Run it only at a commit where the live parity tests against the reference
implementation pass (``tests/test_driver_parity.py``,
``tests/test_parity_reference.py``; they need the reference checkout). The
fixture then carries that agreement to machines that do not have the
reference: pristine weights, the two token sets, and what the solve returned.
"""
from __future__ import annotations

import json
import pathlib

import torch
import transformers
from safetensors.torch import save_file

import rspd

HERE = pathlib.Path(__file__).resolve().parent / "fixtures"
DOSES = [0.1, 1.0]
RULES = [0.01, "erank"]


def config():
    return transformers.MistralConfig(
        vocab_size=128, hidden_size=16, intermediate_size=40, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        bos_token_id=1, eos_token_id=2, pad_token_id=0)


def solve(model, target, retained, solver):
    """``{setting: {matrix: {L_X_rel, L_Y_rel, edit_fro}}}`` for every solved setting."""
    rules = RULES if solver == "readout" else None
    rep = rspd.api.erase(model, target=target, retained=retained, mu_per_token=DOSES,
                         solver=solver, rank_rule=rules, ridge_rel=1e-10, apply=False)
    out = {}
    for s in rep.settings:
        rep.apply_dose(s["mu_per_token"], s["rank_rule"])
        key = f"{s['mu_per_token']}|{s['rank_rule']}"
        out[key] = {n: {"L_X_rel": r["L_X_rel"], "L_Y_rel": r["L_Y_rel"], "edit_fro": r["edit_fro"]}
                    for n, r in rep.matrices.items()}
    rep.close()
    return out


def main():
    torch.manual_seed(0)
    model = transformers.MistralForCausalLM(config()).eval()
    g = torch.Generator().manual_seed(1)
    target = torch.randint(4, 66, (3, 24), generator=g)
    retained = torch.randint(66, 128, (8, 32), generator=g)
    HERE.mkdir(exist_ok=True)
    tensors = {f"model.{k}": v.detach().clone() for k, v in model.state_dict().items()}
    tensors.update(target=target, retained=retained)
    save_file(tensors, str(HERE / "golden_tiny_mistral.safetensors"))
    expected = {solver: solve(model, target, retained, solver) for solver in ("readout", "grams")}
    meta = {"torch": torch.__version__, "transformers": transformers.__version__,
            "doses": DOSES, "rank_rules": [str(r) for r in RULES], "expected": expected}
    (HERE / "golden_tiny_mistral.json").write_text(json.dumps(meta, indent=1) + "\n")
    print("wrote", HERE)


if __name__ == "__main__":
    main()
