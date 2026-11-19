"""The core package imports with torch alone: ``transformers`` / ``safetensors``
are imported lazily where used. Run in a subprocess with
both blocked by a meta-path finder."""
from __future__ import annotations

import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

CODE = r"""
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("transformers", "safetensors", "huggingface_hub"):
            raise ImportError(f"blocked: {name}")
        return None
sys.meta_path.insert(0, _Block())
import torch
import rspd, rspd.models, rspd.capture, rspd.pipeline, rspd.api, rspd.cli
import rspd.models.export, rspd.models.inspect, rspd.models.quirks
assert not any(m.split(".")[0] in ("transformers", "safetensors") for m in sys.modules)
# a plain torch model with decoder-style names works end to end without transformers
from types import SimpleNamespace
class Blk(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.mlp = torch.nn.Module()
        s.mlp.up_proj = torch.nn.Linear(6, 10, bias=False)
        s.mlp.down_proj = torch.nn.Linear(10, 6, bias=False)
    def forward(s, x):
        return x + s.mlp.down_proj(torch.relu(s.mlp.up_proj(x)))
class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        torch.manual_seed(0)
        s.emb = torch.nn.Embedding(50, 6)
        s.layers = torch.nn.ModuleList([Blk(), Blk()])
        s.config = SimpleNamespace(hidden_size=6)
    def get_input_embeddings(s):
        return s.emb
    def forward(s, input_ids=None, use_cache=None):
        x = s.emb(input_ids)
        for b in s.layers:
            x = b(x)
        return x
m = Net()
before = {n: p.detach().clone() for n, p in m.named_parameters()}
g = torch.Generator().manual_seed(0)
rep = rspd.api.erase(m, target=[torch.randint(0, 25, (12,), generator=g)],
                  retained=torch.randint(25, 50, (4, 16), generator=g), mu_per_token=1.0)
assert len(rep.matrices) == 4 and rep.applied
rspd.api.restore(m)
assert all(torch.equal(before[n], p) for n, p in m.named_parameters())
info = rspd.inspect_model(m)
assert set(info["input_keys"]) == {"L00.mlp_in", "L00.down_in", "L01.mlp_in", "L01.down_in"}
assert not any(m.split(".")[0] in ("transformers", "safetensors") for m in sys.modules)
print("OK")
"""


def test_core_import_and_erase_without_transformers():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True,
                       env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin"}, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), r.stderr[-2000:]
