# rspd

Closed-form selective erasure of a behaviour from a model's linear layers, by subtraction.

For each selected weight `W`, with `X` the input activations of the behaviour to erase
(the **target** set) and `Y` those of the behaviour to keep (the **retained** set), `rspd`
computes the erased component

    M = W XᵀX (XᵀX + μ YᵀY)⁺

and writes `W_new = W − M`. No gradient descent. The dose is `mu_per_token`; the leaks
`L_X` (target response surviving) and `L_Y` (retained response disturbed) are reported for
every edited matrix. Rank truncation acts on `M` through the COALA projector over `X`.

Agents: read `AGENTS.md` at the repository root first. Method: `docs/math.md`. API:
`docs/api.md`.

## Install

The repository is private; there is no public release. Install from git at a pinned commit
(your git needs read access to the repository, over HTTPS credentials or SSH):

```bash
pip install "rspd[hf] @ git+https://github.com/ThatE10/rspd@<sha>"
# core only (torch): pip install "git+https://github.com/ThatE10/rspd@<sha>"
# over SSH:          pip install "rspd[hf] @ git+ssh://git@github.com/ThatE10/rspd.git@<sha>"
```

| extra | adds | needed for |
| --- | --- | --- |
| (none) | `torch>=2.0` | the tensor API: solve, truncation, `apply_erasure`, leaks, dose, COALA |
| `hf` | `transformers`, `safetensors` | `rspd.api.erase` / `inspect_model` / export on Hugging Face models, the `rspd` CLI |
| `test` | `pytest`, `numpy` | the test suite |

`import rspd` needs only `torch`; `transformers` is imported where it is used. Building a
wheel and pinning from another repository: `docs/releasing.md`.

## Minimal example

```python runnable
import tempfile
import torch
import transformers
import rspd

torch.manual_seed(0)                       # a tiny random GPT-2; use any HF causal LM
model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
    vocab_size=128, n_embd=16, n_layer=3, n_head=2, n_positions=64,
    bos_token_id=0, eos_token_id=0)).eval()
target = torch.randint(4, 66, (3, 24))     # X: token ids of the behaviour to erase
retained = torch.randint(66, 128, (6, 32)) # Y: token ids of the behaviour to keep

print(rspd.inspect_model(model)["totals"])               # what would be edited, Gram bytes
report = rspd.api.erase(model, target=target, retained=retained,
                     mu_per_token=[0.3, 1.0, 3.0], apply_mu_per_token=1.0)
print(report.matrices["L01.mlp.c_fc"]["L_X_rel"],         # leaks of every edited matrix
      report.matrices["L01.mlp.c_fc"]["L_Y_rel"])
report.apply_dose(3.0)                     # re-dose without a second pass over the data
rspd.export_checkpoint(model, None, tempfile.mkdtemp())  # save the edited model
rspd.api.restore(model)                       # bit for bit
report.close()
```

On raw tensors:

```python runnable
W, X, Y = torch.randn(8, 6), torch.randn(40, 6), torch.randn(200, 6)
dose = rspd.core.dose_from_mu_per_token(1.0, n_x=X.shape[0], n_y=Y.shape[0])
comp = rspd.core.erased_component(W, X, Y, mu=dose.mu)
W_new = rspd.core.apply_erasure(W, comp)                      # W − M, the only edit
print(rspd.core.leaks(W, comp, X, Y)["L_X_rel"], dose.as_dict())
```

Command line (`[hf]` extra), each command printing one JSON document:

```bash
rspd inspect MODEL [--layers SPEC]
rspd erase MODEL --target target.txt --retained retained.txt --mu-per-token 0.3 1 3 \
    --apply-mu-per-token 1 --out edited/ --report report.json
rspd erase MODEL --target target.txt --retained retained.txt --mu-per-token 0.3 1 3 \
    --no-apply --spectrum full --report grid.json      # solve only: leaks of every dose
rspd report report.json
```

`MODEL` is a hub id or a local checkpoint directory; the text files hold one document per
line.

## Model families

Verified end to end on tiny random-weight models of every family below (inspect, erase with
both solvers, leaks for every matrix, bit-for-bit restore, export), and used at full scale by
the recipes of `rspd[experiments]`:

| family | matrices per block | notes |
| --- | --- | --- |
| GPT-2 | `attn.c_attn` (fused QKV), `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj` | `Conv1D` stores `W` transposed; handled |
| Mistral (incl. Zephyr), Llama, Qwen3 | `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` | q/k/v and gate/up share one Gram each |
| Gemma-3 (text and multimodal) | as above | vision tower excluded |
| Gemma-4 (text and multimodal, incl. KV-shared layers) | as above (no `k_proj`/`v_proj` on KV-shared layers) | towers excluded; export backfills the KV-shared tensors |

Other architectures work when their projections are `nn.Linear` (or `Conv1D`) and you pass
a `layers=` regex whose group 1 is the block index and group 2 the matrix tail. Biased
projections need `subtract_bias=True` (the bias is never edited).

## Credits

Rank truncation of `M`, and the separate compression utilities (`fit_coala`,
`compress_weight`), use the context-aware low-rank projector of **COALA**: Uliana Parkina
and Maxim Rakhuba, *COALA: Numerically Stable and Efficient Framework for Context-Aware
Low-Rank Approximation*, arXiv:2507.07580, NeurIPS 2025 (Proposition 1). The implementation
is written independently from the paper's statement. The effective-rank rule uses Roy and
Vetterli, *The effective rank: A measure of effective dimensionality*, EUSIPCO 2007. See
`CITATION.cff`.
