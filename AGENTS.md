# AGENTS.md

How to use `rspd` correctly without reading its source. `rspd` erases a behaviour from a
model's linear layers in closed form: for each selected weight `W` it computes an erased
component `M` from the activations of a **target** set (to erase) and a **retained** set
(to keep), and writes `W_new = W − M`. The math is in `docs/math.md`, the full API in
`docs/api.md`.

Python blocks marked `python runnable` in this file are executed by the test suite on a tiny
random-weight GPT-2 with no network (`tests/test_docs_core.py`).

## Setup

Either install from git at a pinned commit (the repository is private; your git needs read
access):

```bash
pip install "rspd[hf] @ git+https://github.com/ThatE10/rspd@<sha>"
```

or, inside a checkout whose environment already has `torch` and `transformers`, put the
sources on the path; the CLI is then `python -m rspd.cli` instead of `rspd`:

```bash
export PYTHONPATH=$PWD/src:$PWD/packages/rspd[experiments]/src
python -c "import rspd; print(rspd.__version__)"
python -m rspd.cli inspect gpt2        # == `rspd inspect gpt2` when installed
python -m pytest -q                     # the whole suite, from the repository root
```

`import rspd` needs only `torch`; the model-level API (`erase`, `inspect_model`, export,
the CLI) needs `transformers` and `safetensors` (the `[hf]` extra).

## The five rules, and why

1. **`X` is the target set (erased); `Y` is the retained set (kept). Never swap them.**
   Every formula is asymmetric: `M` reproduces `W` on `X` and is penalised on `Y`. Swapped
   sets erase what you meant to keep, and nothing fails loudly. In the API: `target=` is `X`,
   `retained=` is `Y`, and `L_X` / `L_Y` are their leaks.
2. **The edit is always `W_new = W − M`, through `rspd.core.apply_erasure`.** Never write a
   fitted map or the "kept map" (`W − M` restricted to the batch) into a model. Both edits
   agree on the activations you collected, but off the span of `X ∪ Y` the subtraction keeps
   `W`'s action and the kept map zeroes it (`rspd.core.CORE_TABLE`). `apply_erasure` raises
   `ErasureConventionError` for anything that is not an `ErasedComponent`.
3. **Rank truncation acts on `M`, never on `W` or the kept map.** `M_r = U_r U_rᵀ M` with
   `U_r` the leading left singular vectors of `M Xᵀ` (the COALA projector of `M` over the
   target; Parkina & Rakhuba, arXiv:2507.07580). Truncating `M` only coarsens *what is
   removed*; `W − M_r` still keeps `W Y` up to the leak. Truncating `W` or the kept map would
   throw away retained behaviour.
4. **The leaks `L_X`, `L_Y` are measured and reported for every edited matrix.**
   Disentanglement of the two sets is an empirical property of your data at each layer. It is
   never assumed to be zero, so it is always in the report.
5. **The dose is `mu_per_token`. `mu_raw` and `mu` are derived and only reported.** The
   solve's weight `mu` depends on how many tokens each set happens to hold; `mu_per_token` is
   the per-token trade-off and means the same thing at any set size. Never pass `mu` or
   `mu_raw` to a user-facing entry point (none accepts them).

## The dose

`m = mu_per_token`. The solve minimises, per token,

    E_x ‖(W − M) x‖²  +  (1/m) E_y ‖M y‖²

so `1/m` is the per-token Lagrange multiplier on the retained disturbance: **larger `m`
erases more**. The derived quantities (reported in `report.dose`) are
`mu = n_x / (m · n_y)` and `mu_raw = 1/mu = m · n_y / n_x`, with `n_x`, `n_y` the target and
retained token counts.

What a dose does, per direction. Let `γ` be the generalised eigenvalues of the per-token
target covariance against the per-token retained covariance (`C_X v = γ C_Y v`, with
`C_X = XᵀX/n_x`, `C_Y = YᵀY/n_y`). `γ` says how much more target than retained a direction
carries per token. The edit erases fraction

    f = γ m / (1 + γ m)

of each generalised eigendirection: half-erasure at `γ m = 1`, fraction `f` at
`m = f / ((1 − f) γ)`, nothing where `γ = 0`. This is exact at full rank with no ridge
(`ridge_rel=0`, no `rank_rule`) and approximate otherwise.

How to choose `m` from `γ` is step 3 of the workflow below.

## Standard workflow

The examples below assume `model` is a resident Hugging Face causal LM in `eval()` mode,
`tokenizer` its tokenizer (or `None` when you pass token ids), and `target` / `retained`
token ids. To run them with no download:

```python runnable
import torch
import transformers

torch.manual_seed(0)
model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
    vocab_size=128, n_embd=16, n_layer=3, n_head=2, n_positions=64,
    bos_token_id=0, eos_token_id=0)).eval()
tokenizer = None                                    # we pass token ids below
g = torch.Generator().manual_seed(1)
target = [torch.randint(4, 66, (24,), generator=g) for _ in range(3)]   # X: 3 sequences
retained = torch.randint(66, 128, (6, 32), generator=g)                 # Y: 6 sequences
```

Given a model name instead (downloads the weights; any HF causal LM):

```python
model = transformers.AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float32).eval()
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
target = ["one document of the behaviour to erase", "another one"]        # text needs tokenizer
retained = ["one document of the behaviour to keep", "another one"]
report = rspd.api.erase(model, tokenizer, target=target, retained=retained, mu_per_token=1.0)
```

Load in float32 when you can: on a bf16 model the edit is formed in fp32 and cast once, and a
small dose can fall below bf16 resolution (`survival` < 1 in the report).

`target` / `retained` accept a `LongTensor` (`[T]` one sequence, `[B, T]` B sequences), a
list of 1-D tensors or int lists, or (with `tokenizer=`) strings. Each sequence is its own
forward pass (no padding); every token of every sequence is a row of `X` (or `Y`).

### 1. Inspect

```python runnable
import rspd

info = rspd.inspect_model(model)          # also takes a hub id / local dir: config only, no weights
print(info["selector"]["preset"])          # "gpt2" (auto); decoders get "decoder"
print(info["totals"])                      # matrices, blocks, input_keys, edited_parameters, Gram bytes
print(list(info["input_keys"])[:4])        # one float64 Gram per input key
print(info["quirks"], info["warnings"])
```

Matrices are named `L{block:02d}.{tail}`: `L03.self_attn.q_proj`, `L03.mlp.down_proj` on
standard decoders (seven per block), `L00.attn.c_attn`, `L00.mlp.c_fc` on GPT-2 (four per
block; the fused `c_attn` is one matrix). Matrices that read the same input share an input
key and one Gram: `L03.attn_in` (q, k, v), `L03.o_in`, `L03.mlp_in` (gate, up),
`L03.down_in`; on GPT-2 each matrix is its own key. `totals["gram_bytes_readout"]` is the
memory the retained Grams need (`gram_bytes_grams`, twice that, for `solver="grams"`).

### 2. Choose layers

`layers=` (same argument for `inspect_model` and `erase`):

| value | selects |
| --- | --- |
| `None` | the first matching preset, every block |
| `"decoder"` / `"gpt2"` | that preset |
| `[1, 2]` | those blocks of the auto preset |
| `{"types": [...], "blocks": [...]}` | filter by matrix type (`"down_proj"`, `"c_fc"`) or whole tail (`"mlp.c_proj"`), and/or blocks; also keys `preset`, `regex`, `exclude` |
| any other string | a regex over `model.named_modules()` paths with group 1 = block index, group 2 = matrix tail |

A type matches every tail that ends in it: on GPT-2 `"c_proj"` selects both `attn.c_proj`
and `mlp.c_proj`; use the tail to pick one.

```python runnable
layers = {"types": ["mlp.c_fc", "mlp.c_proj"], "blocks": [1, 2]}
print(list(rspd.inspect_model(model, layers)["matrices"]))
# ['L01.mlp.c_fc', 'L01.mlp.c_proj', 'L02.mlp.c_fc', 'L02.mlp.c_proj']
```

### 3. Choose doses

Use the specificity spectrum `γ`:

* `report.specificity[key]["specificity"]` (readout solver) gives, per input key, the top
  eight `γ`, the median and how many exceed 1, 10 and 100. If few `γ` exceed 1 at a layer,
  the target is not separable there at any moderate dose.
* `rspd.core.suggest_mu_per_token(spectrum, fractions, quantiles)` returns the dose that erases
  the `q`-quantile direction by fraction `f`, for every `(q, f)`, plus a sorted grid. It
  needs the **full** spectrum: pass `spectrum="full"` to `erase` (readout solver) and hand
  it `report.specificity[key]["specificity"]` or `report.spectrum(key_or_matrix)`. A
  solve-only run (`apply=False`) at any dose gives you the spectrum; it does not depend on
  the dose:

```python runnable
probe = rspd.api.erase(model, target=target, retained=retained, layers=layers,
                    mu_per_token=1.0, apply=False, spectrum="full")
grid = rspd.core.suggest_mu_per_token(probe.spectrum("L01.mlp.c_fc"),
                                  fractions=(0.5, 0.9), quantiles=(0.5, 0.9))
print(grid["mu_per_token"])                             # pass this grid to erase
# grid["rows"]: one {"quantile", "gamma", "fraction", "mu_per_token"} per (q, f),
# quantile-major in the order given; pick a specific dose by its (q, f):
m_med_90 = next(r["mu_per_token"] for r in grid["rows"]
                if r["quantile"] == 0.5 and r["fraction"] == 0.9)
probe.close()
```

Which spectrum to pass: there is one per input key (matrices reading the same input share it),
and `suggest_mu_per_token` suggests one global grid from whichever you give it. Pass the key of
the layer you most need to erase at, or concatenate several keys' spectra to cover them all.
Per-key doses are not supported yet (issue #8).

Without a report, for one matrix from raw activations:

```python runnable
import torch
import rspd
from rspd.capture import harvest_inputs_concat
from rspd.models import HFLinearHarness, resolve_selector

h = HFLinearHarness(model, selector=resolve_selector(model, {"types": ["mlp.c_fc"], "blocks": [1]}))
name = h.all_names[0]                                   # "L01.mlp.c_fc"
seqs = lambda s: [torch.as_tensor(t).reshape(1, -1) for t in s]
acts_x, n_x = harvest_inputs_concat(h, seqs(target))    # {name: [n_x, d_in]} on CPU
acts_y, n_y = harvest_inputs_concat(h, seqs(retained))
Yd = acts_y[name].double()
spec = rspd.core.specificity(Yd.T @ Yd, acts_x[name], n_y, full=True)   # G_Y, X, n_y
grid = rspd.core.suggest_mu_per_token(spec, fractions=(0.5, 0.9), quantiles=(0.5, 0.9))
print(spec["top"][:3], grid["mu_per_token"])            # pass grid["mu_per_token"] to erase
```

A list of doses costs one set of Gram passes; every dose is solved and any can be applied
afterwards without touching the data again. Solving a grid and choosing by the leaks (and
by your own evaluation) is the normal workflow.

### 4. Erase

```python runnable
report = rspd.api.erase(model, target=target, retained=retained, layers=layers,
                     mu_per_token=[0.3, 1.0, 3.0], apply_mu_per_token=1.0)
print(report.applied, report.dose["mu_per_token"], report.dose["mu"], report.n_x, report.n_y)
```

The model is edited in place. `erase` solves every listed dose and applies exactly one:
with one dose it is applied; with several, pass `apply_mu_per_token=` or `apply=False`
(apply later with `report.apply_dose(m)`).

Rank truncation (readout solver only): `rank_rule=` takes a `lam_rel` in (0, 1), `"erank"`,
or a list. Every rule is sliced from one SVD per matrix and dose, and the full-rank edit is
always solved too. With more than one rule, `apply_rank_rule=` names the applied one
(`None` = full rank).

```python runnable
report.close()                                  # restores the model; see step 6
report = rspd.api.erase(model, target=target, retained=retained, layers=layers,
                     mu_per_token=1.0, rank_rule=[0.1, "erank"], apply_rank_rule="erank")
row = report.matrices["L01.mlp.c_fc"]
print(report.rank_rule, row["rank"], row["rank_bound"], row["erank"])
report.apply_dose(1.0, None)                    # the full-rank edit W − M of the same solve
report.apply_dose(1.0, 0.1)                     # W − M_r under lam_rel = 0.1
report.close()
report = rspd.api.erase(model, target=target, retained=retained, layers=layers,
                     mu_per_token=[0.3, 1.0, 3.0], apply_mu_per_token=1.0)
```

`solver="grams"` streams the target into a Gram instead of holding its activations: use it
when `n_x × d_in` activations per input key do not fit in host RAM. Same `M`, no
`rank_rule`, twice the Gram memory.

Command line (each command prints one JSON document; on failure it prints
`{"error": ..., "type": ...}` and exits 2):

```bash
rspd inspect gpt2 --layers '{"types": ["mlp.c_fc"]}'
rspd erase gpt2 --target target.txt --retained retained.txt \
    --mu-per-token 0.3 1 3 --apply-mu-per-token 1 --layers 1,2 \
    --dtype float32 --out edited/ --report report.json
rspd report report.json --top 5
```

`--target` / `--retained` are text files, one document per non-empty line, tokenised with
the model's tokenizer and cut to `--max-length` (default 512) tokens. `--dtype` defaults to
`bfloat16`, `--device` to `cpu`. `--out` writes the edited checkpoint and the tokenizer;
`--report` writes the report JSON. Repeat `--rank-rule` for several rules and pick one with
`--apply-rank-rule` (`full` = full rank). A list of doses (or rules) needs
`--apply-mu-per-token` (`--apply-rank-rule`), or `--no-apply` for a solve-only run that
applies nothing and writes only the report (no `--out`). `--spectrum full` keeps the whole
`γ` spectrum per input key in the report.

### 5. Read the report

```python runnable
row = report.matrices["L01.mlp.c_fc"]
print(row["L_X_rel"], row["L_Y_rel"], row["solve_path"], row["survival"])
print(report.leaks_by_type["c_fc"]["L_X_rel"])          # {median, min, max} over matrices
for m, d in report.by_dose.items():                     # every solved dose, keyed repr(float)
    print(m, d["leaks_by_type"]["c_fc"]["L_X_rel"]["median"])
print(report.edit["survival"], len(report.warnings))
import json, os, tempfile
path = os.path.join(tempfile.mkdtemp(), "report.json")
report.to_json(path)                                    # also returns the JSON string
assert json.load(open(path))["dose"]["mu_per_token"] == 1.0
```

`EditReport` fields (all JSON-serialisable via `to_dict()` / `to_json(path=None)`):

| field | meaning |
| --- | --- |
| `convention` | `"erase by subtraction: W_new = W - M (X = target, Y = retained)"` |
| `solver` | `"readout"` or `"grams"` |
| `applied` | whether this report's edit is currently in the model |
| `dose` | the applied dose: `mu_per_token`, derived `mu_raw` and `mu`, `n_x`, `n_y`, `formula`; `None` if nothing applied |
| `doses` | every solved dose, same form |
| `rank_rule` / `rank_rules` | the applied rule (`None` = full rank) / every solved rule |
| `settings` | every solved `(mu_per_token, rank_rule)` pair, the valid arguments of `apply_dose` |
| `ridge_rel` | the requested relative ridge |
| `n_x`, `n_y` | target / retained token counts |
| `matrices` | per edited matrix, at the applied setting (row fields below), in block order, then forward order within a block |
| `leaks_by_type` | `{type: {"n", "L_X_rel": {median,min,max}, "L_Y_rel": {...}}}` at the applied setting |
| `by_dose` | `{repr(mu_per_token): {"dose", "leaks": {matrix: {L_X_rel, L_Y_rel}}, "leaks_by_type", "truncated": {str(rule): {"leaks", "leaks_by_type", "rank_by_type"}}}}` for every solved setting |
| `edit` | the applied edit: `survival`, `rel_err`, `delta_fro_sum` (Σ‖M‖_F), `by_type` (survival per type) |
| `input_keys` | `{key: {members, d_in, gram_bytes, device}}` |
| `quirks` | model properties that changed handling (Conv1D layout, biased projections, multimodal towers, KV-shared layers, shared inputs, bf16 edits) |
| `selector` | the resolved layer selector |
| `pass_plan` | the Gram passes: `[{keys, gram_bytes, deepest_layer}]` |
| `specificity` | per input key: token counts, `d_in`, `lam_min`/`lam_max`/`lam_ratio` of `G_Y` (conditioning), `specificity` (`γ`: `top`, `median`, `n_above`, `n`, and with `spectrum="full"` the whole `spectrum`; readout only), `solve_path` and `ridge_delta` per dose index |
| `stats` | per-pass timings, `gram_budget_bytes`, sequence counts, `scratch_dir` |
| `warnings` | strings: one summary of every non-PD solve (counts by solver path), disabled resume, non-finite leaks |
| `non_pd` | per non-PD solve: `{matrix, input_key, mu_per_token, solve_path, ridge_delta, requested_ridge_delta}` |

`report.spectrum(name)` returns the full `γ` spectrum of an input key or edited matrix
(needs `spectrum="full"`).

Per-matrix row (`report.matrices[name]`):

| field | meaning |
| --- | --- |
| `L_X_rel` | `‖(W − M) Xᵀ‖ / ‖W Xᵀ‖`: target response that **survives** (0 = fully erased, 1 = untouched) |
| `L_Y_rel` | `‖M Yᵀ‖ / ‖W Yᵀ‖`: retained response **disturbed** (0 = untouched) |
| `target_surviving_rel`, `retained_disturbed_rel` | `L_X_rel²` (target energy left), `L_Y_rel²` (retained energy disturbed) |
| `erased_rel`, `retained_rel` | **deprecated** aliases of `target_surviving_rel` / `retained_disturbed_rel` (same values; `erased_rel` is the target energy *left*, despite its name) |
| `delta_fro`, `w_fro` | `‖M‖_F`, `‖W‖_F` |
| `edit_fro`, `survival` | `‖W_new − W‖_F` as written, and `edit_fro / ‖M‖_F` (below 1 when the weight dtype rounded part of the edit away) |
| `rank`, `rank_bound` | kept rank of `M_r` (`None` = full rank) and its ceiling (`min(d_out, d_in, n_x)` at full rank, the spectrum length when truncated) |
| `erank` | effective rank of `M Xᵀ` (`None` unless a rank rule was solved) |
| `solve_path`, `ridge_delta` | `cholesky` / `cholesky_ridge` / `eigh`, and the ridge `δ` actually used |
| `input_key`, `d_out`, `d_in` | the shared input and shape |

Reading the leaks:

| `L_X_rel` | `L_Y_rel` | reading |
| --- | --- | --- |
| near 0 | near 0 | disentangled at this matrix: target response removed, retained untouched |
| near 0 | large | overreach: the sets share directions and the dose removes them; lower `mu_per_token` or enlarge the retained set |
| large | near 0 | under-dosed, or the target is not distinct here; raise `mu_per_token` and check `γ` |
| similar, mid-range | | entangled sets (`γ` near 1 everywhere): `f ≈ m/(1+m)` on both sides |

The leaks are computed per matrix on activations of the **pristine** model, as if that matrix
were edited alone; edits of earlier layers change later inputs. They are a diagnostic of each
edit, not a measurement of the edited model's behaviour: evaluate the edited model too.

### 6. Re-dose, restore, export

```python runnable
rspd.api.restore(model)                   # back to the original weights
pristine = {k: v.clone() for k, v in model.state_dict().items()}   # optional: to verify restore
report.apply_dose(3.0)                 # restores the pristine weights, applies dose 3.0
assert report.dose["mu_per_token"] == 3.0
names = rspd.api.restore(model)           # bit for bit; returns the restored matrix names
assert names and not report.applied
assert all(torch.equal(v, pristine[k]) for k, v in model.state_dict().items())   # the check
report.apply_dose(1.0)                 # any solved (dose, rule) again, no new pass over the data

out = rspd.export_checkpoint(model, tokenizer, tempfile.mkdtemp())   # save_pretrained of the edited model
print(out["dir"], out["dtype"], out["tokenizer_saved"], out["warnings"])
report.close()                         # restores the model and frees the solve
assert not report.applied
```

`rspd.export_checkpoint(model, tokenizer, out_dir, source=None)` (also
`rspd.models.export_checkpoint`) writes the model as it is now
(edited) in its own dtype, the tokenizer beside it, and for models with KV-shared layers
backfills the tensors `transformers` does not save, from `source` (default
`model.config._name_or_path`). Export **before** `close()`: `close()` restores the model.
`apply_dose` and `restore` on a report need its live solve; a report loaded from JSON or
closed raises `RuntimeError("this EditReport has no live solve attached ...")`.

## Low-level API

On raw tensors (batch-first `X [n_x, d_in]`, `Y [n_y, d_in]`, `W [d_out, d_in]`):

```python runnable
W = torch.randn(8, 6, dtype=torch.float64)
X, Y = torch.randn(40, 6, dtype=torch.float64), torch.randn(200, 6, dtype=torch.float64)
dose = rspd.core.dose_from_mu_per_token(1.0, n_x=X.shape[0], n_y=Y.shape[0])
comp = rspd.core.erased_component(W, X, Y, mu=dose.mu)        # M = A @ B, an ErasedComponent
W_new = rspd.core.apply_erasure(W, comp)                       # W − M
lk = rspd.core.leaks(W, comp, X, Y)
print(lk["L_X_rel"], lk["L_Y_rel"], dose.as_dict()["mu_raw"])
M_r = rspd.core.erased_component(W, X, Y, mu=dose.mu, rank=2)  # COALA truncation of M over X
```

`dose.mu` is the only place `mu` is used, and only by these low-level solves. For a Gram of
`Y` use `erased_component_gram` / `leaks_gram`; for Grams of both sets
`factor_grams` + `erased_component_grams` / `leaks_grams`.

## Failure modes

| situation | what happens | what to do |
| --- | --- | --- |
| something other than an `ErasedComponent` handed to `apply_erasure` / `leaks` / `truncate_component` (a `KeptMap`, a tensor, factors of `W`) | `ErasureConventionError` (a `TypeError`) with the core table | compute `M` with `erased_component*` and subtract it |
| `erase` on a model that carries an applied erasure | `RuntimeError: this model carries an applied erasure held by EditReport #N (...): call rspd.api.restore(model) (or that report's close()) first ...` | `rspd.api.restore(model)` or the holder's `close()`, then erase |
| `apply_dose` on one report while another report's edit is applied | `RuntimeError: this model carries an applied erasure held by EditReport #N (...), not by this report (EditReport #M ...) ...` | `rspd.api.restore(model)`, then `apply_dose` |
| several doses with `apply=True` and no choice | `ValueError: 2 doses solved; pass apply_mu_per_token=<one of [...]> or apply=False ...` (CLI: `pass --apply-mu-per-token <one of [...]> or --no-apply`) | pass `apply_mu_per_token=` or `apply=False` |
| several rank rules with `apply=True` and no choice | `ValueError: 2 rank rules solved; pass apply_rank_rule=<one of [...], or None for the full-rank edit> ...` | pass `apply_rank_rule=` (`None` = full rank) |
| `apply_dose` with an unsolved setting | `ValueError: mu_per_token 3.0 was not solved; solved: [...]` | use a value from `report.settings` |
| `rank_rule` with `solver="grams"` | `ValueError: rank_rule needs solver='readout' ...` | use the readout solver |
| `rank_rule` outside (0, 1) | `ValueError: rank_rule lam_rel must be in (0, 1) ...` | 0 is the full-rank edit; use `None` |
| `mu_per_token <= 0` | `ValueError: mu_per_token must be finite and > 0` | |
| one input key's Gram exceeds the per-device budget | `ValueError: input 'L..' (d_in N) needs B bytes ... of Gram, more than the ... per-device budget on DEV; no pass can hold it. Raise gram_budget_gib or free device memory.` | raise `gram_budget_gib`, or put the Grams elsewhere with `device=` |
| Grams exceed the budget in total | not an error: more Gram passes (`report.pass_plan`), each resuming from the cached residual stream | |
| `scratch_dir` disk too small | `RuntimeError: scratch_dir ... has X GiB free but this erasure spills Y GiB ...` | larger disk, fewer doses, no `rank_rule` |
| `G_μ` not numerically positive definite (common: LayerNorm outputs are mean-centred, so `Y` spans fewer than `d_in` directions) | not an error: the solve falls back to an `eigh` pseudo-inverse (or, with `ridge_rel > 0`, escalates the ridge ×10 up to `1e-6` first); `solve_path` says which, every such solve is listed in `report.non_pd`, and ONE summary warning `G_mu is not numerically positive definite at the requested ridge for N matrix solve(s) (eigh: ...) ...` is added per run | usually fine; a small `ridge_rel` (e.g. `1e-6`) gives `cholesky_ridge` and makes the dose formula approximate |
| a biased `nn.Linear` selected (GPT-2's `Conv1D` biases are handled automatically: the constant is subtracted at harvest, no flag needed) | `ValueError: ... has a bias; the solve assumes a pure matmul ... Pass subtract_bias=True ...` | `subtract_bias=True` (the bias is never edited) |
| selector matches nothing / a non-linear module | `ValueError: no target projections discovered ...` / `TypeError: ... not nn.Linear/Conv1D` | fix `layers=` against `model.named_modules()` |
| hooks do not bracket `x Wᵀ` | `AssertionError: bracketing probe violated ...` | model must be in `eval()`; the selector must end at the projection |
| resumed pass not bitwise equal to a full forward | `ResumeGuardError` | `resume_from_cache=False` |
| `apply_dose` / `restore` on a closed or JSON-loaded report | `RuntimeError: this EditReport has no live solve attached ...` | re-run `rspd.api.erase` |

The edit lock belongs to the **model**: at most one report's edit is applied at a time.
While one is, every other report's `apply_dose` and every new `erase` on that model refuse
and name the holder; `rspd.api.restore(model)` restores the holder's edit bit for bit
(whichever report applied it) and releases the lock, after which any open report may
`apply_dose` again. `close()` reports you no longer need: each keeps its solve in memory.

```python runnable
try:
    rspd.api.erase(model, target=target, retained=retained, mu_per_token=[0.3, 1.0])
except ValueError as e:
    print(e)                                      # 2 doses solved; pass apply_mu_per_token=...
r = rspd.api.erase(model, target=target, retained=retained, layers=layers, mu_per_token=1.0)
try:
    rspd.api.erase(model, target=target, retained=retained, mu_per_token=1.0)
except RuntimeError as e:
    print(e)                                      # ... applied erasure held by EditReport #N ...
try:
    rspd.core.apply_erasure(W, W_new)                 # a bare tensor is not an ErasedComponent
except rspd.core.ErasureConventionError as e:
    print(str(e).splitlines()[0])
r.close()
```

Memory: the readout solver holds, per input key, one float64 `d_in × d_in` Gram (only the
keys of the current pass), the target activations `n_x × d_in` in the model dtype, and one
fp32 readout `n_x × d_in` per dose; the grams solver holds two Grams per key and one dense
fp32 `d_out × d_in` component per matrix and dose. `scratch_dir=` moves the pristine weights,
readouts and components from host RAM to disk. `gram_budget_gib` defaults to every Gram at
once on CPU and half the free memory on CUDA. `progress=lambda event, info: ...` receives
`plan`, `pass_start`, `chunk`, `solve`, `pass_done`, `warning`, `applied`.

## Experiments package (`rspd-exp`)

`rspd[experiments]` holds the data builders, evaluation, baselines (RMU, KUDA, SimNPO) and the
reproduction recipes of the published results. See `docs/experiments.md`.

```bash
rspd-exp list                                     # recipes: exports, published weights, TP
rspd-exp show zephyr_wmdp [--json]                # exports, doses, rank rules, published numbers
rspd-exp reproduce zephyr_wmdp --mode cheap --dry-run          # resolve the plan; no network
rspd-exp reproduce zephyr_wmdp --mode cheap --export rmu       # download published weights, evaluate
rspd-exp reproduce gpt2_trilingual_recovery --mode full        # re-run the methods, then evaluate
rspd-exp compare zephyr_wmdp results.json [--export ID] [--json]
```

`compare` exits 0 only when every published number of the exports present in the results
file is matched within its tolerance **and** every row's tensor-parallel degree equals the
recipe's `tensor_parallel_size`. **The TP-must-match rule:** tensor parallelism changes the
reduction order of the sharded matmuls and therefore the logprobs, so two numbers are
comparable only at the same TP. Every result row records `tp` (`None` for single-device HF
evaluation); never put rows of different TP in one table, and evaluate a baseline and its
edited model at the same TP. The software stack is part of the configuration too: reproduce
under `constraints.txt`.

## Deliberately not in the package

| not here | status |
| --- | --- |
| clustering of erased components and circuit discovery / isolation | research-only; not released |
| ASVD and the pull-back `[W X]_r X⁺` | issue #10 |
| the synthetic shapes toy experiments | research-only, issue #9 |
| a vLLM activation-capture backend (capture runs on resident HF models only) | deferred, issue #11 |
| per-input-key doses chosen from the `γ` spectrum | issue #8 (`suggest_mu_per_token` only suggests one global grid) |

Do not reimplement these inside `rspd`; the core stays: solve, COALA truncation of `M`,
`W − M`, leaks, dose, Gram streaming, HF capture.
