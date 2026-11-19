# rspd[experiments]

Data builders, the evaluation harness, the baseline methods and the reproduction recipes of
the published `rspd` results. Convention as in `rspd`: `X` = target (erased), `Y` =
retained (kept), the edit is `W − M`, the dose is `mu_per_token`. Agents: read `AGENTS.md`
at the repository root first; the API is in `docs/api.md`.

## Install

```bash
pip install "rspd[experiments]"                # + eval, muse for the evaluation harnesses
```

Until the first release, install from git:
`pip install "rspd[experiments,eval,muse] @ git+https://github.com/ThatE10/rspd@<sha>"`.

To reproduce published numbers, install from a checkout under the pinned stack:

```bash
git clone https://github.com/ThatE10/rspd && cd rspd && git checkout $SHA
pip install -c constraints.txt \
  ".[experiments,eval,muse]"
export RSPD_DATA_DIR=/data/rspd        # default ~/.cache/rspd: datasets/, logs/, models/
```

| extra | adds | needed for |
| --- | --- | --- |
| `experiments` | `accelerate`, `datasets`, `pyyaml`, `numpy` | data builders, method runners, recipes, `compare` |
| `eval` | `lm_eval`, `vllm` (pinned) | `eval.lm_eval` (WMDP, MMLU, WikiText, Global-MMLU) |
| `muse` | `rouge_score`, `scikit-learn`, `scipy` | the MUSE evaluator |
| `test` | `pytest` | the test suite |

The software stack and the tensor-parallel (TP) degree are part of the experimental
configuration. `constraints.txt` pins the stack the published numbers were produced on;
results on another stack are not numerically comparable. TP changes the reduction order of
sharded matmuls and so the logprobs: every result row records `tp`, and two numbers belong
in one table only at the same TP.

## Minimal example

Recipes, plans and comparisons run offline:

```python runnable
import os, tempfile
os.environ.setdefault("RSPD_DATA_DIR", tempfile.mkdtemp())   # where plans would write

from rspd.experiments.recipes import load_recipe, plan
from rspd.experiments.recipes.compare import compare, format_report
from rspd.experiments.results import ResultRow, read_results, write_results

recipe = load_recipe("zephyr_wmdp")
print(recipe["model"], "TP", recipe["tensor_parallel_size"])
p = plan(recipe, "cheap", exports=["rmu"])                     # resolves steps, no network
print([s["step"] for s in p["steps"]])                         # ['download', 'eval', 'eval']

rows = [ResultRow(task="wmdp_bio", metric="acc", value=0.335, model="rmu", tp=1)]
path = write_results(os.path.join(tempfile.mkdtemp(), "results.json"), rows)
rep = compare(recipe, read_results(path))
print(format_report(rep))                                      # 1 pass, 3 missing
off_tp = [ResultRow(task="wmdp_bio", metric="acc", value=0.335, model="rmu", tp=2)]
assert compare(recipe, off_tp)["tp_mismatch"]                   # a TP-2 row never matches a TP-1 recipe
```

## Recipes (`rspd-exp`)

```bash
rspd-exp list                                            # every recipe, its exports and TP
rspd-exp show zephyr_wmdp                                # exports, doses, published numbers
rspd-exp reproduce zephyr_wmdp --mode cheap --dry-run    # the resolved plan as JSON, no network
rspd-exp reproduce zephyr_wmdp --mode cheap --export rmu # download published weights, evaluate
rspd-exp reproduce gpt2_trilingual_recovery --mode full  # re-run every method, then evaluate
rspd-exp compare zephyr_wmdp $RSPD_DATA_DIR/logs/recipes/zephyr_wmdp/results.json
```

`recipes/<result>.yaml` records, per export, the model, method, data spec, method
hyperparameters (the dose as `mu_per_token`; a legacy dose is converted once at load by
`recipes.legacy.mu_per_token_from_legacy`), the evaluation and its TP, a hardware hint, the
published numbers with tolerances, and the published edited weights
(`recipes/hf_weights.csv`; private weight repositories need `HF_TOKEN`). `cheap` mode skips
exports without published weights. `compare` exits 0 only when every published number of the
exports present is matched within tolerance and every row's `tp` equals the recipe's.

| recipe | what |
| --- | --- |
| `zephyr_wmdp` | WMDP bio/cyber on Zephyr-7B-beta: SRSPD vs RMU, KUDA, SimNPO |
| `gpt2_trilingual_recovery` | GPT-2 124M trilingual: erasure, then retrain-to-recover |
| `gemma4_e4b_dutch_gmmlu`, `gemma4_31b_dutch_gmmlu` | Dutch unlearning on Gemma-4, Global-MMLU nl/de/en |
| `muse_news`, `muse_books` | MUSE: SRSPD sweeps vs KUDA and SimNPO |

## Data (`rspd.experiments.data`)

```python
from rspd.experiments.data import DataSpec, build_erasure_data
spec = DataSpec.from_yaml("data.yaml")      # target / retained / held-out; shape in data/spec.py
data = build_erasure_data(tokenizer, spec)  # downloads the corpora; X, Y as token ids
rspd.api.erase(model, target=data.target_ids, retained=data.retained_ids, mu_per_token=[0.01])
```

| module | what |
| --- | --- |
| `spec`, `target` | `DataSpec` / `TargetSpec` / `RetainedSpec` / `HeldoutSpec`; `build_erasure_data` -> `X`, `Y`, held-out readouts as token ids |
| `corpora` | registry: WMDP bio/cyber forget + retain, WikiText, MUSE News/Books (pinned), nl/zh Wikipedia, config-level `local_jsonl`; MUSE windows; tokenisation helpers |
| `retained_corpus` | the retained set `Y`: hub / JSONL buckets to an exact token budget, hashed manifest, content-addressed token cache |
| `budget` | equal-token-budget plumbing for the gradient baselines |
| `gpt2_corpus`, `gpt2_unlearning` | trilingual GPT-2 pretraining corpus + BPE; the GPT-2 unlearning sets |
| `dutch_sets` | the Dutch-unlearning sets for a large instruct model |
| `specs/*.yaml` | the published dataset-builder configs (`spec_path(name)`) |

## Evaluation (`rspd.experiments.eval`)

| module | what |
| --- | --- |
| `lm_eval` | lm-evaluation-harness wrapper (vLLM by default, HF fallback) + local Global-MMLU en/de/nl/zh tasks; records `tp` |
| `krd` | Knowledge Removal Degree, single-domain and the WMDP bio/cyber harmonic mean |
| `muse`, `rouge_l` | the MUSE evaluator (VerbMem, KnowMem, PrivLeak), fp32 |
| `ce`, `heldout` | teacher-forced CE and the shared held-out readout |
| `trainer`, `recovery` | the GPT-2 training loop and the retrain-to-recover study |
| `tracking` | optional Weights & Biases logging, off by default |

Every evaluator writes `results.json` in the schema of `rspd.experiments.results`: rows
`{"task", "metric", "value", "model", "tp", "n", "meta"}`.

## Methods (`rspd.experiments.methods`)

Each runner takes a YAML path or a mapping (`run(cfg)`) and writes
`logs/methods/<name>/metrics.json` and `models/methods/<name>/` under the data root. Nothing
is pushed to a model hub; tracking is off unless the config enables it.
`rspd.experiments.methods.get(name)` imports a runner by its recipe name.

| runner | what |
| --- | --- |
| `srspd` | data spec -> one `rspd.api.erase` at every `mu_per_token` (both solvers, rank rules) -> one export + held-out readout per (dose, rule) |
| `rmu`, `kuda`, `simnpo` | ports of the baselines (credits below) |
| `upstream_kuda`, `upstream_kuda_muse`, `upstream_simnpo` | the authors' own code at pinned commits, run verbatim with documented result-neutral shims |

`rspd.experiments.models` holds the loaders, block-parameter resolution and
`export_checkpoint` (the core export plus a multimodal model's processor).

## Credits

* **COALA** (rank truncation in `rspd`): Parkina & Rakhuba, arXiv:2507.07580, NeurIPS 2025.
* **RMU**: Li et al., *The WMDP Benchmark: Measuring and Reducing Malicious Use With
  Unlearning*, arXiv:2403.03218; ported from github.com/centerforaisafety/wmdp (MIT).
* **KUDA**: Fang et al., *KUDA: Knowledge Unlearning by Deviating Representation for Large
  Language Models*, arXiv:2602.19275; ported from github.com/AceNagi/KUDA (MIT).
* **SimNPO**: Fan et al., *Simplicity Prevails: Rethinking Negative Preference Optimization
  for LLM Unlearning*, arXiv:2410.07163; ported from github.com/OPTML-Group/Unlearn-Simple
  (MIT). It builds on NPO (Zhang et al., arXiv:2404.05868); no NPO code is included.
* **MUSE**: Shi et al., *MUSE: Machine Unlearning Six-Way Evaluation for Language Models*,
  arXiv:2407.06460; the evaluator is a re-implementation (data CC-BY-4.0).
* **KRD** follows the WMDP metric of the KUDA paper (arXiv:2602.19275, Appendix C-A).
* Evaluation runs on EleutherAI's lm-evaluation-harness; Global-MMLU data is
  `CohereLabs/Global-MMLU`.
