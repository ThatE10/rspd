# srspd-experiments

Data builders, the evaluation harness, the baseline methods and the reproduction
recipes of the published `srspd` results. Convention as in `srspd`: `X` = target (erased), `Y` =
retained (kept), edit `W − M`, dose `mu_per_token`.

```bash
pip install -c packages/srspd-experiments/constraints.txt 'packages/srspd-experiments[eval,muse]'
export SRSPD_DATA_DIR=/data/srspd        # default ~/.cache/srspd: datasets/, logs/, models/
```

The stack and the tensor-parallel degree change results; `constraints.txt` pins the
published stack and every result row records `tp`.

## Data (`srspd_experiments.data`)

```python
from srspd_experiments.data import DataSpec, build_erasure_data
spec = DataSpec.from_yaml("data.yaml")      # target / retained / readouts, see data/spec.py
data = build_erasure_data(tokenizer, spec)  # X, Y and held-out readouts as token ids
srspd.erase(model, target=data.target_ids, retained=data.retained_ids, mu_per_token=[0.01])
```

| module | what |
| --- | --- |
| `corpora` | registry: WMDP bio/cyber forget + retain, WikiText, MUSE News/Books (pinned), nl/zh Wikipedia, config `local_jsonl`; MUSE windows; tokenisation helpers |
| `retained_corpus` | the retained set `Y`: hub / JSONL buckets to an exact token budget, hashed manifest, content-addressed token cache |
| `target`, `spec` | the target set `X`, readouts, `DataSpec` dataclasses |
| `budget` | equal-token-budget plumbing for the gradient baselines |
| `gpt2_corpus`, `gpt2_unlearning` | trilingual GPT-2 pretraining corpus + BPE; the GPT-2 unlearning sets (download the published JSONL; a rebuild depends on the live Wikipedia API) |
| `dutch_sets` | the Dutch-unlearning sets for a large instruct model (target cut to N tokens, retained JSONL) |
| `specs/*.yaml` | the published data specs (`spec_path(name)`) |

## Evaluation (`srspd_experiments.eval`)

| module | what |
| --- | --- |
| `lm_eval` | lm-evaluation-harness wrapper (vLLM default, HF fallback) + local Global-MMLU en/de/nl/zh five-shot tasks |
| `krd` | KRD: single-domain and the WMDP bio/cyber harmonic mean |
| `muse`, `rouge_l` | the MUSE evaluator (VerbMem, KnowMem, PrivLeak), fp32 |
| `ce`, `heldout` | teacher-forced CE and the shared held-out readout |
| `trainer`, `recovery` | the GPT-2 training loop and the retrain-to-recover study |
| `tracking` | optional W&B logging, off by default |

Every evaluator writes `results.json` in the common schema of `srspd_experiments.results`:
rows `{"task", "metric", "value", "model", "tp", "n", "meta"}`.

## Methods (`srspd_experiments.methods`)

Each runner takes a YAML path or a mapping and writes `logs/methods/<name>/metrics.json`
and `models/methods/<name>/` under the data root. Nothing is pushed to a model hub;
tracking is off unless the config enables it.

| module | what | credit |
| --- | --- | --- |
| `srspd` | data spec -> `srspd.erase` at every `mu_per_token` (both solvers, rank rules) -> one export + held-out readout per dose | -- |
| `rmu` | RMU (port) | Li et al., arXiv:2403.03218; github.com/centerforaisafety/wmdp (MIT) |
| `kuda` | KUDA (port) | Fang et al., arXiv:2602.19275; github.com/AceNagi/KUDA (MIT) |
| `simnpo` | SimNPO (port) | Fan et al., arXiv:2410.07163; github.com/OPTML-Group/Unlearn-Simple (MIT) |
| `upstream_kuda`, `upstream_kuda_muse`, `upstream_simnpo` | the authors' own code at pinned commits, run verbatim (documented result-neutral shims) | as above |

`srspd_experiments.models` holds the loaders, block-parameter resolution and
`export_checkpoint` (the core export plus a multimodal model's processor).

## Recipes (`srspd-exp`)

```bash
srspd-exp list
srspd-exp show zephyr_wmdp
srspd-exp reproduce zephyr_wmdp --mode cheap --dry-run   # resolve everything, no network
srspd-exp reproduce zephyr_wmdp --mode cheap --export rmu   # download the published weights, evaluate
srspd-exp reproduce gpt2_trilingual_recovery --mode full    # train every run, then evaluate
srspd-exp compare zephyr_wmdp ~/.cache/srspd/logs/recipes/zephyr_wmdp/results.json
```

`recipes/<result>.yaml` records, per export, the model, method, data spec, method
hyperparameters (dose as `mu_per_token`; a legacy dose is converted at load by
`recipes.legacy.mu_per_token_from_legacy`), the evaluation and its tensor-parallel
degree, a hardware hint, the published numbers with tolerances, and the published
edited weights (`recipes/hf_weights.csv`; private repos need `HF_TOKEN`). `cheap`
mode skips exports whose weights are not published.
