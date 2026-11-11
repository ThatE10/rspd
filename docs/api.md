# API reference

Every public symbol of `srspd` (`srspd.__all__`, `srspd.models.__all__`,
`srspd.capture.__all__`), the `srspd` and `srspd-exp` command lines, and the package-level
API of `srspd-experiments`. Hand-written; `packages/srspd/tests/test_docs_core.py` and
`packages/srspd-experiments/tests/test_docs_experiments.py` fail when a public symbol, a
subcommand or an option is missing here. Conventions throughout: `X` target, `Y` retained,
batch-first `[n, d_in]`; `W [d_out, d_in]`; the edit is `W − M`; the dose is `mu_per_token`.
Math: `docs/math.md`. Workflow: `AGENTS.md`.

## `srspd` — model-level

| symbol | semantics |
| --- | --- |
| `erase(model, tokenizer=None, *, target, retained, mu_per_token, apply=True, apply_mu_per_token=None, apply_rank_rule=<unset>, layers=None, solver="readout", rank_rule=None, ridge_rel=0.0, device=None, gram_budget_gib=None, scratch_dir=None, progress=None, subtract_bias=False, resume_from_cache=True, eig_extremes=True, lam_svd_dtype="float32", check_bracketing=True) -> EditReport` | Harvest `X` from `target` and stream `G_Y` from `retained`, solve `M` for every selected matrix at every dose (and rank rule), apply one setting in place (`W − M`), measure leaks per matrix. Refuses a model carrying an applied erasure. |
| `restore(model) -> list` | Put every weight `erase` edited on `model` back bit for bit; returns the restored names (`[]` if nothing applied). |
| `inspect_model(model_or_id, layers=None, *, subtract_bias=False, gram_budget_gib=None, trust_remote_code=False) -> dict` | What `erase` would edit: selector, matrices (shape, dtype, input key), input keys with Gram bytes per solver, totals, quirks, warnings; `pass_plan` with a budget. A hub id / local dir is built on the `meta` device (no weights). |
| `EditReport` | Dataclass returned by `erase`; fields in `AGENTS.md` ("Read the report"). Methods: `apply_dose(mu_per_token, rule=None)` restores then applies a solved setting and returns the report; `restore()`; `close()` restores the model and frees the solve; `to_dict()`; `to_json(path=None, **json_kw) -> str`. |

## `srspd` — tensors

| symbol | semantics |
| --- | --- |
| `erased_component(W, X, Y, *, mu, rank=None, lam=0.0, lam_rel=None, ridge_rel=0.0, rcond=None) -> ErasedComponent` | `M = W XᵀX (XᵀX + μYᵀY)⁺` from stacked activations; optional COALA truncation of `M` over `X` (`rank` wins, then `lam_rel`, then `lam`). |
| `erased_component_gram(W, X, G_Y=None, *, mu=None, rank=None, lam=0.0, lam_rel=None, ridge_rel=0.0, rcond=None, S=None, spectrum=True) -> ErasedComponent` | Same from the retained Gram `G_Y`; `M = (W Xᵀ) S`. Pass a precomputed readout `S` to skip the factorisation; `spectrum=False` skips the SVD when not truncating. |
| `erased_component_grams(W, G_X, G_Y, *, mu, ridge_rel=0.0, rcond=None, factor=None) -> ErasedComponent` | Two-Gram form `M = W G_X (G_X + μG_Y + δI)⁺`, dense (`A = M`, `B = I`), no truncation. |
| `readout_gram(G_Y, X, mu, *, rcond=None, ridge_rel=0.0) -> (S, info)` | The target readout `S = X G_μ⁺` (`[n_x, d_in]`), shared by every matrix reading this input; `info["solver"]` is `cholesky` / `cholesky_ridge` / `eigh` / `none`. |
| `factor_grams(G_X, G_Y, mu, *, rcond=None, ridge_rel=0.0) -> (GramMuFactor, info)` | Factor `G_X + μG_Y + δI` once per input and dose for `erased_component_grams(factor=)`; same PD test, ridge ladder and fallback as the readout. |
| `truncate_component(comp_full, X, specs, *, svd_dtype=torch.float32) -> list` | Several truncations (`lam_rel` floats in [0, 1) and/or `"erank"`) of one full-rank readout-form component from one SVD of `M Xᵀ`. |
| `apply_erasure(W, component) -> Tensor` | `W − A @ B` for an `ErasedComponent`, in `W`'s dtype/device. Anything else raises `ErasureConventionError`. |
| `leaks(W, component, X, Y) -> dict` | `L_X_rel`, `L_Y_rel`, `erased_rel` (= `L_X_rel²`), `retained_rel` (= `L_Y_rel²`), `delta_fro`, `w_fro`, energies. |
| `leaks_gram(W, component, X, G_Y, *, ret_energy=None) -> dict` | Same keys from `G_Y` (plus `ret_energy`), never forming `M Yᵀ`. |
| `leaks_grams(W, component, G_X, G_Y, *, ret_energy=None, target_energy=None) -> dict` | Same from both Grams (plus `mx_energy`). |
| `specificity(G_Y, X, n_y, k=8, *, ridge_rel=0.0, full=False) -> dict` | Generalised eigenvalues `γ` of `C_X` vs `C_Y`: `top` (k largest), `median`, `n_above` (`{"1","10","100"}`), `n`, `solver`, and with `full=True` the whole `spectrum`. |
| `ErasedComponent(A, B, info)` | NamedTuple, `M = A @ B`; `.dense()`. The only thing `apply_erasure` subtracts. |
| `KeptMap(A, B, info)` | NamedTuple, the complementary kept map `M_keep` (diagnostics only; `apply_erasure` refuses it). |
| `ErasureConventionError` | `TypeError` subclass raised when a non-`ErasedComponent` is applied or measured; the message carries `CORE_TABLE`. |
| `CORE_TABLE` | String: the subtraction vs kept-map table (`docs/math.md`). |
| `fit_coala(W, A, T) -> (A_fac, B_fac, sigma)` | COALA factor pair of `W` on inputs `A` with response `T = A Wᵀ`: `A_fac[:, :r] @ B_fac[:r] = U_r U_rᵀ W`. Compression only. |
| `compress_weight(W, A, T, r) -> Tensor` | `U_r U_rᵀ W`, the rank-`r` COALA replacement of `W`. Never an erasure. |
| `Dose(mu_per_token, n_x, n_y)` | Frozen dataclass; properties `mu` (the solve's weight) and `mu_raw` (= `1/mu`), derived; `erased_fraction(gamma)`; `as_dict()`. |
| `dose_from_mu_per_token(m, n_x, n_y) -> Dose` | The dose for `mu_per_token = m` on `n_x` target and `n_y` retained tokens. |
| `suggest_mu_per_token(gamma, fractions=(0.1, 0.5, 0.9), quantiles=(0.5, 0.9, 0.99)) -> dict` | `m = f / ((1 − f) γ_q)` for every quantile `q` and fraction `f`: `rows`, `mu_per_token` (sorted distinct finite grid), `formula`. `gamma`: a spectrum or `specificity(..., full=True)`. |
| `erased_fraction(gamma, mu_per_token)` | `γm / (1 + γm)` (float or float64 tensor). |
| `effective_rank(A) -> float` | Roy–Vetterli effective rank of a matrix or a 1-D singular spectrum. |
| `__version__` | Package version string. |

## `srspd.models`

| symbol | semantics |
| --- | --- |
| `HFLinearHarness(model, layers_regex=None, exclude_regex=None, *, subtract_bias=False, expect_d_in=None, selector=None)` | The selected projections by name (`module_by_name`, `path_by_name`, `all_names`); `get_weight(name)` as `[d_out, d_in]`. Used with `srspd.capture.harvest_inputs_concat`. |
| `as_dout_din(mod) -> Tensor` | A projection's weight as `[d_out, d_in]` (transposes `Conv1D`). |
| `write_dout_din(mod, value)` | Write a `[d_out, d_in]` matrix into the module's native layout and dtype. |
| `is_conv1d(mod) -> bool` | Whether `mod` is a `transformers` `Conv1D`. |
| `input_key(name) -> str` | Matrix name to its input key (`L05.self_attn.q_proj` -> `L05.attn_in`). |
| `key_layer(key) -> int` | Block index of an input key (`-1` if none). |
| `group_by_input(names) -> dict` | `{input key: [member names]}`. |
| `inspect_model(...)` | As `srspd.inspect_model`. |
| `PristineStore(harness, scratch_dir=None)` | RAM- or disk-backed pristine weights: `write(name, value)`, `pristine(name)`, `restore_all()`, `is_edited`. |
| `QUIRKS` | Registry `{name: Quirk}`: `conv1d_transposed`, `biased_projection`, `multimodal_towers`, `kv_shared_layers`, `shared_input_check`, `bf16_edit_in_fp32`. |
| `Quirk(name, description, detect, handled_by)` | One registry entry; `detect(model, harness)` returns a detail dict or `None`. |
| `detect_quirks(model, harness) -> dict` | `{quirk: detail}` for every quirk that applies. |
| `register_quirk(q) -> Quirk` | Add or replace a registry entry. |
| `PRESETS` | `{"decoder": (regex, exclude), "gpt2": (regex, exclude)}`, tried in order when `layers=None`. |
| `LayerSelector(layers_regex, exclude_regex, types=None, blocks=None, preset=None)` | A resolved selection rule; `keep(block, tail)`, `as_dict()`. |
| `build_layers_regex(indices, base=DEFAULT_LAYERS_REGEX) -> str` | `base` narrowed to exactly these block indices. |
| `resolve_selector(model, layers=None) -> LayerSelector` | The `layers=` argument (None, preset, regex, block list, mapping, selector) to a selector. |

`srspd.models.export.export_checkpoint(model, tokenizer, out_dir, *, source=None, token=None) -> dict`
writes the model as it is now with `save_pretrained` (own dtype), the tokenizer, and the
KV-shared tensor backfill from `source`; returns `{dir, bytes, wall_s, dtype,
tokenizer_saved, kv_shared_backfilled, warnings}`. `srspd.report.weight_delta_report(base,
edited, ...)` compares two state dicts or modules (Frobenius change per tensor, block
and type).

## `srspd.capture`

| symbol | semantics |
| --- | --- |
| `ActivationSource` | Protocol: `accumulate(sequences, keys_to_rep, sink, **options) -> dict` streams sequences through a model into a `GramSink`. |
| `GramSink` | Protocol: `add(key, x)` accumulates `[tokens, d_in]` activations; `result() -> {key: (G float64, n_tokens)}`. |
| `Fp64GramSink(dims, device_of_key)` | The in-place float64 Gram sink. |
| `MemorySpill()` | Tensor store over a dict of CPU tensors (`put`, `get`, `has`). |
| `TensorSpill(root)` | Tensor store, one `.pt` file per tag under `root`. |
| `gram_bytes(d_in, bytes_per_elem=8) -> int` | Bytes of one `d_in × d_in` Gram. |
| `plan_gram_groups(keys_with_dim_and_device, budget_bytes_per_device, bytes_per_elem=8) -> list` | Split input keys (in layer order) into passes under a per-device budget; raises when one key alone exceeds it. |
| `BoundaryCache()` | The residual stream cached at a pass boundary, for resuming the next pass. |
| `HFHookSource(harness)` | `ActivationSource` over a resident HF model by forward hooks (early stop, boundary resume, shared-input bitwise check). |
| `ResumeGuardError` | A resumed forward did not reproduce the full forward bitwise; pass `resume_from_cache=False`. |
| `harvest_inputs_concat(harness, ids_list, *, expect_dtype=None, names=None) -> (data, n_tokens)` | One forward per `[1, T]` sequence; `{name: X [n_tokens, d_in]}` on CPU, members of one key sharing one tensor. |
| `harvest_inputs_native(harness, ids, names=None, *, expect_dtype=None) -> (data, report)` | One forward; the inputs each matrix received, plus a shared/mismatch report. |

## `srspd` command line

Every command prints one JSON document on stdout; on failure it prints
`{"error": ..., "type": ...}` and exits with status 2.

`srspd inspect MODEL` — `inspect_model` on a hub id or local dir (config only).

| option | meaning |
| --- | --- |
| `--layers SPEC` | preset name, regex, comma-separated block indices (`1,2`) or a JSON object (`{"types": ["down_proj"], "blocks": [3]}`) |
| `--subtract-bias` | accept biased `nn.Linear` projections |
| `--gram-budget-gib G` | also plan the Gram passes under this per-device budget |

`srspd erase MODEL --target FILE --retained FILE --mu-per-token M [M ...]` — load the model
and tokenizer, read one document per non-empty line, erase, print the report.

| option | meaning |
| --- | --- |
| `--target FILE` / `--retained FILE` | text files of `X` / `Y` documents, one per line (required) |
| `--mu-per-token M [M ...]` | the dose(s) (required) |
| `--apply-mu-per-token M` | which dose to apply (required with several) |
| `--solver readout|grams` | as `solver=` (default `readout`) |
| `--rank-rule R` | a `lam_rel` in (0, 1) or `erank`; repeat for several |
| `--apply-rank-rule R|full` | which rule to apply (required with several); `full` = full rank |
| `--ridge-rel R` | relative ridge (default 0) |
| `--layers SPEC` | as for `inspect` |
| `--dtype bfloat16|float16|float32` | load and export dtype (default `bfloat16`) |
| `--device DEV` | default `cpu` |
| `--max-length N` | tokens per document (default 512) |
| `--gram-budget-gib G` | float64 Gram budget per device |
| `--scratch-dir DIR` | spill to disk instead of host RAM |
| `--subtract-bias` | accept biased projections |
| `--out DIR` | write the edited checkpoint and tokenizer here (KV-shared backfill from `MODEL`) |
| `--report PATH` | also write the report JSON here |

`srspd report PATH` — summarise a saved report: solver, dose, leaks by type, the worst
matrices by each leak, survival, quirks, warnings.

| option | meaning |
| --- | --- |
| `--top N` | how many worst matrices to list per leak (default 5) |

## `srspd-exp` command line

`srspd-exp list` — every shipped recipe, its export count, how many have published weights,
its TP and title.

`srspd-exp show RESULT` — one recipe: model, TP, evaluation, hardware hints, and per export
its method, `mu_per_token`, rank rule, weights and published numbers.

| option | meaning |
| --- | --- |
| `--json` | print the loaded recipe as JSON (also for `compare`: the comparison as JSON) |

`srspd-exp reproduce RESULT --mode cheap|full` — `cheap` downloads the published edited
weights and evaluates them; `full` re-runs each method runner, then evaluates.

| option | meaning |
| --- | --- |
| `--mode cheap|full` | required |
| `--dry-run` | print the resolved plan as JSON; no network, no model |
| `--export ID` | restrict to this export (repeatable) |
| `--authors-python PY` | interpreter carrying the authors' stack (KUDA's MUSE code) |

`srspd-exp compare RESULT RESULTS_JSON` — check a results file against the recipe's
published numbers; exit 0 only when every published number of the exports present passes
within tolerance and every row's `tp` equals the recipe's `tensor_parallel_size`.

| option | meaning |
| --- | --- |
| `--export ID` | read every row as this export (when the file's `model` field is a path) |
| `--json` | the comparison as JSON |

## `srspd_experiments` — package level

`srspd_experiments.data`:

| symbol | semantics |
| --- | --- |
| `DataSpec(target, retained, corpora={}, heldout_retain_corpora=("wikitext",), retain_max_length=512, heldout_mix_tokens=2048, heldout=None)` | What `X`, `Y` and the readouts are; `DataSpec.from_yaml(path)`. |
| `TargetSpec(corpora, max_rows=None, max_tokens=None, min_len=50, max_length=768, heldout_rows=8, heldout_from=None)` | The target set `X`. |
| `RetainedSpec(buckets, chunk_tokens=1024, seed=0, add_bos=True, chat_template=None)` | The retained set `Y` (retained-corpus buckets). |
| `HeldoutSpec(rows=32, max_length=1024, tail=(), min_len=0)` | The shared baseline held-out readout. |
| `ErasureData(target_ids, retained_ids, heldout_ids, manifest)` | Token ids ready for `srspd.erase`, plus the data manifest. |
| `build_erasure_data(tokenizer, spec, *, cache_dir=None, log=print) -> ErasureData` | Build `X`, `Y` and the readouts (downloads and caches corpora). |
| `spec_path(name) -> str` | Path of a shipped dataset-builder config (`data/specs/<name>.yaml`). |
| `SPECS_DIR` | Directory of the shipped configs. |

`srspd_experiments.methods`:

| symbol | semantics |
| --- | --- |
| `METHODS` | `{recipe method name: runner module path}` (`srspd`, `rmu`, `kuda`, `simnpo`, the three authors'-code runners). |
| `get(method)` | Import a runner module; each has `run(config, ...) -> dict` writing under the data root. `methods.srspd.run(config, *, model=None, tokenizer=None, out_root=None, log_root=None, progress=None)` does one `srspd.erase` for every dose and rule and exports each. |

`srspd_experiments.models`:

| symbol | semantics |
| --- | --- |
| `MIN_TRANSFORMERS` | Oldest supported `transformers` (major, minor). |
| `require_modern_stack()` | Return `torch` after checking `transformers >= MIN_TRANSFORMERS`. |
| `load_config(path) -> dict` | A YAML run config (`name`, `model` required). |
| `load_model(cfg, source)` | Load a checkpoint per the config's dtype / device map. |
| `load_tokenizer_and_processor(model_id)` | `(tokenizer, processor or None)`. |
| `load_tokenizer(tokenizer_id, *, padding_side=None, truncation_side=None)` | The tokenizer alone (pad = eos when absent). |
| `load_causal_lm(model_id, *, dtype="bfloat16", device_map="auto", padding_side=None, truncation_side=None, tokenizer_id=None)` | `(model, tokenizer)`; a causal LM first, an image-text-to-text wrapper as fallback. |
| `resolve_model_source(model_ref, root=None) -> str` | A data-root-relative checkpoint reference to an absolute path. |
| `model_fingerprint(model_src, sample_bytes=1048576)` | Cheap content fingerprint of a local checkpoint. |
| `get_decoder_layers(model)` | The decoder's block `ModuleList`. |
| `get_block_module(model, layer_id)` | One decoder block. |
| `get_mlp_out_module(model, layer_id, name="mlp.down_proj")` | The MLP output projection of a block. |
| `get_down_proj_module(model, layer_id)` | `mlp.down_proj` of a block. |
| `weight_layout(module) -> str` | `"in_out"` for `Conv1D`, `"out_in"` for `nn.Linear`. |
| `get_named_block_params(model, layer_ids, param_names=("mlp.down_proj.weight",))` | `{qualified name: Parameter}` by attribute name. |
| `get_named_down_proj(model, layer_ids)` | The same at `mlp.down_proj.weight`. |
| `MLP_OUT_MODULE_NAMES` | `("mlp.down_proj", "mlp.c_proj")`. |
| `DEFAULT_MLP_OUT_MODULE` | `"mlp.down_proj"`. |
| `DEFAULT_BLOCK_PARAM_NAMES` | `("mlp.down_proj.weight",)`. |
| `save_model(model, tokenizer, out_dir, *, processor=None, source=None) -> str` | `save_pretrained` model + tokenizer (+ processor). |
| `save_processor(model, tokenizer, out_dir, *, processor=None, source=None) -> bool` | Save a multimodal processor beside a checkpoint. |
| `export_checkpoint(model, tokenizer, out_dir, *, source=None, processor=None, save_source_processor="auto", token=None) -> dict` | `srspd.models.export.export_checkpoint` plus a multimodal model's processor. |
| `is_multimodal(model) -> bool` | Whether the config carries a non-text tower. |
| `write_metrics(out_dir, metrics) -> str` | Write `metrics.json`. |

`srspd_experiments.paths` (data root: argument, else `$SRSPD_DATA_DIR`, else `~/.cache/srspd`):

| symbol | semantics |
| --- | --- |
| `data_dir(root=None, *, create=True)` | The data root. |
| `datasets_dir(*parts, root=None, create=True)` | `<root>/datasets/...`. |
| `logs_dir(*parts, root=None, create=True)` | `<root>/logs/...`. |
| `models_dir(*parts, root=None, create=True)` | `<root>/models/...`. |
| `resolve_data_ref(ref, root=None) -> str` | Data-root-relative reference to an absolute path. |
| `data_ref(path, root=None) -> str` | The inverse. |
| `resolve_model_path(model_path, root=None) -> str` | A `models/...` checkpoint reference to an absolute path. |
| `hf_token()` | `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` when set. |

`srspd_experiments.recipes` (re-exported from `.catalog` and `.legacy`) and
`srspd_experiments.recipes.compare`:

| symbol | semantics |
| --- | --- |
| `RecipeError` | A recipe that does not validate (`ValueError`). |
| `list_results() -> list` | Ids of the shipped recipes. |
| `load_recipe(result, *, path=None) -> dict` | Load and validate; every dose resolved to `mu_per_token`. |
| `iter_exports(recipe)` | `(run, export)` pairs, the baseline as run `None`. |
| `find_export(recipe, export_id)` | One export's `(run, export)`. |
| `weight_gaps(recipe) -> list` | Exports without published weights. |
| `plan(recipe, mode, *, root=None, exports=None) -> dict` | The resolved `reproduce` steps; no network. `recipes.reproduce.execute(plan, ...)` runs them. |
| `compare(recipe, rows, *, export=None) -> dict` | Checks, `n_pass` / `n_fail` / `n_missing`, `tp_mismatch`, `ok`. |
| `format_report(rep) -> str` | The comparison as text. |
| `mu_per_token_from_legacy(mu_raw, n_x, n_y) -> float` | The one translation of a legacy dose: `mu_raw · n_x / n_y`. Recipes only; never an entry-point dose. |

`srspd_experiments.results` (schema `srspd-experiments/results/v1`):

| symbol | semantics |
| --- | --- |
| `ResultRow(task, metric, value, model=None, tp=None, n=None, meta={})` | One number; `tp` is the tensor-parallel degree (`None` = unsharded HF). |
| `write_results(path, rows, **header) -> str` | Write a result file atomically. |
| `read_results(path) -> list` | Read one (schema checked). |
| `index(rows) -> dict` | `{(task, metric): row}`; duplicates raise. |

Evaluators (`srspd_experiments.eval.*`) each expose `resolve_config(cfg)` and `run(cfg, ...)`
writing `results.json`: `lm_eval.run(cfg, *, out_dir=None, root=None, tasks_override=None,
runner=None)` (`lm_eval.tensor_parallel_size(cfg)` is the TP a config runs at),
`muse.run(cfg, ...)`, `recovery.run(cfg, ...)`; `krd.krd_from_results(...)` and
`krd.wmdp_krd_from_results(...)` compute KRD from lm-eval output.
