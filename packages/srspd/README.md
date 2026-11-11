# srspd

The slim core: closed-form erasure by subtraction (`W_new = W − M`), COALA rank truncation
of `M`, leaks, the `mu_per_token` dose, COALA compression, and capture on resident HF
models. See the repository README.

```python
import srspd
dose = srspd.dose_from_mu_per_token(1.0, n_x=X.shape[0], n_y=Y.shape[0])
comp = srspd.erased_component(W, X, Y, mu=dose.mu)
W_new = srspd.apply_erasure(W, comp)
print(srspd.leaks(W, comp, X, Y)["L_X_rel"], dose.as_dict())
```

On a resident HF model (`pip install 'srspd[hf]'`):

```python
info = srspd.inspect_model(model)            # matrices, input keys, Gram bytes, quirks
report = srspd.erase(model, target=target_ids, retained=retained_ids,
                     mu_per_token=[0.3, 1.0, 3.0], apply_mu_per_token=1.0)
report.matrices["L03.mlp.down_proj"]["L_X_rel"]   # leaks for every edited matrix
report.apply_dose(3.0)                       # re-dose without a second pass over the data
srspd.restore(model)                         # bit for bit
print(report.to_json())
```

`target` / `retained` are token ids (a `[T]` or `[B, T]` LongTensor, or a list of
sequences); each sequence is its own forward. `solver="readout"` holds the target
activations and supports `rank_rule` (`lam_rel` or `"erank"`, or a list of them: every
rule is truncated from the same solve, one shared SVD per matrix and dose, so a list costs
no extra Gram pass; pick the applied one with `apply_rank_rule=`, then
`report.apply_dose(m, rule)` for any other, `rule=None` being the full-rank edit);
`solver="grams"` streams the target into a Gram for a target too large to hold. Command line:
`srspd inspect MODEL`, `srspd erase MODEL --target t.txt --retained r.txt --mu-per-token 1
--out DIR`, `srspd report report.json`; each prints JSON.
