# srspd

The slim core: closed-form erasure by subtraction (`W_new = W − M`), COALA rank truncation
of `M`, leaks, the `mu_per_token` dose, and COALA compression. See the repository README.

```python
import srspd
dose = srspd.dose_from_mu_per_token(1.0, n_x=X.shape[0], n_y=Y.shape[0])
comp = srspd.erased_component(W, X, Y, mu=dose.mu)
W_new = srspd.apply_erasure(W, comp)
print(srspd.leaks(W, comp, X, Y)["L_X_rel"], dose.as_dict())
```
