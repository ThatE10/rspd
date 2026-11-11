# AGENTS.md

The convention, which every function and doc in this repository follows:

1. `X` is the **target** set (erased). `Y` is the **retained** set (kept). Never swap them.
2. The edit is always `W_new = W − M`, through `srspd.apply_erasure`. Never write a fitted or kept map into a model.
3. Rank truncation acts on `M` (COALA projector on `M` over `X`), never on `W` or the kept map.
4. Leaks `L_X`, `L_Y` are measured and reported for every edited matrix; disentanglement is never assumed.
5. The dose is `mu_per_token`; `mu_raw` and `mu` are derived from the set sizes and only reported.

(Full agent guide is tracked in issue #7.)
