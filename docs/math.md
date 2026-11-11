# The method

This page states what `srspd` computes and why. Notation is fixed throughout the package
and never swapped.

| symbol | meaning |
| --- | --- |
| `W` | a pristine linear weight, `[d_out, d_in]` (GPT-2 `Conv1D` weights are transposed into this layout) |
| `X` | the **target** set: input activations of the behaviour to erase, batch-first `[n_x, d_in]` |
| `Y` | the **retained** set: input activations of the behaviour to keep, batch-first `[n_y, d_in]` |
| `G_X = XᵀX`, `G_Y = YᵀY` | the `d_in × d_in` Grams (at scale `Y` is only ever held as `G_Y`) |
| `μ` | the weight on the retained term of the objective (derived from the dose, below) |
| `G_μ = XᵀX + μ YᵀY` | the weighted batch Gram |
| `P = G_μ G_μ⁺` | the orthogonal projector onto span(X ∪ Y) |
| `M` | the **erased component** |
| `m` | the dose, `mu_per_token` |

## Objective and closed form

The erased component is the map that reproduces `W` on the target and is as small as
possible on the retained set:

    M = argmin_M  ‖W Xᵀ − M Xᵀ‖²_F + μ ‖M Yᵀ‖²_F

The objective is a convex quadratic in `M`. Setting its gradient to zero gives the normal
equations `M (XᵀX + μ YᵀY) = W XᵀX`, whose minimum-norm solution is

    M = W XᵀX G_μ⁺.

The edit is always

    W_new = W − M.

There is no gradient descent: one Gram per input, one factorisation per input and dose,
one matrix product per edited weight.

### Readout form (`solver="readout"`)

The target is held as activations. The **readout** `S = X G_μ⁺` (`[n_x, d_in]`) is the
least-squares readout of the target that is blind to the retained set, and

    M = (W Xᵀ) S.

One `S` serves every matrix that reads the same input (q/k/v share one, gate/up share
one). This form also exposes the target response `M Xᵀ`, which rank truncation needs.
Functions: `srspd.readout_gram`, `srspd.erased_component_gram`, `srspd.erased_component`.

### Two-Gram form (`solver="grams"`)

For a target too large to hold as activations, the target is streamed into `G_X` by the
same pass that forms `G_Y`, and

    M = W G_X (G_X + μ G_Y + δ I)⁺.

It is the same `M` (equal to float64 round-off for the same inputs and ridge). `M` is
returned dense (`A = M`, `B = I`) and cannot be rank-truncated, because `M Xᵀ` is never
formed. Functions: `srspd.factor_grams`, `srspd.erased_component_grams`,
`srspd.leaks_grams`.

## The kept map and the core table

The complementary **kept map** `M_keep = W μ YᵀY G_μ⁺` satisfies `M + M_keep = W P`. On the
batch the two possible edits agree; off it they do not:

| edit | on the target `X` | on the retained set `Y` | off the batch (`z ⊥ span(X ∪ Y)`) |
| --- | --- | --- | --- |
| `W_new = W − M` | `L_X` | `W Y − L_Y` | `W z` |
| `W_new = M_keep` | `L_X` | `W Y − L_Y` | `0` |

`(W − M) − M_keep = W (I − P)`: the original weight's action on every input direction
the batch does not span. Neither set spans the input space, so this term is not zero.
Writing the kept map into a model throws it away, and under rank truncation the kept map
also loses retained behaviour inside the span. That is why the only edit the package
writes is `W − M` (`srspd.apply_erasure`), and why `apply_erasure` refuses a `KeptMap`.
The table is available at run time as `srspd.CORE_TABLE`.

## Leaks

Disentanglement of `X` and `Y` is measured, never assumed. For every edited matrix:

    L_X_rel = ‖(W − M) Xᵀ‖_F / ‖W Xᵀ‖_F     target response that survives the edit
    L_Y_rel = ‖M Yᵀ‖_F       / ‖W Yᵀ‖_F     retained response the edit disturbs

Both are zero only when span(X) and span(Y) are disentangled and `M` is untruncated. Their
squares are reported as `erased_rel = L_X_rel²` and `retained_rel = L_Y_rel²` (the names
follow the code; read `erased_rel` as "fraction of target energy left"). From the Grams:
`‖M Yᵀ‖² = tr(M G_Y Mᵀ)`, so `L_Y` never needs `Y` itself (`srspd.leaks_gram`,
`srspd.leaks_grams`).

## The specificity spectrum

Let `C_X = XᵀX / n_x` and `C_Y = YᵀY / n_y` be the per-token covariances. Their
generalised eigenvalues `γ`,

    C_X v = γ C_Y v,

say how much more target than retained each input direction carries, per token. A
direction with `γ = 100` is a hundred times more target than retained; one with `γ ≪ 1`
is mostly retained. At most `min(n_x, d_in)` values are nonzero (the target spans at most
`n_x` directions); the rest are exactly 0 and not reported. `srspd.specificity` computes
them as the squared singular values of `X L_y⁻ᵀ / √n_x` with `L_y = chol(C_Y)` (one
triangular solve), falling back to an `eigh` pseudo-inverse square root when `C_Y` is not
positive definite. `srspd.erase` reports the top values, the median and the counts above
1, 10 and 100 per input key (`report.specificity[key]["specificity"]`).

## The dose

`μ` depends on how many tokens each set happens to hold, so it is not the dose a user
gives. Divide the objective by `n_x`:

    (1/n_x) ‖(W − M) Xᵀ‖² + (μ n_y / n_x) (1/n_y) ‖M Yᵀ‖²
      = E_x ‖(W − M) x‖² + (1/m) E_y ‖M y‖²,      m = n_x / (μ n_y),

where `E_x`, `E_y` are means over target and retained tokens. `1/m` is the **per-token
Lagrange multiplier** on the retained disturbance. The dose is `m = mu_per_token`; the
other two parametrisations are derived and only reported:

    μ      = n_x / (m n_y)          (the solve's weight on the retained term)
    mu_raw = 1/μ = m n_y / n_x      (the weight on the target term at unit retained weight)

Larger `m` erases more. `srspd.dose_from_mu_per_token(m, n_x, n_y)` returns a `Dose`
whose `.mu` is what the low-level solves take.

### What a dose erases

Substitute the covariances: `G_μ = (n_x / m)(m C_X + C_Y)`, so

    M = W · T,     T = XᵀX G_μ⁻¹ = m C_X (m C_X + C_Y)⁻¹.

For a generalised eigenvector `v_i` (`C_X v_i = γ_i C_Y v_i`), `(m C_X + C_Y) v_i =
(1 + γ_i m) C_Y v_i`, hence `T (C_Y v_i) = m C_X v_i / (1 + γ_i m) = f_i · C_Y v_i` with

    f_i = γ_i m / (1 + γ_i m).

So in the basis `{C_Y v_i}` the edit is diagonal: the erased component takes fraction
`f_i` of `W`'s action on direction `i`, and `W − M` keeps `1 − f_i = 1 / (1 + γ_i m)`.

* Half-erasure is at `γ m = 1`.
* A direction is erased by fraction `f` at `m = f / ((1 − f) γ)`.
* Directions with `γ = 0` are untouched at every dose.

This is exact at full rank with no ridge (`ridge_rel = 0`, no truncation, `C_Y`
nonsingular). A ridge adds `δ‖M‖²`, which is isotropic in input space and not diagonal in
the `γ` basis; a rank truncation selects directions by the SVD of `M Xᵀ`, which is weighted
by `W`. Both make the relation approximate.

`srspd.suggest_mu_per_token(gamma, fractions, quantiles)` inverts it: for every quantile
`q` of the `γ` spectrum and erased fraction `f` it returns `m = f / ((1 − f) γ_q)`
(`inf` when `γ_q = 0`), plus the sorted distinct finite values as a ready dose grid.

## Rank truncation of `M` (COALA)

Truncation acts on the erased component, never on `W` or the kept map:

    M_r = U_r U_rᵀ M,     U_r = the leading r left singular vectors of M Xᵀ.

This is the context-aware projector of **COALA** (Parkina & Rakhuba, *COALA: Numerically
Stable and Efficient Framework for Context-Aware Low-Rank Approximation*, arXiv:2507.07580,
NeurIPS 2025, Proposition 1), applied to `M` with the target as context. By Eckart–Young
on the response, `M_r Xᵀ = [M Xᵀ]_r`: the truncation keeps the best rank-`r` part of what
`M` does on the target. Off the target's span `M_r` is an orthogonal projection of `M`'s
output, not zero. Since `range(M) = range(M Xᵀ)`, `M_r = M` at full rank. The edit is still
`W_new = W − M_r`, and its leaks are measured like any other edit.

Rank rules (`rank_rule=` in `srspd.erase`, `specs` in `srspd.truncate_component`):

| rule | kept rank |
| --- | --- |
| `lam_rel` in (0, 1) | `r = #{j : σ_j² > lam_rel · σ_1²}` (a hard threshold on each matrix's own spectrum) |
| `"erank"` | `r = max(1, round(erank(σ)))`, with `erank(σ) = exp(−Σ p_j ln p_j)`, `p_j = σ_j / Σ σ` (Roy & Vetterli, EUSIPCO 2007) |

Every rule of one matrix and dose is sliced from one thin SVD of `M Xᵀ`, so a list of
rules costs one SVD, not one per rule.

### Compression is a different operation

`srspd.compress_weight` / `srspd.fit_coala` apply the same COALA projector to `W` itself
(`W_r = U_r U_rᵀ W`, `U_r` from `W Xᵀ`) and the result **replaces** the weight. It is the
only weight replacement in the package and it is never an erasure.

## Regularisation and numerics

* `ridge_rel > 0` solves with `G_μ + δ I`, `δ = ridge_rel · max(diag(μ YᵀY))`. On `M` this
  is a ridge toward zero (`+ δ‖M‖²`); `W − M = M_keep` still holds exactly.
* The positive-definiteness test is Cholesky plus the squared pivot ratio against
  `rcond = max(d_in · ε_float64, 1e-12)` (a singular PSD Gram often passes Cholesky in
  floating point with a round-off pivot).
* If Cholesky fails and `ridge_rel > 0`, `δ` escalates ×10 up to `1e-6` relative; if it
  still fails (or `ridge_rel = 0`), the solve falls back to an `eigh` pseudo-inverse over
  the eigenvalues above `rcond · λ_max`. The path (`cholesky`, `cholesky_ridge`, `eigh`) and
  the `δ` actually used are reported per matrix.
* Every solve runs in float64: a Gram squares the condition number
  (`κ(G) = κ([X; Y])²`), which is decisive in float32. On a bf16/fp16 model the edit is
  formed in fp32 and cast once; the fraction of `−M` that survives the cast is reported as
  `survival`.
