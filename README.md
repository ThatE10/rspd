# srspd

Closed-form selective erasure of behaviour from a model's linear layers, by subtraction.

For each edited weight `W`, with `X` the activations of the behaviour to erase (the **target** set)
and `Y` the activations of the behaviour to keep (the **retained** set), SRSPD computes the
erased component

    M = W XᵀX (XᵀX + μ YᵀY)⁺

and writes `W_new = W − M`. There is no gradient descent. The dose is given as `mu_per_token`.
Every edited matrix reports its leaks, which measure how well the target and retained sets
are disentangled.

This repository holds two distributions:

| distribution | path | what it is |
| --- | --- | --- |
| `srspd` | `packages/srspd` | the slim core: solve, COALA truncation, edit, leaks, Gram streaming, HF model capture |
| `srspd-experiments` | `packages/srspd-experiments` | data builders, evaluation, baseline methods, reproduction recipes |

Status: private, pre-release. Work is tracked in this repository's issues.

## Credits

Rank truncation and compression use the context-aware projector of **COALA**:
Parkina & Rakhuba, *COALA: Numerically Stable and Efficient Framework for Context-Aware
Low-Rank Approximation*, arXiv:2507.07580 (NeurIPS 2025). See `CITATION.cff`.
