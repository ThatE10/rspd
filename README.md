# rspd

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
| `rspd` | `packages/rspd` | the slim core: solve, COALA truncation, edit, leaks, Gram streaming, HF model capture |
| `rspd[experiments]` | `packages/rspd[experiments]` | data builders, evaluation, baseline methods, reproduction recipes |

Status: private, pre-release; there is no public release (`docs/releasing.md`). Work is
tracked in this repository's issues.

## Documentation

| file | for |
| --- | --- |
| `AGENTS.md` | agents (and anyone new): the five convention rules, the dose, the standard workflow, the report, failure modes |
| `docs/usage.md`, `docs/experiments.md` | install, extras, minimal examples, supported models, credits |
| `docs/math.md` | the method: objective, closed forms, leaks, specificity, the dose derivation, COALA truncation |
| `docs/api.md` | every public symbol and CLI option of both packages |
| `docs/releasing.md` | wheels, tags, pinning from another repository |
| `docs/agent_acceptance.md` | the fresh-agent check `AGENTS.md` must pass |

Tests (both packages, including the documentation tests), from the repository root:

    PYTHONPATH=src:packages/rspd[experiments]/src python -m pytest -q

## Credits

Rank truncation and compression use the context-aware projector of **COALA**:
Parkina & Rakhuba, *COALA: Numerically Stable and Efficient Framework for Context-Aware
Low-Rank Approximation*, arXiv:2507.07580 (NeurIPS 2025). See `CITATION.cff`.
