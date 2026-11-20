# Representation Space-aware Parameter Decomposition (RSPD)

**Localize features within the parameter space.**

<p align="center">
  <img src="https://raw.githubusercontent.com/ThatE10/rspd/v0.0.1/assets/demo.gif" alt="A point moves through two overlapping sets X and Y in an input activation space. The layer W carries the whole space across. The component M carries only what X alone uses." width="460">
</p>

> **Status.** Unsupervised variant coming soon!  

## Install

```bash
pip install rspd
```

This requires installs `torch`, `transformers` and `safetensors` in order for the HuggingFace integration. `rspd[experiments]` adds the data builders, baselines and evaluation used in the paper.

The slim mathematical core exist in `rspd.core` needing only `torch`. For that alone: `pip install --no-deps rspd`, with `torch` already installed.

## Quickstart

```python runnable
from transformers import AutoModelForCausalLM, AutoTokenizer
from rspd import SRSPD

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dutch, english = ["De kat zit op de mat."], ["The cat sits on the mat."]  # placeholders, see below

srspd = SRSPD(model, tokenizer)                    # attach
components = srspd.fit(target=dutch, retained=english, mu_per_token=0.3)
components.apply()                                 # W <- W - M in every layer
components.restore()                               # bit-for-bit undo
```

`target` is the behaviour to remove and `retained` is the behaviour to keep. The one-sentence lists above only show the shape of the call. A real fit wants about a thousand tokens of target and tens of thousands of retained text, as in the [Dutch example](examples/erase_dutch.ipynb). `fit` is only computational. The fit functions returns one `Component` per weight matrix, with its factors `A` and `B` (`M = A @ B`) and the error. `mu_per_token` is the dose: larger removes more.

## How it works

A linear layer only ever acts on the activations it is given. For a set of activations `X`, the layer's response is `W X`, and

```
rank(W X) ≤ min(rank(W), rank(X))
```

So however large `W` is, the part of it that a behaviour uses is no bigger than the manifold that behaviour activations acts on. RSPD reads that part off the activations themselves instead of off the weights. This is the idea behind [parameter decomposition](https://arxiv.org/pdf/2506.20790), which inspired the technique: a network's weights are a sum of components, each doing one thing.

SRSPD, the supervised variant, takes two sets of activations: a **target** `X` to remove and a **retained** set `Y` to keep. It asks for the component `M` that reproduces the layer on the target and is as quiet as possible on the retained set:

```
M = argmin  ‖W Xᵀ − M Xᵀ‖² + μ ‖M Yᵀ‖²        which has the closed form        M = W XᵀX (XᵀX + μ YᵀY)⁺
```

(Activations are stored one per row, hence the transposes.) There is no training. The solve needs one pass over each set to collect activations and one factorisation per layer input. Subtracting the component, `W ← W − M`, removes the target behaviour and leaves the retained one in place; adding it back restores the model exactly.

Every fit also measures its **leaks** for each matrix: how much of the layer's response to the target survives the subtraction, and how much of its response to the retained set was disturbed. The full derivation is in [`docs/math.md`](docs/math.md).

## Examples

| | | |
| --- | --- | --- |
| [Quickstart on a toy](examples/quickstart.ipynb) | Two overlapping sets are found using SRSPD (runs on the CPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThatE10/rspd/blob/main/examples/quickstart.ipynb) |
| [Erase Dutch from an open language model](examples/erase_dutch.ipynb) | Remove dutch language from a model and keep the others, on a free Cloud GPU | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ThatE10/rspd/blob/main/examples/erase_dutch.ipynb) |

## Documentation

- [`docs/usage.md`](docs/usage.md): install options, supported model families, the command line.
- [`docs/math.md`](docs/math.md): the objective, the closed forms, leaks, the dose, rank truncation.
- [`docs/api.md`](docs/api.md): every public name.
- [`docs/experiments.md`](docs/experiments.md): `rspd[experiments]`.
- [`AGENTS.md`](AGENTS.md): the conventions and the full workflow, written for coding agents.

## Citation

If you use `rspd`, please cite it. [`CITATION.cff`](CITATION.cff) has the machine-readable form.

```bibtex
@software{rspd,
  title = {rspd: Representation Space-aware Parameter Decomposition},
  url   = {https://github.com/ThatE10/rspd},
  year  = {2026}
}
```

Rank truncation uses the context-aware projector of COALA: Parkina and Rakhuba, *COALA: Numerically Stable and Efficient Framework for Context-Aware Low-Rank Approximation*, arXiv:2507.07580 (NeurIPS 2025).
