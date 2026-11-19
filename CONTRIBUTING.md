# Contributing

## Setup

```bash
git clone https://github.com/ThatE10/rspd && cd rspd
pip install -e ".[hf,experiments,test,examples]"
python -m pytest -q
```

The suite runs on a CPU in about two minutes. It includes the documentation examples (every
block marked `python runnable` in `README.md`, `AGENTS.md` and `docs/`) and the notebooks under
`examples/`, which are executed top to bottom.

## Layout

| path | what |
| --- | --- |
| `src/rspd/core/` | the mathematics, on plain tensors. No model classes and no model libraries: a test enforces it. |
| `src/rspd/srspd.py`, `api.py`, `pipeline.py` | the supervised variant on a live model |
| `src/rspd/models/`, `capture/` | finding the layers of a model, collecting activations, model-family quirks, export |
| `src/rspd/toy.py` | the toy of the quickstart and the README animation |
| `src/rspd/experiments/` | `rspd[experiments]`: data, baselines, evaluation, recipes |
| `examples/` | notebooks |
| `assets/make_demo.py` | the README animation |

## Conventions

- `X` is the target set (removed), `Y` the retained set (kept). The edit is always `W − M`. The
  dose is `mu_per_token`. `AGENTS.md` has the full list.
- The top level of `rspd` exports a handful of names. New public names go in `docs/api.md`.
- A change to an example notebook is re-run and committed with its outputs.

## The README animation

```bash
pip install manim            # also needs LaTeX and ffmpeg
python assets/make_demo.py   # writes assets/demo.gif
```

## Parity with the reference implementation

Three test files compare this package bit for bit with the research code it was distilled
from. They run only when `RSPD_REFERENCE_REPO` names a checkout of that (private) repository,
and skip otherwise. `tests/test_golden.py` carries the same agreement as a frozen record for
everyone else; regenerate it with `python tests/make_golden.py` only at a commit where the live
tests pass.
