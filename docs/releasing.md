# Releasing

One distribution, `rspd`, with the experiments code as the extra `rspd[experiments]`. The
version lives in two places that must agree: `pyproject.toml` and `rspd.__version__`.

## Versions

`X.Y.Z`. A change to the public interface (`rspd.__all__`, `rspd.core.__all__`, the command
line) or to the numbers a fit returns is at least a `Y` bump. A change of a pin in
`constraints.txt` is a `Y` bump: the pinned stack is the one the published experiment numbers
were produced on.

## Before a release

1. `python -m pytest -q` passes locally, and CI is green on `main` for every row of the matrix
   (Python 3.10 to 3.12; the oldest supported, the pinned and the newest `torch` and
   `transformers`).
2. Run both notebooks under `examples/` on the hardware they are written for and commit them
   with their outputs:
   - `quickstart.ipynb`: any CPU.
   - `erase_dutch.ipynb`: a free Colab GPU (T4). Record the wall-clock and peak memory it
     printed in the release notes. CI only runs this notebook's cells on a tiny random model.
3. Re-render the animation if `rspd.toy` or `assets/make_demo.py` changed:
   `python assets/make_demo.py`.
4. Set the version in `pyproject.toml` and `src/rspd/__init__.py`, and the install line of each
   notebook to `rspd==X.Y.Z` (with the extras it uses).
5. On PyPI the README's relative image link does not resolve: point `assets/demo.gif` at the
   tagged raw URL for the release, or accept a missing image on the PyPI page.

## Cutting it

```bash
git tag -a vX.Y.Z -m "rspd X.Y.Z"
git push origin vX.Y.Z
```

The `Release` workflow builds the sdist and wheel and checks that the tag matches the
version. It uploads to PyPI only when the repository variable `PYPI_PUBLISH` is `true` and a
trusted publisher for the workflow is configured on PyPI; until then the built files are kept
as a workflow artifact.

To build by hand:

```bash
git clean -xdf build dist src/*.egg-info
python -m build
twine check dist/*
```

## Installing an unreleased commit

```bash
pip install "rspd[hf] @ git+https://github.com/ThatE10/rspd@<sha>"
pip install "rspd[experiments,eval,muse] @ git+https://github.com/ThatE10/rspd@<sha>"
```

Add `-c constraints.txt` to reproduce published experiment numbers.
