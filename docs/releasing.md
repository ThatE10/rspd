# Releasing

**The repository is private and there is no public release.** Nothing is uploaded to PyPI,
TestPyPI or any other index (TestPyPI is public), and no licence has been chosen yet. A
"release" here is a git tag plus wheels built reproducibly from it, kept wherever the
consumer can read them. Other repositories pin a commit or tag directly from git.

## Versions and tags

Both distributions carry their version in `pyproject.toml` (`[project] version`), and
`srspd` also in `srspd.__version__`; `srspd-experiments` in
`srspd_experiments.__version__`. They are released together from one commit.

| tag | meaning |
| --- | --- |
| `v<X.Y.Z>` | both packages at version `X.Y.Z` (the version in both `pyproject.toml` files and both `__version__`) |

Pre-1.0 (`0.y.z`): bump `y` for any change to a public signature, a default, or a result
(a solve, a leak, a dose conversion); bump `z` for fixes that change no number and no
signature. Results of `srspd-experiments` also depend on the pinned stack in
`packages/srspd-experiments/constraints.txt`; changing a pin is a `y` bump.

## Cutting a release

1. On a clean checkout of the commit to release, set the version in
   `packages/srspd/pyproject.toml`, `packages/srspd/src/srspd/__init__.py`,
   `packages/srspd-experiments/pyproject.toml` and
   `packages/srspd-experiments/src/srspd_experiments/__init__.py`; commit.
2. Run the full suite from the repository root:

   ```bash
   PYTHONPATH=packages/srspd/src:packages/srspd-experiments/src python -m pytest -q
   ```

3. Tag (annotated) and push the tag:

   ```bash
   git tag -a v0.1.0 -m "srspd 0.1.0"
   git push origin v0.1.0
   ```

4. Build the wheels (below) from the tag and keep them with the tag (for example as
   private release assets).

## Building wheels reproducibly

```bash
git checkout v0.1.0
git clean -xdf packages/          # deletes untracked files under packages/: a stale build/ or *.egg-info/ can carry deleted files into a wheel
export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)   # fixed timestamps inside the wheel
python -m pip wheel --no-deps -w dist/ packages/srspd packages/srspd-experiments
sha256sum dist/*.whl
```

This produces `dist/srspd-<version>-py3-none-any.whl` and
`dist/srspd_experiments-<version>-py3-none-any.whl` (pure Python). `--no-deps` builds only
these two; their dependencies come from the consumer's index at install time. With the
same commit, a clean tree, the same `SOURCE_DATE_EPOCH` and the same setuptools, two builds
give byte-identical wheels. Add `--no-build-isolation` to build with the setuptools already
installed instead of fetching one (offline builds; then record the setuptools version).
`srspd-experiments`' wheel includes its data specs, evaluation tasks and recipes
(`[tool.setuptools.package-data]`).

## Pinning from another repository

From git, at a tag or a full commit SHA (the consumer's git needs read access):

```text
# requirements.txt
srspd[hf] @ git+https://github.com/ThatE10/srspd@v0.1.0#subdirectory=packages/srspd
srspd-experiments @ git+https://github.com/ThatE10/srspd@v0.1.0#subdirectory=packages/srspd-experiments
```

```toml
# pyproject.toml
dependencies = [
  "srspd[hf] @ git+https://github.com/ThatE10/srspd@<full-sha>#subdirectory=packages/srspd",
]
```

A full SHA is immutable; a tag can be moved, so pin a SHA where reproducibility matters.
Pin `srspd` and `srspd-experiments` at the same ref: `srspd-experiments` requires
`srspd[hf]` with no version bound, and no index can supply it. Over SSH use
`git+ssh://git@github.com/ThatE10/srspd.git@<ref>#subdirectory=...`.

From a built wheel: `pip install dist/srspd-0.1.0-py3-none-any.whl` (and the experiments
wheel after it), or `srspd @ file:///abs/path/srspd-0.1.0-py3-none-any.whl` in a
requirements file.
