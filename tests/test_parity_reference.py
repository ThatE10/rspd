"""Parity with the reference implementation the package was distilled from.

If the reference repository is present (``$RSPD_REFERENCE_REPO``), its ``origin/main`` solve and effective-rank modules
are exported with ``git show`` into a temporary package (never imported from a
working tree, which may sit on another branch) and the package's solve is
checked against them to float64 round-off (<= 1e-12 relative) on random
problems, a singular batch Gram (the eigh path) and a ridge case. Skipped
cleanly when the reference or git is absent.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import types

import pytest

torch = pytest.importorskip("torch")

import rspd  # noqa: E402

from reference_modules import reference_repo  # noqa: E402

REPO = reference_repo()
REF = "origin/main"
FILES = {"erase": "rspd/erase.py", "erank": "rspd/erank.py"}
PKG = "_rspd_parity_ref"
TOL = 1e-12
F64 = torch.float64


def _git_show(path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{REF}:{path}"],
                          check=True, capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    if REPO is None:
        pytest.skip("no reference checkout: set RSPD_REFERENCE_REPO to run the live parity tests")
    try:
        src = {k: _git_show(v) for k, v in FILES.items()}
    except subprocess.CalledProcessError as e:
        pytest.skip(f"cannot read the reference at {REF}: {e.stderr.strip()[:200]}")
    d = tmp_path_factory.mktemp("ref") / PKG
    d.mkdir()
    (d / "__init__.py").write_text("")
    for k, text in src.items():
        # the reference imports its effective rank as ``rspd.erank``; point it at the export
        (d / f"{k}.py").write_text(text.replace("from rspd.erank import",
                                                f"from {PKG}.erank import"))
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [str(d)]
    sys.modules[PKG] = pkg
    mods = {}
    try:
        for k in ("erank", "erase"):
            name = f"{PKG}.{k}"
            spec = importlib.util.spec_from_file_location(name, d / f"{k}.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            mods[k] = mod
        yield mods["erase"]
    finally:
        for k in [m for m in sys.modules if m == PKG or m.startswith(PKG + ".")]:
            del sys.modules[k]


def _rel(a, b):
    a, b = a.to(F64), b.to(F64)
    return float((a - b).norm()) / max(float(b.norm()), 1e-300)


def _problem(seed, d_out=11, d_in=9, n_x=7, n_y=25, singular=False):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(d_out, d_in, generator=g, dtype=F64)
    X = torch.randn(n_x, d_in, generator=g, dtype=F64)
    Y = torch.randn(n_y, d_in, generator=g, dtype=F64)
    if singular:            # X and Y in a common 5-dim subspace of d_in: G_mu singular
        B = torch.linalg.qr(torch.randn(d_in, 5, generator=g, dtype=F64)).Q
        X, Y = X @ B @ B.T, Y @ B @ B.T
    return W, X, Y


CASES = [dict(seed=0), dict(seed=1, d_out=5, d_in=12, n_x=20, n_y=8),
         dict(seed=2, singular=True), dict(seed=3, ridge_rel=1e-6),
         dict(seed=4, singular=True, ridge_rel=1e-8)]
IDS = ["random", "wide-target", "singular", "ridge", "singular-ridge"]


def _split(case):
    c = dict(case)
    ridge = c.pop("ridge_rel", 0.0)
    return _problem(**c), ridge


def _same_info(a, b, keys):
    for k in keys:
        x, y = a[k], b[k]
        if isinstance(x, float) and x != x:
            assert y != y, k
        else:
            assert x == y, k


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("mu", [0.1, 1.0, 30.0])
def test_erased_component_forms(ref, case, mu):
    (W, X, Y), ridge = _split(case)
    G_X, G_Y = X.T @ X, Y.T @ Y
    for rank in (None, 2):
        a = rspd.core.erased_component(W, X, Y, mu=mu, rank=rank, ridge_rel=ridge)
        b = ref.erased_component(W, X, Y, mu=mu, rank=rank, ridge_rel=ridge)
        assert _rel(a.dense(), b.dense()) <= TOL
        _same_info(a.info, b.info, ("solver", "rank", "k", "truncated", "formula", "mu",
                                    "ridge_delta"))
    for lam_rel in (None, 0.05):
        a = rspd.core.erased_component_gram(W, X, G_Y, mu=mu, lam_rel=lam_rel, ridge_rel=ridge)
        b = ref.erased_component_gram(W, X, G_Y, mu=mu, lam_rel=lam_rel, ridge_rel=ridge)
        assert _rel(a.A @ a.B, b.A @ b.B) <= TOL
        assert _rel(a.info["sigma"], b.info["sigma"]) <= TOL
        assert a.info["fit"] == pytest.approx(b.info["fit"], rel=TOL, abs=1e-300)
        _same_info(a.info, b.info, ("solver", "rank", "k", "ridge_delta"))
    a = rspd.core.erased_component_grams(W, G_X, G_Y, mu=mu, ridge_rel=ridge)
    b = ref.erased_component_grams(W, G_X, G_Y, mu=mu, ridge_rel=ridge)
    assert _rel(a.A, b.A) <= TOL and torch.equal(a.B, b.B)
    _same_info(a.info, b.info, ("solver", "ridge_delta", "rank", "B_is_identity", "formula"))
    Sa, ia = rspd.core.readout_gram(G_Y, X, mu, ridge_rel=ridge)
    Sb, ib = ref.readout_gram(G_Y, X, mu, ridge_rel=ridge)
    assert _rel(Sa, Sb) <= TOL and ia == ib


@pytest.mark.parametrize("case", CASES, ids=IDS)
@pytest.mark.parametrize("svd_dtype", [torch.float64, torch.float32], ids=["f64", "f32"])
def test_truncate_component(ref, case, svd_dtype):
    (W, X, Y), ridge = _split(case)
    G_Y = Y.T @ Y
    specs = ["erank", 0.0, 1e-3, 0.1, 0.5]
    fa = rspd.core.erased_component_gram(W, X, G_Y, mu=2.0, ridge_rel=ridge, spectrum=False)
    fb = ref.erased_component_gram(W, X, G_Y, mu=2.0, ridge_rel=ridge, spectrum=False)
    ca = rspd.core.truncate_component(fa, X, specs, svd_dtype=svd_dtype)
    cb = ref.truncate_component(fb, X, specs, svd_dtype=svd_dtype)
    for a, b in zip(ca, cb):
        assert _rel(a.dense(), b.dense()) <= TOL
        assert _rel(a.info["sigma"], b.info["sigma"]) <= TOL
        _same_info(a.info, b.info, ("rank", "k", "rule", "lam_rel", "erank", "svd_dtype",
                                    "formula", "solver"))


#: package-only leak keys -> the reference key holding the same value.
NEW_NAMES = {new: old for old, new in rspd.core.solve.ENERGY_ALIASES.items()}


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_leaks(ref, case):
    (W, X, Y), ridge = _split(case)
    G_X, G_Y = X.T @ X, Y.T @ Y
    for rank in (None, 3):
        a = rspd.core.erased_component(W, X, Y, mu=1.0, rank=rank, ridge_rel=ridge)
        b = ref.erased_component(W, X, Y, mu=1.0, rank=rank, ridge_rel=ridge)
        for fa, fb, args in ((rspd.core.leaks_gram, ref.leaks_gram, (X, G_Y)),
                             (rspd.core.leaks, ref.leaks, (X, Y))):
            la, lb = fa(W, a, *args), fb(W, b, *args)
            # the package adds descriptive names of two reference keys (same values)
            assert set(la) - set(NEW_NAMES) == set(lb)
            for k in la:
                assert la[k] == pytest.approx(lb[NEW_NAMES.get(k, k)], rel=TOL, abs=1e-300), k
    a = rspd.core.erased_component_grams(W, G_X, G_Y, mu=1.0, ridge_rel=ridge)
    b = ref.erased_component_grams(W, G_X, G_Y, mu=1.0, ridge_rel=ridge)
    la, lb = rspd.core.leaks_grams(W, a, G_X, G_Y), ref.leaks_grams(W, b, G_X, G_Y)
    for k in la:
        assert la[k] == pytest.approx(lb[NEW_NAMES.get(k, k)], rel=TOL, abs=1e-300), k


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_specificity(ref, case):
    (W, X, Y), ridge = _split(case)
    a = rspd.core.specificity(Y.T @ Y, X, Y.shape[0], k=4, ridge_rel=ridge, full=True)
    b = ref.specificity_gram(Y.T @ Y, X, Y.shape[0], k=4, ridge_rel=ridge)
    assert a["solver"] == b["solver"] and a["n"] == b["n"] and a["n_above"] == b["n_above"]
    assert a["top"] == pytest.approx(b["top"], rel=TOL, abs=1e-300)
    assert a["median"] == pytest.approx(b["median"], rel=TOL, abs=1e-300)
