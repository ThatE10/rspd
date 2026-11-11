"""``srspd.dose``: the dose conversions and the per-direction erased fraction
``f = gamma m / (1 + gamma m)`` that :func:`srspd.suggest_mu_per_token` inverts."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

import srspd  # noqa: E402
from srspd import Dose, dose_from_mu_per_token, specificity, suggest_mu_per_token  # noqa: E402
from srspd.dose import erased_fraction  # noqa: E402


@pytest.mark.parametrize("m,n_x,n_y", [(0.7, 40, 300), (1.0, 1000, 1000), (3e-3, 7, 123456)])
def test_conversions_round_trip(m, n_x, n_y):
    d = dose_from_mu_per_token(m, n_x, n_y)
    assert isinstance(d, Dose) and d.mu_per_token == m and (d.n_x, d.n_y) == (n_x, n_y)
    assert d.mu == pytest.approx(n_x / (m * n_y), rel=1e-14)
    assert d.mu * d.mu_raw == pytest.approx(1.0, rel=1e-14)
    # back from the solve's mu to the user-facing dose
    assert n_x / (d.mu * n_y) == pytest.approx(m, rel=1e-14)
    assert d.mu_raw * n_x / n_y == pytest.approx(m, rel=1e-14)
    rep = d.as_dict()
    assert set(rep) >= {"mu_per_token", "mu_raw", "mu", "n_x", "n_y"}
    assert rep["mu"] == d.mu and rep["mu_raw"] == d.mu_raw and rep["mu_per_token"] == m
    with pytest.raises(Exception):
        d.mu_per_token = 2.0                                   # frozen


def test_dose_validation():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            dose_from_mu_per_token(bad, 10, 10)
    with pytest.raises(ValueError):
        dose_from_mu_per_token(1.0, 0, 10)
    with pytest.raises(ValueError):
        dose_from_mu_per_token(1.0, 10, 2.5)


def _problem(seed=0, d=6, nx=40, ny=300):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((nx, d)) * rng.uniform(0.2, 3.0, d)
    Y = rng.standard_normal((ny, d)) * rng.uniform(0.2, 3.0, d)
    return torch.from_numpy(X), torch.from_numpy(Y)


def test_erased_fraction_is_the_spectrum_of_the_solve():
    """eig(X^T X (X^T X + mu Y^T Y)^{-1}) = gamma m / (1 + gamma m), to 1e-10."""
    X, Y = _problem()
    mpt = 0.7
    nx, ny = X.shape[0], Y.shape[0]
    dose = dose_from_mu_per_token(mpt, nx, ny)
    G_X, G_Y = X.T @ X, Y.T @ Y
    K = G_X @ torch.linalg.inv(G_X + dose.mu * G_Y)
    ev = torch.linalg.eigvals(K)
    assert float(ev.imag.abs().max()) < 1e-10
    ev = torch.sort(ev.real, descending=True).values
    sp = specificity(G_Y, X, ny, k=6, full=True)
    assert sp["solver"] == "cholesky" and sp["n"] == 6 and len(sp["spectrum"]) == 6
    assert sp["top"] == sp["spectrum"][:6]
    gamma = torch.tensor(sp["spectrum"], dtype=torch.float64)
    want = erased_fraction(gamma, mpt)
    assert torch.allclose(ev, want, atol=1e-10, rtol=0)
    assert torch.allclose(dose.erased_fraction(gamma), want, atol=0, rtol=0)
    # half-erasure at gamma m = 1
    assert erased_fraction(1.0 / mpt, mpt) == pytest.approx(0.5)


def test_full_spectrum_flag_keeps_the_summary():
    X, Y = _problem(seed=1, d=8, nx=5, ny=100)
    a = specificity(Y.T @ Y, X, Y.shape[0], k=3)
    b = specificity(Y.T @ Y, X, Y.shape[0], k=3, full=True)
    assert "spectrum" not in a and {k: b[k] for k in a} == a
    assert len(b["spectrum"]) == b["n"] == 5
    assert srspd.solve.specificity_gram is specificity


@pytest.mark.parametrize("fractions,quantiles", [((0.1, 0.5, 0.9), (0.5, 0.9, 0.99)),
                                                 ((0.25,), (0.0, 1.0)), ((0.5,), (0.75,))])
def test_suggest_puts_the_quantile_at_the_fraction(fractions, quantiles):
    X, Y = _problem(seed=3, d=12, nx=200, ny=500)
    sp = specificity(Y.T @ Y, X, Y.shape[0], full=True)
    out = suggest_mu_per_token(sp, fractions=fractions, quantiles=quantiles)
    g = torch.tensor(sp["spectrum"], dtype=torch.float64)
    assert len(out["rows"]) == len(fractions) * len(quantiles)
    for row in out["rows"]:
        gq = float(torch.quantile(g, row["quantile"]))
        assert row["gamma"] == pytest.approx(gq, rel=1e-14)
        assert erased_fraction(gq, row["mu_per_token"]) == pytest.approx(row["fraction"],
                                                                         rel=1e-12)
    assert out["mu_per_token"] == sorted({r["mu_per_token"] for r in out["rows"]})
    # a spectrum given directly is the same as the dict
    assert suggest_mu_per_token(sp["spectrum"], fractions, quantiles) == out


def test_suggest_end_to_end_on_the_solve():
    """At the suggested dose, the q-quantile generalised direction of the solve
    is erased by the requested fraction (full rank, no ridge)."""
    X, Y = _problem(seed=4, d=6, nx=60, ny=400)
    nx, ny = X.shape[0], Y.shape[0]
    sp = specificity(Y.T @ Y, X, ny, full=True)
    out = suggest_mu_per_token(sp, fractions=(0.3,), quantiles=(1.0,))   # the top direction
    (row,) = out["rows"]
    dose = dose_from_mu_per_token(row["mu_per_token"], nx, ny)
    G_X, G_Y = X.T @ X, Y.T @ Y
    ev = torch.linalg.eigvals(G_X @ torch.linalg.inv(G_X + dose.mu * G_Y)).real
    assert float(ev.max()) == pytest.approx(0.3, abs=1e-10)


def test_suggest_validation_and_zero_gamma():
    with pytest.raises(ValueError, match="full gamma spectrum"):
        suggest_mu_per_token({"top": [1.0]})
    with pytest.raises(ValueError):
        suggest_mu_per_token([])
    with pytest.raises(ValueError):
        suggest_mu_per_token([1.0, -1.0])
    with pytest.raises(ValueError):
        suggest_mu_per_token([1.0], fractions=(1.0,))
    with pytest.raises(ValueError):
        suggest_mu_per_token([1.0], quantiles=(1.5,))
    out = suggest_mu_per_token([0.0, 0.0, 4.0], fractions=(0.5,), quantiles=(0.0, 1.0))
    assert math.isinf(out["rows"][0]["mu_per_token"])
    assert out["rows"][1]["mu_per_token"] == pytest.approx(0.25)
    assert out["mu_per_token"] == [pytest.approx(0.25)]
