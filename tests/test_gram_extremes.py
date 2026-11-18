"""``rspd.core.solve.gram_eig_extremes`` (the Gram conditioning check) and the CPU
fallback of the dense ``eigh`` in ``rspd.core.solve``."""
import pytest

torch = pytest.importorskip("torch")


def test_gram_eig_extremes_matches_eigvalsh():
    from rspd.core.solve import gram_eig_extremes
    g = torch.Generator().manual_seed(3)
    A = torch.randn(400, 40, dtype=torch.float64, generator=g)
    A[:, :5] *= 30.0                      # a spread spectrum, kappa ~1e3-1e4
    A[:, -3:] *= 0.05
    G = A.T @ A
    ev = torch.linalg.eigvalsh(G)
    lo, hi, info = gram_eig_extremes(G, iters=2000, tol=1e-12)
    assert info["pd"]
    assert abs(hi - float(ev[-1])) / float(ev[-1]) < 1e-6
    assert abs(lo - float(ev[0])) / float(ev[0]) < 1e-6
    # Rayleigh-quotient estimates: lam_max from below, lam_min from above
    assert hi <= float(ev[-1]) * (1 + 1e-12) and lo >= float(ev[0]) * (1 - 1e-12)


def test_gram_eig_extremes_not_pd_reports_zero():
    from rspd.core.solve import gram_eig_extremes
    g = torch.Generator().manual_seed(4)
    A = torch.randn(10, 30, dtype=torch.float64, generator=g)   # rank 10 < 30
    lo, hi, info = gram_eig_extremes(A.T @ A)
    assert lo == 0.0 and hi > 0.0 and info["pd"] is False


def test_eigh_oom_safe_falls_back_to_cpu(monkeypatch):
    """A CUDA OOM in the dense fallback eigh is retried on the CPU."""
    import rspd.core.solve as solve
    real = torch.linalg.eigh
    calls = {"n": 0}

    def flaky(G):
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return real(G)

    monkeypatch.setattr(torch.linalg, "eigh", flaky)
    g = torch.Generator().manual_seed(5)
    A = torch.randn(20, 8, dtype=torch.float64, generator=g)
    ev, V = solve._eigh_oom_safe(A.T @ A)
    monkeypatch.setattr(torch.linalg, "eigh", real)
    ev_ref, _ = real(A.T @ A)
    assert calls["n"] == 2 and torch.allclose(ev, ev_ref)
