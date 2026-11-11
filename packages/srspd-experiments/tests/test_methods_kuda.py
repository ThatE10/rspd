import importlib
"""Unit tests for :mod:`srspd_experiments.methods.kuda` — KUDA's two substitutions.

CPU, synthetic, no checkpoint, no network, tiny tensors. What is worth pinning
before an H100 is booked:

* the **relaxed null-space projector** — symmetric, idempotent, and spanning
  exactly the directions whose singular value is below ``tau`` (with the
  argmin fallback when nothing clears the threshold). A projector built on the
  wrong side of the threshold would confine the update to the directions the
  retain corpus *uses*, which is the opposite of the method;
* the **representation-deviation loss** — monotonically decreasing as the
  cosine goes to -1, and matching the ported expression term for term;
* the **gradient projection** — after it, every gradient row lies inside the
  null subspace (``grad @ (I - P) == 0``), which is the whole guarantee the
  method rests on;
* **how much retain data the covariance pass sees** — ``sample_p`` is upstream's
  ``--sample_p``, a *percentage* of the retain corpus, not a row count. Because
  ``tau`` is an absolute threshold on that covariance's spectrum, a starved
  sample does not add noise, it selects a different subspace;
* **config validation** — it runs before the models load, so every bad key must
  raise a ``ValueError`` naming itself.
"""
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402



d = importlib.import_module("srspd_experiments.methods.kuda")

from recipe_configs import config, legacy  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
def _synthetic_cov(n=8, big=4, *, seed=0):
    """``(cov, U, S)``: a symmetric PSD ``n x n`` second moment with ``big``
    large eigenvalues and ``n - big`` tiny ones, built from a known orthonormal
    basis so the expected null subspace is exact."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    s = torch.tensor([10.0, 7.0, 4.0, 1.0][:big] + [1e-4, 5e-5, 2e-5, 1e-6][:n - big],
                     dtype=torch.float64)
    cov = Q @ torch.diag(s) @ Q.T
    return cov, Q, s


def _base_config():
    return {
        "name": "unit_kuda",
        "model": "HuggingFaceH4/zephyr-7b-beta",
        "dtype": "bfloat16",
        "kuda": {
            "beta": 2.0, "tau": 2.0e-3, "alpha": [1.0, 1.0], "lr": 1.0e-5,
            "batch_size": 4, "max_num_batches": 150, "epoch": 1,
            "layer_id": 9, "layer_ids": [6, 7, 8, 9],
            "forget_corpora": ["bio-forget-corpus", "cyber-forget-corpus"],
            "retain_corpora": ["wikitext", "wikitext"],
        },
    }


# --------------------------------------------------------------------------- #
# 1. The relaxed null-space projector                                          #
# --------------------------------------------------------------------------- #
def test_projector_is_a_symmetric_idempotent_projection():
    cov, _, _ = _synthetic_cov()
    P, info = d.build_projector(cov, tau=1e-2, dtype=torch.float64)
    assert torch.allclose(P, P.T, atol=1e-10), "P must be symmetric"
    assert torch.allclose(P @ P, P, atol=1e-10), "P must be idempotent"
    # A projector's trace is its rank.
    assert pytest.approx(float(torch.trace(P)), abs=1e-8) == info["n_dirs"]


def test_projector_spans_exactly_the_sub_tau_directions():
    """``P`` must keep the low-energy directions and kill the high-energy ones —
    the update may move only where the retain corpus barely lives."""
    n, big, tau = 8, 4, 1e-2
    cov, Q, s = _synthetic_cov(n=n, big=big)
    P, info = d.build_projector(cov, tau=tau, dtype=torch.float64)

    assert info["n_dirs"] == int((s < tau).sum()) == n - big
    assert info["d"] == n
    assert info["frac_kept"] == pytest.approx((n - big) / n)
    assert "cpu" in info["svd_path"]          # no CUDA in the test environment

    for i in range(n):
        u = Q[:, i]
        if s[i] < tau:                         # kept: P acts as the identity
            assert torch.allclose(P @ u, u, atol=1e-8), f"direction {i} should be kept"
        else:                                  # dropped: P annihilates it
            assert torch.allclose(P @ u, torch.zeros_like(u), atol=1e-8), \
                f"direction {i} should be projected out"


def test_projector_falls_back_to_argmin_when_nothing_clears_tau():
    """An empty null subspace would zero every gradient, so the single smallest
    direction is kept instead (upstream's ``if small.numel()==0`` branch)."""
    cov, Q, s = _synthetic_cov()
    P, info = d.build_projector(cov, tau=1e-12, dtype=torch.float64)
    assert info["n_dirs"] == 1
    smallest = Q[:, int(torch.argmin(s))]
    assert torch.allclose(P @ smallest, smallest, atol=1e-8)
    assert torch.allclose(P @ Q[:, 0], torch.zeros(P.shape[0], dtype=P.dtype), atol=1e-8)


def test_projector_default_dtype_is_bf16():
    cov, _, _ = _synthetic_cov()
    P, _ = d.build_projector(cov, tau=1e-2)
    assert P.dtype is torch.bfloat16          # 410 MB/layer at Zephyr's width


def test_svd_fallback_reports_the_path_that_ran():
    cov, _, _ = _synthetic_cov()
    _, S, path = d._svd_with_fallback(cov, prefer="cpu")
    assert path == "cpu/float64"
    assert torch.all(S[:-1] >= S[1:]), "singular values come back descending"
    # Asking for CUDA on a CPU-only box resolves to the CPU path, not a crash.
    _, _, path2 = d._svd_with_fallback(cov, prefer="cuda" if torch.cuda.is_available()
                                       else "auto")
    assert "float" in path2


#: Every config mutation that must change the COVARIANCE key. ``tau`` is
#: deliberately absent — it belongs to the projector key alone.
_COV_KEY_MUTATIONS = {
    "model": lambda c: c.__setitem__("model", "google/gemma-3-4b-it"),
    "retain_corpora": lambda c: c["kuda"].__setitem__(
        "retain_corpora", ["bio-retain-corpus", "wikitext"]),
    # sample_p is a PERCENTAGE: 50 is half the corpus, and max_cov_texts is a
    # separate, independently keyed knob.
    "sample_p": lambda c: c["kuda"].__setitem__("nullspace", {"sample_p": 50}),
    "sample_p_fractional": lambda c: c["kuda"].__setitem__("nullspace",
                                                           {"sample_p": 99.5}),
    "max_cov_texts": lambda c: c["kuda"].__setitem__("nullspace",
                                                     {"max_cov_texts": 1000}),
    "max_length": lambda c: c["kuda"].__setitem__("nullspace", {"max_length": 256}),
    # Result-affecting once padding_side is "left": left padding shifts every
    # sequence's RoPE position ids by the batch's own longest text.
    "batch_size": lambda c: c["kuda"].__setitem__("nullspace", {"batch_size": 8}),
    # The accumulation dtype moves membership of a tau cut. (The two *devices*
    # are keyed too, but by their RESOLVED value — see the dedicated test
    # below, since on a CPU-only box "auto" and "cpu" describe the same run.)
    "cov_dtype": lambda c: c["kuda"].__setitem__("nullspace",
                                                 {"cov_dtype": "float64"}),
    "min_len": lambda c: c["kuda"].__setitem__("min_len", 80),
    "dtype": lambda c: c.__setitem__("dtype", "float32"),
}


@pytest.mark.parametrize("field", sorted(_COV_KEY_MUTATIONS))
def test_cov_key_changes_with_every_field_that_changes_the_covariance(field):
    base = d._cov_key(d.resolve_kuda_config(_base_config()), 6)
    cfg = _base_config()
    _COV_KEY_MUTATIONS[field](cfg)
    assert d._cov_key(d.resolve_kuda_config(cfg), 6) != base, \
        f"{field} changes what the covariance pass sees but not its cache key"


def test_cov_key_is_stable_and_per_layer():
    R = d.resolve_kuda_config(_base_config())
    base = d._cov_key(R, 6)
    assert base == d._cov_key(d.resolve_kuda_config(_base_config()), 6)
    assert base != d._cov_key(R, 7)                      # layer


def test_the_RESOLVED_svd_and_cov_devices_are_part_of_the_covariance_key(monkeypatch):
    """The SVD's precision follows its device (float32 on CUDA, float64 on the
    CPU: ``_svd_with_fallback``), and an fp32-vs-fp64 spectrum moves membership
    of a ``tau = 2e-3`` cut — the driver already records ``svd_path`` in ``info``
    because "which path ran is part of the record", so the key cannot omit it.

    What is keyed is the **resolved** device, not the config string: on a
    CPU-only box ``auto`` and ``cpu`` describe the same computation and must
    share an entry, while on a GPU box they do not."""
    assert d._resolved_device("cpu") == "cpu"

    def _cov_key_for(cov_device, svd_device):
        cfg = _base_config()
        cfg["kuda"]["nullspace"] = {"cov_device": cov_device, "svd_device": svd_device}
        return d._cov_key(d.resolve_kuda_config(cfg), 6)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert d._resolved_device("auto") == "cpu" and d._resolved_device("cuda") == "cpu"
    assert _cov_key_for("auto", "auto") == _cov_key_for("cpu", "cpu")
    cpu_box = _cov_key_for("auto", "auto")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert d._resolved_device("auto") == "cuda" and d._resolved_device("cuda") == "cuda"
    gpu_box = _cov_key_for("auto", "auto")
    assert gpu_box != cpu_box, "the same config on a GPU box is a different SVD"
    assert _cov_key_for("cpu", "auto") != gpu_box     # cov_device alone moves it
    assert _cov_key_for("auto", "cpu") != gpu_box     # svd_device alone moves it
    assert _cov_key_for("cpu", "cpu") == cpu_box      # pinned to CPU == the CPU box
    # `svd_device: auto` follows the covariance, because _svd_with_fallback
    # starts from cov.device — so pinning the covariance to the CPU takes the
    # SVD with it even on a GPU box, and the key must say so.
    assert d.resolved_ns_devices({"cov_device": "cpu", "svd_device": "auto"}) \
        == ("cpu", "cpu")
    assert d.resolved_ns_devices({"cov_device": "auto", "svd_device": "auto"}) \
        == ("cuda", "cuda")
    assert d.resolved_ns_devices({"cov_device": "cpu", "svd_device": "cuda"}) \
        == ("cpu", "cuda")
    assert _cov_key_for("cpu", "auto") == cpu_box


def test_the_tokenizer_sides_are_part_of_the_covariance_key(monkeypatch):
    """They are not config keys (upstream forces them), but they decide which
    positions the covariance is taken over, so a future change to either must
    invalidate the cache rather than silently reuse it."""
    R = d.resolve_kuda_config(_base_config())
    base = d._cov_key(R, 6)
    for field, other in (("padding_side", "right"), ("truncation_side", "left")):
        flipped = dict(R, **{field: other})
        assert d._cov_key(flipped, 6) != base, f"{field} is missing from the cov key"


def test_tau_moves_the_projector_key_and_NOT_the_covariance_key():
    """The whole point of the split: a tau sweep must re-cut a cached spectrum,
    not repay the covariance pass and a d^3 SVD at every grid point. Upstream
    splits it the same way (layer_stats.py keys the second moment without tau,
    nullspace.py keys the projector with it)."""
    R = d.resolve_kuda_config(_base_config())
    cfg = _base_config()
    cfg["kuda"]["tau"] = 3.0e-3
    R2 = d.resolve_kuda_config(cfg)
    assert R["tau"] != R2["tau"]
    assert d._cov_key(R2, 6) == d._cov_key(R, 6), "tau must not touch the cov key"
    assert d._proj_key(R2, 6) != d._proj_key(R, 6), "tau must move the proj key"


@pytest.mark.parametrize("field", sorted(_COV_KEY_MUTATIONS))
def test_the_projector_key_inherits_everything_the_covariance_key_carries(field):
    """``_proj_key`` is ``_cov_key`` + tau, so nothing that invalidates the
    covariance may leave the projector's own entry standing."""
    base = d._proj_key(d.resolve_kuda_config(_base_config()), 6)
    cfg = _base_config()
    _COV_KEY_MUTATIONS[field](cfg)
    assert d._proj_key(d.resolve_kuda_config(cfg), 6) != base


def test_retain_corpus_ORDER_is_part_of_the_covariance_key():
    """``select_cov_texts`` dedups in INSERTION order, so with two distinct
    retain corpora ``[A, B]`` and ``[B, A]`` select differently-ordered texts —
    and differently *chosen* ones once ``max_cov_texts`` truncates the
    concatenation. The key used to be ``sorted(set(...))``, which collided."""
    def _cfg(order):
        c = _base_config()
        c["kuda"]["retain_corpora"] = list(order)
        c["kuda"]["nullspace"] = {"max_cov_texts": 4}
        return d.resolve_kuda_config(c)

    ab = _cfg(["wikitext", "bio-retain-corpus"])
    ba = _cfg(["bio-retain-corpus", "wikitext"])
    assert d.cov_corpora(ab) == ["wikitext", "bio-retain-corpus"]
    assert d.cov_corpora(ba) == ["bio-retain-corpus", "wikitext"]
    assert d._cov_key(ab, 6) != d._cov_key(ba, 6)
    assert d._proj_key(ab, 6) != d._proj_key(ba, 6)
    # And a *repeated* corpus still dedups to one entry, as the loader does.
    dup = _cfg(["wikitext", "wikitext"])
    assert d.cov_corpora(dup) == ["wikitext"]


def test_the_two_keys_are_bumped_past_every_v2_cache_entry():
    """The bf16 covariance alone invalidates every projector on disk, so the
    version must have moved off 2."""
    assert d.CACHE_VERSION == 3


def test_cache_key_cannot_collide_with_the_old_row_count_semantics():
    """``sample_p: 100`` changed meaning (100 rows -> 100% of the corpus), so the
    key must change even though its value did not: a v1 projector built from ~100
    documents must never be served to a v2 run that asked for the whole corpus."""
    R = d.resolve_kuda_config(_base_config())
    assert R["nullspace"]["sample_p"] == 100.0
    ns = R["nullspace"]
    legacy = {
        "model": R["model"],
        "retain_corpora": sorted(set(R["retain_corpora"])),
        "layer_id": 6,
        "module_name": R["module_name"],
        "tau": float(R["tau"]),
        "sample_p": int(ns["sample_p"]),          # the v1 payload, verbatim
        "max_length": int(ns["max_length"]),
        "min_len": int(R["min_len"]),
        "dtype": R["dtype"],
        "version": 1,
    }
    import hashlib as _h
    import json as _j
    old = _h.sha256(_j.dumps(legacy, sort_keys=True).encode()).hexdigest()[:16]
    assert d._cov_key(R, 6) != old


# --------------------------------------------------------------------------- #
# 2. The representation-deviation objective                                    #
# --------------------------------------------------------------------------- #
def _acts_at_cosine(cos_target, *, batch=2, tokens=3, dim=5):
    """A pair of activation tensors whose per-token cosine is exactly
    ``cos_target`` — built in a 2-D plane so the angle is controlled."""
    frozen = torch.zeros(batch, tokens, dim)
    updated = torch.zeros(batch, tokens, dim)
    frozen[..., 0] = 1.0
    updated[..., 0] = cos_target
    updated[..., 1] = (1.0 - cos_target ** 2) ** 0.5
    return updated, frozen


@pytest.mark.parametrize("beta", [1.0, 2.0, 5.0])
def test_deviation_loss_decreases_as_cosine_goes_to_minus_one(beta):
    losses, cosines = [], []
    for target in (1.0, 0.5, 0.0, -0.5, -1.0):
        upd, frz = _acts_at_cosine(target)
        loss, cos = d.deviation_loss(upd, frz, beta)
        losses.append(float(loss))
        cosines.append(cos)
    assert cosines == pytest.approx([1.0, 0.5, 0.0, -0.5, -1.0], abs=1e-5)
    assert all(a > b for a, b in zip(losses, losses[1:])), \
        f"loss must fall monotonically as cos -> -1, got {losses}"


def test_deviation_loss_matches_the_ported_expression():
    upd, frz = _acts_at_cosine(-0.25)
    beta = 2.0
    loss, cos = d.deviation_loss(upd, frz, beta)
    ref_cos = F.cosine_similarity(upd, frz, dim=-1).mean(-1)
    ref = -F.logsigmoid(beta * (1.0 - ref_cos)).mean() * 2 / beta
    assert float(loss) == pytest.approx(float(ref), rel=1e-6)
    assert cos == pytest.approx(-0.25, abs=1e-5)


def test_retain_loss_is_alpha_weighted_mse_and_zero_at_the_frozen_point():
    a = torch.randn(2, 3, 4)
    assert float(d.retain_loss(a, a.clone(), 1.0)) == pytest.approx(0.0, abs=1e-9)
    b = a + 0.5
    assert float(d.retain_loss(b, a, 2.0)) == pytest.approx(
        2.0 * float(F.mse_loss(b, a)), rel=1e-6)


def test_forward_with_cache_handles_tuple_and_tensor_block_outputs():
    """transformers 4.x decoder blocks return a tuple, several 5.x ones return a
    bare tensor; the hook must take the hidden states either way."""
    class Tuple(torch.nn.Module):
        def forward(self, x):
            return (x * 2, None)

    class Bare(torch.nn.Module):
        def forward(self, x):
            return x * 3

    for mod, k in ((Tuple(), 2.0), (Bare(), 3.0)):
        wrapper = torch.nn.Sequential(mod)
        x = torch.randn(1, 2, 3)
        got = d.forward_with_cache(wrapper, {"input": x}, mod, no_grad=True)
        assert torch.allclose(got, x * k)


# --------------------------------------------------------------------------- #
# 3. The gradient projection                                                   #
# --------------------------------------------------------------------------- #
def test_project_gradients_leaves_every_grad_inside_the_null_subspace():
    n, big, tau = 8, 4, 1e-2
    cov, Q, s = _synthetic_cov(n=n, big=big)
    P, info = d.build_projector(cov, tau=tau, dtype=torch.float64)

    # down_proj.weight is (d_model, d_intermediate); P is (d_int, d_int).
    w = torch.nn.Parameter(torch.randn(3, n, dtype=torch.float64))
    w.grad = torch.randn(3, n, dtype=torch.float64)
    raw = w.grad.clone()

    n_proj = d.project_gradients({"L6.weight": w}, {"L6.weight": P})
    assert n_proj == 1
    assert w.grad.shape == raw.shape
    assert not torch.allclose(w.grad, raw), "a real projection must change the grad"

    complement = torch.eye(n, dtype=torch.float64) - P
    assert torch.allclose(w.grad @ complement, torch.zeros_like(w.grad), atol=1e-9), \
        "the projected gradient must have no component outside the null subspace"
    # Projecting twice is projecting once.
    d.project_gradients({"L6.weight": w}, {"L6.weight": P})
    assert torch.allclose(w.grad, raw @ P, atol=1e-9)
    # And the surviving component is exactly the sub-tau part of the raw grad.
    kept = Q[:, [i for i in range(n) if s[i] < tau]]
    assert torch.allclose(w.grad, raw @ kept @ kept.T, atol=1e-9)
    assert info["n_dirs"] == kept.shape[1]


def test_project_gradients_skips_none_and_refuses_an_unknown_parameter():
    P = torch.eye(3, dtype=torch.float64)
    w = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.float64))
    assert d.project_gradients({"a": w}, {"a": P}) == 0          # grad is None
    w.grad = torch.randn(2, 3, dtype=torch.float64)
    with pytest.raises(KeyError):
        d.project_gradients({"a": w}, {"b": P})


def test_a_projected_sgd_step_only_moves_the_weight_in_null_directions():
    """End to end on a toy parameter: with a step that is linear in the gradient
    (plain SGD), the projection confines the *weight change* to the null
    subspace, not merely the gradient.

    The driver's optimizer is AdamW (as upstream's is), whose elementwise
    second-moment preconditioning does **not** commute with a projection — so
    with AdamW the guarantee is exactly the one KUDA states, on the gradient.
    SGD is used here to test the geometry itself rather than Adam's rescaling.
    """
    n, big, tau = 8, 4, 1e-2
    cov, Q, s = _synthetic_cov(n=n, big=big)
    P, _ = d.build_projector(cov, tau=tau, dtype=torch.float64)
    w = torch.nn.Parameter(torch.randn(3, n, dtype=torch.float64))
    before = w.detach().clone()
    opt = torch.optim.SGD([w], lr=1e-2)
    (w * torch.randn(3, n, dtype=torch.float64)).sum().backward()
    d.project_gradients({"w": w}, {"w": P})
    opt.step()
    delta = (w.detach() - before)
    big_dirs = Q[:, [i for i in range(n) if s[i] >= tau]]
    assert not torch.allclose(delta, torch.zeros_like(delta)), "the step did nothing"
    assert torch.allclose(delta @ big_dirs, torch.zeros(3, big, dtype=torch.float64),
                          atol=1e-12), "the step leaked into a high-energy direction"


# --------------------------------------------------------------------------- #
# 3b. The covariance pass and the projector cache, on a synthetic decoder      #
# --------------------------------------------------------------------------- #
D_MODEL = 4
PAD_ID = 0


class _MLP(torch.nn.Module):
    """``down_proj`` sees its block's hidden state unchanged, so the captured
    input features are exactly the embeddings — the expected covariance is then
    computable without re-using the code under test."""

    def __init__(self):
        super().__init__()
        self.down_proj = torch.nn.Linear(D_MODEL, D_MODEL, bias=False)

    def forward(self, h):
        return self.down_proj(h)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, h):
        return (h + 0.0 * self.mlp(h),)     # identity, but down_proj still runs


class _Inner(torch.nn.Module):
    def __init__(self, n_layers, vocab):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, D_MODEL)
        self.layers = torch.nn.ModuleList([_Block() for _ in range(n_layers)])


class _FakeLM(torch.nn.Module):
    """Minimal ``model.model.layers[i].mlp.down_proj`` shape — what
    ``common.get_down_proj_module`` resolves on Mistral/Zephyr."""

    def __init__(self, n_layers=2, vocab=7):
        super().__init__()
        self.model = _Inner(n_layers, vocab)

    def forward(self, input_ids, attention_mask=None, **kw):
        h = self.model.embed(input_ids)
        for blk in self.model.layers:
            h = blk(h)[0]
        return h


class _FakeTokenizer:
    """Whitespace tokenizer: word i -> id i+1, PAD = 0, right-padded."""

    def __call__(self, texts, return_tensors=None, padding=None, truncation=None,
                 max_length=None):
        rows = [[(ord(c) % 6) + 1 for c in t][:max_length or 8] for t in texts]
        width = max(len(r) for r in rows)
        ids = torch.full((len(rows), width), PAD_ID, dtype=torch.long)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, r in enumerate(rows):
            ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_covariance_is_the_masked_uncentered_second_moment():
    """The exactness check, run at ``cov_dtype: float64``.

    The *default* is bfloat16 (upstream's ``mom2_dtype``) and is pinned by
    :func:`test_the_covariance_accumulates_in_bfloat16_by_default`; float64 is
    asked for explicitly here so the masking and the normalisation can be
    checked against an independently computed second moment rather than against
    a quantised one.
    """
    torch.manual_seed(0)
    model, tok = _FakeLM(), _FakeTokenizer()
    texts = ["abcde", "fg", "hijklm"]
    covs, n_tokens = d.accumulate_down_proj_cov(model, tok, [0, 1], texts,
                                                batch_size=2, max_length=8,
                                                cov_dtype="float64")
    assert set(covs) == {0, 1}
    assert covs[0].dtype is torch.float64 and covs[0].shape == (D_MODEL, D_MODEL)

    # Expected: mean over NON-PAD tokens of e e^T, e the embedding of that token.
    enc = tok(texts, max_length=8)
    ids = enc["input_ids"][enc["attention_mask"].bool()]
    X = model.model.embed(ids).double()
    assert n_tokens == X.shape[0] == sum(min(len(t), 8) for t in texts)
    assert torch.allclose(covs[0], X.T @ X / X.shape[0], atol=1e-10)
    # PAD positions really are excluded: including them would change the answer.
    all_ids = enc["input_ids"].reshape(-1)
    Xp = model.model.embed(all_ids).double()
    assert not torch.allclose(covs[0], Xp.T @ Xp / Xp.shape[0], atol=1e-6)
    # Symmetric PSD, as a second moment must be.
    assert torch.allclose(covs[0], covs[0].T, atol=1e-12)
    assert float(torch.linalg.eigvalsh(covs[0]).min()) > -1e-10


def test_covariance_pass_forces_right_padding_and_restores_the_side():
    """The covariance MUST be built with RIGHT padding, whatever the training
    ``padding_side`` is, and the tokenizer's own side must be put back afterwards.

    Left padding here (the training side, forced by upstream)
    passes no ``position_ids``, so transformers-5.x assigns RoPE positions
    ``arange(L)`` and shifts every real token by the batch's pad count — corrupting
    the ``mlp.down_proj`` input features and collapsing the spectrum until ~80% of
    singular values fall below ``tau``, i.e. a near-identity projector that gives
    KUDA no retain protection. Upstream builds this second moment through ROME's
    right-padded ``TokenizedDataset`` with natural positions; this pins that.
    """
    class _RecordingTokenizer(_FakeTokenizer):
        def __init__(self):
            self.padding_side = "left"          # the training side
            self.seen = []

        def __call__(self, *a, **kw):
            self.seen.append(self.padding_side)  # what the pass actually used
            return super().__call__(*a, **kw)

    torch.manual_seed(0)
    model, tok = _FakeLM(), _RecordingTokenizer()
    d.accumulate_down_proj_cov(model, tok, [0], ["abcde", "fg"],
                               batch_size=2, max_length=8)
    assert tok.seen and all(side == "right" for side in tok.seen), tok.seen
    assert tok.padding_side == "left"           # restored to the training side


def test_the_covariance_accumulates_in_bfloat16_by_default():
    """Upstream's ``mom2_dtype`` is ``bfloat16`` (hyparams/*.json ->
    layer_stats.py:117,163 -> runningstats.py:256-260, which allocates ``mom2``
    in the features' own dtype). The default must match it — the published
    ``tau = 2e-3`` was tuned against a bf16 spectrum — and the float64 path must
    stay reachable and give a *different*, more accurate answer."""
    torch.manual_seed(0)
    model, tok = _FakeLM(), _FakeTokenizer()
    texts = ["abcde", "fg", "hijklm"]
    kw = dict(batch_size=2, max_length=8)

    default, _ = d.accumulate_down_proj_cov(model, tok, [0], texts, **kw)
    assert default[0].dtype is torch.bfloat16
    named, _ = d.accumulate_down_proj_cov(model, tok, [0], texts,
                                          cov_dtype="bfloat16", **kw)
    assert torch.equal(default[0], named[0])          # the default IS bfloat16

    exact, _ = d.accumulate_down_proj_cov(model, tok, [0], texts,
                                          cov_dtype="float64", **kw)
    assert exact[0].dtype is torch.float64
    # Same matrix to within bf16's ~8-bit mantissa, and NOT the same matrix.
    assert torch.allclose(default[0].double(), exact[0], atol=5e-2)
    assert not torch.equal(default[0].double(), exact[0])


def test_cov_dtype_resolves_by_name_and_rejects_anything_else():
    assert d.DEFAULT_COV_DTYPE == "bfloat16"
    assert d.resolve_cov_dtype("bfloat16") is torch.bfloat16
    assert d.resolve_cov_dtype("float64") is torch.float64
    assert d.resolve_cov_dtype(torch.float32) is torch.float32   # passthrough
    for bad in ("float8", "int8", None, 64):
        with pytest.raises(ValueError, match="cov_dtype"):
            d.resolve_cov_dtype(bad)
    # And it resolves from the config, with bfloat16 as the default.
    assert d.resolve_kuda_config(_base_config())["nullspace"]["cov_dtype"] == "bfloat16"
    cfg = _base_config()
    cfg["kuda"]["nullspace"] = {"cov_dtype": "float64"}
    assert d.resolve_kuda_config(cfg)["nullspace"]["cov_dtype"] == "float64"
    cfg["kuda"]["nullspace"] = {"cov_dtype": "float8"}
    with pytest.raises(ValueError, match="cov_dtype"):
        d.resolve_kuda_config(cfg)


def _cache_cfg():
    cfg = _base_config()
    cfg["kuda"].update({"layer_id": 1, "layer_ids": [0, 1], "tau": 1.0})
    cfg["kuda"]["nullspace"] = {"sample_p": 100, "max_length": 8, "batch_size": 2}
    return cfg


def _cache_files(tmp_path):
    return (sorted(p.name for p in tmp_path.glob("kuda_cov_L*.pt")),
            sorted(p.name for p in tmp_path.glob("kuda_nullspace_L*.pt")))


def test_projectors_are_built_once_and_then_read_from_the_disk_cache(tmp_path,
                                                                     monkeypatch):
    monkeypatch.setattr(d.C, "load_corpus_texts_all",
                        lambda corpus, **kw: ["abcde", "fghij", "klmno"])
    model, tok = _FakeLM(), _FakeTokenizer()
    R = d.resolve_kuda_config(_cache_cfg())

    P1, info1 = d.load_or_build_projectors(model, tok, R, cache_dir=str(tmp_path))
    assert set(P1) == {0, 1}
    assert all(info1[l]["cached"] is False for l in (0, 1))
    assert all(info1[l]["cov_cached"] is False for l in (0, 1))
    covs, projs = _cache_files(tmp_path)
    assert len(covs) == len(projs) == 2       # one of each tier, per layer

    P2, info2 = d.load_or_build_projectors(model, tok, R, cache_dir=str(tmp_path))
    assert all(info2[l]["cached"] is True for l in (0, 1))
    for l in (0, 1):
        assert torch.equal(P2[l].double(), P1[l].double())
        assert info2[l]["n_dirs"] == info1[l]["n_dirs"] >= 1
    assert _cache_files(tmp_path) == (covs, projs)      # nothing new written


def test_the_spectrum_tier_stores_U_and_S_and_NOT_the_covariance(tmp_path, monkeypatch):
    """``U``/``S`` re-cut any ``tau``, and every input that would make the SVD
    come out differently (``cov_dtype``, the resolved device pair, everything
    the covariance pass consumes) is already in ``cov_key`` — so in exactly the
    cases where a stored ``cov`` could be reused, a rebuild is the correct
    behaviour. Keeping it would cost another ``d²`` per layer per entry on a
    shared artifacts volume and buy nothing."""
    monkeypatch.setattr(d.C, "load_corpus_texts_all",
                        lambda corpus, **kw: ["abcde", "fghij", "klmno"])
    model, tok = _FakeLM(), _FakeTokenizer()
    R = d.resolve_kuda_config(_cache_cfg())
    d.load_or_build_projectors(model, tok, R, cache_dir=str(tmp_path))

    blob = torch.load(d.cov_cache_path(str(tmp_path), R, 0), map_location="cpu",
                      weights_only=True)
    assert set(blob) == {"U", "S", "info"}
    assert "cov" not in blob
    assert blob["U"].shape == (D_MODEL, D_MODEL) and blob["S"].shape == (D_MODEL,)
    assert blob["info"]["cov_dtype"] == "bfloat16"
    # And it really is enough: the tier-2 path rebuilds a projector from it.
    cfg = _cache_cfg()
    cfg["kuda"]["tau"] = 1e-9
    _, info = d.load_or_build_projectors(model, tok, d.resolve_kuda_config(cfg),
                                         cache_dir=str(tmp_path))
    assert info[0]["cov_cached"] is True and info[0]["n_dirs"] == 1


def test_a_tau_sweep_recuts_the_CACHED_spectrum_instead_of_rebuilding_it(tmp_path,
                                                                         monkeypatch):
    """The point of splitting the key. A new ``tau`` must produce a new
    projector entry and reuse the covariance entry — no second covariance pass,
    no second ``d x d`` SVD. Upstream splits its own cache the same way."""
    calls = []
    real = d.accumulate_down_proj_cov
    monkeypatch.setattr(d.C, "load_corpus_texts_all",
                        lambda corpus, **kw: ["abcde", "fghij", "klmno"])
    monkeypatch.setattr(d, "accumulate_down_proj_cov",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    model, tok = _FakeLM(), _FakeTokenizer()

    cfg = _cache_cfg()
    R = d.resolve_kuda_config(cfg)
    P1, _ = d.load_or_build_projectors(model, tok, R, cache_dir=str(tmp_path))
    assert len(calls) == 1
    covs, projs = _cache_files(tmp_path)

    cfg["kuda"]["tau"] = 1e-9                      # -> the argmin fallback
    R2 = d.resolve_kuda_config(cfg)
    P2, info2 = d.load_or_build_projectors(model, tok, R2, cache_dir=str(tmp_path))
    assert len(calls) == 1, "a tau change repaid the covariance pass"
    for l in (0, 1):
        assert info2[l]["cached"] is False          # a new projector ...
        assert info2[l]["cov_cached"] is True       # ... off the cached spectrum
        assert info2[l]["n_dirs"] == 1              # nothing clears tau=1e-9
        assert not torch.equal(P2[l].double(), P1[l].double())
        # The covariance metadata survives the round trip.
        assert info2[l]["n_samples"] == 3 and info2[l]["svd_path"].startswith("cpu")
    covs2, projs2 = _cache_files(tmp_path)
    assert covs2 == covs, "a tau change wrote a second covariance entry"
    assert len(projs2) == 4, "a tau change must write its own projector entry"

    # And the new projector is itself cached: a third call hits tier 1.
    _, info3 = d.load_or_build_projectors(model, tok, R2, cache_dir=str(tmp_path))
    assert all(info3[l]["cached"] is True for l in (0, 1))
    assert len(calls) == 1


def test_the_cache_records_both_keys_for_the_run_record(tmp_path, monkeypatch):
    monkeypatch.setattr(d.C, "load_corpus_texts_all",
                        lambda corpus, **kw: ["abcde", "fghij", "klmno"])
    model, tok = _FakeLM(), _FakeTokenizer()
    R = d.resolve_kuda_config(_cache_cfg())
    _, info = d.load_or_build_projectors(model, tok, R, cache_dir=str(tmp_path))
    for l in (0, 1):
        assert info[l]["cov_key"] == d._cov_key(R, l)
        assert info[l]["proj_key"] == d._proj_key(R, l)
        assert info[l]["cov_key"] in info[l]["cov_cache_path"]
        assert info[l]["proj_key"] in info[l]["cache_path"]
        assert info[l]["cov_dtype"] == "bfloat16"


def test_a_different_max_cov_texts_is_a_different_projector_cache_entry(tmp_path,
                                                                        monkeypatch):
    """The RSPD-only cap changes what the covariance saw, so it must change the
    key — otherwise a cheap capped probe would poison the full run."""
    monkeypatch.setattr(d.C, "load_corpus_texts_all",
                        lambda corpus, **kw: ["abcde", "fghij", "klmno"])
    model, tok = _FakeLM(), _FakeTokenizer()
    cfg = _cache_cfg()
    R = d.resolve_kuda_config(cfg)
    _, info1 = d.load_or_build_projectors(model, tok, R, cache_dir=str(tmp_path))
    assert info1[0]["n_samples"] == 3 and info1[0]["sample_p_percent"] == 100.0

    cfg["kuda"]["nullspace"]["max_cov_texts"] = 2
    R2 = d.resolve_kuda_config(cfg)
    _, info2 = d.load_or_build_projectors(model, tok, R2, cache_dir=str(tmp_path))
    # Unlike tau, the cap changes the COVARIANCE, so both tiers rebuild.
    assert all(info2[l]["cached"] is False for l in (0, 1))   # not a stale hit
    assert all(info2[l]["cov_cached"] is False for l in (0, 1))
    assert info2[0]["n_samples"] == 2 and info2[0]["max_cov_texts"] == 2
    covs, projs = _cache_files(tmp_path)
    assert len(covs) == len(projs) == 4


# --------------------------------------------------------------------------- #
# 3c. sample_p is a PERCENTAGE of the retain corpus (upstream's --sample_p)    #
#                                                                              #
# It used to be an absolute row count, which built a d x d covariance from ~100 #
# documents. tau is an ABSOLUTE threshold on that covariance's spectrum, so the #
# starved version did not merely add noise — it selected a different, far more  #
# permissive null subspace. These tests pin the corrected semantics.            #
# --------------------------------------------------------------------------- #
def _texts(n):
    return [f"text-{i:04d}" for i in range(n)]


def _select(monkeypatch, corpus_texts, **nullspace):
    """``select_cov_texts`` over a stubbed corpus loader. ``corpus_texts`` maps
    corpus name -> the rows that corpus yields (post-``min_len``)."""
    seen = {}

    def fake(corpus, *, max_rows=None):
        seen[corpus] = {"max_rows": max_rows}
        rows = corpus_texts[corpus]
        return rows if max_rows is None else rows[:max_rows]

    monkeypatch.setattr(d.C, "load_corpus_texts_all", fake)
    retain = list(dict.fromkeys(corpus_texts))
    cfg = _base_config()
    cfg["kuda"]["retain_corpora"] = retain
    cfg["kuda"]["forget_corpora"] = ["bio-forget-corpus",
                                     "cyber-forget-corpus"][:len(retain)]
    cfg["kuda"]["alpha"] = [1.0] * len(retain)
    cfg["kuda"]["nullspace"] = dict(nullspace)
    R = d.resolve_kuda_config(cfg)
    return d.select_cov_texts(R), seen


def test_sample_p_selects_a_percentage_of_the_retain_corpus(monkeypatch):
    rows = _texts(10)
    got, _ = _select(monkeypatch, {"wikitext": rows}, sample_p=50)
    assert got == rows[:5]                       # 50% of 10 -> 5, a prefix


def test_sample_p_100_is_the_whole_corpus(monkeypatch):
    rows = _texts(10)
    got, seen = _select(monkeypatch, {"wikitext": rows}, sample_p=100)
    assert got == rows
    # Uncapped: the loader is asked for everything, not for `sample_p` rows.
    assert seen["wikitext"]["max_rows"] is None


def test_sample_p_rounds_up_so_a_tiny_percentage_still_samples_something(monkeypatch):
    rows = _texts(10)
    # ceil(10 * 7 / 100) = 1, ceil(10 * 11 / 100) = 2, ceil(10 * 0.5 / 100) = 1
    assert _select(monkeypatch, {"wikitext": rows}, sample_p=7)[0] == rows[:1]
    assert _select(monkeypatch, {"wikitext": rows}, sample_p=11)[0] == rows[:2]
    assert _select(monkeypatch, {"wikitext": rows}, sample_p=0.5)[0] == rows[:1]
    # 33% of 100 is exactly 33, not 34 — no floating-point drift upward.
    assert len(_select(monkeypatch, {"wikitext": _texts(100)}, sample_p=33)[0]) == 33


def test_sample_p_is_sequential_and_deterministic(monkeypatch):
    rows = _texts(50)
    a, _ = _select(monkeypatch, {"wikitext": rows}, sample_p=20)
    b, _ = _select(monkeypatch, {"wikitext": rows}, sample_p=20)
    assert a == b == rows[:10]                   # a prefix; upstream never shuffles


def test_the_percentage_applies_per_corpus_so_no_retain_corpus_is_dropped(monkeypatch):
    a, b = _texts(10), [f"other-{i}" for i in range(20)]
    got, _ = _select(monkeypatch, {"wikitext": a, "bio-retain-corpus": b}, sample_p=50)
    assert got == a[:5] + b[:10]                 # both sampled, order preserved


def test_max_cov_texts_caps_after_the_percentage(monkeypatch):
    rows = _texts(100)
    got, _ = _select(monkeypatch, {"wikitext": rows}, sample_p=50, max_cov_texts=7)
    assert got == rows[:7]                       # 50 selected, then capped to 7
    # A cap above the selection is a no-op, NOT a way to get more than sample_p.
    got, _ = _select(monkeypatch, {"wikitext": rows}, sample_p=50, max_cov_texts=90)
    assert len(got) == 50
    # Default is uncapped.
    got, _ = _select(monkeypatch, {"wikitext": rows}, sample_p=50)
    assert len(got) == 50


def test_a_sub_100_sample_p_is_flagged_as_diverging_from_upstream():
    """The deviation is inert at 100 and silently activates below it, which is
    the shape of bug that surfaces three weeks later as an unexplainable
    number. The guard makes it impossible to run one without noticing."""
    assert d.sample_p_divergence_warning(100) is None
    assert d.sample_p_divergence_warning(100.0) is None

    for pct in (99.9, 50, 7, 0.5):
        note = d.sample_p_divergence_warning(pct)
        assert note is not None, f"{pct}% must warn"
        assert "WARNING" in note and "sample_p" in note
        # It must name BOTH divergences, not just the obvious one ...
        assert "random.sample" in note and "PREFIX" in note
        assert "int()" in note and "ceil()" in note
        # ... cite where the claim comes from, and say what it costs.
        assert "utils.py:265-269" in note
        assert "NOT a faithful replication" in note
        assert f"{float(pct):g}%" in note


def test_resolve_warns_on_a_sub_100_sample_p_and_records_the_verdict(capsys):
    """It fires from ``resolve_kuda_config``, which runs before either model
    loads, so the warning costs seconds rather than a checkpoint download — and
    it covers the ``--sample-p`` CLI override as well as the config key."""
    R = d.resolve_kuda_config(_base_config())
    assert R["nullspace"]["sample_p_is_upstream_faithful"] is True
    assert "WARNING" not in capsys.readouterr().out

    cfg = _base_config()
    cfg["kuda"]["nullspace"] = {"sample_p": 10}
    R = d.resolve_kuda_config(cfg)
    assert R["nullspace"]["sample_p_is_upstream_faithful"] is False
    assert "WARNING" in capsys.readouterr().out

    # The CLI override goes through the same gate.
    R = d.resolve_kuda_config(_base_config(), sample_p=7)
    assert R["nullspace"]["sample_p_is_upstream_faithful"] is False
    assert "WARNING" in capsys.readouterr().out


def test_the_sample_p_guard_warns_rather_than_refusing():
    """Deliberate: ``--sample-p`` is a documented smoke-run flag and a cheap
    covariance is how the pipeline gets sanity-checked before an H100 is booked,
    so a sub-100% config must still *run*. The record, not the refusal, is what
    stops it becoming a published number."""
    cfg = _base_config()
    cfg["kuda"]["nullspace"] = {"sample_p": 1}
    R = d.resolve_kuda_config(cfg)          # must NOT raise
    assert R["nullspace"]["sample_p"] == 1.0
    # Same for the CLI override, which exists precisely for smoke runs.
    assert d.resolve_kuda_config(_base_config(), sample_p=1)["nullspace"]["sample_p"] \
        == 1.0
    # What IS refused is a value outside upstream's units — a different failure.
    with pytest.raises(ValueError, match="sample_p"):
        d.resolve_kuda_config(_base_config(), sample_p=0)


def test_the_faithfulness_verdict_does_not_leak_into_the_cache_key():
    """It is a *record*, derived from ``sample_p``, which is already keyed —
    adding it to the payload would be a second name for the same fact."""
    R = d.resolve_kuda_config(_base_config())
    before = d._cov_key(R, 6)
    R["nullspace"]["sample_p_is_upstream_faithful"] = False
    assert d._cov_key(R, 6) == before


def test_max_cov_texts_at_100_percent_lets_the_loader_stop_early(monkeypatch):
    """At 100% the prefix is the same whether the cap is applied in the loader or
    after it, so the loader is told to stop — the point of the knob is cost."""
    rows = _texts(1000)
    got, seen = _select(monkeypatch, {"wikitext": rows}, sample_p=100, max_cov_texts=4)
    assert got == rows[:4] and seen["wikitext"]["max_rows"] == 4


def test_below_100_percent_the_loader_must_see_the_whole_corpus(monkeypatch):
    """The denominator is the full corpus size, so an early stop would silently
    turn "50% of 1000" into "50% of the cap"."""
    rows = _texts(1000)
    got, seen = _select(monkeypatch, {"wikitext": rows}, sample_p=50, max_cov_texts=4)
    assert seen["wikitext"]["max_rows"] is None and got == rows[:4]


# --------------------------------------------------------------------------- #
# 3d. Escaping the forget objective's stationary point (kuda.init_jitter)      #
#                                                                              #
# cos(u, f) is MAXIMISED at u == f, so its gradient there is exactly zero      #
# (f/|f|^2 - f/|f|^2), and the retain MSE is at its own minimum at the same    #
# point. The updated model starts bit-identical to the frozen one, so step 0   #
# sits on an exact stationary point of the whole objective and the only thing  #
# that moves the weights is bf16 rounding noise. Upstream has this property    #
# too; init_jitter is an RSPD addition and 0.0 must reproduce upstream EXACTLY.#
# --------------------------------------------------------------------------- #
def _jitter_params(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"model.layers.6.mlp.down_proj.weight":
            torch.nn.Parameter(torch.randn(4, 6, generator=g)),
            "model.layers.7.mlp.down_proj.weight":
            torch.nn.Parameter(torch.randn(4, 6, generator=g) * 3.0)}


def test_zero_init_jitter_is_an_exact_no_op():
    """Not "approximately unchanged" — bit-identical, and no RNG consumed, so
    published KUDA stays reachable from this driver."""
    params = _jitter_params()
    before = {k: p.detach().clone() for k, p in params.items()}
    torch.manual_seed(1234)
    rng_before = torch.randn(3)

    assert d.apply_init_jitter(params, 0.0, seed=42) == {}
    for k, p in params.items():
        assert torch.equal(p.detach(), before[k])

    # The global stream is untouched: the same draw comes back.
    torch.manual_seed(1234)
    assert torch.equal(torch.randn(3), rng_before)
    # And the default is 0.0, so a config that says nothing reproduces upstream.
    assert d.DEFAULT_INIT_JITTER == 0.0
    assert d.resolve_kuda_config(_base_config())["init_jitter"] == 0.0


def test_init_jitter_is_deterministic_under_a_fixed_seed():
    a, b = _jitter_params(), _jitter_params()
    ia = d.apply_init_jitter(a, 1e-3, seed=42)
    ib = d.apply_init_jitter(b, 1e-3, seed=42)
    for k in a:
        assert torch.equal(a[k].detach(), b[k].detach()), f"{k} is not reproducible"
    assert ia == ib

    # A different seed is a different perturbation ...
    c = _jitter_params()
    d.apply_init_jitter(c, 1e-3, seed=43)
    assert not torch.equal(c["model.layers.6.mlp.down_proj.weight"].detach(),
                           a["model.layers.6.mlp.down_proj.weight"].detach())
    # ... and the draw does NOT depend on the global stream's position, which is
    # what makes it reproducible across runs (model loading, wandb.init and the
    # corpus loaders sit between set_seeds() and here).
    e = _jitter_params()
    torch.manual_seed(999)
    torch.randn(17)                      # advance the global generator
    d.apply_init_jitter(e, 1e-3, seed=42)
    for k in a:
        assert torch.equal(e[k].detach(), a[k].detach())


def test_init_jitter_scales_with_each_tensors_own_std():
    """``p += randn_like(p) * init_jitter * p.std()`` — per tensor, so a big
    weight and a small one are perturbed by the same *relative* amount."""
    params = _jitter_params()
    before = {k: p.detach().clone() for k, p in params.items()}
    info = d.apply_init_jitter(params, 1e-2, seed=7)

    assert set(info) == set(params)
    for k, p in params.items():
        delta = p.detach() - before[k]
        assert not torch.equal(delta, torch.zeros_like(delta))
        # The delta's std is init_jitter x the weight's own std, to sampling error.
        assert float(delta.std()) == pytest.approx(1e-2 * float(before[k].std()),
                                                   rel=0.4)
        assert info[k]["weight_std"] == pytest.approx(float(before[k].std()), rel=1e-5)
        assert info[k]["delta_norm"] == pytest.approx(float(delta.norm()), rel=1e-5)
        assert info[k]["rel_delta"] == pytest.approx(
            float(delta.norm()) / float(before[k].norm()), rel=1e-5)
    # The second tensor is 3x bigger, so its absolute delta is ~3x bigger but its
    # RELATIVE one is the same.
    a, b = (info["model.layers.6.mlp.down_proj.weight"],
            info["model.layers.7.mlp.down_proj.weight"])
    assert b["delta_norm"] > 2.0 * a["delta_norm"]
    assert b["rel_delta"] == pytest.approx(a["rel_delta"], rel=0.4)


def test_init_jitter_is_applied_in_place_without_autograd():
    params = _jitter_params()
    for p in params.values():
        p.requires_grad_(True)
    d.apply_init_jitter(params, 1e-3, seed=42)
    for p in params.values():
        assert p.requires_grad and p.grad is None
        assert p.grad_fn is None, "the jitter must not enter the autograd graph"


@pytest.mark.parametrize("value, expected", [(0, 0.0), (0.0, 0.0),
                                             (1.0e-3, 1.0e-3), (1, 1.0)])
def test_init_jitter_resolves_from_the_config(value, expected):
    cfg = _base_config()
    cfg["kuda"]["init_jitter"] = value
    R = d.resolve_kuda_config(cfg)
    assert R["init_jitter"] == pytest.approx(expected)
    assert isinstance(R["init_jitter"], float)


@pytest.mark.parametrize("bad", [-1, -1e-9, True, False, "1e-3", None, [1e-3]])
def test_init_jitter_rejects_anything_that_is_not_a_nonnegative_number(bad):
    cfg = _base_config()
    cfg["kuda"]["init_jitter"] = bad
    with pytest.raises(ValueError, match="init_jitter"):
        d.resolve_kuda_config(cfg)


# --------------------------------------------------------------------------- #
# 3e. The step dynamics the driver logs                                        #
# --------------------------------------------------------------------------- #
def test_grad_norm_is_the_l2_over_every_target_gradient():
    params = _jitter_params()
    for p in params.values():
        p.grad = torch.randn_like(p)
    expected = torch.cat([p.grad.reshape(-1) for p in params.values()]).norm()
    assert d.grad_norm(params) == pytest.approx(float(expected), rel=1e-5)
    # A parameter with no gradient contributes nothing, rather than crashing.
    params["model.layers.6.mlp.down_proj.weight"].grad = None
    only = params["model.layers.7.mlp.down_proj.weight"].grad.norm()
    assert d.grad_norm(params) == pytest.approx(float(only), rel=1e-5)
    for p in params.values():
        p.grad = None
    assert d.grad_norm(params) == 0.0


def test_grad_norm_detects_the_stationary_point_a_zero_gradient_looks_like():
    params = _jitter_params()
    for p in params.values():
        p.grad = torch.zeros_like(p)
    assert d.grad_norm(params) < d.STATIONARY_GRAD_NORM


def test_the_projection_can_only_shrink_the_gradient_norm():
    n = 8
    cov, _, _ = _synthetic_cov(n=n, big=4)
    P, _ = d.build_projector(cov, tau=1e-2, dtype=torch.float64)
    w = torch.nn.Parameter(torch.randn(3, n, dtype=torch.float64))
    w.grad = torch.randn(3, n, dtype=torch.float64)
    raw = d.grad_norm({"w": w})
    d.project_gradients({"w": w}, {"w": P})
    assert d.grad_norm({"w": w}) < raw          # P is an orthogonal projection


def test_rel_weight_delta_is_the_worst_relative_movement():
    params = _jitter_params()
    reference = {k: p.detach().clone() for k, p in params.items()}
    assert d.rel_weight_delta(params, reference) == 0.0

    with torch.no_grad():
        params["model.layers.7.mlp.down_proj.weight"].mul_(1.5)
    big = params["model.layers.7.mlp.down_proj.weight"]
    expected = float((big.detach() - reference[
        "model.layers.7.mlp.down_proj.weight"]).norm()
        / reference["model.layers.7.mlp.down_proj.weight"].norm())
    # The max, not the mean: the untouched tensor must not dilute it.
    assert d.rel_weight_delta(params, reference) == pytest.approx(expected, rel=1e-5)
    assert expected == pytest.approx(0.5, rel=1e-5)


# --------------------------------------------------------------------------- #
# 3f. Forget max_length — 768 for BOTH topics, as upstream actually runs       #
# --------------------------------------------------------------------------- #
def test_kuda_asks_for_768_tokens_on_every_forget_topic():
    """Upstream ``unlearning/unlearn.py:109-111`` reads::

        # 512 for bio, 768 for cyber
        # max_length = 512 if topic_idx == 0 else 768
        max_length = 768

    The per-topic line is commented out, so 768 is what actually runs. This
    driver used to follow the comment."""
    assert d.FORGET_MAX_LENGTH == 768
    R = d.resolve_kuda_config(_base_config())
    assert R["forget_corpora"] == ["bio-forget-corpus", "cyber-forget-corpus"]
    assert R["forget_max_length"] == [768, 768]
    # Including bio, whose topic entry in the SHARED table says 512 — that table
    # describes RMU, whose upstream really does use the split, and KUDA must not
    # consult it.
    from srspd_experiments.methods import _common
    assert _common.FORGET_MAX_LENGTH["bio"] == 512
    assert R["forget_max_length"][0] != _common.FORGET_MAX_LENGTH["bio"]
    # One entry per forget corpus, whatever the topic ordering is.
    cfg = _base_config()
    cfg["kuda"]["forget_corpora"] = ["cyber-forget-corpus", "bio-forget-corpus"]
    assert d.resolve_kuda_config(cfg)["forget_max_length"] == [768, 768]


def test_rmu_keeps_its_per_topic_split_and_kuda_does_not():
    """The two drivers disagree on purpose: CAIS's RMU genuinely uses 512/768,
    AceNagi's KUDA does not. A future edit that "unifies" them breaks one."""
    rmu = importlib.import_module("srspd_experiments.methods.rmu")
    assert rmu.forget_max_lengths(["bio-forget-corpus", "cyber-forget-corpus"]) \
        == [512, 768]
    assert d.resolve_kuda_config(_base_config())["forget_max_length"] == [768, 768]


# --------------------------------------------------------------------------- #
# 3g. The tokenizer sides upstream forces                                      #
# --------------------------------------------------------------------------- #
def test_the_tokenizer_sides_are_resolved_and_recorded():
    """``deviation_loss`` and ``retain_loss`` both average over EVERY position
    including pads, so which end the pads sit on and which end a long document
    is cut from are part of the objective. Upstream forces left/right
    (``utils.py:102`` / ``unlearn.py:72``); the two models ship different
    defaults, so inheriting them was an uncontrolled confound."""
    assert (d.TOKENIZER_PADDING_SIDE, d.TOKENIZER_TRUNCATION_SIDE) == ("left", "right")
    R = d.resolve_kuda_config(_base_config())
    assert (R["padding_side"], R["truncation_side"]) == ("left", "right")
    # Not config keys — upstream forces them, so a config cannot drift off them.
    cfg = _base_config()
    cfg["kuda"]["padding_side"] = "right"
    assert d.resolve_kuda_config(cfg)["padding_side"] == "left"


# --------------------------------------------------------------------------- #
# 4. Config validation (runs before either model loads)                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value, expected", [
    (100, 100.0), (100.0, 100.0), (50, 50.0), (0.5, 0.5), (1, 1.0), (1e-6, 1e-6),
])
def test_sample_p_accepts_any_percentage_in_the_half_open_unit_interval(value,
                                                                        expected):
    cfg = _base_config()
    cfg["kuda"]["nullspace"] = {"sample_p": value}
    R = d.resolve_kuda_config(cfg)
    assert R["nullspace"]["sample_p"] == pytest.approx(expected)
    assert isinstance(R["nullspace"]["sample_p"], float)


@pytest.mark.parametrize("value", [0, 0.0, -1, -0.5, 101, 100.001, 1000,
                                   True, False, "100", None, [100]])
def test_sample_p_rejects_anything_outside_0_to_100(value):
    """Including the old row counts (200, 1000) — a config written against the
    superseded "absolute count" meaning must fail loudly, not be reinterpreted."""
    cfg = _base_config()
    cfg["kuda"]["nullspace"] = {"sample_p": value}
    with pytest.raises(ValueError) as e:
        d.resolve_kuda_config(cfg)
    assert "sample_p" in str(e.value) and "PERCENTAGE" in str(e.value)


def test_max_cov_texts_defaults_to_none_and_wants_a_positive_int():
    R = d.resolve_kuda_config(_base_config())
    assert R["nullspace"]["max_cov_texts"] is None is d.DEFAULT_MAX_COV_TEXTS
    cfg = _base_config()
    cfg["kuda"]["nullspace"] = {"max_cov_texts": 2048}
    assert d.resolve_kuda_config(cfg)["nullspace"]["max_cov_texts"] == 2048
    for bad in (0, -1, 1.5, True, "2048"):
        cfg["kuda"]["nullspace"] = {"max_cov_texts": bad}
        with pytest.raises(ValueError) as e:
            d.resolve_kuda_config(cfg)
        assert "max_cov_texts" in str(e.value)



def test_resolve_accepts_the_shipped_shape_and_resolves_defaults():
    R = d.resolve_kuda_config(_base_config())
    assert R["beta"] == 2.0 and R["tau"] == 2.0e-3
    assert R["layer_id"] == 9 and R["layer_ids"] == [6, 7, 8, 9]
    assert R["alpha"] == [1.0, 1.0]
    assert R["seed"] == 42 and R["epoch"] == 1
    assert R["module_name"] == "mlp.down_proj"
    assert R["retain_max_length"] == 512
    # sample_p is a PERCENTAGE: the default 100 means the whole retain corpus,
    # which is what upstream's Zephyr scripts pass.
    assert R["nullspace"]["sample_p"] == d.DEFAULT_SAMPLE_P == 100.0
    assert R["nullspace"]["max_cov_texts"] is None
    assert R["nullspace"]["svd_device"] == "auto" and R["nullspace"]["use_cache"] is True
    assert R["push_to_hub"] is False          # nothing is ever pushed (no-op either way)


def test_cli_overrides_win_over_the_config():
    R = d.resolve_kuda_config(_base_config(), max_num_batches=3, sample_p=7,
                              push_to_hub=False)
    assert R["max_num_batches"] == 3
    assert R["nullspace"]["sample_p"] == 7.0      # 7 PERCENT, a smoke-run sample
    assert R["push_to_hub"] is False
    # The override is validated on the same (0, 100] scale as the config key.
    with pytest.raises(ValueError, match="sample_p"):
        d.resolve_kuda_config(_base_config(), sample_p=500)


@pytest.mark.parametrize("mutate, needle", [
    (lambda c: c.pop("name"), "name"),
    (lambda c: c.pop("model"), "model"),
    (lambda c: c.__setitem__("dtype", "int8"), "dtype"),
    (lambda c: c.pop("kuda"), "kuda"),
    (lambda c: c["kuda"].__setitem__("beta", 0.0), "kuda.beta"),
    (lambda c: c["kuda"].__setitem__("beta", -1.0), "kuda.beta"),
    (lambda c: c["kuda"].__setitem__("tau", "small"), "kuda.tau"),
    (lambda c: c["kuda"].__setitem__("lr", 0), "kuda.lr"),
    (lambda c: c["kuda"].__setitem__("batch_size", 0), "kuda.batch_size"),
    (lambda c: c["kuda"].__setitem__("max_num_batches", -5), "kuda.max_num_batches"),
    (lambda c: c["kuda"].__setitem__("epoch", 0), "kuda.epoch"),
    (lambda c: c["kuda"].__setitem__("alpha", [1.0]), "kuda.alpha"),
    (lambda c: c["kuda"].__setitem__("alpha", [1.0, -1.0]), "kuda.alpha[1]"),
    (lambda c: c["kuda"].__setitem__("layer_ids", []), "kuda.layer_ids"),
    (lambda c: c["kuda"].__setitem__("layer_ids", [6, 6, 9]), "duplicates"),
    (lambda c: c["kuda"].__setitem__("layer_id", 20), "kuda.layer_id"),
    (lambda c: c["kuda"].__setitem__("layer_id", -1), "kuda.layer_id"),
    (lambda c: c["kuda"].__setitem__("module_name", "mlp.up_proj"), "module_name"),
    (lambda c: c["kuda"].__setitem__("forget_corpora", ["nope"]), "forget_corpora"),
    (lambda c: c["kuda"].__setitem__("retain_corpora", ["wikitext"]), "same length"),
    (lambda c: c["kuda"].__setitem__("nullspace", {"svd_device": "tpu"}), "svd_device"),
    (lambda c: c["kuda"].__setitem__("nullspace", {"sample_p": 0}), "sample_p"),
    (lambda c: c["kuda"].__setitem__("nullspace", {"sample_p": 101}), "sample_p"),
    (lambda c: c["kuda"].__setitem__("nullspace", {"max_cov_texts": 0}),
     "max_cov_texts"),
])
def test_bad_config_raises_valueerror_naming_the_key(mutate, needle):
    cfg = _base_config()
    mutate(cfg)
    with pytest.raises(ValueError) as e:
        d.resolve_kuda_config(cfg)
    assert needle in str(e.value)


def test_default_repo_name():
    assert d.default_repo_name("google/gemma-3-4b-it") == "rspd-wmdp-kuda-gemma-3-4b-it"
    assert d.default_repo_name("HuggingFaceH4/zephyr-7b-beta").startswith("rspd-wmdp-kuda-")


# --------------------------------------------------------------------------- #
# 5. The shipped configs                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stem", ["zephyr_kuda", "zephyr_kuda_faithful"])
def test_shipped_configs_validate_and_carry_the_expected_blocks(stem):
    cfg = legacy(stem)
    R = d.resolve_kuda_config(cfg)
    assert R["name"] == cfg["name"]
    assert R["dtype"] == "bfloat16"
    assert R["push_to_hub"] is False and "wandb" not in cfg
    # The upstream Zephyr hyperparameters, carried to both models.
    assert (R["beta"], R["tau"], R["lr"]) == (2.0, 2.0e-3, 1.0e-5)
    assert (R["batch_size"], R["max_num_batches"], R["epoch"]) == (4, 150, 1)
    assert R["layer_id"] == 9 and R["layer_ids"] == [6, 7, 8, 9]
    # 100 PERCENT of wikitext — upstream's --sample_p 100, uncapped.
    assert R["seed"] == 42 and R["nullspace"]["sample_p"] == 100.0
    # ... which is the only value at which the retain selection matches
    # upstream's (see sample_p_divergence_warning), so both shipped configs must
    # come back clean.
    assert R["nullspace"]["sample_p_is_upstream_faithful"] is True
    assert d.sample_p_divergence_warning(R["nullspace"]["sample_p"]) is None
    assert R["nullspace"]["max_cov_texts"] is None
    assert R["forget_corpora"] == ["bio-forget-corpus", "cyber-forget-corpus"]
    assert R["retain_corpora"] == ["wikitext", "wikitext"]
    # bfloat16 is upstream's mom2_dtype, and the published tau=2e-3 was tuned
    # against that spectrum — both shipped configs say so explicitly rather
    # than relying on the default.
    assert cfg["kuda"]["nullspace"]["cov_dtype"] == "bfloat16"
    assert R["nullspace"]["cov_dtype"] == "bfloat16"
    # The first port escaped the stationary point with a 1e-3 jitter; the
    # faithful port pins 0.0 (upstream exact) after the jitter was found to
    # destroy MMLU retention. Each config says so explicitly.
    expected_jitter = {"zephyr_kuda": 1.0e-3, "zephyr_kuda_faithful": 0.0}[stem]
    assert cfg["kuda"]["init_jitter"] == expected_jitter
    assert R["init_jitter"] == expected_jitter
    # 768 for both topics (upstream hard-codes it) and upstream's two sides.
    assert R["forget_max_length"] == [768, 768]
    assert (R["padding_side"], R["truncation_side"]) == ("left", "right")


# --------------------------------------------------------------------------- #
# Learning-rate schedule and weight decay (upstream MUSE: HF Trainer defaults)   #
# --------------------------------------------------------------------------- #
def test_constant_schedule_is_the_default_and_holds_lr():
    R = d.resolve_kuda_config(_base_config())
    assert R["lr_schedule"] == "constant" == d.DEFAULT_LR_SCHEDULE
    assert R["schedule_epochs"] == R["epoch"] == 1
    # torch.optim.AdamW's own default -- what the driver always passed implicitly.
    assert R["weight_decay"] == 0.01 == d.DEFAULT_WEIGHT_DECAY
    assert R["weight_decay"] == torch.optim.AdamW([torch.zeros(1, requires_grad=True)]) \
        .defaults["weight_decay"]
    for step in (0, 1, 7, 10_000):
        assert d.lr_at_step(step, base_lr=5e-5, total_steps=12, schedule="constant") == 5e-5


def test_linear_decay_matches_hf_get_linear_schedule_with_warmup_at_zero_warmup():
    """``lr_t = lr * (T - t) / T``: ``lr`` at step 0, 0 at step T, straight in between."""
    T, lr = 30, 5e-5
    got = [d.lr_at_step(t, base_lr=lr, total_steps=T, schedule="linear_decay")
           for t in range(T + 2)]
    # HF: LambdaLR(lambda s: max(0, (T - s) / max(1, T)))  (num_warmup_steps=0)
    want = [lr * max(0.0, (T - t) / max(1, T)) for t in range(T + 2)]
    assert got == pytest.approx(want)
    assert got[0] == lr and got[T] == 0.0 and got[T + 1] == 0.0
    assert got[T // 3] == pytest.approx(lr * 2 / 3)
    assert got == sorted(got, reverse=True)


def test_linear_decay_over_a_longer_horizon_stops_part_way_down():
    """epoch 2 of a 3-epoch schedule: the last step the run takes sits at lr/3
    (upstream MUSE-News reports the 2nd checkpoint of a 3-epoch linear decay)."""
    steps_per_epoch, lr = 203, 5e-5
    T = 3 * steps_per_epoch
    last_run_step = 2 * steps_per_epoch - 1
    assert d.lr_at_step(last_run_step, base_lr=lr, total_steps=T, schedule="linear_decay") \
        == pytest.approx(lr * (T - last_run_step) / T)
    assert d.lr_at_step(2 * steps_per_epoch, base_lr=lr, total_steps=T,
                        schedule="linear_decay") == pytest.approx(lr / 3)


def test_lr_at_step_rejects_unknown_schedules_and_empty_horizons():
    with pytest.raises(ValueError, match="unknown lr schedule"):
        d.lr_at_step(0, base_lr=1e-5, total_steps=10, schedule="cosine")
    with pytest.raises(ValueError, match="total_steps"):
        d.lr_at_step(0, base_lr=1e-5, total_steps=0, schedule="linear_decay")


def test_resolver_validates_lr_schedule_schedule_epochs_and_weight_decay():
    cfg = _base_config()
    cfg["kuda"].update({"epoch": 2, "lr_schedule": "linear_decay", "schedule_epochs": 3,
                        "weight_decay": 0.01})
    R = d.resolve_kuda_config(cfg)
    assert (R["lr_schedule"], R["epoch"], R["schedule_epochs"]) == ("linear_decay", 2, 3)
    assert R["weight_decay"] == 0.01

    bad = _base_config(); bad["kuda"]["lr_schedule"] = "linear"
    with pytest.raises(ValueError, match="kuda.lr_schedule"):
        d.resolve_kuda_config(bad)
    bad = _base_config(); bad["kuda"].update({"epoch": 3, "schedule_epochs": 2})
    with pytest.raises(ValueError, match="schedule_epochs.*>= kuda.epoch"):
        d.resolve_kuda_config(bad)
    bad = _base_config(); bad["kuda"]["schedule_epochs"] = 0
    with pytest.raises(ValueError, match="kuda.schedule_epochs"):
        d.resolve_kuda_config(bad)
    bad = _base_config(); bad["kuda"]["weight_decay"] = -0.1
    with pytest.raises(ValueError, match="kuda.weight_decay"):
        d.resolve_kuda_config(bad)
    # schedule_epochs is allowed to exceed the run under `constant` too (inert).
    ok = _base_config(); ok["kuda"].update({"epoch": 1, "schedule_epochs": 5})
    assert d.resolve_kuda_config(ok)["lr_schedule"] == "constant"


def test_the_schedule_keys_do_not_touch_the_projector_cache_keys():
    base = d.resolve_kuda_config(_base_config())
    cfg = _base_config()
    cfg["kuda"].update({"lr_schedule": "linear_decay", "schedule_epochs": 4,
                        "weight_decay": 0.0})
    other = d.resolve_kuda_config(cfg)
    assert d._cov_key(base, 6) == d._cov_key(other, 6)
    assert d._proj_key(base, 6) == d._proj_key(other, 6)



@pytest.mark.parametrize("side,epochs,horizon", [("news", 2, 3), ("books", 5, 5)])
def test_shipped_muse_sched_configs_differ_only_in_schedule_name_repo_and_tag(side, epochs,
                                                                             horizon):
    plain = config(f"muse_{side}", "kuda")
    sched = config(f"muse_{side}", "kuda_sched")
    Rp = d.resolve_kuda_config(plain)
    Rs = d.resolve_kuda_config(sched)
    assert Rp["lr_schedule"] == "constant" and Rp["schedule_epochs"] == Rp["epoch"] == epochs
    assert Rs["lr_schedule"] == "linear_decay"
    assert (Rs["epoch"], Rs["schedule_epochs"]) == (epochs, horizon)
    assert Rs["weight_decay"] == 0.01 == Rp["weight_decay"]
    assert Rs["name"] == f"muse_{side}_kuda_sched"
    for c in (plain, sched):
        c.pop("name")
        for key in ("lr_schedule", "schedule_epochs", "weight_decay"):
            c["kuda"].pop(key, None)
    assert plain == sched
