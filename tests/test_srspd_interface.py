"""The public interface: ``SRSPD`` (attach, fit), the ``Component`` /
``Components`` output type, the handful of top-level names, and the rule that
``rspd.core`` is plain-tensor mathematics with no model code."""
from __future__ import annotations

import pathlib
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import rspd  # noqa: E402
import tiny_models as tm  # noqa: E402
from rspd import SRSPD, Component, Components  # noqa: E402

LAYERS = {"types": ["mlp.c_fc", "mlp.c_proj"], "blocks": [1, 2]}
CORE = pathlib.Path(rspd.__file__).parent / "core"


def _params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _bitwise(m, before):
    return all(torch.equal(before[n], p) for n, p in m.named_parameters())


def test_the_top_level_exports_a_handful_of_names():
    assert rspd.__all__ == ["SRSPD", "Component", "Components", "inspect_model",
                            "export_checkpoint", "__version__"]
    for old in ("erase", "restore", "erased_component", "apply_erasure", "leaks", "Dose"):
        assert not hasattr(rspd, old), old


def test_quickstart_attach_fit_apply_restore():
    model = tm.gpt2()
    target, retained = tm.token_sets()
    pristine = _params(model)
    srspd = SRSPD(model, layers=LAYERS)
    components = srspd.fit(target=target, retained=retained, mu_per_token=1.0)
    assert isinstance(components, Components)
    assert _bitwise(model, pristine), "fit must not change the model"
    assert not components.applied and len(components) == 4 == len(components.names)
    c = components[components.names[0]]
    assert isinstance(c, Component) and c.name == components.names[0]
    assert c.A.shape[1] == c.B.shape[0] == c.rank
    assert set(c.leaks) == {"L_X_rel", "L_Y_rel"}
    assert set(components.leaks) == set(components.names)
    assert [x.name for x in components] == components.names and components[0].name == c.name

    assert components.apply() is components and components.applied
    assert not _bitwise(model, pristine)
    restored = components.restore()
    assert len(restored) == 4 and _bitwise(model, pristine) and not components.applied
    components.close()


def test_fit_matches_the_function_it_wraps_bit_for_bit():
    target, retained = tm.token_sets()
    a, b = tm.gpt2(), tm.gpt2()
    rep = rspd.api.erase(a, target=target, retained=retained, layers=LAYERS, mu_per_token=0.5)
    comps = SRSPD(b, layers=LAYERS).fit(target, retained, 0.5).apply()
    assert _bitwise(b, _params(a))
    assert comps.report.matrices.keys() == rep.matrices.keys()
    for n, row in rep.matrices.items():
        assert comps.leaks[n]["L_X_rel"] == row["L_X_rel"]
    rep.close()
    comps.close()


def test_a_component_is_what_apply_subtracts():
    model = tm.gpt2()
    target, retained = tm.token_sets()
    comps = SRSPD(model, layers=LAYERS).fit(target, retained, 1.0)
    h = comps.report._session.h
    W = {n: h.get_weight(n).detach().clone() for n in comps.names}
    by_hand = comps.apply_to(W)
    comps.apply()
    for n in comps.names:
        assert torch.allclose(h.get_weight(n), by_hand[n], rtol=0, atol=1e-6), n
        assert not torch.equal(h.get_weight(n), W[n])
    comps.close()


def test_settings_doses_and_rank_rules():
    model = tm.gpt2()
    target, retained = tm.token_sets()
    pristine = _params(model)
    comps = SRSPD(model, layers=LAYERS).fit(target, retained, [0.3, 3.0], rank_rule="erank")
    assert comps.settings == [(0.3, None), (0.3, "erank"), (3.0, None), (3.0, "erank")]
    assert comps.setting == (0.3, "erank")                 # the one solved rule is the default
    name = comps.names[0]
    weak, strong = comps.at((0.3, None))[name], comps.at((3.0, None))[name]
    assert strong.dense().norm() > weak.dense().norm()     # a larger dose removes more
    assert comps.at((3.0, "erank"))[name].rank <= strong.rank
    comps.apply(3.0)                                       # a bare dose, at the default rule
    assert comps.setting == (3.0, "erank") and comps.report.rank_rule == "erank"
    comps.apply((0.3, None))
    assert comps.report.dose["mu_per_token"] == 0.3 and comps.report.rank_rule is None
    with pytest.raises(ValueError, match="not solved"):
        comps.apply(7.0)
    comps.close()
    assert _bitwise(model, pristine)


def test_text_sets_and_inspect():
    model, tok = tm.gpt2(), tm.word_tokenizer()
    srspd = SRSPD(model, tok, layers=LAYERS)
    info = srspd.inspect()
    assert len(info["matrices"]) == 4
    comps = srspd.fit(target=["w1 w2 w3 w4 w5 w6", "w7 w8 w9 w3 w2"],
                      retained=["w10 w11 w12 w13 w14 w15 w16", "w17 w18 w19 w20 w21 w22"] * 3,
                      mu_per_token=1.0)
    assert comps.names == list(info["matrices"])
    with comps:
        comps.apply()
    assert not comps.applied


def test_components_from_tensors_need_no_model():
    torch.manual_seed(0)
    W = torch.randn(5, 7, dtype=torch.float64)
    X, Y = torch.randn(6, 7, dtype=torch.float64), torch.randn(30, 7, dtype=torch.float64)
    m = rspd.core.erased_component(W, X, Y, mu=1.0)
    assert isinstance(m, Component) and rspd.core.ErasedComponent is Component
    lk = rspd.core.leaks(W, m, X, Y)
    comps = Components.from_components([Component(m.A, m.B, dict(m.info, name="layer", leaks=lk))])
    assert comps.names == ["layer"] and comps["layer"].leaks == lk and not comps.applied
    new = comps.apply_to({"layer": W})["layer"]
    assert torch.equal(new, rspd.core.apply_erasure(W, m))
    with pytest.raises(RuntimeError, match="not attached to a model"):
        comps.apply()
    with pytest.raises(KeyError):
        comps["other"]
    with pytest.raises(ValueError, match="name"):
        Components.from_components([m])


def test_several_components_per_matrix_share_the_container():
    """The shape a variant that splits a matrix into several pieces returns."""
    torch.manual_seed(1)
    W = torch.randn(4, 6)
    parts = [Component(torch.randn(4, 1), torch.randn(1, 6), {"name": "layer"}) for _ in range(3)]
    comps = Components.from_components(parts)
    assert len(comps) == 3 and comps.names == ["layer"] and len(comps["layer"]) == 3
    want = W - sum(p.dense() for p in parts)
    assert torch.allclose(comps.apply_to({"layer": W})["layer"], want, atol=1e-6)


def test_core_is_plain_tensor_mathematics():
    """``rspd.core`` must stay portable: no model classes, no model libraries,
    nothing imported from the model-handling subpackages."""
    banned = re.compile(r"\btransformers\b|\bsafetensors\b|torch\.nn\b|\bnn\.Module\b|"
                        r"from \.\.(models|capture|api|pipeline)|from rspd\.(models|capture)")
    hits = []
    for p in sorted(CORE.glob("*.py")):
        code = "\n".join(ln for ln in p.read_text().splitlines()
                         if not ln.lstrip().startswith("#"))
        code = re.sub(r'"""(?s:.*?)"""', "", code)          # prose may name them
        hits += [f"{p.name}: {m.group(0)}" for m in banned.finditer(code)]
    assert not hits, hits
    assert {p.name for p in CORE.glob("*.py")} >= {"types.py", "solve.py", "coala.py",
                                                    "dose.py", "erank.py"}
