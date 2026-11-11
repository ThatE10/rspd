"""``srspd.erase`` with a LIST of rank rules: one set of Gram passes, one solve
per dose, every rule truncated from the one shared SVD of ``M X^T`` -- and every
``(dose, rule)`` setting bitwise equal to a separate single-rule call."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

import srspd  # noqa: E402
import tiny_models as tm  # noqa: E402
from srspd import api  # noqa: E402

DOSES = [0.5, 4.0]
RULES = [0.01, 0.3, "erank"]
BUDGET = 8 * 64 * 64 / 2 ** 30                 # one d_in-64 Gram: several passes


def _params(m):
    return {n: p.detach().clone() for n, p in m.named_parameters()}


def _same(a, b):
    return set(a) == set(b) and all(torch.equal(a[n], b[n]) for n in a)


def _counting(monkeypatch):
    calls = []
    real = api.run_gram_passes

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(api, "run_gram_passes", spy)
    return calls


@pytest.mark.parametrize("make", [tm.mistral, tm.gpt2], ids=["mistral", "gpt2"])
def test_every_setting_bitwise_equals_a_single_rule_call(make, monkeypatch):
    tgt, ret = tm.token_sets()
    calls = _counting(monkeypatch)
    events = []
    m = make()
    pristine = _params(m)
    multi = srspd.erase(m, target=tgt, retained=ret, mu_per_token=DOSES, rank_rule=RULES,
                        apply=False, gram_budget_gib=BUDGET,
                        progress=lambda e, i: events.append(e))
    assert len(calls) == 1 and events.count("plan") == 1
    assert events.count("pass_start") == len(multi.pass_plan) > 1
    assert multi.rank_rules == RULES
    assert multi.settings == [{"mu_per_token": d, "rank_rule": r}
                              for d in DOSES for r in [None] + RULES]
    for d in DOSES:
        assert set(multi.by_dose[repr(d)]["truncated"]) == {str(r) for r in RULES}

    for d in DOSES:
        for rule in [None] + RULES:
            multi.apply_dose(d, rule)
            assert multi.rank_rule == rule and multi.dose["mu_per_token"] == d
            got = _params(m)
            got_rows = {n: dict(r) for n, r in multi.matrices.items()}
            got_trunc = multi.by_dose[repr(d)]["truncated"]
            srspd.restore(m)
            assert _same(_params(m), pristine)

            ref_model = make()
            single = srspd.erase(ref_model, target=tgt, retained=ret, mu_per_token=d,
                                 rank_rule=rule, gram_budget_gib=BUDGET)
            assert single.rank_rule == rule
            assert _same(got, _params(ref_model)), (d, rule)
            want = single.matrices
            if rule is None:                     # erank(M X^T) is recorded whenever a rule ran
                assert all(r["erank"] is None for r in want.values())
                assert all(r["erank"] > 0 for r in got_rows.values())
                got_rows = {n: {**r, "erank": None} for n, r in got_rows.items()}
            assert got_rows == want, (d, rule)
            if rule is not None:
                key = str(rule)
                assert got_trunc[key] == single.by_dose[repr(d)]["truncated"][key]
            single.close()
    multi.close()
    assert _same(_params(m), pristine)


def test_apply_needs_an_explicit_rule_with_several(monkeypatch):
    tgt, ret = tm.token_sets()
    m = tm.mistral()
    before = _params(m)
    with pytest.raises(ValueError, match="apply_rank_rule"):
        srspd.erase(m, target=tgt, retained=ret, mu_per_token=1.0, rank_rule=[0.1, "erank"])
    assert _same(_params(m), before)
    with pytest.raises(ValueError, match="was not solved"):
        srspd.erase(m, target=tgt, retained=ret, mu_per_token=1.0, rank_rule=[0.1, "erank"],
                    apply_rank_rule=0.2)
    assert _same(_params(m), before)

    rep = srspd.erase(m, target=tgt, retained=ret, mu_per_token=[1.0, 2.0],
                      rank_rule=[0.1, "erank"], apply_mu_per_token=2.0,
                      apply_rank_rule="erank")
    assert rep.applied and rep.rank_rule == "erank" and rep.dose["mu_per_token"] == 2.0
    w_erank = _params(m)
    rep.apply_dose(2.0)                                    # rule None: the full-rank edit
    assert rep.rank_rule is None and not _same(_params(m), w_erank)
    w_full = _params(m)
    srspd.restore(m)
    full = srspd.erase(m, target=tgt, retained=ret, mu_per_token=[1.0, 2.0],
                       rank_rule=[0.1, "erank"], apply_mu_per_token=2.0, apply_rank_rule=None)
    assert full.rank_rule is None and _same(_params(m), w_full)
    with pytest.raises(ValueError, match="was not solved"):
        full.apply_dose(2.0, 0.5)
    with pytest.raises(ValueError, match="lam_rel"):
        full.apply_dose(2.0, 1.5)
    full.close()
    rep.close()
    assert _same(_params(m), before)

    # duplicates collapse, None entries are the full-rank edit, one rule applies by default
    one = srspd.erase(m, target=tgt, retained=ret, mu_per_token=1.0,
                      rank_rule=[None, 0.1, 0.1])
    assert one.rank_rules == [0.1] and one.rank_rule == 0.1
    one.close()
    with pytest.raises(ValueError, match="readout"):
        srspd.erase(m, target=tgt, retained=ret, mu_per_token=1.0, rank_rule=[0.1, "erank"],
                    solver="grams")
