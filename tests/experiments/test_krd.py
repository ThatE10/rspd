"""Unit tests for ``rspd.experiments.eval.krd`` -- single-domain KRD and the
two-domain WMDP harmonic mean from lm-eval results, on synthetic accuracies.
No GPU, no network, no model.
"""
import math

import pytest

from rspd.experiments.eval import krd as k


def _payload(**accs):
    """An lm-eval-shaped payload: {"results": {task: {"acc,none": a, ...}}}."""
    return {"results": {t: {"acc,none": a, "acc_stderr,none": 0.01, "alias": t}
                        for t, a in accs.items()}}


class TestKrdFormula:
    def test_full_removal_is_one(self):
        # driven all the way to chance
        assert k.krd(0.80, 0.25) == pytest.approx(1.0)

    def test_no_movement_is_zero(self):
        assert k.krd(0.80, 0.80) == pytest.approx(0.0)

    def test_partial(self):
        # (0.80 - 0.60) / (0.80 - 0.25) = 0.2 / 0.55
        assert k.krd(0.80, 0.60) == pytest.approx(0.2 / 0.55)

    def test_clipped_above_one_when_below_chance(self):
        assert k.krd(0.80, 0.10) == 1.0            # fell below chance -> clip to 1

    def test_clipped_to_zero_when_accuracy_rose(self):
        assert k.krd(0.80, 0.90) == 0.0            # improved -> clip to 0

    def test_no_headroom_is_zero(self):
        assert k.krd(0.25, 0.10) == 0.0            # baseline already at/below chance
        assert k.krd(0.20, 0.10) == 0.0

    def test_custom_chance(self):
        assert k.krd(0.75, 0.50, chance=0.5) == pytest.approx(1.0)


class TestAccuracyExtraction:
    def test_prefers_acc_none(self):
        res = {"t": {"acc,none": 0.7, "acc_norm,none": 0.4}}
        assert k.task_accuracy(res, "t") == 0.7

    def test_falls_back_through_keys(self):
        assert k.task_accuracy({"t": {"acc_norm,none": 0.42}}, "t") == 0.42
        assert k.task_accuracy({"t": {"acc": 0.31}}, "t") == 0.31

    def test_missing_task_or_metric_is_none(self):
        assert k.task_accuracy({}, "t") is None
        assert k.task_accuracy({"t": {"perplexity,none": 12.0}}, "t") is None
        assert k.task_accuracy({"t": {"acc,none": True}}, "t") is None   # bool is not an accuracy


class TestKrdFromResults:
    def test_per_task_and_mean_over_dutch_tasks(self):
        base = _payload(global_mmlu_full_nl=0.60, global_mmlu_nl_local=0.50,
                        global_mmlu_full_en=0.70)
        edit = _payload(global_mmlu_full_nl=0.30, global_mmlu_nl_local=0.50,
                        global_mmlu_full_en=0.69)
        out = k.krd_from_results(base, edit)
        assert set(out["per_task"]) == set(k.DUTCH_TASKS)
        assert out["per_task"]["global_mmlu_full_nl"]["krd"] == pytest.approx(
            (0.60 - 0.30) / (0.60 - 0.25))
        assert out["per_task"]["global_mmlu_nl_local"]["krd"] == 0.0    # unchanged
        assert out["mean_krd"] == pytest.approx(
            ((0.60 - 0.30) / (0.60 - 0.25) + 0.0) / 2)
        assert out["skipped"] == []

    def test_missing_task_is_skipped_not_fatal(self):
        base = _payload(global_mmlu_full_nl=0.60)
        edit = _payload(global_mmlu_full_nl=0.40)
        out = k.krd_from_results(base, edit)
        assert list(out["per_task"]) == ["global_mmlu_full_nl"]
        assert out["skipped"] == ["global_mmlu_nl_local"]
        assert out["mean_krd"] == pytest.approx((0.60 - 0.40) / (0.60 - 0.25))

    def test_all_missing_gives_none_mean(self):
        out = k.krd_from_results(_payload(other=0.5), _payload(other=0.4))
        assert out["per_task"] == {} and out["mean_krd"] is None

    def test_accepts_bare_results_map(self):
        base = {"global_mmlu_full_nl": {"acc,none": 0.6}}
        edit = {"global_mmlu_full_nl": {"acc,none": 0.25}}
        out = k.krd_from_results(base, edit)
        assert out["per_task"]["global_mmlu_full_nl"]["krd"] == pytest.approx(1.0)


class TestTwoDomain:
    def test_harmonic_mean_of_bio_and_cyber(self):
        a, b = k.krd(0.64, 0.30), k.krd(0.44, 0.30)
        assert k.krd_two_domain(0.64, 0.30, 0.44, 0.30) == pytest.approx(2 / (1 / a + 1 / b))

    def test_zero_when_either_domain_is_untouched(self):
        assert k.krd_two_domain(0.64, 0.25, 0.44, 0.50) == 0.0

    def test_nan_propagates(self):
        assert math.isnan(k.krd_two_domain(0.64, float("nan"), 0.44, 0.3))

    def test_from_results(self):
        base = _payload(wmdp_bio=0.64, wmdp_cyber=0.44)
        edit = _payload(wmdp_bio=0.25, wmdp_cyber=0.25)
        out = k.wmdp_krd_from_results(base, edit)
        assert out["krd"] == pytest.approx(1.0) and out["krd_bio"] == pytest.approx(1.0)
        assert out["acc"]["wmdp_cyber_base"] == 0.44
        out = k.wmdp_krd_from_results(base, _payload(wmdp_bio=0.3))
        assert math.isnan(out["krd"]) and math.isnan(out["acc"]["wmdp_cyber_edit"])
