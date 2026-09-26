"""Unit tests for calibration metrics and temperature scaling.

The properties pinned are the ones a calibration report silently depends on: that the proper scores
are computed correctly, that temperature scaling recovers a known temperature and never changes the
argmax, that ECE is zero for a perfectly calibrated forecaster, and that the cross-fitted numbers
really are out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import softmax

from vifeedback.evaluation import calibration as C


def _calibrated_sample(n: int = 20000, t_true: float = 1.0, seed: int = 0):
    """Labels drawn FROM the model's own probabilities: calibrated by construction at T=t_true."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=2.5, size=(n, 3))
    p = softmax(logits / t_true, axis=1)
    y = np.array([rng.choice(3, p=row) for row in p])
    return logits, y


class TestProperScores:
    def test_nll_of_a_perfect_prediction_is_zero(self) -> None:
        p = np.eye(3)[[0, 1, 2]]
        assert C.nll(p, np.array([0, 1, 2])) == pytest.approx(0.0, abs=1e-9)

    def test_nll_of_uniform_is_log_k(self) -> None:
        p = np.full((4, 3), 1 / 3)
        assert C.nll(p, np.array([0, 1, 2, 0])) == pytest.approx(np.log(3))

    def test_brier_bounds(self) -> None:
        y = np.array([0, 1])
        assert C.brier(np.eye(3)[y], y) == pytest.approx(0.0)
        assert C.brier(np.eye(3)[[1, 2]], y) == pytest.approx(2.0)  # confidently wrong


class TestECE:
    def test_near_zero_for_a_calibrated_forecaster(self) -> None:
        logits, y = _calibrated_sample()
        r = C.reliability(softmax(logits, axis=1), y, n_bins=15)
        assert r["ece"] < 0.02

    def test_detects_overconfidence(self) -> None:
        logits, y = _calibrated_sample()
        over = softmax(logits * 3.0, axis=1)  # sharpened: confident beyond its accuracy
        assert C.reliability(over, y)["ece"] > 0.08

    def test_bins_cover_every_example_including_confidence_one(self) -> None:
        p = np.array([[1.0, 0.0, 0.0], [0.5, 0.3, 0.2], [0.34, 0.33, 0.33]])
        for scheme in ("equal_width", "equal_mass"):
            r = C.reliability(p, np.array([0, 1, 2]), n_bins=5, scheme=scheme)
            assert sum(b["n"] for b in r["bins"]) == 3

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="binning"):
            C.reliability(np.eye(3), np.arange(3), scheme="nope")


class TestTemperature:
    @pytest.mark.parametrize("t_true", [0.5, 1.0, 2.0])
    def test_recovers_a_known_temperature(self, t_true: float) -> None:
        logits, y = _calibrated_sample(t_true=t_true)
        assert C.fit_temperature(logits, y) == pytest.approx(t_true, rel=0.08)

    def test_never_changes_the_argmax(self) -> None:
        logits, _ = _calibrated_sample(n=500)
        for t in (0.1, 0.7, 3.0):
            assert (C.apply_temperature(logits, t).argmax(1) == logits.argmax(1)).all()

    def test_log_probs_are_valid_logits(self) -> None:
        p = softmax(np.random.default_rng(1).normal(size=(50, 3)), axis=1)
        assert np.allclose(C.apply_temperature(C.probs_to_logits(p), 1.0), p)

    def test_cross_fit_improves_nll_of_an_overconfident_model(self) -> None:
        logits, y = _calibrated_sample(n=4000)
        r = C.cross_fit_temperature(logits * 3.0, y)
        assert r["argmax_preserved"]
        assert r["temperature_scaled"]["nll"] < r["uncalibrated"]["nll"]
        for t in r["temperatures"].values():
            assert t == pytest.approx(3.0, rel=0.15)

    def test_cross_fit_halves_are_disjoint_and_stratified(self) -> None:
        y = np.array([0] * 50 + [1] * 10 + [2] * 40)
        a, b = C._halves(y, seed=0)
        assert not set(a) & set(b) and len(a) + len(b) == len(y)
        assert (y[a] == 1).sum() == 5


class TestRiskCoverage:
    def test_full_coverage_risk_is_the_error_rate(self) -> None:
        logits, y = _calibrated_sample(n=2000)
        p = softmax(logits, axis=1)
        r = C.risk_coverage(p, y, coverages=(1.0,))[0]
        assert r["risk"] == pytest.approx((p.argmax(1) != y).mean())
        assert all(v == 1.0 for v in r["class_coverage"].values())

    def test_risk_falls_as_coverage_falls_for_a_calibrated_model(self) -> None:
        logits, y = _calibrated_sample(n=5000)
        risks = [r["risk"] for r in C.risk_coverage(softmax(logits, axis=1), y)]
        assert risks == sorted(risks, reverse=True)

    def test_reports_per_class_coverage(self) -> None:
        """A rule that drops one class entirely must show it."""
        p = np.array([[0.9, 0.05, 0.05]] * 8 + [[0.4, 0.35, 0.25]] * 2)
        y = np.array([0] * 8 + [1] * 2)
        r = C.risk_coverage(p, y, coverages=(0.8,))[0]
        assert r["class_coverage"][1] == 0.0
        assert r["class_coverage"][0] == 1.0
