"""Unit tests for the evaluation harness.

The harness is built in Phase 1 and reused unchanged by every later phase, so a bug here silently
corrupts every number in the project. Correctness is pinned against sklearn, including the
degenerate cases that actually occur on this dataset: a class that is never predicted, and a class
absent from a bootstrap resample.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from vifeedback.constants import n_classes
from vifeedback.evaluation import bootstrap as B
from vifeedback.evaluation import metrics as M


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def _random_case(rng, n=500, k=3, skew=None):
    p = skew if skew is not None else np.ones(k) / k
    y_true = rng.choice(k, size=n, p=p)
    y_pred = np.where(rng.random(n) < 0.75, y_true, rng.choice(k, size=n))
    return y_true, y_pred


class TestAgainstSklearn:
    @pytest.mark.parametrize("task", ["sentiment", "topic"])
    def test_matches_sklearn_on_random_cases(self, rng, task):
        k = n_classes(task)
        for _ in range(20):
            y_true, y_pred = _random_case(rng, n=400, k=k)
            m = M.evaluate(y_true, y_pred, task)
            assert m["macro_f1"] == pytest.approx(
                f1_score(y_true, y_pred, average="macro", labels=range(k), zero_division=0)
            )
            assert m["weighted_f1"] == pytest.approx(
                f1_score(y_true, y_pred, average="weighted", labels=range(k), zero_division=0)
            )
            assert m["accuracy"] == pytest.approx(accuracy_score(y_true, y_pred))
            assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y_true, y_pred))
            assert m["mcc"] == pytest.approx(matthews_corrcoef(y_true, y_pred), abs=1e-9)

    def test_per_class_matches_sklearn(self, rng):
        y_true, y_pred = _random_case(rng, n=600, k=3)
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=range(3), zero_division=0
        )
        m = M.evaluate(y_true, y_pred, "sentiment")
        for i, name in enumerate(m["labels"]):
            c = m["per_class"][name]
            assert c["precision"] == pytest.approx(p[i])
            assert c["recall"] == pytest.approx(r[i])
            assert c["f1"] == pytest.approx(f[i])
            assert c["support"] == s[i]

    def test_confusion_matrix_orientation(self, rng):
        """Rows are true, columns predicted. A transposed matrix still looks plausible."""
        y_true, y_pred = _random_case(rng, n=300, k=4)
        m = M.evaluate(y_true, y_pred, "topic")
        assert np.array_equal(
            np.array(m["confusion"]), confusion_matrix(y_true, y_pred, labels=range(4))
        )


class TestDegenerateCases:
    """The cases that actually occur on UIT-VSFC."""

    def test_classifier_that_never_predicts_neutral(self):
        """The motivating example, computed on the real UIT-VSFC test distribution.

        These exact values are quoted in docs/DATA_CARD.md § 4, docs/EVALUATION_PROTOCOL.md § 1 and
        ADR-001. Pinning them here means the project's central claim cannot silently drift.
        """
        n_neg, n_neu, n_pos = 1409, 167, 1590  # the real test split
        y_true = np.array([0] * n_neg + [1] * n_neu + [2] * n_pos)
        # Perfect on negative/positive; neutral split evenly into the two majority classes.
        y_pred = np.array([0] * n_neg + [0] * 84 + [2] * 83 + [2] * n_pos)

        m = M.evaluate(y_true, y_pred, "sentiment")
        assert m["per_class"]["neutral"]["f1"] == 0.0
        assert m["per_class"]["neutral"]["predicted"] == 0
        assert m["accuracy"] == pytest.approx(0.947, abs=0.001)
        assert m["weighted_f1"] == pytest.approx(0.922, abs=0.001)
        assert m["macro_f1"] == pytest.approx(0.649, abs=0.002)

    def test_macro_f1_is_the_only_metric_that_moves(self):
        """Quantifies the framing: adding the neutral class costs macro-F1 ~5x what it costs
        weighted F1. This ratio is the argument for the headline metric choice."""
        n_neg, n_neu, n_pos = 1409, 167, 1590
        y_true = np.array([0] * n_neg + [1] * n_neu + [2] * n_pos)
        blind = np.array([0] * n_neg + [0] * 84 + [2] * 83 + [2] * n_pos)
        perfect = y_true.copy()

        b = M.evaluate(y_true, blind, "sentiment")
        p_ = M.evaluate(y_true, perfect, "sentiment")
        weighted_cost = p_["weighted_f1"] - b["weighted_f1"]
        macro_cost = p_["macro_f1"] - b["macro_f1"]
        assert macro_cost / weighted_cost > 4.0

    def test_class_with_no_predictions_gets_zero_not_nan(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 0, 0, 2, 2])
        m = M.evaluate(y_true, y_pred, "sentiment")
        assert m["per_class"]["neutral"]["f1"] == 0.0
        assert np.isfinite(m["macro_f1"])

    def test_class_absent_from_the_sample(self):
        """A bootstrap resample can drop the 167-example neutral class entirely."""
        y_true = np.array([0, 0, 2, 2])
        y_pred = np.array([0, 2, 2, 2])
        m = M.evaluate(y_true, y_pred, "sentiment")
        assert len(m["per_class"]) == 3
        assert m["per_class"]["neutral"]["support"] == 0
        assert np.isfinite(m["macro_f1"])

    def test_perfect_and_worst_predictions(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        assert M.evaluate(y, y, "sentiment")["macro_f1"] == pytest.approx(1.0)
        assert M.evaluate(y, np.zeros_like(y), "sentiment")["macro_f1"] < 0.25

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            M.evaluate(np.zeros(5), np.zeros(4), "sentiment")


class TestProbabilityMetrics:
    def test_log_loss_matches_sklearn(self, rng):
        from sklearn.metrics import log_loss as sk_log_loss

        y_true = rng.choice(3, size=200)
        logits = rng.normal(size=(200, 3))
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        m = M.evaluate(y_true, prob.argmax(axis=1), "sentiment", y_prob=prob)
        assert m["log_loss"] == pytest.approx(sk_log_loss(y_true, prob, labels=range(3)), rel=1e-6)

    def test_perfect_calibration_has_near_zero_ece(self):
        y_true = np.array([0] * 100 + [1] * 100)
        prob = np.zeros((200, 3))
        prob[:100, 0] = 1.0
        prob[100:, 1] = 1.0
        m = M.evaluate(y_true, prob.argmax(axis=1), "sentiment", y_prob=prob)
        assert m["ece_10bin"] == pytest.approx(0.0, abs=1e-9)


class TestBootstrap:
    def test_ci_is_deterministic_under_a_fixed_seed(self, rng):
        y_true, y_pred = _random_case(rng, n=300, k=3)
        a = B.bootstrap_ci(y_true, y_pred, 3, n_resamples=200, seed=7)
        b = B.bootstrap_ci(y_true, y_pred, 3, n_resamples=200, seed=7)
        assert a == b

    def test_ci_brackets_the_point_estimate(self, rng):
        y_true, y_pred = _random_case(rng, n=800, k=3)
        r = B.bootstrap_ci(y_true, y_pred, 3, n_resamples=500, seed=1)
        assert r["ci_low"] <= r["point"] <= r["ci_high"]
        assert r["point"] == pytest.approx(M.macro_f1(y_true, y_pred, 3))

    def test_paired_bootstrap_detects_a_real_difference(self, rng):
        y_true = rng.choice(3, size=1000)
        good = np.where(rng.random(1000) < 0.90, y_true, rng.choice(3, size=1000))
        bad = np.where(rng.random(1000) < 0.55, y_true, rng.choice(3, size=1000))
        r = B.paired_bootstrap(y_true, good, bad, 3, n_resamples=500, seed=3)
        assert r["observed_diff"] > 0
        assert r["significant"]
        assert r["p_value"] < 0.05

    def test_paired_bootstrap_finds_no_difference_between_identical_models(self, rng):
        y_true, y_pred = _random_case(rng, n=600, k=3)
        r = B.paired_bootstrap(y_true, y_pred, y_pred.copy(), 3, n_resamples=300, seed=5)
        assert r["observed_diff"] == 0.0
        assert not r["significant"]

    def test_p_value_is_never_exactly_zero(self, rng):
        """An empirical bootstrap cannot evidence p = 0; +1 smoothing enforces it."""
        y_true = rng.choice(3, size=500)
        r = B.paired_bootstrap(y_true, y_true, rng.choice(3, size=500), 3, n_resamples=200, seed=9)
        assert r["p_value"] > 0


class TestMcNemarAndFDR:
    def test_mcnemar_agrees_with_direction(self, rng):
        y_true = rng.choice(3, size=500)
        good = np.where(rng.random(500) < 0.9, y_true, rng.choice(3, size=500))
        bad = np.where(rng.random(500) < 0.5, y_true, rng.choice(3, size=500))
        r = B.mcnemar(y_true, good, bad)
        assert r["n10_only_a_correct"] > r["n01_only_b_correct"]
        assert r["significant"]

    def test_mcnemar_on_identical_predictions(self, rng):
        y_true = rng.choice(3, size=100)
        r = B.mcnemar(y_true, y_true, y_true)
        assert r["p_value"] == 1.0 and not r["significant"]

    def test_benjamini_hochberg_is_more_conservative_than_raw_alpha(self):
        p = [0.001, 0.02, 0.03, 0.04, 0.9]
        out = B.benjamini_hochberg(p, alpha=0.05)
        assert out[0] is True
        assert out[-1] is False
        assert sum(out) <= sum(x < 0.05 for x in p)


class TestPriorTuningHonesty:
    """Regression cover for ADR-015.

    Tuning decision priors on the same data they are scored on is optimistically biased. On
    UIT-VSFC the bias accounted for the *entire* apparent gain on PhoBERT, so the distinction is
    load-bearing rather than pedantic.
    """

    def test_crossfitted_is_not_more_optimistic_than_fit_on_eval(self, rng):
        from vifeedback.models.baseline_tfidf import prior_tuning_report

        y = rng.choice(3, size=600, p=[0.46, 0.04, 0.50])  # the real UIT-VSFC prior
        logits = rng.normal(size=(600, 3))
        logits[np.arange(600), y] += 1.5  # weakly informative probabilities
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

        r = prior_tuning_report(prob, y, 3, seed=0)
        assert r["fit_on_eval"] >= r["crossfitted"] - 1e-9, (
            "fitting and scoring on the same data must not look worse than cross-fitting"
        )
        assert r["optimism_bias"] >= -1e-9

    def test_crossfit_returns_one_prediction_per_example(self, rng):
        from vifeedback.models.baseline_tfidf import crossfit_class_priors

        y = rng.choice(3, size=200, p=[0.46, 0.04, 0.50])
        prob = rng.dirichlet([1, 1, 1], size=200)
        pred = crossfit_class_priors(prob, y, 3, seed=0)
        assert pred.shape == y.shape
        assert set(np.unique(pred)).issubset({0, 1, 2})

    def test_tuning_cannot_reduce_fit_on_eval_macro_f1(self, rng):
        """Sanity: the search starts from w = 1 and keeps the best, so it is monotone on its own
        objective. Any violation means the optimizer is broken."""
        from vifeedback.models.baseline_tfidf import tune_class_priors

        y = rng.choice(3, size=400, p=[0.46, 0.04, 0.50])
        prob = rng.dirichlet([2, 1, 2], size=400)
        w = tune_class_priors(prob, y, 3, seed=0)
        assert M.macro_f1(y, (prob * w).argmax(axis=1), 3) >= M.macro_f1(y, prob.argmax(axis=1), 3)
