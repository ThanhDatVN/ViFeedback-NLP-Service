"""Bootstrap confidence intervals and paired significance testing.

docs/EVALUATION_PROTOCOL.md § 3. The paired bootstrap is the primary test: resampling the *same*
test indices for both models removes between-example variance, which is what gives the test power on
a 3,166-example set.

Implementation note: everything runs off the flat code array `y_true * k + y_pred`, so one resample
costs a single `bincount`. Resamples are generated in chunks to bound peak memory — 10,000 x 3,166
int64 indices would be ~250 MB in one allocation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vifeedback.constants import BOOTSTRAP_ALPHA, BOOTSTRAP_RESAMPLES
from vifeedback.evaluation.metrics import macro_f1_from_confusion

_CHUNK = 500


def _codes(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    return np.asarray(y_true, dtype=np.int64) * k + np.asarray(y_pred, dtype=np.int64)


def _resample_scores(
    codes: np.ndarray,
    k: int,
    n_resamples: int,
    rng: np.random.Generator,
    metric: Callable[[np.ndarray], float],
    indices_out: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Scores over `n_resamples` bootstrap resamples of `codes`.

    `indices_out` collects the drawn indices so a paired test can reuse the identical resamples for
    the second model — the pairing is the whole point of the test.
    """
    n = len(codes)
    scores = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        size = min(_CHUNK, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n), dtype=np.int64)
        if indices_out is not None:
            indices_out.append(idx)
        for j in range(size):
            cm = np.bincount(codes[idx[j]], minlength=k * k).reshape(k, k)
            scores[done + j] = metric(cm)
        done += size
    return scores


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
    metric: Callable[[np.ndarray], float] = macro_f1_from_confusion,
) -> dict[str, float]:
    """Percentile bootstrap CI for a confusion-matrix metric. Deterministic given `seed`."""
    rng = np.random.default_rng(seed)
    scores = _resample_scores(_codes(y_true, y_pred, k), k, n_resamples, rng, metric)
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(
            metric(np.bincount(_codes(y_true, y_pred, k), minlength=k * k).reshape(k, k))
        ),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "std": float(scores.std(ddof=1)),
        "n_resamples": n_resamples,
        "alpha": alpha,
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    k: int,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = BOOTSTRAP_ALPHA,
    seed: int = 0,
    metric: Callable[[np.ndarray], float] = macro_f1_from_confusion,
) -> dict[str, Any]:
    """Test whether model A beats model B, resampling identical indices for both.

    Reports the observed difference, its CI, and a two-sided bootstrap p-value. Significant at
    `alpha` iff the CI excludes zero.
    """
    y_true = np.asarray(y_true)
    codes_a = _codes(y_true, pred_a, k)
    codes_b = _codes(y_true, pred_b, k)
    n = len(y_true)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_resamples, dtype=np.float64)
    done = 0
    while done < n_resamples:
        size = min(_CHUNK, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n), dtype=np.int64)
        for j in range(size):
            sel = idx[j]
            cm_a = np.bincount(codes_a[sel], minlength=k * k).reshape(k, k)
            cm_b = np.bincount(codes_b[sel], minlength=k * k).reshape(k, k)
            diffs[done + j] = metric(cm_a) - metric(cm_b)
        done += size

    score_a = metric(np.bincount(codes_a, minlength=k * k).reshape(k, k))
    score_b = metric(np.bincount(codes_b, minlength=k * k).reshape(k, k))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Two-sided p-value: how often the resampled difference lands on the other side of zero.
    # +1 smoothing keeps p > 0 — an empirical bootstrap cannot evidence p = 0.
    tail = (diffs <= 0).sum() if score_a > score_b else (diffs >= 0).sum()
    p_value = min(1.0, 2.0 * (tail + 1) / (n_resamples + 1))

    return {
        "score_a": float(score_a),
        "score_b": float(score_b),
        "observed_diff": float(score_a - score_b),
        "mean_diff": float(diffs.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(p_value),
        "significant": bool(lo > 0 or hi < 0),
        "n_resamples": n_resamples,
        "alpha": alpha,
    }


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar test on the paired correct/incorrect table (secondary test).

    Uses the exact binomial rather than the chi-square approximation; the discordant counts here can
    be small, and the approximation is unreliable below ~25.
    """
    from scipy.stats import binomtest

    a_ok = np.asarray(pred_a) == np.asarray(y_true)
    b_ok = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int((~a_ok & b_ok).sum())  # only B right
    n10 = int((a_ok & ~b_ok).sum())  # only A right

    if n01 + n10 == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "significant": False}

    p = float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return {
        "n01_only_b_correct": n01,
        "n10_only_a_correct": n10,
        "p_value": p,
        "significant": bool(p < BOOTSTRAP_ALPHA),
    }


def benjamini_hochberg(p_values: list[float], alpha: float = BOOTSTRAP_ALPHA) -> list[bool]:
    """FDR correction for a family of comparisons reported together (§ 3, multiple comparisons)."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    cutoff = np.where(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(m, dtype=bool)
    out[order[:cutoff]] = True
    return out.tolist()
