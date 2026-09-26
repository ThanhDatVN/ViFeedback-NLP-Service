"""Calibration and selective prediction — experiment E09/E10 of docs/REVIEW_AND_RESEARCH_PLAN.md.

Protocol, fixed before any number is looked at (docs/EVALUATION_PROTOCOL.md § Calibration):

* **Fit and score on different data.** A temperature fitted on the examples it is scored on reports
  its own training loss. `cross_fit_temperature` fits on one stratified half of the given set and
  scores the other, then swaps, so every reported number is out-of-sample for its temperature.
* **Proper scores first.** NLL and Brier are the headline. ECE is reported with its bin count and
  binning scheme, because the same predictions can give visibly different ECE under different bins.
* **Temperature scaling preserves the argmax.** It cannot change accuracy or F1, so F1 is not its
  success criterion (Guo et al., 2017). A "calibration method" that moves F1 is doing something
  else and must be reported as such.
* **Abstention reports coverage by class.** Rejecting nearly every neutral example can buy a good
  aggregate risk-coverage curve while making the system useless for the class that matters.

Only numpy and scipy are used, so this runs where scikit-learn's native extension is blocked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

# --- Proper scoring rules -------------------------------------------------------------------------


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    """Mean negative log-likelihood of the gold class."""
    p = np.clip(probs[np.arange(len(y)), np.asarray(y)], 1e-12, 1.0)
    return float(-np.log(p).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot target, summed over classes."""
    onehot = np.eye(probs.shape[1])[np.asarray(y)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


# --- Binned calibration error -----------------------------------------------------------------------


def reliability(
    probs: np.ndarray, y: np.ndarray, n_bins: int = 15, scheme: str = "equal_width"
) -> dict[str, Any]:
    """Top-label reliability table and ECE.

    `equal_width` bins [0,1] uniformly — the scheme of Guo et al. and the usual default.
    `equal_mass` puts the same number of examples in every bin, which avoids near-empty
    high-uncertainty bins on a model that is confident on most inputs. Both are reported by
    `summary()`; neither is the "true" ECE.
    """
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == np.asarray(y)).astype(float)
    n = len(conf)

    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif scheme == "equal_mass":
        edges = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown binning scheme {scheme!r}")

    # Right-closed bins so a confidence of exactly 1.0 lands in the last bin.
    idx = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    rows, ece, mce = [], 0.0, 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        acc, cf, w = correct[m].mean(), conf[m].mean(), m.sum() / n
        gap = abs(acc - cf)
        ece += w * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin_low": float(edges[b]),
                "bin_high": float(edges[b + 1]),
                "n": int(m.sum()),
                "accuracy": float(acc),
                "confidence": float(cf),
            }
        )
    return {"ece": float(ece), "mce": float(mce), "n_bins": n_bins, "scheme": scheme, "bins": rows}


def classwise_ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[int, float]:
    """One-vs-rest ECE per class. Top-label ECE is dominated by the majority classes; a model can
    be well calibrated on average and badly overconfident about neutral."""
    y = np.asarray(y)
    out: dict[int, float] = {}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for c in range(probs.shape[1]):
        pc = probs[:, c]
        hit = (y == c).astype(float)
        idx = np.clip(np.searchsorted(edges, pc, side="left") - 1, 0, n_bins - 1)
        e = 0.0
        for b in range(n_bins):
            m = idx == b
            if m.any():
                e += m.mean() * abs(hit[m].mean() - pc[m].mean())
        out[c] = float(e)
    return out


# --- Temperature scaling --------------------------------------------------------------------------


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """The scalar T minimizing NLL of softmax(logits / T). Searched in log-space, T in [0.05, 20]."""
    y = np.asarray(y)

    def loss(log_t: float) -> float:
        lp = log_softmax(logits / np.exp(log_t), axis=1)
        return float(-lp[np.arange(len(y)), y].mean())

    res = minimize_scalar(loss, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def apply_temperature(logits: np.ndarray, t: float) -> np.ndarray:
    return softmax(logits / t, axis=1)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    """Log-probabilities are valid logits: softmax(log p) = p. Lets saved probabilities be
    temperature-scaled without the raw logits."""
    return np.log(np.clip(probs, 1e-12, 1.0))


def _halves(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a, b = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        a.append(idx[: len(idx) // 2])
        b.append(idx[len(idx) // 2 :])
    return np.concatenate(a), np.concatenate(b)


def cross_fit_temperature(
    logits: np.ndarray, y: np.ndarray, *, seed: int = 42, n_bins: int = 15
) -> dict[str, Any]:
    """Fit T on one stratified half, score the other, swap; pool the out-of-sample probabilities.

    Returns the uncalibrated and calibrated summaries on identical examples, plus both fitted
    temperatures — if they differ a lot, a single T is not a stable property of this model.
    """
    y = np.asarray(y)
    a, b = _halves(y, seed)
    t_a, t_b = fit_temperature(logits[a], y[a]), fit_temperature(logits[b], y[b])
    cal = np.empty_like(logits, dtype=np.float64)
    cal[b] = apply_temperature(logits[b], t_a)  # fitted on a, scored on b
    cal[a] = apply_temperature(logits[a], t_b)
    raw = softmax(logits, axis=1)
    return {
        "temperatures": {"fit_on_half_a": t_a, "fit_on_half_b": t_b},
        "uncalibrated": summary(raw, y, n_bins=n_bins),
        "temperature_scaled": summary(cal, y, n_bins=n_bins),
        "argmax_preserved": bool((raw.argmax(1) == cal.argmax(1)).all()),
        "calibrated_probs": cal,
    }


# --- Selective prediction -------------------------------------------------------------------------


def risk_coverage(
    probs: np.ndarray, y: np.ndarray, coverages: tuple[float, ...] = (1.0, 0.95, 0.9, 0.8, 0.7)
) -> list[dict[str, Any]]:
    """Error rate on the retained examples when keeping the most confident `coverage` fraction,
    with the per-class share of each gold class that is retained.

    The per-class column is the point: an abstention rule that keeps 90% of all examples but only
    40% of the neutral ones has not solved neutral, it has hidden it.
    """
    y = np.asarray(y)
    conf = probs.max(axis=1)
    order = np.argsort(-conf, kind="stable")
    pred = probs.argmax(axis=1)
    out = []
    for cov in coverages:
        keep = order[: max(1, round(cov * len(y)))]
        kept = np.zeros(len(y), dtype=bool)
        kept[keep] = True
        out.append(
            {
                "coverage": cov,
                "threshold": float(conf[keep].min()),
                "risk": float((pred[kept] != y[kept]).mean()),
                "class_coverage": {int(c): float(kept[y == c].mean()) for c in np.unique(y)},
            }
        )
    return out


def summary(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> dict[str, Any]:
    width = reliability(probs, y, n_bins, "equal_width")
    mass = reliability(probs, y, n_bins, "equal_mass")
    return {
        "n": len(y),
        "nll": nll(probs, y),
        "brier": brier(probs, y),
        "ece_equal_width": width["ece"],
        "ece_equal_mass": mass["ece"],
        "mce_equal_width": width["mce"],
        "n_bins": n_bins,
        "classwise_ece": classwise_ece(probs, y, n_bins),
        "accuracy": float((probs.argmax(1) == np.asarray(y)).mean()),
        "mean_confidence": float(probs.max(1).mean()),
        "reliability_equal_width": width["bins"],
    }
