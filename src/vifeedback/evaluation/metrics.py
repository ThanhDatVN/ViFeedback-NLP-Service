"""The metric suite every model in this project is measured with.

Built once, in Phase 1, against cheap models — so its bugs surface in seconds rather than after a
40-minute GPU run. Every later phase imports this module unchanged.

The contract (docs/EVALUATION_PROTOCOL.md § 1): macro-F1 is the headline, weighted F1 and accuracy
are always printed beside it, and per-class F1 always carries its support.

Metrics are computed from a confusion matrix rather than by calling sklearn per metric, because the
bootstrap needs tens of thousands of evaluations and a confusion matrix is the cheapest sufficient
statistic. Correctness is pinned against sklearn in `tests/unit/test_metrics.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vifeedback.constants import label_names, n_classes


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """k x k confusion matrix, rows = true, cols = predicted."""
    codes = np.asarray(y_true, dtype=np.int64) * k + np.asarray(y_pred, dtype=np.int64)
    return np.bincount(codes, minlength=k * k).reshape(k, k)


def prf_from_confusion(cm: np.ndarray) -> dict[str, np.ndarray]:
    """Per-class precision, recall, F1 and support.

    Zero-division convention matches sklearn's default (`zero_division=0`): a class with no
    predictions gets precision 0, and a class with no support gets recall 0. Both cases occur here —
    a baseline that never predicts `neutral` is the motivating example for the whole project — so
    this is load-bearing, not boilerplate.
    """
    tp = np.diag(cm).astype(np.float64)
    pred = cm.sum(axis=0).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred > 0, tp / pred, 0.0)
        recall = np.where(support > 0, tp / support, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """Unweighted mean of per-class F1 — the headline metric.

    Averaged over *all* declared classes, including any absent from a bootstrap resample, which is
    what makes the minority class impossible to hide.
    """
    return float(prf_from_confusion(cm)["f1"].mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    return macro_f1_from_confusion(confusion(y_true, y_pred, k))


def _mcc_from_confusion(cm: np.ndarray) -> float:
    """Multi-class Matthews correlation coefficient (Gorodkin's R_K)."""
    c = np.diag(cm).sum().astype(np.float64)
    s = cm.sum().astype(np.float64)
    t = cm.sum(axis=1).astype(np.float64)  # true totals
    p = cm.sum(axis=0).astype(np.float64)  # predicted totals
    num = c * s - t @ p
    den = np.sqrt(max(s**2 - p @ p, 0.0)) * np.sqrt(max(s**2 - t @ t, 0.0))
    return float(num / den) if den > 0 else 0.0


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str,
    y_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """The full metric suite for one set of predictions."""
    k = n_classes(task)
    names = label_names(task)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")

    cm = confusion(y_true, y_pred, k)
    prf = prf_from_confusion(cm)
    support = prf["support"]
    n = float(support.sum())

    weights = support / n if n > 0 else np.zeros_like(support)
    accuracy = float(np.diag(cm).sum() / n) if n > 0 else 0.0

    out: dict[str, Any] = {
        "n": int(n),
        "macro_f1": float(prf["f1"].mean()),
        "weighted_f1": float(prf["f1"] @ weights),
        "micro_f1": accuracy,  # identical to accuracy in single-label multi-class
        "accuracy": accuracy,
        "balanced_accuracy": float(prf["recall"].mean()),
        "mcc": _mcc_from_confusion(cm),
        "g_mean": float(np.prod(prf["recall"]) ** (1.0 / k)),
        "cohen_kappa": cohen_kappa(y_true, y_pred, k),
        "per_class": {
            name: {
                "precision": float(prf["precision"][i]),
                "recall": float(prf["recall"][i]),
                "f1": float(prf["f1"][i]),
                "support": int(support[i]),
                "predicted": int(cm[:, i].sum()),
            }
            for i, name in enumerate(names)
        },
        "confusion": cm.tolist(),
        "confusion_normalized": np.divide(
            cm, support[:, None], out=np.zeros(cm.shape), where=support[:, None] > 0
        )
        .round(4)
        .tolist(),
        "labels": names,
    }

    if task in ORDINAL_TASKS:
        out.update(ordinal_metrics(y_true, y_pred, k))

    if y_prob is not None:
        prob = np.asarray(y_prob, dtype=np.float64)
        out.update(_probability_metrics(y_true, prob, k))
        ap = average_precision_per_class(y_true, prob, k)
        for i, name in enumerate(names):
            out["per_class"][name]["average_precision"] = ap[i]
    return out


def _probability_metrics(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> dict[str, Any]:
    """Log loss and expected calibration error.

    A service that returns probabilities should be able to say how well calibrated they are; ECE is
    cheap here and becomes the Phase 8 calibration baseline.
    """
    eps = 1e-12
    p = np.clip(y_prob, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)

    log_loss = float(-np.log(p[np.arange(len(y_true)), y_true]).mean())

    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y_true).astype(np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(conf, bins[1:-1]), 0, 9)
    ece = 0.0
    for b in range(10):
        m = idx == b
        if m.any():
            ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())

    return {"log_loss": log_loss, "ece_10bin": float(ece)}


def format_report(metrics: dict[str, Any], title: str = "") -> str:
    """The canonical text block from docs/EVALUATION_PROTOCOL.md § 1."""
    lines = []
    if title:
        lines.append(title)
    lines.append(f"  Macro-F1      {metrics['macro_f1']:.3f}")
    if "macro_f1_ci" in metrics:
        lo, hi = metrics["macro_f1_ci"]
        lines[-1] += f"   [95% CI {lo:.3f}-{hi:.3f}]"
    lines.append(f"  Weighted F1   {metrics['weighted_f1']:.3f}")
    lines.append(f"  Accuracy      {metrics['accuracy']:.3f}")
    lines.append(f"  Bal. accuracy {metrics['balanced_accuracy']:.3f}")
    lines.append(f"  MCC           {metrics['mcc']:.3f}")
    first = True
    for name, m in metrics["per_class"].items():
        prefix = "  Per class    " if first else "               "
        first = False
        lines.append(
            f"{prefix} {name:<17s} P {m['precision']:.3f}  R {m['recall']:.3f}  "
            f"F1 {m['f1']:.3f}  (n={m['support']:>5d}, predicted {m['predicted']:>5d})"
        )
    return "\n".join(lines)


# --- Ordinal metrics ----------------------------------------------------------------------------
# Sentiment is ordered: negative < neutral < positive. Macro-F1 treats all errors as equal, but a
# `positive -> negative` error is plainly worse than `positive -> neutral`. Measured at Gate G3,
# 59.4% of all errors involve the middle class, and true neutrals split almost exactly evenly
# between the two poles (20.8% / 20.3%) - the signature of ordinal confusion. These metrics make
# that structure measurable; macro-F1 cannot express it. See docs/PROPOSALS.md F1.

ORDINAL_TASKS = frozenset({"sentiment"})


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error over ordinal label indices. 0 is perfect; 2 is maximally wrong here."""
    return float(np.abs(np.asarray(y_true, np.float64) - np.asarray(y_pred, np.float64)).mean())


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Cohen's kappa with quadratic weights — penalizes distant errors quadratically.

    1.0 is perfect, 0.0 is chance, negative is worse than chance. The standard metric for ordinal
    agreement, and the right companion to macro-F1 on an ordered label set.
    """
    cm = confusion(y_true, y_pred, k).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0

    idx = np.arange(k)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2

    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    num = (weights * cm).sum()
    den = (weights * expected).sum()
    return float(1.0 - num / den) if den > 0 else 0.0


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of predictions within one ordinal step of the truth.

    Separates "the model is roughly right about polarity" from "the model inverted the sentiment",
    which a single accuracy figure merges.
    """
    return float((np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= 1).mean())


def ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> dict[str, float]:
    return {
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "qwk": quadratic_weighted_kappa(y_true, y_pred, k),
        "adjacent_accuracy": adjacent_accuracy(y_true, y_pred),
    }


# --- Imbalance-aware and threshold-free metrics ---------------------------------------------------
# Added after an audit found the suite incomplete in a way that mattered: neutral F1 is the number
# this project turns on, and it was being reported without a confidence interval while macro-F1 had
# one. A point estimate on 73 dev examples without an interval invites exactly the over-reading the
# rest of the protocol is built to prevent.


def g_mean(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Geometric mean of per-class recall — the standard imbalance metric.

    Unlike the arithmetic mean (balanced accuracy), the geometric mean goes to **zero** if any single
    class is entirely missed. On a 3-class problem with a 4% class, that is the behaviour we want:
    a model that ignores neutral should score 0, not 0.67.
    """
    recall = prf_from_confusion(confusion(y_true, y_pred, k))["recall"]
    return float(np.prod(recall) ** (1.0 / k))


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Unweighted chance-corrected agreement.

    Reported so model performance can be compared against the corpus's published inter-annotator
    agreement (91.20% sentiment, 71.07% topic) on the same footing. A topic macro-F1 of 0.80 means
    something very different against 71% human agreement than against 91%.
    """
    cm = confusion(y_true, y_pred, k).astype(np.float64)
    n = cm.sum()
    if n == 0:
        return 0.0
    po = np.diag(cm).sum() / n
    pe = (cm.sum(axis=1) @ cm.sum(axis=0)) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 0.0


def average_precision_per_class(y_true: np.ndarray, y_prob: np.ndarray, k: int) -> dict[int, float]:
    """One-vs-rest average precision (area under the precision-recall curve).

    **Threshold-free**, which matters here: F1 measures the model at the argmax decision rule, and
    ADR-015 showed that decision rule cannot be tuned on this dataset. AP asks a different and
    fairer question — *does the model rank neutral examples above the rest?* — separating ranking
    quality from the decision rule it is stuck with.
    """
    from sklearn.metrics import average_precision_score

    out = {}
    for c in range(k):
        binary = (np.asarray(y_true) == c).astype(int)
        if binary.sum() == 0:
            out[c] = float("nan")
            continue
        out[c] = float(average_precision_score(binary, y_prob[:, c]))
    return out


def per_class_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Bootstrap CI for **each class's** F1, not just the macro average.

    The gap this closes: neutral F1 is the number the project turns on, and it rests on 73 dev
    examples. Reporting it as a bare point estimate beside a macro-F1 that *does* carry an interval
    was an inconsistency in the protocol, not a stylistic choice.
    """
    y_true = np.asarray(y_true, np.int64)
    y_pred = np.asarray(y_pred, np.int64)
    codes = y_true * k + y_pred
    n = len(codes)

    rng = np.random.default_rng(seed)
    scores = np.empty((n_resamples, k), dtype=np.float64)
    done = 0
    while done < n_resamples:
        size = min(500, n_resamples - done)
        idx = rng.integers(0, n, size=(size, n), dtype=np.int64)
        for j in range(size):
            cm = np.bincount(codes[idx[j]], minlength=k * k).reshape(k, k)
            scores[done + j] = prf_from_confusion(cm)["f1"]
        done += size

    point = prf_from_confusion(confusion(y_true, y_pred, k))["f1"]
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return {
        c: {
            "f1": float(point[c]),
            "ci_low": float(lo[c]),
            "ci_high": float(hi[c]),
            "ci_width": float(hi[c] - lo[c]),
            "std": float(scores[:, c].std(ddof=1)),
        }
        for c in range(k)
    }


def metrics_by_length_bucket(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: list[str],
    task: str,
    edges: tuple[int, ...] = (0, 5, 10, 20, 10_000),
) -> pd.DataFrame:
    """Macro-F1 and accuracy sliced by sentence length.

    Not cosmetic. Neutral sentences average 9.8 syllables against 16.9 for negative, and 19.2% of
    them are under 5 syllables. Without this slice, an error attributed to *meaning* may really be
    an artifact of *brevity* — a very short sentence carries little evidence either way, and the
    error analysis would be describing the wrong cause.
    """
    from itertools import pairwise

    from vifeedback.preprocess import syllable_count

    lengths = np.array([syllable_count(t) for t in texts])
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    rows = []
    for lo, hi in pairwise(edges):
        m = (lengths >= lo) & (lengths < hi)
        if m.sum() == 0:
            continue
        sub = evaluate(y_true[m], y_pred[m], task)
        row = {
            "bucket": f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+",
            "n": int(m.sum()),
            "share": round(float(m.mean()), 3),
            "macro_f1": round(sub["macro_f1"], 4),
            "accuracy": round(sub["accuracy"], 4),
        }
        for name in label_names(task):
            row[f"support_{name}"] = sub["per_class"][name]["support"]
            row[f"f1_{name}"] = round(sub["per_class"][name]["f1"], 4)
        rows.append(row)
    return pd.DataFrame(rows)
