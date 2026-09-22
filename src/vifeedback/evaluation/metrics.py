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

    if y_prob is not None:
        out.update(_probability_metrics(y_true, np.asarray(y_prob, dtype=np.float64), k))
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
