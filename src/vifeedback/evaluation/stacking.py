"""Sparse + dense stacking for topic — Cycle 2 H5 (E08).

TF-IDF beats PhoBERT on `facility`, the class with the most distinctive vocabulary. Stacking asks
whether the two models' errors are different enough to combine. The protocol keeps every fitted
quantity off the evaluation data:

* both base models produce **out-of-fold** features over train, on the same folds;
* the sparse model's C is fixed (1.0), not tuned on dev;
* the meta-model is fitted on the out-of-fold features only;
* validation features come from the 5 fold models (dense, averaged) and a sparse model refitted on
  all of train, and are used once, for the comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def sparse_model(seed: int = 42):
    """TF-IDF word+char union + LinearSVC, C fixed at 1.0 (B4 without dev selection)."""
    from vifeedback.models.baseline_tfidf import LADDER_BY_KEY, build

    return build(LADDER_BY_KEY["b4"], seed=seed)


def sparse_oof_scores(
    raw_texts: list[str], y: np.ndarray, fold_id: np.ndarray, seed: int = 42
) -> np.ndarray:
    """Out-of-fold LinearSVC decision scores, one fold at a time, on the dense model's folds."""
    y = np.asarray(y)
    texts = np.asarray(raw_texts, dtype=object)
    scores = np.full((len(y), len(np.unique(y))), np.nan)
    for f in np.unique(fold_id):
        tr, ho = fold_id != f, fold_id == f
        m = sparse_model(seed).fit(texts[tr], y[tr])
        scores[ho] = m.decision_function(texts[ho])
    assert not np.isnan(scores).any()
    return scores


def features(dense_probs: np.ndarray, sparse_scores: np.ndarray) -> np.ndarray:
    """Meta-features: dense log-probabilities next to sparse decision scores."""
    return np.hstack([np.log(np.clip(dense_probs, 1e-9, 1.0)), sparse_scores])


def fit_meta(x: np.ndarray, y: np.ndarray, seed: int = 42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
    ).fit(x, y)


def compare(y: np.ndarray, preds: dict[str, np.ndarray], task: str) -> dict[str, Any]:
    """Per-model metrics plus the declared paired comparison (stacked vs dense)."""
    from vifeedback.constants import label_names
    from vifeedback.evaluation import bootstrap as B
    from vifeedback.evaluation import metrics as M

    k = len(label_names(task))
    out: dict[str, Any] = {
        name: {
            "macro_f1": M.macro_f1(y, p, k),
            "per_class_f1": {c: v["f1"] for c, v in M.evaluate(y, p, task)["per_class"].items()},
        }
        for name, p in preds.items()
    }
    pb = B.paired_bootstrap(y, preds["stacked"], preds["dense"], k, n_resamples=5000, seed=42)
    out["paired_stacked_minus_dense"] = pb
    d, s = out["dense"]["per_class_f1"], out["stacked"]["per_class_f1"]
    out["supported"] = bool(
        pb["observed_diff"] >= 0.005
        and pb["ci_low"] > 0
        and s["facility"] >= d["facility"]
        and s["others"] >= d["others"]
    )
    if "sparse" in preds:
        out["descriptive_facility_sparse_minus_dense"] = facility_claim(
            y, preds["sparse"], preds["dense"], task
        )
    return out


def facility_claim(
    y: np.ndarray, sparse_pred: np.ndarray, dense_pred: np.ndarray, task: str = "topic"
) -> dict[str, Any]:
    """cycle2.yaml v2, descriptive: is 'TF-IDF beats PhoBERT on facility' more than noise?

    A paired example bootstrap of facility F1 on the same validation rows. Decides nothing.
    """
    from vifeedback.constants import label_names
    from vifeedback.evaluation import bootstrap as B

    names = label_names(task)
    c = names.index("facility")

    def facility_f1(cm: np.ndarray) -> float:
        tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
        return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0

    return B.paired_bootstrap(
        y, sparse_pred, dense_pred, len(names), n_resamples=5000, seed=42, metric=facility_f1
    )
