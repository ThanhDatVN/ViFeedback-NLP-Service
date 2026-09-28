"""B4': a topic-aware scope detector (configs/experiments/cycle4.yaml v3, NEXT_PLAN v5).

The served `in_scope` (Mahalanobis on the sentence feature) tracks the institution, not the topic
(ADR-032). This fits a detector on labelled topics instead: UIT-VSFC and in-scope NEU-ESC posts are
in scope, NEU-ESC spam, news, jobs and club posts are not. Two declared candidates, chosen on NEU-ESC
validation and confirmed once on NEU-ESC test:

* ``feature_logistic``: logistic regression on the served ONNX graph's sentence feature;
* ``tfidf_logistic``: logistic regression on TF-IDF word unigrams and bigrams.

Scores are "higher = more in scope", like the served score, so the service's threshold logic holds.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from vifeedback import paths

OUT = paths.RESULTS / "studies" / "cycle4" / "b4prime"
GRID = {"feature_logistic": (0.01, 0.1, 1.0, 10.0), "tfidf_logistic": (0.1, 1.0, 10.0)}


def _neu(split: str, transform) -> dict[str, Any]:
    from vifeedback.evaluation import external as X
    from vifeedback.training.domain import OFF_TOPIC

    ne = X.load_neu_esc(split)
    return {
        "x": transform(ne.text.tolist()),
        "in_scope": (~ne.topic.isin(OFF_TOPIC)).to_numpy(),
        "topic": ne.topic.to_numpy(),
    }


def data() -> dict[str, Any]:
    """Every set the study needs, as the service would see it (model input, labels, sources)."""
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.domain import serving_transform

    transform = serving_transform()
    uit_tr = load_variant("seg_pyvi", "train").sentence.tolist()
    uit_dv = load_variant("seg_pyvi", "validation").sentence.tolist()
    u4 = json.loads(
        (paths.RESULTS / "studies" / "ood" / "u4_offtopic_dev.json").read_text(encoding="utf-8")
    )
    return {
        "uit_train": uit_tr,
        "uit_validation": uit_dv,
        "neu_train": _neu("train", transform),
        "neu_validation": _neu("validation", transform),
        "neu_test": _neu("test", transform),
        "u4": transform([r["text"] for r in u4["rows"]]),
    }


def served_features(texts: list[str]) -> np.ndarray:
    """The served graph's `features` output (what a deployed detector would read)."""
    from vifeedback.inference.onnx_export import OnnxClassifier

    d = paths.MODELS / "serve" / "sentiment"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    clf = OnnxClassifier(d, max_length=manifest.get("max_length", 96))
    return np.concatenate(
        [clf.logits_and_features(texts[i : i + 64])[1] for i in range(0, len(texts), 64)]
    )


def served_mahalanobis(features: np.ndarray) -> np.ndarray:
    """The score the service returns today (for the report)."""
    from vifeedback.serving import pipeline as SP

    d = paths.MODELS / "serve" / "sentiment"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))

    class _Clf:
        has_features = True

    ood = SP.load_ood(d, manifest["ood"], _Clf())
    dist = np.stack(
        [
            np.einsum("ij,jk,ik->i", features - m, ood["precision"], features - m)
            for m in ood["means"]
        ],
        axis=1,
    )
    return -dist.min(axis=1)


def _auroc(in_scope: np.ndarray, off: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(
        roc_auc_score(np.r_[np.ones(len(in_scope)), np.zeros(len(off))], np.r_[in_scope, off])
    )


def fit(kind: str, c: float, x_train: Any, y_train: np.ndarray):
    """A fitted scorer: text or feature rows -> decision value (higher = in scope)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000)
    if kind == "feature_logistic":
        model = make_pipeline(StandardScaler(), lr)
    else:
        from sklearn.feature_extraction.text import TfidfVectorizer

        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2), lr)
    model.fit(x_train, y_train)
    return model


def run() -> dict[str, Any]:
    """Selection on validation, then the declared rule on NEU-ESC test (logged)."""
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.training.domain import log_test_use

    sets = data()
    tr_text = sets["uit_train"] + sets["neu_train"]["x"]
    y_tr = np.r_[
        np.ones(len(sets["uit_train"]), dtype=int), sets["neu_train"]["in_scope"].astype(int)
    ]
    neu_dv, neu_te = sets["neu_validation"], sets["neu_test"]

    # Features of the served graph, computed once.
    f = {
        "train": served_features(tr_text),
        "uit_validation": served_features(sets["uit_validation"]),
        "neu_validation": served_features(neu_dv["x"]),
        "neu_test": served_features(neu_te["x"]),
        "u4": served_features(sets["u4"]),
    }
    views = {
        "feature_logistic": {k: v for k, v in f.items()},
        "tfidf_logistic": {
            "train": tr_text,
            "uit_validation": sets["uit_validation"],
            "neu_validation": neu_dv["x"],
            "neu_test": neu_te["x"],
            "u4": sets["u4"],
        },
    }

    # Selection on NEU-ESC validation (within-source AUROC).
    grid: list[dict[str, Any]] = []
    for kind, cs in GRID.items():
        for c in cs:
            model = fit(kind, c, views[kind]["train"], y_tr)
            s = model.decision_function(views[kind]["neu_validation"])
            grid.append(
                {
                    "candidate": kind,
                    "C": c,
                    "neu_validation_auroc": _auroc(s[neu_dv["in_scope"]], s[~neu_dv["in_scope"]]),
                }
            )
    best = max(grid, key=lambda r: float(r["neu_validation_auroc"]))
    kind, c = str(best["candidate"]), float(best["C"])
    model = fit(kind, c, views[kind]["train"], y_tr)
    view = views[kind]

    def score(name: str) -> np.ndarray:
        return model.decision_function(view[name])

    s_uit_dv, s_neu_dv = score("uit_validation"), score("neu_validation")
    threshold = float(np.quantile(np.r_[s_uit_dv, s_neu_dv[neu_dv["in_scope"]]], 0.05))

    log_test_use("cycle4.yaml v3 B4' rule", [f"{kind} C={c}"])
    s_te = score("neu_test")
    ins, off = neu_te["in_scope"], ~neu_te["in_scope"]
    values = {
        "1_within_source_auroc": _auroc(s_te[ins], s_te[off]),
        "2_in_scope_flagged": float((s_te[ins] < threshold).mean()),
        "3_off_topic_caught": float((s_te[off] < threshold).mean()),
        "4_uit_validation_flagged": float((s_uit_dv < threshold).mean()),
    }
    limits = {  # (comparison, bound) as declared
        "1_within_source_auroc": (">=", 0.85),
        "2_in_scope_flagged": ("<=", 0.10),
        "3_off_topic_caught": (">=", 0.50),
        "4_uit_validation_flagged": ("<=", 0.05),
    }
    rules: dict[str, dict[str, Any]] = {
        k: {
            "value": v,
            "limit": f"{limits[k][0]} {limits[k][1]:.2f}",
            "passed": v >= limits[k][1] if limits[k][0] == ">=" else v <= limits[k][1],
        }
        for k, v in values.items()
    }

    # Reported: today's served score on the same sets; U4; per-topic flag rates.
    m_te, m_dv = served_mahalanobis(f["neu_test"]), served_mahalanobis(f["uit_validation"])
    s_u4 = score("u4")
    per_topic = {
        t: {
            "n": int((neu_te["topic"] == t).sum()),
            "flagged": float((s_te[neu_te["topic"] == t] < threshold).mean()),
        }
        for t in sorted(set(neu_te["topic"]))
    }
    out = {
        "declared_in": "configs/experiments/cycle4.yaml v3 B4prime_scope_detector",
        "selection": {"grid": grid, "chosen": {"candidate": kind, "C": c}},
        "threshold": threshold,
        "rules": rules,
        "passed": all(r["passed"] for r in rules.values()),
        "reported": {
            "served_mahalanobis_within_neu_test_auroc": _auroc(m_te[ins], m_te[off]),
            "u4_vs_uit_validation_auroc": _auroc(s_uit_dv, s_u4),
            "u4_caught": float((s_u4 < threshold).mean()),
            "served_mahalanobis_uit_validation_threshold_check": float(np.quantile(m_dv, 0.05)),
            "per_topic_neu_test": per_topic,
        },
        "sizes": {
            "train": len(tr_text),
            "train_off_topic": int((y_tr == 0).sum()),
            "neu_validation": len(neu_dv["x"]),
            "neu_test": len(neu_te["x"]),
            "u4": len(sets["u4"]),
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # The fitted detector, for a release step if the rule holds (models/ is git-ignored).
    import pickle

    dst = paths.MODELS / "scope"
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / f"b4prime_{kind}.pkl", "wb") as fh:
        pickle.dump({"model": model, "kind": kind, "C": c, "threshold": threshold}, fh)
    return out


__all__ = ["OUT", "data", "fit", "run", "served_features", "served_mahalanobis"]
