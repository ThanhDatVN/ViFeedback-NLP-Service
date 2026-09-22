"""The classical baseline ladder — Phase 1.

Each rung exists to make the next number interpretable (docs/ROADMAP.md Phase 1):

* B0 / B0b establish the macro-F1 *floor*, so "0.65" can be read as "barely above trivial".
* B1 is the conventional word-level TF-IDF baseline everyone reports.
* B2 uses character n-grams, which for Vietnamese user text survive typos and missing diacritics
  that destroy word features — the hypothesis being that B2 > B1 on macro-F1.
* B5 isolates the *imbalance* effect from the *model* effect, which is the distinction that decides
  whether Phase 4 Tier A is worth a week.

Selection is on the official dev split throughout, never on test, via `PredefinedSplit`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


@dataclass(frozen=True)
class BaselineSpec:
    """One rung of the ladder. `search_space` is swept with dev as the validation fold."""

    key: str
    label: str
    features: str  # "none" | "word" | "char" | "union"
    classifier: str  # "majority" | "stratified" | "logreg" | "linearsvc"
    class_weight: str | None = None
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    supports_proba: bool = True


_WORD_SPACE = {
    "features__ngram_range": [(1, 1), (1, 2)],
    "features__min_df": [1, 2, 3],
    "features__sublinear_tf": [True, False],
    "clf__C": [0.5, 1.0, 3.0, 10.0],
}
_CHAR_SPACE = {
    "features__ngram_range": [(2, 4), (3, 5), (2, 5)],
    "features__min_df": [2, 3],
    "features__sublinear_tf": [True, False],
    "clf__C": [0.5, 1.0, 3.0, 10.0],
}

LADDER: tuple[BaselineSpec, ...] = (
    BaselineSpec("b0", "B0 majority class", "none", "majority", supports_proba=True),
    BaselineSpec("b0b", "B0b stratified random", "none", "stratified", supports_proba=True),
    BaselineSpec("b1", "B1 TF-IDF word 1-2g + LR", "word", "logreg", search_space=_WORD_SPACE),
    BaselineSpec("b2", "B2 TF-IDF char_wb 3-5g + LR", "char", "logreg", search_space=_CHAR_SPACE),
    BaselineSpec(
        "b3",
        "B3 TF-IDF word+char union + LR",
        "union",
        "logreg",
        search_space={"clf__C": [0.5, 1.0, 3.0, 10.0]},
    ),
    BaselineSpec(
        "b4",
        "B4 TF-IDF union + LinearSVC",
        "union",
        "linearsvc",
        search_space={"clf__C": [0.1, 0.5, 1.0, 3.0]},
        supports_proba=False,
    ),
    BaselineSpec(
        "b5",
        "B5 TF-IDF union + LR, class-weighted",
        "union",
        "logreg",
        class_weight="balanced",
        search_space={"clf__C": [0.5, 1.0, 3.0, 10.0]},
    ),
)

LADDER_BY_KEY = {s.key: s for s in LADDER}


def _vectorizer(kind: str):
    if kind == "word":
        return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    if kind == "char":
        # char_wb keeps n-grams inside word boundaries, which suits whitespace-delimited
        # Vietnamese syllables far better than unconstrained `char`.
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True)
    if kind == "union":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown feature kind {kind!r}")


def build(spec: BaselineSpec, seed: int = 42) -> Pipeline:
    if spec.classifier in ("majority", "stratified"):
        strategy = "most_frequent" if spec.classifier == "majority" else "stratified"
        return Pipeline([("clf", DummyClassifier(strategy=strategy, random_state=seed))])

    clf: Any
    if spec.classifier == "logreg":
        clf = LogisticRegression(
            max_iter=2000, C=1.0, class_weight=spec.class_weight, random_state=seed, n_jobs=-1
        )
    elif spec.classifier == "linearsvc":
        clf = LinearSVC(C=1.0, class_weight=spec.class_weight, random_state=seed, max_iter=5000)
    else:
        raise ValueError(f"unknown classifier {spec.classifier!r}")

    return Pipeline([("features", _vectorizer(spec.features)), ("clf", clf)])


def fit_with_dev_selection(
    spec: BaselineSpec,
    x_train,
    y_train,
    x_dev,
    y_dev,
    task: str,
    seed: int = 42,
    n_iter: int = 12,
) -> dict[str, Any]:
    """Fit on train, select hyperparameters on the official dev split.

    `PredefinedSplit` is what enforces the protocol: train rows carry fold -1 (never validated), dev
    rows carry fold 0. sklearn therefore fits on train only and scores on dev only — no CV folds that
    would quietly mix the two.
    """
    from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

    t0 = time.perf_counter()
    pipe = build(spec, seed=seed)

    if not spec.search_space:
        pipe.fit(x_train, y_train)
        return {"model": pipe, "best_params": {}, "fit_seconds": time.perf_counter() - t0}

    x_all = list(x_train) + list(x_dev)
    y_all = np.concatenate([np.asarray(y_train), np.asarray(y_dev)])
    fold = np.concatenate([np.full(len(x_train), -1), np.zeros(len(x_dev))])

    # Scored on macro-F1 because that is what the project optimizes; scoring on accuracy here would
    # select the hyperparameters that best ignore the neutral class.
    search = RandomizedSearchCV(
        pipe,
        spec.search_space,
        n_iter=min(n_iter, _space_size(spec.search_space)),
        scoring="f1_macro",
        cv=PredefinedSplit(fold),
        random_state=seed,
        refit=False,
        n_jobs=1,
    )
    search.fit(x_all, y_all)

    best = build(spec, seed=seed)
    best.set_params(**search.best_params_)
    best.fit(x_train, y_train)

    return {
        "model": best,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "dev_macro_f1_search": float(search.best_score_),
        "fit_seconds": time.perf_counter() - t0,
    }


def _space_size(space: dict[str, list[Any]]) -> int:
    n = 1
    for v in space.values():
        n *= len(v)
    return n


def predict_proba(model: Pipeline, x, k: int) -> np.ndarray | None:
    """Class probabilities, or None for models without them (LinearSVC)."""
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return model.predict_proba(x)
    return None


# --- Decision-threshold tuning ------------------------------------------------------------------


def tune_class_priors(
    prob_dev: np.ndarray,
    y_dev: np.ndarray,
    k: int,
    seed: int = 42,
    n_random: int = 400,
    n_refine: int = 3,
) -> np.ndarray:
    """Per-class multiplicative weights `w` maximizing dev macro-F1 under `argmax(p * w)`.

    The multi-class generalization of threshold tuning, and a legitimate, nearly free macro-F1
    lever. Its diagnostic value is the point: if tuning `w` recovers most of the neutral-class F1,
    the baseline's failure was a *decision-rule* problem; if it does not, the failure is in the
    *representation*, and only a better model will fix it. Almost no public UIT-VSFC notebook draws
    that distinction.

    Random search then coordinate refinement — the objective is piecewise constant, so gradients are
    useless and a coarse-to-fine search is the right tool.
    """
    from vifeedback.evaluation.metrics import macro_f1

    rng = np.random.default_rng(seed)

    def score(w: np.ndarray) -> float:
        return macro_f1(y_dev, (prob_dev * w).argmax(axis=1), k)

    best_w = np.ones(k)
    best = score(best_w)

    # Log-uniform over [1/32, 32] per class: the minority class needs a large boost.
    for _ in range(n_random):
        w = np.exp(rng.uniform(np.log(1 / 32), np.log(32), size=k))
        w = w / w[0]
        s = score(w)
        if s > best:
            best, best_w = s, w

    for _ in range(n_refine):
        for c in range(k):
            grid = best_w[c] * np.exp(np.linspace(np.log(0.5), np.log(2.0), 25))
            for v in grid:
                w = best_w.copy()
                w[c] = v
                s = score(w)
                if s > best:
                    best, best_w = s, w

    return best_w


def crossfit_class_priors(
    prob: np.ndarray,
    y: np.ndarray,
    k: int,
    n_folds: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Out-of-fold predictions under per-fold prior tuning — the *honest* estimate.

    Tuning priors on a set and scoring on the same set is optimistically biased, and on UIT-VSFC the
    bias is not small: measured at **+0.012 to +0.036 macro-F1** across every baseline, and it
    accounts for the *entire* apparent gain on PhoBERT (ADR-015). The cause is sample size — dev holds
    73 neutral examples, which is too few to estimate a decision threshold that generalizes.

    Use this whenever a tuned-threshold number is reported. `tune_class_priors` remains the right
    function for *fitting* the priors that will actually be deployed; it is the wrong function for
    *estimating what they will be worth*.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(y)
    pred = np.empty_like(y)
    for train_idx, test_idx in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(
        prob, y
    ):
        w = tune_class_priors(prob[train_idx], y[train_idx], k, seed=seed)
        pred[test_idx] = (prob[test_idx] * w).argmax(axis=1)
    return pred


def prior_tuning_report(
    prob: np.ndarray, y: np.ndarray, k: int, seed: int = 42
) -> dict[str, float]:
    """Untuned, fit-on-eval (optimistic) and cross-fitted (honest) macro-F1, with the bias."""
    from vifeedback.evaluation.metrics import macro_f1

    untuned = macro_f1(y, prob.argmax(axis=1), k)
    w = tune_class_priors(prob, y, k, seed=seed)
    optimistic = macro_f1(y, (prob * w).argmax(axis=1), k)
    honest = macro_f1(y, crossfit_class_priors(prob, y, k, seed=seed), k)
    return {
        "untuned": untuned,
        "fit_on_eval": optimistic,
        "crossfitted": honest,
        "optimism_bias": optimistic - honest,
        "honest_gain": honest - untuned,
    }
