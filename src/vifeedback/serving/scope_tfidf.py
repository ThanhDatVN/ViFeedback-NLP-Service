"""The topic-aware scope detector at serving time, without scikit-learn (cycle4.yaml v3 B4', ADR-034).

B4' chose logistic regression on TF-IDF word unigrams and bigrams. The runtime image has no
scikit-learn, so this re-implements `TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True,
min_df=2)` followed by `LogisticRegression.decision_function` on the fitted vocabulary:

lowercase -> tokens by the fitted token pattern -> unigrams then bigrams (space-joined) -> counts of
known terms -> 1 + log(count) -> times idf -> l2 normalisation -> dot with the coefficients + the
intercept. The export step (`serve add-scope`) checks it equal to scikit-learn on every evaluation
text before it is released.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np


class TfidfScope:
    """Decision value, higher = more in scope; `in_scope` is `decision >= threshold`."""

    def __init__(
        self,
        terms: list[str],
        idf: np.ndarray,
        coef: np.ndarray,
        intercept: float,
        threshold: float,
        token_pattern: str = r"(?u)\b\w\w+\b",
        ngram_range: tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.index = {t: i for i, t in enumerate(terms)}
        self.idf = np.asarray(idf, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.threshold = float(threshold)
        self.pattern = re.compile(token_pattern)
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.lowercase = lowercase

    @classmethod
    def load(cls, path: Path) -> TfidfScope:
        with np.load(path, allow_pickle=False) as z:
            return cls(
                terms=[str(t) for t in z["terms"]],
                idf=z["idf"],
                coef=z["coef"],
                intercept=float(z["intercept"]),
                threshold=float(z["threshold"]),
                token_pattern=str(z["token_pattern"]),
                ngram_range=(int(z["ngram_min"]), int(z["ngram_max"])),
                sublinear_tf=bool(z["sublinear_tf"]),
                lowercase=bool(z["lowercase"]),
            )

    def save(self, path: Path) -> None:
        terms = sorted(self.index, key=self.index.__getitem__)
        np.savez(
            path,
            terms=np.array(terms),
            idf=self.idf,
            coef=self.coef,
            intercept=np.float64(self.intercept),
            threshold=np.float64(self.threshold),
            token_pattern=np.array(self.pattern.pattern),
            ngram_min=np.int64(self.ngram_range[0]),
            ngram_max=np.int64(self.ngram_range[1]),
            sublinear_tf=np.bool_(self.sublinear_tf),
            lowercase=np.bool_(self.lowercase),
        )

    def _terms(self, text: str) -> list[str]:
        tokens = self.pattern.findall(text.lower() if self.lowercase else text)
        lo, hi = self.ngram_range
        out: list[str] = []
        for n in range(lo, min(hi, len(tokens)) + 1):
            out.extend(" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
        return out

    def decision(self, texts: list[str]) -> np.ndarray:
        scores = np.empty(len(texts))
        for k, text in enumerate(texts):
            counts = Counter(self.index[t] for t in self._terms(text) if t in self.index)
            if not counts:
                scores[k] = self.intercept
                continue
            idx = np.fromiter(counts.keys(), dtype=np.int64)
            tf = np.fromiter(counts.values(), dtype=np.float64)
            if self.sublinear_tf:
                tf = 1.0 + np.log(tf)
            v = tf * self.idf[idx]
            norm = np.sqrt(np.dot(v, v))
            scores[k] = (np.dot(v, self.coef[idx]) / norm if norm > 0 else 0.0) + self.intercept
        return scores

    @classmethod
    def from_sklearn(cls, pipeline, threshold: float) -> TfidfScope:
        """From the fitted `make_pipeline(TfidfVectorizer, LogisticRegression)` of B4'."""
        vec, lr = pipeline.steps[0][1], pipeline.steps[-1][1]
        if vec.analyzer != "word" or vec.norm != "l2" or not vec.use_idf or vec.stop_words:
            raise ValueError(
                "only word n-grams with l2-normalised idf and no stop words are served"
            )
        terms = sorted(vec.vocabulary_, key=vec.vocabulary_.__getitem__)
        return cls(
            terms=terms,
            idf=vec.idf_,
            coef=lr.coef_[0],
            intercept=float(lr.intercept_[0]),
            threshold=threshold,
            token_pattern=vec.token_pattern,
            ngram_range=tuple(vec.ngram_range),
            sublinear_tf=bool(vec.sublinear_tf),
            lowercase=bool(vec.lowercase),
        )
