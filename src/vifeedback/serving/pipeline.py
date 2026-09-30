"""The service's per-request computation, shared by the API and the latency benchmark.

Raw text in: lowercase NFC (the training corpus is), diacritics restored on essentially unaccented
input (ADR-031), word segmentation, the ONNX graph, and, when the release declares one, the
out-of-scope score. Keeping it in one module means `study latency` times exactly what is served
(NEXT_PLAN v5, A1). Nothing here imports pandas: the runtime image does not ship it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback.preprocess.normalize import model_text


def _checked(path: Path, sha256: str) -> Path:
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError("sha256 does not match the manifest")
    return path


def load_restorer(model_dir: Path, spec: dict[str, Any]) -> Any:
    """The manifest's diacritic restorer, refused unless its SHA-256 matches."""
    from vifeedback.preprocess.diacritics import Restorer

    return Restorer.load(_checked(Path(model_dir) / spec["file"], spec["sha256"]))


def load_ood(model_dir: Path, spec: dict[str, Any], clf: Any) -> dict[str, Any]:
    """The manifest's out-of-scope parameters, refused unless the hash matches and the graph has
    the `features` output they were fitted on."""
    path = _checked(Path(model_dir) / spec["file"], spec["sha256"])
    if not clf.has_features:
        raise ValueError("the graph has no 'features' output")
    with np.load(path) as z:
        return {
            "means": z["means"],
            "precision": z["precision"],
            "threshold": float(z["threshold"]),
        }


def load_scope(model_dir: Path, spec: dict[str, Any]) -> Any:
    """The manifest's topic-aware scope detector (ADR-034), refused unless its SHA-256 matches."""
    from vifeedback.serving.scope_tfidf import TfidfScope

    return TfidfScope.load(_checked(Path(model_dir) / spec["file"], spec["sha256"]))


def prepare(
    texts: Sequence[str],
    restorer: Callable[[str], str] | None = None,
    segmenter: Callable[[list[str]], list[str]] | None = None,
) -> list[str]:
    """Raw text to the model's input: lowercase NFC, restore diacritics, segment."""
    normalized = [model_text(x) for x in texts]
    if restorer is not None:  # rewrites only essentially unaccented input
        normalized = [restorer(x) for x in normalized]
    return segmenter(normalized) if segmenter is not None else normalized


def _softmax(logits: np.ndarray, temperature: float | None) -> np.ndarray:
    z = logits / temperature if temperature else logits
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def score(
    clf: Any,
    model_input: list[str],
    ood: dict[str, Any] | None = None,
    scope: Any = None,
    temperature: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Label ids, probabilities, and the scope score (None without one); higher = more in scope.

    With `scope` (ADR-034) the score is the topic-aware detector's decision value, compared with
    `scope.threshold`. Otherwise, with `ood` (ADR-031), it is the negative Mahalanobis distance of
    the sentence feature to the nearest class mean, compared with `ood["threshold"]`.

    With `temperature` (ADR-041) the probabilities are softmax(logits / T), T fitted on validation;
    the labels are unchanged, only the confidence is calibrated.
    """
    if ood is None:
        if temperature:
            probs = _softmax(clf.logits(model_input), temperature)
            ids = probs.argmax(axis=1)
        else:
            ids, probs = clf.predict(model_input)
        return ids, probs, (scope.decision(model_input) if scope is not None else None)
    logits, feats = clf.logits_and_features(model_input)
    logits = logits / temperature if temperature else logits
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    dist = np.stack(
        [np.einsum("ij,jk,ik->i", feats - m, ood["precision"], feats - m) for m in ood["means"]],
        axis=1,
    )
    return probs.argmax(axis=1), probs, -dist.min(axis=1)
