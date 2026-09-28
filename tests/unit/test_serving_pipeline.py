"""serving.pipeline: the per-request computation the API and `study latency` share (NEXT_PLAN v5, A1)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from vifeedback.serving import pipeline as SP


class _Clf:
    has_features = True

    def __init__(self):
        self.logits = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
        self.feats = np.array([[1.0, 0.0], [0.0, 5.0]])

    def logits_and_features(self, texts):
        return self.logits[: len(texts)], self.feats[: len(texts)]

    def predict(self, texts):
        return np.array([0, 2])[: len(texts)], np.full((len(texts), 3), 1 / 3)


def test_prepare_lowercases_restores_then_segments():
    seen = []

    def seg(xs):
        seen.extend(xs)
        return [x.replace(" ", "_") for x in xs]

    out = SP.prepare(["Thầy Dạy", "ok"], restorer=lambda x: x + "!", segmenter=seg)
    assert seen == ["thầy dạy!", "ok!"] and out == ["thầy_dạy!", "ok!"]


def test_score_without_ood_is_the_plain_prediction():
    ids, probs, scope = SP.score(_Clf(), ["a", "b"])
    assert list(ids) == [0, 2] and scope is None and probs.shape == (2, 3)


def test_score_with_ood_is_softmax_and_negative_nearest_mahalanobis():
    ood = {"means": np.array([[0.0, 0.0], [0.0, 4.0]]), "precision": np.eye(2), "threshold": -3.0}
    ids, probs, scope = SP.score(_Clf(), ["a", "b"], ood)
    assert list(ids) == [0, 2]
    assert np.allclose(probs.sum(axis=1), 1.0)
    # row 0: distances 1 and 17 -> -1; row 1: distances 25 and 1 -> -1
    assert np.allclose(scope, [-1.0, -1.0])


def test_loaders_refuse_a_file_that_does_not_match_its_hash(tmp_path):
    np.savez(tmp_path / "ood.npz", means=np.zeros((3, 2)), precision=np.eye(2), threshold=-1.0)
    good = hashlib.sha256((tmp_path / "ood.npz").read_bytes()).hexdigest()
    assert SP.load_ood(tmp_path, {"file": "ood.npz", "sha256": good}, _Clf())["threshold"] == -1.0
    with pytest.raises(ValueError, match="sha256"):
        SP.load_ood(tmp_path, {"file": "ood.npz", "sha256": "0" * 64}, _Clf())
    no_features = _Clf()
    no_features.has_features = False
    with pytest.raises(ValueError, match="features"):
        SP.load_ood(tmp_path, {"file": "ood.npz", "sha256": good}, no_features)
