"""The service's out-of-scope score (ADR-031): hash-checked, and in_scope follows the threshold.

No model: a stub classifier returns logits and a feature per sentence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pyvi")

from fastapi.testclient import TestClient


class _StubWithFeatures:
    """'far' sentences get a feature far from every class mean."""

    has_features = True

    def __init__(self, model_dir, threads=None, max_length=96, model_file=None):
        self.path = Path(model_dir) / "model.onnx"

    def logits_and_features(self, texts):
        feats = np.array([[50.0, 50.0] if "far" in t else [0.1, 0.0] for t in texts])
        return np.tile([0.0, 0.0, 3.0], (len(texts), 1)), feats

    def predict(self, texts):  # pragma: no cover - not used when the score is present
        raise AssertionError("the out-of-scope path must use logits_and_features")


def _model_dir(tmp_path, good_hash: bool):
    d = tmp_path / "serve" / "sentiment"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"stub")
    np.savez(
        d / "ood.npz",
        means=np.zeros((3, 2), np.float32),
        precision=np.eye(2, dtype=np.float32),
        threshold=np.float32(-4.0),
    )
    sha = hashlib.sha256((d / "ood.npz").read_bytes()).hexdigest()
    manifest = {
        "task": "sentiment",
        "model_file": "model.onnx",
        "preprocessing": "seg_pyvi",
        "ood": {"file": "ood.npz", "sha256": sha if good_hash else "0" * 64},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "serve"


@pytest.fixture
def app_with(monkeypatch):
    import vifeedback.inference.onnx_export as OX
    import vifeedback.serving.app as app_module

    monkeypatch.setattr(OX, "OnnxClassifier", _StubWithFeatures)

    def make(model_dir):
        monkeypatch.setattr(app_module, "MODEL_DIR", model_dir)
        app_module._state["models"] = {}
        return TestClient(app_module.app)

    return make


def test_in_scope_follows_the_threshold(tmp_path, app_with):
    with app_with(_model_dir(tmp_path, good_hash=True)) as client:
        assert client.get("/readyz").status_code == 200
        preds = client.post(
            "/v1/classify", json={"texts": ["thầy dạy hay", "far away text"]}
        ).json()
        near, far = preds["predictions"]
        assert near["in_scope"] is True and far["in_scope"] is False
        assert near["scope_score"] > far["scope_score"]
        assert near["label"] == "positive"  # a label is always returned


def test_mismatched_score_file_makes_the_service_not_ready(tmp_path, app_with):
    with app_with(_model_dir(tmp_path, good_hash=False)) as client:
        r = client.get("/readyz")
        assert r.status_code == 503 and "out-of-scope" in r.json()["detail"]
