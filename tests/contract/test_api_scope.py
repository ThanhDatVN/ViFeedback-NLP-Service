"""The topic-aware scope detector in the service (ADR-034): hash-checked, preferred over the
Mahalanobis score, and `in_scope` follows its threshold. No model: a stub classifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pyvi")

from fastapi.testclient import TestClient

from vifeedback.serving.scope_tfidf import TfidfScope


class _Stub:
    has_features = True

    def __init__(self, model_dir, threads=None, max_length=96, model_file=None):
        self.path = Path(model_dir) / "model.onnx"

    def predict(self, texts):
        return np.full(len(texts), 2), np.tile([0.1, 0.1, 0.8], (len(texts), 1))

    def logits_and_features(self, texts):  # pragma: no cover - the detector takes precedence
        raise AssertionError("with a scope detector the Mahalanobis path must not run")


def _model_dir(tmp_path, good_hash: bool = True, with_ood: bool = True):
    d = tmp_path / "serve" / "sentiment"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"stub")
    # "vàng" and "bóng_đá" push a text out of scope; everything else stays in.
    TfidfScope(
        terms=["giảng_viên", "vàng", "bóng_đá"],
        idf=np.ones(3),
        coef=np.array([1.0, -3.0, -3.0]),
        intercept=0.5,
        threshold=0.0,
    ).save(d / "scope.npz")
    sha = hashlib.sha256((d / "scope.npz").read_bytes()).hexdigest()
    manifest = {
        "task": "sentiment",
        "model_file": "model.onnx",
        "preprocessing": "seg_pyvi",
        "scope": {"file": "scope.npz", "sha256": sha if good_hash else "0" * 64},
    }
    if with_ood:  # a stale Mahalanobis entry must be ignored when a detector is present
        np.savez(d / "ood.npz", means=np.zeros((3, 2)), precision=np.eye(2), threshold=-4.0)
        manifest["ood"] = {
            "file": "ood.npz",
            "sha256": hashlib.sha256((d / "ood.npz").read_bytes()).hexdigest(),
        }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "serve"


@pytest.fixture
def app_with(monkeypatch):
    import vifeedback.inference.onnx_export as OX
    import vifeedback.serving.app as app_module

    monkeypatch.setattr(OX, "OnnxClassifier", _Stub)

    def make(model_dir):
        monkeypatch.setattr(app_module, "MODEL_DIR", model_dir)
        app_module._state["models"] = {}
        return TestClient(app_module.app)

    return make


def test_the_detector_decides_in_scope(tmp_path, app_with):
    with app_with(_model_dir(tmp_path)) as client:
        assert client.get("/readyz").status_code == 200
        preds = client.post(
            "/v1/classify", json={"texts": ["giảng viên nhiệt tình", "giá vàng tăng"]}
        ).json()["predictions"]
        assert preds[0]["in_scope"] is True and preds[1]["in_scope"] is False
        assert preds[0]["scope_score"] > 0 > preds[1]["scope_score"]


def test_a_mismatched_detector_makes_the_service_not_ready(tmp_path, app_with):
    with app_with(_model_dir(tmp_path, good_hash=False)) as client:
        r = client.get("/readyz")
        assert r.status_code == 503 and "scope detector" in r.json()["detail"]
