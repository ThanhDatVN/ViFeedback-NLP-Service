"""The service's diacritic restorer (ADR-031): used only when the manifest's hash matches.

No model is needed: the ONNX classifier is replaced by a stub that records what it was given.
"""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("pyvi")

from fastapi.testclient import TestClient

from vifeedback.preprocess.diacritics import Restorer

TRAIN = ["thầy dạy rất dễ hiểu"] * 5 + ["cô dạy rất hay"] * 5


class _StubClassifier:
    seen: ClassVar[list[str]] = []

    def __init__(self, model_dir, threads=None, max_length=96, model_file=None):
        from pathlib import Path

        self.path = Path(model_dir) / "model.onnx"

    def predict(self, texts):
        _StubClassifier.seen = list(texts)
        return np.full(len(texts), 2), np.tile([0.1, 0.1, 0.8], (len(texts), 1))


def _model_dir(tmp_path, good_hash: bool):
    d = tmp_path / "serve" / "sentiment"
    d.mkdir(parents=True)
    (d / "model.onnx").write_bytes(b"stub")
    r = Restorer.fit(TRAIN, max_share_restored=0.0)
    r.save(d / "restorer.json")
    sha = hashlib.sha256((d / "restorer.json").read_bytes()).hexdigest()
    manifest = {
        "task": "sentiment",
        "model_file": "model.onnx",
        "preprocessing": "seg_pyvi",
        "max_length": 96,
        "restorer": {"file": "restorer.json", "sha256": sha if good_hash else "0" * 64},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path / "serve"


@pytest.fixture
def app_with(monkeypatch):
    import vifeedback.inference.onnx_export as OX
    import vifeedback.serving.app as app_module

    monkeypatch.setattr(OX, "OnnxClassifier", _StubClassifier)

    def make(model_dir):
        monkeypatch.setattr(app_module, "MODEL_DIR", model_dir)
        app_module._state["models"] = {}
        return TestClient(app_module.app)

    return make


def test_matching_restorer_is_used_for_unaccented_input(tmp_path, app_with):
    with app_with(_model_dir(tmp_path, good_hash=True)) as client:
        assert client.get("/readyz").status_code == 200
        client.post("/v1/classify", json={"texts": ["Thay day rat de hieu"]})
        assert "thầy" in _StubClassifier.seen[0] and "dạy" in _StubClassifier.seen[0]
        client.post("/v1/classify", json={"texts": ["thầy dạy rất dễ hiểu"]})
        assert "thầy" in _StubClassifier.seen[0]  # accented input passes through


def test_mismatched_restorer_makes_the_service_not_ready(tmp_path, app_with):
    with app_with(_model_dir(tmp_path, good_hash=False)) as client:
        r = client.get("/readyz")
        assert r.status_code == 503
        assert "restorer" in r.json()["detail"]
