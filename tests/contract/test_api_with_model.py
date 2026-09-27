"""API tests against a real released artifact — review R4.

The contract tests in test_api.py run without a model by design. That leaves the path that matters
most, a request answered by a real model, untested. These tests run it whenever a release exists
under `models/serve/<task>/` (see `vifeedback serve export`) and skip otherwise, as in CI, where no
540 MB artifact is available.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient

from vifeedback import paths

SERVE = paths.MODELS / "serve"

pytestmark = pytest.mark.skipif(
    not (SERVE / "sentiment" / "manifest.json").exists(),
    reason="no released sentiment artifact under models/serve/ (run `vifeedback serve export`)",
)


@pytest.fixture(scope="module")
def client():
    import vifeedback.serving.app as app_module

    original = app_module.MODEL_DIR
    app_module.MODEL_DIR = SERVE
    try:
        with TestClient(app_module.app) as c:
            yield c
    finally:
        app_module.MODEL_DIR = original


def test_ready_once_a_model_is_loaded(client) -> None:
    body = client.get("/readyz").json()
    assert body["ready"] is True
    assert "sentiment" in body["models_loaded"]


def test_classify_returns_a_valid_distribution(client) -> None:
    r = client.post(
        "/v1/classify",
        json={
            "texts": ["giảng viên nhiệt tình .", "phòng học nóng ."],
            "return_probabilities": True,
        },
    )
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    for p in preds:
        assert p["label"] in {"negative", "neutral", "positive"}
        assert abs(sum(p["probabilities"].values()) - 1.0) < 1e-3
        assert max(p["probabilities"], key=p["probabilities"].get) == p["label"]


@pytest.mark.parametrize(
    ("text", "label"),
    [
        # From data/probes/negation_v1.csv, pos_to_neg: the released seed-42 model gets all 36.
        ("giảng viên nhiệt tình", "positive"),
        ("giảng viên không nhiệt tình", "negative"),
        ("thầy giảng bài dễ hiểu", "positive"),
        ("thầy giảng bài không dễ hiểu", "negative"),
    ],
)
def test_golden_predictions(client, text: str, label: str) -> None:
    """End-to-end: raw text in, served through the manifest's segmenter, label out."""
    r = client.post("/v1/classify", json={"texts": [text]})
    assert r.json()["predictions"][0]["label"] == label


def test_served_file_is_the_manifest_file(client) -> None:
    import json

    import vifeedback.serving.app as app_module

    manifest = json.loads((SERVE / "sentiment" / "manifest.json").read_text(encoding="utf-8"))
    assert app_module._state["models"]["sentiment"].path.name == manifest["model_file"]
