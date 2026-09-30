"""API tests against a real released artifact — review R4.

The contract tests in test_api.py run without a model by design. That leaves the path that matters
most, a request answered by a real model, untested. These tests run it whenever a release exists
under `models/serve/<task>/` (see `vifeedback serve export`) and skip otherwise, as in CI, where no
540 MB artifact is available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("onnxruntime")

from fastapi.testclient import TestClient

from vifeedback import paths

# SERVE_DIR_UNDER_TEST points the suite at another release folder, e.g. a candidate (models/candidate).
SERVE = Path(os.getenv("SERVE_DIR_UNDER_TEST", str(paths.MODELS / "serve")))

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
        # People capitalize; the corpus never does. The service lowercases first (cycle3.yaml).
        ("Giảng viên nhiệt tình", "positive"),
        ("THẦY GIẢNG BÀI KHÔNG DỄ HIỂU", "negative"),
        # Unaccented input goes through the restorer when the release declares one (ADR-031).
        ("thay giang bai rat de hieu", "positive"),
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


def test_version_names_the_released_checkpoint_and_hash(client) -> None:
    """/version identifies the served model from its manifest, not "unversioned" (found by docker-e2e)."""
    import json
    from pathlib import Path

    manifest = json.loads((SERVE / "sentiment" / "manifest.json").read_text(encoding="utf-8"))
    version = client.get("/version").json()["model_version"]
    assert Path(manifest["checkpoint"]).name in version
    assert manifest["sha256"][:12] in version


def test_out_of_scope_score_when_the_release_declares_one(client) -> None:
    """ADR-031/034: every prediction carries the score, and off-topic text scores below course
    feedback (with either the Mahalanobis score or the topic-aware detector)."""
    import json

    manifest = json.loads((SERVE / "sentiment" / "manifest.json").read_text(encoding="utf-8"))
    if not (manifest.get("ood") or manifest.get("scope")):
        pytest.skip("this release has no scope score (serve add-ood / add-scope)")
    feedback = ["giảng viên nhiệt tình", "thầy giảng bài dễ hiểu", "phòng học nóng"]
    off_topic = ["giá vàng hôm nay tăng mạnh", "đội tuyển bóng đá thắng hai không"]
    preds = client.post("/v1/classify", json={"texts": feedback + off_topic}).json()["predictions"]
    assert all(isinstance(p["in_scope"], bool) for p in preds)
    scores = [p["scope_score"] for p in preds]
    assert max(scores[len(feedback) :]) < min(scores[: len(feedback)])
    assert all(p["in_scope"] for p in preds[: len(feedback)])
