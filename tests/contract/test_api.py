"""API contract tests — the shape external callers depend on.

These run **without a model**, deliberately. A service that cannot start, report its own
unreadiness, and reject malformed input before any artifact exists is broken in a way no accuracy
metric would reveal. Model-dependent behaviour is covered separately by the golden-prediction tests.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from vifeedback.serving.app import app  # noqa: E402
from vifeedback.serving.schemas import MAX_BATCH, MAX_CHARS, ClassifyRequest  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestProbes:
    def test_healthz_is_liveness_only(self, client) -> None:
        """Up is not the same as able to serve; /healthz must answer the first question only."""
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readyz_is_false_without_a_model(self, client) -> None:
        """The important one. A readiness probe that greens before the model loads makes an
        orchestrator route traffic to a process that can only fail."""
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["ready"] is False
        assert body["models_loaded"] == []
        assert "no ONNX artifact" in (body["detail"] or "")

    def test_version_reports_the_serving_contract(self, client) -> None:
        body = client.get("/version").json()
        assert body["runtime"] == "onnxruntime"
        assert body["max_length"] == 96  # Gate G0 decision
        assert body["service_version"]


class TestClassifyWithoutAModel:
    def test_returns_503_not_500(self, client) -> None:
        """A missing model is unavailability, not a server fault. The status code is the difference
        between a load balancer retrying elsewhere and an alert firing."""
        r = client.post("/v1/classify", json={"texts": ["giảng viên nhiệt tình ."]})
        assert r.status_code == 503
        assert "not loaded" in r.json()["detail"]


class TestInputValidation:
    @pytest.mark.parametrize(
        ("payload", "why"),
        [
            ({"texts": []}, "empty batch"),
            ({"texts": ["   "]}, "whitespace-only text"),
            ({"texts": [""]}, "empty string"),
            ({"texts": ["ok"], "task": "emotion"}, "unknown task"),
            ({"texts": ["x"] * (MAX_BATCH + 1)}, "batch over the limit"),
            ({"texts": ["a" * (MAX_CHARS + 1)]}, "text over the character limit"),
            ({"nottexts": ["x"]}, "missing required field"),
        ],
    )
    def test_rejected_at_the_edge(self, client, payload, why) -> None:
        """Rejected by the schema, before any model work. Each of these is either a garbage-in
        prediction or a denial-of-service surface."""
        assert client.post("/v1/classify", json=payload).status_code == 422, why

    def test_valid_payload_passes_validation(self) -> None:
        req = ClassifyRequest(texts=["giảng viên nhiệt tình ."], task="topic")
        assert req.task == "topic"
        assert req.return_probabilities is True

    def test_malformed_json_is_rejected(self, client) -> None:
        r = client.post(
            "/v1/classify", content=b"{not json", headers={"content-type": "application/json"}
        )
        assert r.status_code == 422


class TestObservability:
    def test_metrics_is_prometheus_exposition(self, client) -> None:
        body = client.get("/metrics").text
        assert "# HELP vifeedback_ready" in body
        assert "# TYPE vifeedback_ready gauge" in body
        assert "vifeedback_ready 0" in body  # no model loaded
        for line in body.splitlines():
            if line and not line.startswith("#"):
                assert len(line.split()) >= 2, f"malformed metric line: {line!r}"

    def test_request_id_is_echoed_and_generated(self, client) -> None:
        r = client.get("/healthz", headers={"x-request-id": "abc123"})
        assert r.headers["x-request-id"] == "abc123"
        assert client.get("/healthz").headers["x-request-id"]

    def test_response_time_header_is_present(self, client) -> None:
        assert float(client.get("/healthz").headers["x-response-time-ms"]) >= 0


class TestOpenAPI:
    def test_schema_is_generated_and_documents_every_route(self, client) -> None:
        spec = client.get("/openapi.json").json()
        for path in ("/healthz", "/readyz", "/version", "/v1/classify", "/metrics"):
            assert path in spec["paths"], f"{path} missing from the OpenAPI schema"

    def test_classify_response_shape_is_pinned(self, client) -> None:
        """The response shape is the public contract; changing it breaks every caller."""
        props = client.get("/openapi.json").json()["components"]["schemas"]["ClassifyResponse"][
            "properties"
        ]
        assert set(props) >= {"predictions", "task", "model_version", "latency_ms"}
