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
def client(tmp_path_factory):
    """An app pointed at an EMPTY model directory.

    Not the default `models/serve/`: once a real artifact is released there, "without a model"
    stops being true and these tests would assert against the working copy's state instead of the
    code. Found exactly that way, the first time an FP32 release ran on the reference machine.
    """
    import vifeedback.serving.app as app_module

    empty = tmp_path_factory.mktemp("no_models")
    original = app_module.MODEL_DIR
    app_module.MODEL_DIR = empty
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app_module.MODEL_DIR = original


class TestProbes:
    def test_healthz_is_liveness_only(self, client) -> None:
        """Up is not the same as able to serve; /healthz must answer the first question only."""
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_readyz_is_false_without_a_model(self, client) -> None:
        """The important one. A readiness probe that greens before the model loads makes an
        orchestrator route traffic to a process that can only fail.

        HTTP 503, not 200 with `ready: false` (review R5): a conventional probe reads the status
        code only."""
        r = client.get("/readyz")
        assert r.status_code == 503
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

    @pytest.mark.parametrize("given", ['a"},"level":"ERROR', "x" * 65, "a b", "id\\x00"])
    def test_request_id_that_could_forge_a_log_line_is_replaced(self, client, given) -> None:
        """The id is written into a JSON log line; quotes, spaces or a long value are not echoed."""
        rid = client.get("/healthz", headers={"x-request-id": given}).headers["x-request-id"]
        assert rid != given and len(rid) <= 64 and rid.replace("-", "").isalnum()


class TestAbuse:
    """Cheap requests must not buy expensive work."""

    def test_oversized_body_is_refused_before_parsing(self, client) -> None:
        from vifeedback.serving.app import MAX_BODY_BYTES

        r = client.post(
            "/v1/classify",
            content=b"x" * (MAX_BODY_BYTES + 1),
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 413

    def test_oversized_chunked_body_is_refused(self, client) -> None:
        from vifeedback.serving.app import MAX_BODY_BYTES

        def chunks():
            for _ in range(3):
                yield b"x" * (MAX_BODY_BYTES // 2)

        r = client.post(
            "/v1/classify", content=chunks(), headers={"content-type": "application/json"}
        )
        assert r.status_code == 413

    def test_body_within_the_limit_reaches_validation(self, client) -> None:
        r = client.post("/v1/classify", json={"texts": ["a" * 2000] * 64})
        assert r.status_code == 503  # validated, then refused for the missing model, not 413

    def test_version_does_not_capture_the_environment_per_call(self, client, monkeypatch) -> None:
        from vifeedback import env

        def boom(*a, **k):
            raise AssertionError("env.capture ran inside a request")

        monkeypatch.setattr(env, "capture", boom)
        assert client.get("/version").status_code == 200

    def test_readyz_does_not_reveal_the_model_path(self, client) -> None:
        import vifeedback.serving.app as A

        assert str(A.MODEL_DIR) not in (client.get("/readyz").json()["detail"] or "")


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


class TestReadinessSemantics:
    """Review R5 and R6, exercised on the module state directly."""

    def test_missing_segmenter_makes_a_loaded_model_not_ready(self, monkeypatch) -> None:
        import vifeedback.serving.app as A

        monkeypatch.setattr(A, "REQUIRED_TASKS", ("sentiment",))
        monkeypatch.setitem(A._state, "models", {"sentiment": object()})
        monkeypatch.setitem(A._state, "needs_segmenter", True)
        monkeypatch.setitem(A._state, "segmenter", None)
        monkeypatch.setitem(
            A._state, "segmenter_error", "segmenter 'pyvi' unavailable: ImportError"
        )
        assert "pyvi" in A._not_ready_reason()

    def test_every_required_task_must_be_loaded(self, monkeypatch) -> None:
        import vifeedback.serving.app as A

        monkeypatch.setattr(A, "REQUIRED_TASKS", ("sentiment", "topic"))
        monkeypatch.setitem(A._state, "models", {"sentiment": object()})
        monkeypatch.setitem(A._state, "needs_segmenter", False)
        assert "topic" in A._not_ready_reason()

    def test_latency_buffer_is_bounded(self) -> None:
        import vifeedback.serving.app as A

        assert A._latencies.maxlen == 10_000


def test_backend_mapping_agrees_with_the_variant_table() -> None:
    """The app maps variants to segmenters without importing pandas; it must not drift."""
    from vifeedback.preprocess.variants import VARIANTS
    from vifeedback.serving.app import _backend_for

    for name, (backend, _, _) in VARIANTS.items():
        assert _backend_for(name) == backend, name
