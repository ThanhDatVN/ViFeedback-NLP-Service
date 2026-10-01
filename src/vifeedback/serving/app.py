"""FastAPI service — Gate G7.

Serves the ONNX artifact, not PyTorch: the runtime image needs `onnxruntime` only, which is what
keeps it inside the 700 MB target. Preprocessing is `pyvi` (ADR-012) — segmentation is worth
+0.023 macro-F1 and pyvi delivers it for 0.31 ms p95 with no JVM. The segmenter actually
used is the one named in the artifact's manifest.

Design choices worth stating:

* `/readyz` reports false until a model is genuinely loaded, so an orchestrator never routes to a
  process that can only fail;
* per-class prediction counters are exported, because the cheapest production drift signal is the
  output distribution moving away from the 46/4/50 prior measured in the data card;
* the model version is in every response, so a prediction can always be traced to the artifact that
  produced it.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from vifeedback import __version__, paths
from vifeedback.constants import LABELS
from vifeedback.serving import pipeline
from vifeedback.serving.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse,
    Prediction,
    ReadyResponse,
    VersionResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"ts":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
log = logging.getLogger("vifeedback")

MODEL_DIR = Path(os.getenv("MODEL_DIR", str(paths.MODELS / "serve")))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "96"))
THREADS = int(os.getenv("ORT_THREADS", "0")) or None
# Tasks this deployment must serve. Readiness requires every one of them (review R5).
REQUIRED_TASKS = tuple(t for t in os.getenv("REQUIRED_TASKS", "sentiment").split(",") if t)
# The schema bounds a parsed request (64 texts x 2,000 characters, at most about 0.8 MB of JSON
# even with every character escaped); this bounds the bytes read before the JSON parser runs.
MAX_BODY_BYTES = int(os.getenv("MAX_BODY_BYTES", str(1 << 20)))
# A caller's request id is echoed and logged only if it is short and plain.
REQUEST_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")

_state: dict[str, Any] = {
    "models": {},
    "version": "unloaded",
    "segmenter": None,
    "needs_segmenter": False,
    "segmenter_error": None,
    "git_sha": None,
}
_counters: dict[str, int] = {}
# Bounded (review R6): percentiles only ever read the most recent window, so older entries were
# dead memory growing with every request. The total count is kept separately.
_latencies: deque[float] = deque(maxlen=10_000)
_request_count = 0


def _git_sha() -> str | None:
    """The served code's commit, resolved once: GIT_SHA from the build, else `git rev-parse`.

    /version used to capture the whole environment on every call: three git processes,
    nvidia-smi, a CPU probe and a hash of the source tree, a cheap unauthenticated request that
    bought expensive work."""
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
            ).stdout.strip()
            or None
        )
    except Exception:
        return None


def _load() -> None:
    """Load every task model found under MODEL_DIR/<task>/. Missing models leave /readyz false."""
    import json

    _state["git_sha"] = _git_sha()

    from vifeedback.inference.onnx_export import OnnxClassifier

    manifests: dict[str, dict[str, Any]] = {}
    for task in ("sentiment", "topic"):
        d = MODEL_DIR / task
        if (d / "manifest.json").exists() or any(d.glob("*.onnx")):
            try:
                if (d / "manifest.json").exists():
                    manifests[task] = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
                max_len = manifests.get(task, {}).get("max_length", MAX_LENGTH)
                _state["models"][task] = OnnxClassifier(d, THREADS, max_len)
                log.info(f"loaded {task} from {_state['models'][task].path.name}")
            except Exception as e:
                log.error(f"failed to load {task}: {type(e).__name__}: {e}")

    # An explicit VERSION file wins; otherwise the release manifests identify what is served
    # (checkpoint name and the first 12 hex digits of the served file's SHA-256).
    # Pre-model steps a release declares in its manifest (Cycle 3, ADR-031): a diacritic restorer
    # that must match its recorded SHA-256, or the service reports itself not ready.
    _state["restorers"] = {}
    _state["preprocessing_error"] = None
    for task, m in manifests.items():
        spec = m.get("restorer")
        if not spec:
            continue
        try:
            _state["restorers"][task] = pipeline.load_restorer(MODEL_DIR / task, spec)
            log.info(f"loaded {task} diacritic restorer ({spec['sha256'][:12]})")
        except Exception as e:
            _state["preprocessing_error"] = f"restorer for {task} unusable: {type(e).__name__}: {e}"
            log.error(_state["preprocessing_error"] + " — /readyz will report not ready")

    # The out-of-scope score (ADR-031): parameters from train features, hash-checked like the restorer.
    _state["ood"] = {}
    for task, m in manifests.items():
        spec = m.get("ood")
        if not spec:
            continue
        try:
            _state["ood"][task] = pipeline.load_ood(MODEL_DIR / task, spec, _state["models"][task])
            log.info(f"loaded {task} out-of-scope score ({spec['sha256'][:12]})")
        except Exception as e:
            _state["preprocessing_error"] = (
                f"out-of-scope score for {task} unusable: {type(e).__name__}: {e}"
            )
            log.error(_state["preprocessing_error"] + " — /readyz will report not ready")

    # The topic-aware scope detector (ADR-034); when a release declares it, it replaces the
    # Mahalanobis score behind `in_scope`.
    _state["scope"] = {}
    for task, m in manifests.items():
        spec = m.get("scope")
        if not spec:
            continue
        try:
            _state["scope"][task] = pipeline.load_scope(MODEL_DIR / task, spec)
            log.info(f"loaded {task} scope detector ({spec['sha256'][:12]})")
        except Exception as e:
            _state["preprocessing_error"] = (
                f"scope detector for {task} unusable: {type(e).__name__}: {e}"
            )
            log.error(_state["preprocessing_error"] + " — /readyz will report not ready")

    # Calibrated confidence (ADR-041): the release's temperature, fitted on validation.
    _state["temperature"] = {
        task: float(m["temperature"]["value"])
        for task, m in manifests.items()
        if m.get("temperature")
    }

    vf = MODEL_DIR / "VERSION"
    if vf.exists():
        _state["version"] = vf.read_text(encoding="utf-8").strip()
    elif manifests:
        _state["version"] = ", ".join(
            f"{t}={PureWindowsPath(str(m.get('checkpoint', '?'))).name}@{str(m.get('sha256', ''))[:12]}"
            for t, m in sorted(manifests.items())
        )
    else:
        _state["version"] = "unversioned"

    # The artifact's manifest says which preprocessing it was verified with (review R3);
    # SEGMENTER overrides it only when set explicitly.
    backend = os.getenv("SEGMENTER") or _backend_from_manifests(manifests) or "pyvi"
    _state["needs_segmenter"] = backend != "none"
    _state["segmenter_error"] = None
    try:
        from vifeedback.preprocess.segment import get_segmenter

        _state["segmenter"] = get_segmenter(backend)
    except Exception as e:
        # Not a silent fallback (review R5). A model verified on segmented input is not verified on
        # raw text, so the service reports itself not ready instead of serving unverified output.
        _state["segmenter"] = None
        _state["segmenter_error"] = f"segmenter '{backend}' unavailable: {type(e).__name__}"
        log.error(_state["segmenter_error"] + " — /readyz will report not ready")


def _backend_for(preprocessing: str) -> str:
    """Variant name → segmenter backend, without importing preprocess.variants.

    That module pulls in pandas, which the runtime image deliberately does not ship; importing it
    here crashed start-up whenever SEGMENTER was unset. Variant names encode their backend
    (`seg_pyvi`, `norm_seg_vncorenlp`, `raw`), so the mapping needs no table.
    """
    return preprocessing.rsplit("seg_", 1)[1] if "seg_" in preprocessing else "none"


def _backend_from_manifests(manifests: dict[str, dict[str, Any]]) -> str | None:
    backends = {
        _backend_for(m["preprocessing"]) for m in manifests.values() if m.get("preprocessing")
    }
    if len(backends) > 1:
        log.warning(f"task artifacts disagree on preprocessing {sorted(backends)}; using the first")
    return sorted(backends)[0] if backends else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load()
    yield
    _state["models"].clear()


class BodySizeLimit:
    """Refuse a request body over `max_bytes` with HTTP 413 before it is parsed.

    A declared Content-Length is checked up front (the server never delivers more than it
    declares). A chunked body is read up to the limit and replayed, so memory stays bounded.
    """

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def _reject(self, send: Any) -> None:
        body = f'{{"detail":"request body over {self.max_bytes} bytes"}}'.encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None:
            if not length.isdigit() or int(length) > self.max_bytes:
                await self._reject(send)
                return
            await self.app(scope, receive, send)
            return
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":  # the client went away
                return
            chunks.append(message.get("body", b""))
            size += len(chunks[-1])
            if size > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break
        replay = [{"type": "http.request", "body": b"".join(chunks), "more_body": False}]

        async def again() -> Any:
            return replay.pop() if replay else await receive()

        await self.app(scope, again, send)


app = FastAPI(
    title="ViFeedback",
    description="Vietnamese feedback sentiment and topic classification",
    version=__version__,
    lifespan=lifespan,
)
app.add_middleware(BodySizeLimit, max_bytes=MAX_BODY_BYTES)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    # Echoed in a header and written into a JSON log line: anything but a short plain id is
    # replaced, so a caller cannot forge log fields or flood the log.
    given = request.headers.get("x-request-id", "")
    rid = given if REQUEST_ID.fullmatch(given) else str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    response = await call_next(request)
    dt = (time.perf_counter() - t0) * 1000
    response.headers["x-request-id"] = rid
    response.headers["x-response-time-ms"] = f"{dt:.2f}"
    if request.url.path.startswith("/v1"):
        global _request_count
        _request_count += 1
        _latencies.append(dt)
        log.info(f"{rid} {request.method} {request.url.path} {response.status_code} {dt:.1f}ms")
    return response


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness: the process is up. Says nothing about whether it can serve."""
    return HealthResponse(status="ok")


def _not_ready_reason() -> str | None:
    missing = [t for t in REQUIRED_TASKS if t not in _state["models"]]
    if missing:
        return f"no ONNX artifact for required task(s) {missing}"  # the path stays in the log
    if _state["needs_segmenter"] and _state["segmenter"] is None:
        return _state["segmenter_error"] or "required segmenter is not loaded"
    if _state.get("preprocessing_error"):
        return str(_state["preprocessing_error"])
    return None


@app.get("/readyz", response_model=ReadyResponse)
def readyz() -> JSONResponse:
    """Readiness: every required task loaded AND its preprocessing available (review R5).

    Not ready is HTTP 503, so a conventional probe that only reads the status code gets it right.
    """
    reason = _not_ready_reason()
    body = ReadyResponse(
        ready=reason is None, models_loaded=sorted(_state["models"]), detail=reason
    )
    return JSONResponse(status_code=200 if reason is None else 503, content=body.model_dump())


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    return VersionResponse(
        service_version=__version__,
        model_version=_state["version"],
        git_sha=_state.get("git_sha"),
        runtime="onnxruntime",
        max_length=MAX_LENGTH,
    )


@app.post("/v1/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest) -> ClassifyResponse:
    clf = _state["models"].get(req.task)
    if clf is None:
        raise HTTPException(503, f"model for task '{req.task}' is not loaded")
    if _state["needs_segmenter"] and _state["segmenter"] is None:
        raise HTTPException(503, _state["segmenter_error"] or "required segmenter is not loaded")

    t0 = time.perf_counter()
    texts = list(req.texts)
    # Lowercase NFC, restore diacritics on unaccented input (ADR-031), segment, score: the same
    # function `study latency` times.
    model_input = pipeline.prepare(
        texts, _state.get("restorers", {}).get(req.task), _state["segmenter"]
    )
    detector = _state.get("scope", {}).get(req.task)
    ood = None if detector is not None else _state.get("ood", {}).get(req.task)
    temperature = _state.get("temperature", {}).get(req.task)
    ids, probs, scope = pipeline.score(clf, model_input, ood, detector, temperature)
    threshold = (
        detector.threshold if detector is not None else ood["threshold"] if ood is not None else 0.0
    )
    names = [LABELS[req.task][i] for i in sorted(LABELS[req.task])]

    preds = []
    for k, (text, i, p) in enumerate(zip(texts, ids, probs, strict=True)):
        label = names[int(i)]
        _counters[f"{req.task}:{label}"] = _counters.get(f"{req.task}:{label}", 0) + 1
        preds.append(
            Prediction(
                text=text,
                label=label,
                label_id=int(i),
                confidence=round(float(p[int(i)]), 4),
                probabilities=(
                    {n: round(float(v), 4) for n, v in zip(names, p, strict=True)}
                    if req.return_probabilities
                    else None
                ),
                in_scope=None if scope is None else bool(scope[k] >= threshold),
                scope_score=None if scope is None else round(float(scope[k]), 3),
            )
        )

    return ClassifyResponse(
        predictions=preds,
        task=req.task,
        model_version=_state["version"],
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus exposition.

    The per-class counters are the point: the cheapest production drift signal is the output
    distribution drifting from the 46/4/50 sentiment prior recorded in the data card.
    """
    lines = [
        "# HELP vifeedback_predictions_total Predictions by task and label.",
        "# TYPE vifeedback_predictions_total counter",
    ]
    for key, n in sorted(_counters.items()):
        task, label = key.split(":", 1)
        lines.append(f'vifeedback_predictions_total{{task="{task}",label="{label}"}} {n}')

    lines += [
        "# HELP vifeedback_requests_total Requests served on /v1.",
        "# TYPE vifeedback_requests_total counter",
        f"vifeedback_requests_total {_request_count}",
        "# HELP vifeedback_ready Whether a model is loaded.",
        "# TYPE vifeedback_ready gauge",
        f"vifeedback_ready {int(_not_ready_reason() is None)}",
    ]
    if _latencies:
        import numpy as np

        a = np.array(_latencies)
        for q in (50, 95, 99):
            lines += [
                f"# TYPE vifeedback_latency_p{q}_ms gauge",
                f"vifeedback_latency_p{q}_ms {np.percentile(a, q):.3f}",
            ]
    return "\n".join(lines) + "\n"
