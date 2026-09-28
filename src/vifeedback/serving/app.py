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
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from vifeedback import __version__, paths
from vifeedback.constants import LABELS
from vifeedback.preprocess.normalize import model_text
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

_state: dict[str, Any] = {
    "models": {},
    "version": "unloaded",
    "segmenter": None,
    "needs_segmenter": False,
    "segmenter_error": None,
}
_counters: dict[str, int] = {}
# Bounded (review R6): percentiles only ever read the most recent window, so older entries were
# dead memory growing with every request. The total count is kept separately.
_latencies: deque[float] = deque(maxlen=10_000)
_request_count = 0


def _load() -> None:
    """Load every task model found under MODEL_DIR/<task>/. Missing models leave /readyz false."""
    import json

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
        path = MODEL_DIR / task / spec["file"]
        try:
            import hashlib

            from vifeedback.preprocess.diacritics import Restorer

            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError("sha256 does not match the manifest")
            _state["restorers"][task] = Restorer.load(path)
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
            import hashlib

            path = MODEL_DIR / task / spec["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                raise ValueError("sha256 does not match the manifest")
            if not _state["models"][task].has_features:
                raise ValueError("the graph has no 'features' output")
            with np.load(path) as z:
                _state["ood"][task] = {
                    "means": z["means"],
                    "precision": z["precision"],
                    "threshold": float(z["threshold"]),
                }
            log.info(f"loaded {task} out-of-scope score ({spec['sha256'][:12]})")
        except Exception as e:
            _state["preprocessing_error"] = (
                f"out-of-scope score for {task} unusable: {type(e).__name__}: {e}"
            )
            log.error(_state["preprocessing_error"] + " — /readyz will report not ready")

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


app = FastAPI(
    title="ViFeedback",
    description="Vietnamese feedback sentiment and topic classification",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def request_id_and_timing(request: Request, call_next):
    rid = request.headers.get("x-request-id", str(uuid.uuid4())[:8])
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
        return f"no ONNX artifact for required task(s) {missing} under {MODEL_DIR}"
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
    from vifeedback import env

    return VersionResponse(
        service_version=__version__,
        model_version=_state["version"],
        git_sha=(env.capture().get("git") or {}).get("sha"),
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
    seg = _state["segmenter"]
    normalized = [model_text(x) for x in texts]  # the training corpus is lowercase NFC
    restorer = _state.get("restorers", {}).get(req.task)
    if restorer is not None:  # rewrites only essentially unaccented input (ADR-031)
        normalized = [restorer(x) for x in normalized]
    model_input = seg(normalized) if seg is not None else normalized

    ood = _state.get("ood", {}).get(req.task)
    if ood is not None:
        logits, feats = clf.logits_and_features(model_input)
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        ids = probs.argmax(axis=1)
        dist = np.stack(
            [
                np.einsum("ij,jk,ik->i", feats - m, ood["precision"], feats - m)
                for m in ood["means"]
            ],
            axis=1,
        )
        scope = -dist.min(axis=1)
    else:
        ids, probs = clf.predict(model_input)
        scope = None
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
                in_scope=None if scope is None else bool(scope[k] >= ood["threshold"]),
                scope_score=None if scope is None else round(float(scope[k]), 2),
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
