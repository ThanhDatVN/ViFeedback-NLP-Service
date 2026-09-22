"""FastAPI service — Gate G7.

Serves the ONNX artifact, not PyTorch: the runtime image needs `onnxruntime` only, which is what
keeps it inside the 700 MB target. Preprocessing is `pyvi` (ADR-012) — segmentation is worth
+0.023 macro-F1 and pyvi delivers it for 0.31 ms p95 with no JVM.

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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from vifeedback import __version__, paths
from vifeedback.constants import LABELS
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

_state: dict[str, Any] = {"models": {}, "version": "unloaded", "segmenter": None}
_counters: dict[str, int] = {}
_latencies: list[float] = []


def _load() -> None:
    """Load every task model found under MODEL_DIR/<task>/. Missing models leave /readyz false."""
    from vifeedback.inference.onnx_export import OnnxClassifier

    for task in ("sentiment", "topic"):
        d = MODEL_DIR / task
        if (d / "model.onnx").exists() or (d / "model.quant.onnx").exists():
            try:
                _state["models"][task] = OnnxClassifier(d, THREADS, MAX_LENGTH)
                log.info(f"loaded {task} from {_state['models'][task].path.name}")
            except Exception as e:
                log.error(f"failed to load {task}: {type(e).__name__}: {e}")

    vf = MODEL_DIR / "VERSION"
    _state["version"] = vf.read_text(encoding="utf-8").strip() if vf.exists() else "unversioned"

    try:
        from vifeedback.preprocess.segment import get_segmenter

        _state["segmenter"] = get_segmenter(os.getenv("SEGMENTER", "pyvi"))
    except Exception as e:
        # Serving raw text costs ~0.023 macro-F1 (ADR-012). Degrading loudly beats failing to boot.
        log.warning(
            f"segmenter unavailable ({type(e).__name__}) — serving raw text, -0.023 macro-F1"
        )


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


@app.get("/readyz", response_model=ReadyResponse)
def readyz() -> ReadyResponse:
    loaded = sorted(_state["models"])
    return ReadyResponse(
        ready=bool(loaded),
        models_loaded=loaded,
        detail=None if loaded else f"no ONNX artifact under {MODEL_DIR}",
    )


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

    t0 = time.perf_counter()
    texts = list(req.texts)
    seg = _state["segmenter"]
    model_input = seg(texts) if seg is not None else texts

    ids, probs = clf.predict(model_input)
    names = [LABELS[req.task][i] for i in sorted(LABELS[req.task])]

    preds = []
    for text, i, p in zip(texts, ids, probs, strict=True):
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
        f"vifeedback_requests_total {len(_latencies)}",
        "# HELP vifeedback_ready Whether a model is loaded.",
        "# TYPE vifeedback_ready gauge",
        f"vifeedback_ready {int(bool(_state['models']))}",
    ]
    if _latencies:
        import numpy as np

        a = np.array(_latencies[-10_000:])
        for q in (50, 95, 99):
            lines += [
                f"# TYPE vifeedback_latency_p{q}_ms gauge",
                f"vifeedback_latency_p{q}_ms {np.percentile(a, q):.3f}",
            ]
    return "\n".join(lines) + "\n"
