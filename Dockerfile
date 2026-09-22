# Multi-stage. The runtime layer carries onnxruntime and NOT torch: torch is ~2.5 GB and the
# service never needs it, which is the difference between meeting the 700 MB target and missing it
# by a factor of four.

FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
      "onnxruntime>=1.19" "fastapi>=0.115" "uvicorn[standard]>=0.30" \
      "transformers>=4.44" "numpy>=2.0" "pyvi" "pyyaml" \
 && pip install --no-cache-dir --no-deps .


FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="ViFeedback" \
      org.opencontainers.image.description="Vietnamese feedback sentiment and topic classification" \
      org.opencontainers.image.source="https://github.com/ThanhDatVN/ViFeedback-NLP-Service" \
      org.opencontainers.image.licenses="MIT"

# Non-root. The service reads a model and writes nothing.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_DIR=/app/models/serve \
    MAX_LENGTH=96 \
    SEGMENTER=pyvi \
    ORT_THREADS=2 \
    HF_HUB_OFFLINE=1

# Artifacts are mounted, not baked: a 120 MB model in an image layer means rebuilding the image to
# ship a retrain, and the image is then pinned to one model version forever.
RUN mkdir -p /app/models/serve && chown -R app:app /app
USER app

EXPOSE 8000

# Readiness, not liveness: /healthz is true as soon as the process is up, which would mark a
# model-less container healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,json,sys; \
r=json.load(urllib.request.urlopen('http://localhost:8000/readyz',timeout=4)); \
sys.exit(0 if r.get('ready') else 1)"

CMD ["uvicorn", "vifeedback.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
