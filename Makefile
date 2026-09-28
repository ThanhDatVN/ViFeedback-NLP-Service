# ViFeedback — one target per reproducible step.
# Every target is what CI runs and what the README documents; there is no second way to do anything.

.PHONY: help venv install data report test test-all lint format typecheck baseline train export serve docker docker-run docker-e2e bench clean ci
.DEFAULT_GOAL := help

# The project's .venv when it exists (python scripts/setup_venv.py), otherwise whatever `python` is.
VENV_PY := $(firstword $(wildcard .venv/Scripts/python.exe .venv/bin/python))
PY ?= $(if $(VENV_PY),$(VENV_PY),python)
TASK ?= sentiment
MODEL ?= phobert-base
PREP ?= seg_pyvi
SEEDS ?= all
# The served checkpoint (ADR-027: H2-augmented, seed 42); pass CKPT=path to choose another. INT8 needs the `onnx` package (ADR-017/020).
CKPT ?= $(lastword $(sort $(wildcard models/p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42*ckp)))
QUANT ?= none

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

venv:  ## Create .venv with every pinned package (CUDA torch if an NVIDIA GPU is present)
	python scripts/setup_venv.py

install:  ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev]"

data:  ## Fetch UIT-VSFC and verify integrity
	$(PY) -m vifeedback.cli data fetch
	$(PY) -m pytest tests/data -q

report:  ## Phase 0 profiling report + EDA figures
	$(PY) -m vifeedback.cli data report

variants:  ## Materialize preprocessing variants
	$(PY) -m vifeedback.cli data variants --name seg_pyvi
	$(PY) -m vifeedback.cli data variants --name seg_vncorenlp

lint:  ## ruff check + format check
	ruff check src tests
	ruff format --check src tests

format:  ## Apply ruff fixes and formatting
	ruff check --fix src tests
	ruff format src tests

typecheck:  ## mypy
	$(PY) -m mypy

test:  ## Fast tests (excludes slow integration)
	$(PY) -m pytest tests -q -m "not slow"

test-all:  ## Everything, including slow integration tests
	$(PY) -m pytest tests -q

ci: lint typecheck test  ## What CI runs

baseline:  ## TF-IDF ladder for TASK
	$(PY) -m vifeedback.cli baseline run --task $(TASK)

train:  ## Fine-tune MODEL on TASK across SEEDS
	$(PY) -m vifeedback.cli train run --task $(TASK) --model $(MODEL) \
		--preprocessing $(PREP) --seeds $(SEEDS)

export:  ## Release the serving artifact: ONNX with features (staging -> verify on full dev -> swap), restorer, out-of-scope score (ADR-031)
	$(PY) -m vifeedback.cli serve export --task $(TASK) --checkpoint $(CKPT) --quantize $(QUANT) --with-features
	$(PY) -m vifeedback.cli serve add-restorer --task $(TASK)
	$(PY) -m vifeedback.cli serve add-ood --task $(TASK)

bench:  ## CPU latency benchmark on the reference machine
	$(PY) -m vifeedback.cli serve bench --task $(TASK)

serve:  ## Run the API locally on :8000
	uvicorn vifeedback.serving.app:app --host 0.0.0.0 --port 8000

docker:  ## Build the runtime image
	docker build -t vifeedback:latest .

docker-run:  ## Run the container on :8000
	docker run --rm -p 8000:8000 -v "$(PWD)/models:/app/models:ro" vifeedback:latest

docker-e2e:  ## Build the image, mount the released model, check /readyz and golden cases
	$(PY) scripts/docker_e2e.py

notebooks:  ## Re-execute both notebooks in place
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/02_results.ipynb

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
