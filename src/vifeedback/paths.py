"""Canonical filesystem layout.

Every path in the project resolves through this module, so nothing depends on the current working
directory and a script behaves identically from the repo root, from `notebooks/`, or on Colab.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# src/vifeedback/paths.py -> src/vifeedback -> src -> <repo root>
ROOT: Final[Path] = Path(__file__).resolve().parents[2]

CONFIGS: Final[Path] = ROOT / "configs"
DOCS: Final[Path] = ROOT / "docs"

DATA: Final[Path] = ROOT / "data"
DATA_RAW: Final[Path] = DATA / "raw"
DATA_INTERIM: Final[Path] = DATA / "interim"
DATA_PROCESSED: Final[Path] = DATA / "processed"

RESULTS: Final[Path] = ROOT / "results"
RUNS: Final[Path] = RESULTS / "runs"
FIGURES: Final[Path] = RESULTS / "figures"
REGISTRY: Final[Path] = RESULTS / "registry.csv"
TEST_EVAL_LOG: Final[Path] = RESULTS / "test_evaluations.log"

MODELS: Final[Path] = ROOT / "models"

_WRITABLE: Final[tuple[Path, ...]] = (
    DATA_RAW,
    DATA_INTERIM,
    DATA_PROCESSED,
    RUNS,
    FIGURES,
    MODELS,
)


def ensure_dirs() -> None:
    """Create the writable tree. Idempotent; safe to call at the start of any entrypoint."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


def raw_file(split: str) -> Path:
    return DATA_RAW / f"{split}.parquet"


def run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
