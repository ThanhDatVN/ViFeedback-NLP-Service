"""Training: the config, the trainer and the multi-seed runner.

The names below load on first use, so the modules that only read results (a declared rule applied
to prediction files, a split drawn from indices) import without torch.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vifeedback.training.runner import aggregate, format_summary, run_once, run_seeds
    from vifeedback.training.seeding import seed_everything
    from vifeedback.training.trainer import TrainConfig, train

_LAZY = {
    "aggregate": "runner",
    "format_summary": "runner",
    "run_once": "runner",
    "run_seeds": "runner",
    "seed_everything": "seeding",
    "TrainConfig": "trainer",
    "train": "trainer",
}

__all__ = [
    "TrainConfig",
    "aggregate",
    "format_summary",
    "run_once",
    "run_seeds",
    "seed_everything",
    "train",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(f"vifeedback.training.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
