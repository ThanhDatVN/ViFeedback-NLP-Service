"""Command-line entrypoint: `vifeedback <group> <command>` or `python -m vifeedback.cli ...`.

Every experiment in this project is a CLI call, so the laptop and Kaggle run identical code paths and
notebooks stay logic-free (docs/ROADMAP.md § 4).

One module per command group: `data`, `baseline`, `train`, `serve`, `results`. The `study` group is
split by research cycle (`study_cycle1` … `study_cycle4`), matching configs/experiments/cycleN.yaml.
The groups themselves are declared in `_apps`.
"""

from __future__ import annotations

# Imported for their side effect: each module registers its commands on a group from `_apps`.
from vifeedback.cli import (  # noqa: F401
    baseline,
    data,
    results,
    serve,
    study_cycle1,
    study_cycle2,
    study_cycle3,
    study_cycle4,
    train,
)
from vifeedback.cli._apps import app

__all__ = ["app"]
