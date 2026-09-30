"""The closing-gate procedure for one checkpoint on UIT-VSFC test (Cycle 1, reused by Cycle 5).

Test metrics on the official split and on the overlap-excluded slice (configs/data/eval_slices_v1.json),
calibration with the temperature fitted on validation and applied to test, and the four robustness
suites on test. Callers log every test touch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths

SUITES = ("nodiacritic", "nodiacritic-50", "teencode-100", "charnoise-5")


def inputs() -> dict[str, Any]:
    """Test and validation (seg_pyvi), raw test text for the robustness suites, the slice, pyvi."""
    from vifeedback.data.loader import load
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import load_variant

    te, dv = load_variant("seg_pyvi", "test"), load_variant("seg_pyvi", "validation")
    slices = json.loads(
        (paths.CONFIGS / "data" / "eval_slices_v1.json").read_text(encoding="utf-8")
    )
    y_te = te.sentiment.to_numpy()
    return {
        "x_te": te.sentence.tolist(),
        "y_te": y_te,
        "x_dv": dv.sentence.tolist(),
        "y_dv": dv.sentiment.to_numpy(),
        "raw_te": load("test")["sentence"].tolist(),
        "keep": np.setdiff1d(np.arange(len(y_te)), slices["test"]["overlapping_train_indices"]),
        "slice_version": slices["version"],
        "segment": get_segmenter("pyvi"),
    }


def row(ck: Path, gi: dict[str, Any]) -> dict[str, Any]:
    """Everything the closing gate reports for one checkpoint (the test split is read here)."""
    from vifeedback.evaluation import calibration as C
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import robustness as R

    y_te, y_dv, keep = gi["y_te"], gi["y_dv"], gi["keep"]
    p_te = EA.predict_proba(ck, gi["x_te"])
    p_dv = EA.predict_proba(ck, gi["x_dv"])
    t = C.fit_temperature(C.probs_to_logits(p_dv), y_dv)  # fitted on validation only
    cal_te = C.apply_temperature(C.probs_to_logits(p_te), t)
    full = M.evaluate(y_te, p_te.argmax(1), "sentiment", y_prob=p_te)
    sl = M.evaluate(y_te[keep], p_te[keep].argmax(1), "sentiment")
    rob = {}
    for suite in SUITES:
        pert, _ = R.perturb(gi["raw_te"], suite, seed=42)
        pred = EA.predict_proba(ck, gi["segment"](pert)).argmax(1)
        rob[suite] = {
            "macro_f1": M.macro_f1(y_te, pred, 3),
            "pred_share_neutral": float((pred == 1).mean()),
        }
    keys = ("nll", "brier", "ece_equal_width", "ece_equal_mass")
    return {
        "checkpoint": str(ck),
        "test": {
            "macro_f1": full["macro_f1"],
            "weighted_f1": full["weighted_f1"],
            "per_class_f1": {c: v["f1"] for c, v in full["per_class"].items()},
        },
        "test_excluding_train_overlap": {
            "n": len(keep),
            "macro_f1": sl["macro_f1"],
            "neutral_f1": sl["per_class"]["neutral"]["f1"],
        },
        "calibration": {
            "temperature_fit_on_validation": t,
            "uncalibrated": {k: v for k, v in C.summary(p_te, y_te).items() if k in keys},
            "calibrated": {k: v for k, v in C.summary(cal_te, y_te).items() if k in keys},
        },
        "robustness_test": rob,
    }


def log_test_use(checkpoint: Path, phase: str, reason: str) -> None:
    from datetime import UTC, datetime

    with open(paths.TEST_EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}\t{checkpoint.name}\t{phase}\t{reason}\n"
        )
