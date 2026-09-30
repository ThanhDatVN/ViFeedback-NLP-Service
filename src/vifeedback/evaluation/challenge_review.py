"""U2 (NEXT_PLAN v6 F6): challenge v1 scored on the owner-reviewed labels next to the frozen ones.

Challenge v1 stays frozen (its SHA-256 is checked by `study challenge`), so a reviewed label never
replaces a frozen one in the data. This module reports how each system's score moves if the 15
reviewed rows take the owner's decision: `keep` leaves the label, a label name replaces it, and
`ambiguous` drops the row from the reviewed view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vifeedback import paths
from vifeedback.constants import label_names

LABELS = label_names("sentiment")
DECISIONS = {"keep", "ambiguous", *LABELS}
REVIEW = paths.RESULTS / "studies" / "challenge" / "label_review_v1.csv"
CHALLENGE = paths.ROOT / "data" / "challenge" / "challenge_v1.csv"
LLM = paths.RESULTS / "studies" / "llm_reference"
SYSTEMS = {
    "ce": ("encoder", "ce_pred"),
    "augmented (served)": ("encoder", "augmented_pred"),
    "gpt-4o-mini": ("llm", "gpt-4o-mini-2024-07-18"),
    "Qwen3-4B": ("llm", "Qwen__Qwen3-4B"),
}


def read_decisions(path: Path = REVIEW) -> dict[str, str]:
    """id -> decision for the rows the owner has filled; refuses a value outside DECISIONS."""
    df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    got = {
        str(i): str(d).strip().lower()
        for i, d in zip(df["id"], df["owner_decision"], strict=True)
        if str(d).strip()
    }
    bad = sorted({d for d in got.values() if d not in DECISIONS})
    if bad:
        raise ValueError(f"owner_decision must be one of {sorted(DECISIONS)}; found {bad}")
    return got


def reviewed_labels(gold: pd.Series, decisions: dict[str, str]) -> tuple[pd.Series, pd.Series]:
    """The reviewed label per row (indexed by id) and a mask of rows kept in the reviewed view."""
    out = gold.copy()
    keep = pd.Series(True, index=gold.index)
    for i, d in decisions.items():
        if d == "ambiguous":
            keep[i] = False
        elif d != "keep":
            out[i] = d
    return out, keep


def _scores(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    from vifeedback.evaluation import metrics as M

    ids = {c: k for k, c in enumerate(LABELS)}
    yt, yp = y.map(ids).to_numpy(), pred.map(ids).to_numpy()
    ev = M.evaluate(yt, yp, "sentiment")
    return {
        "accuracy": float(np.mean(yt == yp)),
        "macro_f1": ev["macro_f1"],
        "neutral_f1": ev["per_class"]["neutral"]["f1"],
    }


def predictions() -> dict[str, pd.Series]:
    """Each system's zero-shot / seed-42 prediction on challenge v1, indexed by row id."""
    enc = pd.read_csv(paths.RESULTS / "studies" / "challenge" / "predictions.csv").set_index("id")
    out = {}
    for name, (kind, key) in SYSTEMS.items():
        if kind == "encoder":
            out[name] = enc[key]
        else:
            f = LLM / key / "challenge-v4_policy_informal_contrast-k0" / "predictions.csv"
            if f.exists():
                out[name] = pd.read_csv(f).set_index("id")["pred"]
    return out


def sensitivity(decisions: dict[str, str]) -> dict[str, Any]:
    ch = pd.read_csv(CHALLENGE).set_index("id")
    gold = ch["sentiment"]
    unknown = sorted(set(decisions) - set(gold.index))
    if unknown:
        raise ValueError(f"ids not in challenge v1: {unknown}")
    rev, keep = reviewed_labels(gold, decisions)
    scored = gold.isin(LABELS)  # out-of-scope rows ("none") are not scored, as in `study challenge`
    systems = {}
    for name, pred in predictions().items():
        pred = pred.reindex(gold.index)
        frozen = _scores(gold[scored], pred[scored])
        reviewed = _scores(rev[keep & scored], pred[keep & scored])
        systems[name] = {
            "frozen": frozen,
            "reviewed": reviewed,
            "delta": {k: reviewed[k] - frozen[k] for k in frozen},
        }
    changed = [i for i, d in decisions.items() if d not in ("keep", "ambiguous")]
    return {
        "rows": len(gold),
        "scored_rows": int(scored.sum()),
        "decisions": {
            "filled": len(decisions),
            "keep": sum(d == "keep" for d in decisions.values()),
            "relabelled": len(changed),
            "ambiguous": sum(d == "ambiguous" for d in decisions.values()),
        },
        "relabelled_rows": {i: {"frozen": gold[i], "reviewed": rev[i]} for i in changed},
        "systems": systems,
        "note": "challenge v1 stays frozen; this is a sensitivity view (NEXT_PLAN v6 F6)",
    }
