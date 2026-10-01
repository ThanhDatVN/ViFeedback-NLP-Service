"""Cycle 5 H12': held-out in-scope NEU-ESC train posts (cycle5.yaml v6).

The held-out set is drawn once, stratified by label, and written as row indices into the NEU-ESC
train split (no text), so every later step reads the same 3,000 posts.
"""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd

from vifeedback import paths

N_HOLDOUT = 3000
SEED = 44
OUT = paths.RESULTS / "studies" / "cycle5" / "h12p"
INDEX = paths.RESULTS / "studies" / "cycle5" / "h12p_holdout_index.csv"


def holdout_split(frame: pd.DataFrame, off_topic: Collection[str]) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (held-out, training) of the in-scope posts, as cycle5.yaml v6 declares.

    Within each label, its in-scope rows in loader order are ordered by
    default_rng(44).permutation(n_label); the first round(3000 * n_label / n_in_scope) are held out.
    """
    in_scope = np.flatnonzero(~frame.topic.isin(off_topic).to_numpy())
    labels = frame.sentiment.to_numpy()[in_scope]
    held: list[np.ndarray] = []
    for label in sorted(set(labels)):
        rows = in_scope[labels == label]
        k = round(N_HOLDOUT * len(rows) / len(in_scope))
        held.append(rows[np.random.default_rng(SEED).permutation(len(rows))[:k]])
    holdout = np.sort(np.concatenate(held))
    return holdout, np.setdiff1d(in_scope, holdout)


def write_holdout_index() -> Path:
    """Draw the split from NEU-ESC train and write the held-out row positions with their labels."""
    from vifeedback.evaluation import external as X
    from vifeedback.training.domain import OFF_TOPIC

    frame = X.load_neu_esc("train")
    holdout, _ = holdout_split(frame, OFF_TOPIC)
    if INDEX.exists():
        old = pd.read_csv(INDEX)["row"].to_numpy()
        if not np.array_equal(old, holdout):
            raise RuntimeError(f"{INDEX} exists with a different split; it is drawn once")
        return INDEX
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": holdout, "sentiment": frame.sentiment.to_numpy()[holdout]}).to_csv(
        INDEX, index=False
    )
    return INDEX
