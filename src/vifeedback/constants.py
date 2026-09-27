"""Project-wide constants.

Single source of truth for label maps, split sizes and seeds. Every module imports from here so a
label map can never drift between training, evaluation and serving — a transposed `id2label` produces
a plausible-looking confusion matrix and wrong everything (see docs/EVALUATION_PROTOCOL.md).
"""

from __future__ import annotations

from typing import Final

# --- Dataset identity -------------------------------------------------------------------------

HF_DATASET_ID: Final = "uitnlp/vietnamese_students_feedback"
DATASET_NAME: Final = "UIT-VSFC"

# --- Tasks ------------------------------------------------------------------------------------

TASKS: Final = ("sentiment", "topic")

SENTIMENT_LABELS: Final[dict[int, str]] = {0: "negative", 1: "neutral", 2: "positive"}
TOPIC_LABELS: Final[dict[int, str]] = {
    0: "lecturer",
    1: "training_program",
    2: "facility",
    3: "others",
}

LABELS: Final[dict[str, dict[int, str]]] = {
    "sentiment": SENTIMENT_LABELS,
    "topic": TOPIC_LABELS,
}


def label_names(task: str) -> list[str]:
    """Class names in id order — the order every confusion matrix and report must use."""
    mapping = LABELS[task]
    return [mapping[i] for i in sorted(mapping)]


def n_classes(task: str) -> int:
    return len(LABELS[task])


# --- Official split -----------------------------------------------------------------------------
# Asserted in tests/data/. If the upstream source shifts, the build fails rather than silently
# reporting numbers computed against different data. See docs/DATA_CARD.md § 2.

SPLITS: Final = ("train", "validation", "test")

EXPECTED_SPLIT_SIZES: Final[dict[str, int]] = {
    "train": 11_426,
    "validation": 1_583,
    "test": 3_166,
}
EXPECTED_TOTAL: Final = sum(EXPECTED_SPLIT_SIZES.values())  # 16_175

# --- Experiment policy --------------------------------------------------------------------------
# docs/EVALUATION_PROTOCOL.md § 2: fixed, listed, never cherry-picked.

SEEDS: Final[tuple[int, ...]] = (42, 1337, 2024, 7, 31337)
EXPLORATION_SEEDS: Final[tuple[int, ...]] = SEEDS[:3]

BOOTSTRAP_RESAMPLES: Final = 10_000
BOOTSTRAP_ALPHA: Final = 0.05

# Pre-registered accuracy budget for inference optimization (docs/ROADMAP.md O4 / S6).
MACRO_F1_BUDGET_PP: Final = 0.5

# --- Model zoo ----------------------------------------------------------------------------------

MODEL_IDS: Final[dict[str, str]] = {
    "phobert-base": "vinai/phobert-base",
    "phobert-base-v2": "vinai/phobert-base-v2",
    "phobert-large": "vinai/phobert-large",
    "visobert": "uitnlp/visobert",
    "xlmr-base": "FacebookAI/xlm-roberta-base",
}

# Hub commits, pinned after review R11: a model id alone names a moving branch, so a re-run months
# later could silently start from different weights. These are the commits `main` pointed to on
# 2026-09-27; phobert-base's matches the weights every registry row was trained from (local cache).
MODEL_REVISIONS: Final[dict[str, str]] = {
    "phobert-base": "01daacda68afe13d83023d16ec647239e344a1e6",
    "phobert-base-v2": "86cd7fd4c148980922ac11a2cf5e257f2ba639e1",
    "phobert-large": "70e2cfcd3cce29c970aee4954ea34a32bb30afdc",
    "visobert": "196a62afad9cbe4f52a54aabad828b13f0eec59a",
    "xlmr-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
}

# Models pretrained on word-segmented Vietnamese; their model cards require segmented input.
REQUIRES_WORD_SEGMENTATION: Final = frozenset({"phobert-base", "phobert-base-v2", "phobert-large"})
