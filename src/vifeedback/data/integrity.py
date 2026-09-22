"""Dataset integrity and leakage analysis — Phase 0, gate G0.

Implements the checklist in docs/DATA_CARD.md § 7. The leakage check is the one step here that can
invalidate every later number in the project, so its result is published regardless of what it says.

Everything returns plain dicts so the report can be serialized to JSON, asserted in tests, and
rendered into the data card without a second implementation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from vifeedback.constants import (
    EXPECTED_SPLIT_SIZES,
    LABELS,
    SPLITS,
    TASKS,
    label_names,
)
from vifeedback.preprocess.normalize import (
    dedup_key,
    detect_unicode_form,
    strip_diacritics,
    syllable_count,
)

# --- Structural checks --------------------------------------------------------------------------


def check_split_sizes(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    actual = {s: len(dfs[s]) for s in SPLITS}
    return {
        "expected": dict(EXPECTED_SPLIT_SIZES),
        "actual": actual,
        "total": sum(actual.values()),
        "passed": actual == EXPECTED_SPLIT_SIZES,
    }


def check_labels(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Label ids in range, no nulls, every declared class present in every split."""
    out: dict[str, Any] = {"passed": True, "tasks": {}}
    for task in TASKS:
        valid = set(LABELS[task])
        per_split = {}
        for split in SPLITS:
            col = dfs[split][task]
            observed = set(col.dropna().unique().tolist())
            entry = {
                "nulls": int(col.isna().sum()),
                "out_of_range": sorted(int(v) for v in observed - valid),
                "missing_classes": sorted(int(v) for v in valid - observed),
            }
            entry["passed"] = (
                entry["nulls"] == 0 and not entry["out_of_range"] and not entry["missing_classes"]
            )
            out["passed"] &= entry["passed"]
            per_split[split] = entry
        out["tasks"][task] = per_split
    return out


def check_text_quality(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Empty / whitespace-only sentences, and the Unicode normalization form actually present."""
    out: dict[str, Any] = {"passed": True, "splits": {}}
    for split in SPLITS:
        s = dfs[split]["sentence"]
        empty = int((s.fillna("").str.strip() == "").sum())
        forms = s.map(detect_unicode_form).value_counts().to_dict()
        entry = {
            "nulls": int(s.isna().sum()),
            "empty_or_whitespace": empty,
            "unicode_forms": {k: int(v) for k, v in forms.items()},
            "passed": empty == 0 and int(s.isna().sum()) == 0,
        }
        out["passed"] &= entry["passed"]
        out["splits"][split] = entry
    return out


# --- Duplication and leakage --------------------------------------------------------------------


def _keys(df: pd.DataFrame, normalized: bool) -> pd.Series:
    return df["sentence"].map(dedup_key) if normalized else df["sentence"]


def check_duplicates(dfs: dict[str, pd.DataFrame], normalized: bool) -> dict[str, Any]:
    """Within-split duplicates and cross-split overlap.

    `normalized=False` is exact string matching; `normalized=True` uses the aggressive dedup key
    (lowercased, diacritics stripped, punctuation removed) and is what "near-duplicate" means here.
    """
    keys = {s: _keys(dfs[s], normalized) for s in SPLITS}

    within = {}
    for s in SPLITS:
        n_dup = int(len(keys[s]) - keys[s].nunique())
        within[s] = {
            "rows": len(keys[s]),
            "unique": int(keys[s].nunique()),
            "duplicate_rows": n_dup,
            "duplicate_share": round(n_dup / max(len(keys[s]), 1), 4),
        }

    sets = {s: set(keys[s]) for s in SPLITS}
    cross = {}
    for a, b in combinations(SPLITS, 2):
        shared = sets[a] & sets[b]
        # Count affected *rows*, not just distinct keys — a key appearing 5 times in test leaks
        # 5 test rows, which is what actually inflates the headline metric.
        rows_b = int(keys[b].isin(shared).sum())
        rows_a = int(keys[a].isin(shared).sum())
        cross[f"{a}|{b}"] = {
            "shared_keys": len(shared),
            f"rows_in_{a}": rows_a,
            f"rows_in_{b}": rows_b,
            f"share_of_{b}": round(rows_b / max(len(keys[b]), 1), 4),
        }

    return {"mode": "normalized" if normalized else "exact", "within": within, "cross": cross}


def leakage_summary(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The headline leakage number: share of *test* rows also present in train.

    Reported in both modes. A large normalized-but-not-exact gap means the corpus contains formulaic
    restatements rather than literal copies — a different and milder problem, but still one that
    makes the test set easier than an unseen deployment distribution.
    """
    exact = check_duplicates(dfs, normalized=False)
    near = check_duplicates(dfs, normalized=True)
    return {
        "exact": exact,
        "normalized": near,
        "headline": {
            "test_rows_seen_in_train_exact": exact["cross"]["train|test"]["rows_in_test"],
            "test_rows_seen_in_train_normalized": near["cross"]["train|test"]["rows_in_test"],
            "share_of_test_exact": exact["cross"]["train|test"]["share_of_test"],
            "share_of_test_normalized": near["cross"]["train|test"]["share_of_test"],
        },
    }


# --- Distributions ------------------------------------------------------------------------------


def class_distribution(dfs: dict[str, pd.DataFrame], task: str) -> pd.DataFrame:
    """Counts and within-split shares, one row per split, columns in canonical label order."""
    names = label_names(task)
    rows = []
    for split in SPLITS:
        counts = dfs[split][task].value_counts().reindex(range(len(names)), fill_value=0)
        total = int(counts.sum())
        row: dict[str, Any] = {"split": split, "n": total}
        for i, name in enumerate(names):
            row[name] = int(counts.iloc[i])
            row[f"{name}_pct"] = round(100 * int(counts.iloc[i]) / max(total, 1), 2)
        rows.append(row)

    all_counts = (
        pd.concat([dfs[s][task] for s in SPLITS])
        .value_counts()
        .reindex(range(len(names)), fill_value=0)
    )
    total = int(all_counts.sum())
    row = {"split": "ALL", "n": total}
    for i, name in enumerate(names):
        row[name] = int(all_counts.iloc[i])
        row[f"{name}_pct"] = round(100 * int(all_counts.iloc[i]) / max(total, 1), 2)
    rows.append(row)
    return pd.DataFrame(rows)


def joint_distribution(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Sentiment x topic contingency over the full corpus.

    Evidence for or against the multi-task model of Phase 4 Tier E, obtained before spending GPU
    hours on it: near-independence means the shared encoder has little to exploit.
    """
    full = pd.concat([dfs[s] for s in SPLITS], ignore_index=True)
    ct = pd.crosstab(
        full["sentiment"].map(LABELS["sentiment"]),
        full["topic"].map(LABELS["topic"]),
    )
    return ct.reindex(index=label_names("sentiment"), columns=label_names("topic"), fill_value=0)


def cramers_v(contingency: pd.DataFrame) -> float:
    """Association strength in [0, 1] between sentiment and topic. Bias-corrected."""
    from scipy.stats import chi2_contingency

    chi2 = chi2_contingency(contingency.values)[0]
    n = contingency.values.sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float((phi2corr / denom) ** 0.5) if denom > 0 else 0.0


def length_stats(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Character and syllable length percentiles per split.

    Subword-token lengths are measured separately in `profile.py` because they require downloading a
    tokenizer; these two need nothing and already bound the `max_length` decision.
    """
    qs = [0.5, 0.9, 0.95, 0.99, 1.0]
    out: dict[str, Any] = {}
    for split in SPLITS:
        s = dfs[split]["sentence"]
        chars = s.str.len()
        syls = s.map(syllable_count)
        out[split] = {
            "chars": {f"p{int(q * 100)}": int(chars.quantile(q)) for q in qs}
            | {"mean": round(float(chars.mean()), 2)},
            "syllables": {f"p{int(q * 100)}": int(syls.quantile(q)) for q in qs}
            | {"mean": round(float(syls.mean()), 2)},
        }
    return out


def diacritic_stats(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Share of sentences written without diacritics.

    Sizes the `NODIACRITIC` error category before any model is trained, and tells us whether the
    `nodiacritic` perturbation suite tests a realistic distribution shift or an artificial one.
    """
    out: dict[str, Any] = {}
    for split in SPLITS:
        s = dfs[split]["sentence"]
        stripped = s.map(strip_diacritics)
        no_diac = (s == stripped) & s.str.strip().ne("")
        out[split] = {
            "sentences_without_diacritics": int(no_diac.sum()),
            "share": round(float(no_diac.mean()), 4),
        }
    return out


# --- Orchestration ------------------------------------------------------------------------------


def full_report(dfs: dict[str, pd.DataFrame]) -> dict[str, Any]:
    structural = {
        "split_sizes": check_split_sizes(dfs),
        "labels": check_labels(dfs),
        "text_quality": check_text_quality(dfs),
    }
    return {
        "structural": structural,
        "structural_passed": all(v["passed"] for v in structural.values()),
        "leakage": leakage_summary(dfs),
        "class_distribution": {
            t: class_distribution(dfs, t).to_dict(orient="records") for t in TASKS
        },
        "joint_distribution": joint_distribution(dfs).to_dict(),
        "cramers_v_sentiment_topic": round(cramers_v(joint_distribution(dfs)), 4),
        "length_stats": length_stats(dfs),
        "diacritics": diacritic_stats(dfs),
    }
