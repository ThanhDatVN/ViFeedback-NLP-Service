"""The neutral audit's analysis (NEXT_PLAN step 1): agreement, per-stratum rates, the frozen tree.

The annotation itself is human work (docs/ANNOTATION_GUIDE.md). This module only reads filled
sheets. It never invents a label, and it follows the guide's reporting rules: rates are reported
within their stratum, and only the ``random`` stratum estimates a corpus-level rate, with a Wilson
interval.

The decision tree is the one frozen in ``configs/experiments/cycle2.yaml`` (``audit_decision_tree``)
before any annotation existed.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

LABEL_VALUES = {"negative", "neutral", "positive", "ambiguous"}
ASSESSMENTS = ("agree", "ambiguous", "incorrect")
SUBTYPES = {
    "objective_fact",
    "no_opinion",
    "request_suggestion",
    "mixed",
    "insufficient_context",
    "other",
    "n/a",
}
# cycle2.yaml scope: "confident neutral-error strata", the error cells with neutral on either side.
SCOPE_STRATA = (
    "error:neutral->negative",
    "error:neutral->positive",
    "error:negative->neutral",
    "error:positive->neutral",
)
# cycle2.yaml audit_decision_tree, in order.
INCORRECT_THRESHOLD = 0.30
AMBIGUOUS_THRESHOLD = 0.40
MIN_AGREEMENT_ROWS = 50


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pe = sum((a == c).sum() * (b == c).sum() for c in cats) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def validate(sheet: pd.DataFrame) -> list[str]:
    """Problems that make a sheet unusable; empty when it is complete and well-formed."""
    problems = []
    for col, allowed in (
        ("annotator_label", LABEL_VALUES),
        ("gold_assessment", set(ASSESSMENTS)),
    ):
        vals = sheet[col].fillna("").astype(str).str.strip().str.lower()
        if (vals == "").any():
            problems.append(f"{col}: {(vals == '').sum()} rows empty")
        bad = sorted(set(vals[vals != ""]) - allowed)
        if bad:
            problems.append(f"{col}: unknown values {bad}")
    sub = sheet["neutral_subtype"].fillna("").astype(str).str.strip().str.lower()
    bad = sorted(set(sub[sub != ""]) - SUBTYPES)
    if bad:
        problems.append(f"neutral_subtype: unknown values {bad}")
    needs = sheet.annotator_label.str.lower().isin(["neutral", "ambiguous"]) | (
        sheet.gold == "neutral"
    )
    missing = needs & (sub == "")
    if missing.any():
        problems.append(f"neutral_subtype: {int(missing.sum())} rows need a subtype (guide § 3)")
    return problems


def _clean(sheet: pd.DataFrame) -> pd.DataFrame:
    out = sheet.copy()
    for c in ("annotator_label", "gold_assessment", "neutral_subtype"):
        out[c] = out[c].fillna("").astype(str).str.strip().str.lower()
    return out


def per_stratum(sheet: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for stratum, g in sheet.groupby("stratum", sort=True):
        n = len(g)
        out[str(stratum)] = {
            "n": n,
            "gold_assessment": {a: int((g.gold_assessment == a).sum()) for a in ASSESSMENTS},
            "annotator_disagrees_with_gold": int((g.annotator_label != g.gold).sum()),
            "neutral_subtype": {
                str(k): int(v)
                for k, v in g.neutral_subtype.replace("", "n/a").value_counts().items()
            },
        }
    return out


def random_stratum_rates(sheet: pd.DataFrame) -> dict[str, Any]:
    """Corpus-level estimates: the random stratum only, with Wilson intervals (guide § 6.2)."""
    r = sheet[sheet.stratum == "random"]
    n = len(r)
    out: dict[str, Any] = {"n": n}
    for a in ("ambiguous", "incorrect"):
        k = int((r.gold_assessment == a).sum())
        out[a] = {"k": k, "rate": k / n if n else float("nan"), "wilson_95": wilson(k, n)}
    k = int((r.annotator_label != r.gold).sum())
    out["annotator_disagrees_with_gold"] = {
        "k": k,
        "rate": k / n if n else float("nan"),
        "wilson_95": wilson(k, n),
    }
    return out


def agreement(first: pd.DataFrame, second: pd.DataFrame, kind: str) -> dict[str, Any]:
    """Cohen's κ between two passes over the same rows, overall and neutral-vs-rest (guide § 5)."""
    key = ["split", "example_index"]
    m = first.merge(second, on=key, suffixes=("_a", "_b"))
    m = m[(m.annotator_label_a != "") & (m.annotator_label_b != "")]
    random_rows = int((first.stratum == "random").sum())
    covered_random = int((m.stratum_a == "random").sum())
    warnings = []
    if len(m) < MIN_AGREEMENT_ROWS:
        warnings.append(
            f"only {len(m)} doubly-labelled rows; the guide asks for >= {MIN_AGREEMENT_ROWS}"
        )
    if covered_random < random_rows:
        warnings.append(f"{random_rows - covered_random} random-stratum rows lack a second label")
    return {
        "kind": kind,  # "inter" (two people) or "intra" (one person, >= 24 h apart): a weaker figure
        "n": len(m),
        "kappa": cohen_kappa(m.annotator_label_a, m.annotator_label_b) if len(m) else float("nan"),
        "kappa_neutral_vs_rest": cohen_kappa(
            m.annotator_label_a == "neutral", m.annotator_label_b == "neutral"
        )
        if len(m)
        else float("nan"),
        "raw_agreement": float((m.annotator_label_a == m.annotator_label_b).mean())
        if len(m)
        else float("nan"),
        "warnings": warnings,
    }


def decide(sheet: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen tree to the scope strata, rules in declared order."""
    s = sheet[sheet.stratum.isin(SCOPE_STRATA)]
    n = len(s)
    incorrect = float((s.gold_assessment == "incorrect").mean()) if n else float("nan")
    ambiguous = float((s.gold_assessment == "ambiguous").mean()) if n else float("nan")
    if n and incorrect >= INCORRECT_THRESHOLD:
        branch, next_step = (
            "incorrect_gold",
            "E04: train-label correction, 2x2 (labels x recipe), 3 seeds",
        )
    elif n and ambiguous >= AMBIGUOUS_THRESHOLD:
        branch, next_step = (
            "policy_ceiling",
            "one soft-label run, then stop optimizing neutral on this benchmark",
        )
    else:
        branch, next_step = (
            "representation",
            "raw-text or social-domain encoder (BamiBERT / ViSoBERT) or TAPT",
        )
    return {
        "scope_strata": list(SCOPE_STRATA),
        "n": n,
        "share_incorrect": incorrect,
        "share_ambiguous": ambiguous,
        "thresholds": {"incorrect": INCORRECT_THRESHOLD, "ambiguous": AMBIGUOUS_THRESHOLD},
        "branch": branch,
        "next": next_step,
    }


def report(
    first: pd.DataFrame, second: pd.DataFrame | None = None, kind: str = "intra"
) -> dict[str, Any]:
    """The full audit report. Refuses an incomplete first sheet rather than analysing part of it."""
    problems = validate(first)
    if problems:
        raise ValueError("audit sheet is not complete: " + "; ".join(problems))
    a = _clean(first)
    out: dict[str, Any] = {
        "rows": len(a),
        "per_stratum": per_stratum(a),
        "random_stratum": random_stratum_rates(a),
        "decision": decide(a),
    }
    if second is not None:
        out["agreement"] = agreement(a, _clean(second), kind)
    return out
