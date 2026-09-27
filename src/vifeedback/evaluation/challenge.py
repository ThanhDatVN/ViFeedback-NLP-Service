"""Challenge set v1 (Cycle 2): hash-checked loading, per-category scoring, and the H6 serving rule.

The set is constructed, not sampled (data/challenge/README.md), so its aggregate accuracy means
little. Every number is reported per category, and the declared rules read specific categories.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vifeedback import paths
from vifeedback.constants import label_names

CHALLENGE_V1 = paths.DATA / "challenge" / "challenge_v1.csv"
CYCLE2 = paths.CONFIGS / "experiments" / "cycle2.yaml"
TYPED = ("unaccented_typed", "teencode_typed")
OUT_OF_SCOPE = "out_of_scope"

# H6 (cycle2.yaml): switch only if typed-noise accuracy improves with a paired CI above 0, and no
# other scored row group loses more than this much accuracy.
H6_MAX_OTHER_LOSS = 0.02


def declared_sha256() -> str:
    import yaml

    spec = yaml.safe_load(CYCLE2.read_text(encoding="utf-8"))
    return str(spec["evaluation_data"]["challenge_set_v1"]["sha256"])


def load(path: Path = CHALLENGE_V1, verify: bool = True) -> pd.DataFrame:
    """The frozen set. Line endings are normalized first, so a CRLF checkout hashes the same."""
    raw = path.read_bytes().replace(b"\r\n", b"\n")
    if verify:
        got, want = hashlib.sha256(raw).hexdigest(), declared_sha256()
        if got != want:
            raise ValueError(f"{path} sha256 {got[:12]} does not match cycle2.yaml ({want[:12]})")
    df = pd.read_csv(io.BytesIO(raw), keep_default_na=False, dtype=str)
    df["y"] = df.sentiment.map({n: i for i, n in enumerate(label_names("sentiment"))})
    df["scored"] = df.y.notna()
    return df


def pipeline(preprocessing: str = "seg_pyvi"):
    """Raw text -> the model's input, as training and serving build it."""
    from vifeedback.preprocess.normalize import basic_clean
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import VARIANTS

    backend, clean, _ = VARIANTS[preprocessing]
    seg = get_segmenter(backend)
    return lambda texts: seg([basic_clean(t) for t in texts] if clean else list(texts))


def _accuracy(cm: np.ndarray) -> float:
    return float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan")


def category_report(df: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    """Accuracy per category, plus macro-F1 and per-class F1 over all scored rows."""
    from vifeedback.evaluation import metrics as M

    s = df.scored.to_numpy()
    y, p = df.y.to_numpy()[s].astype(int), np.asarray(pred)[s]
    overall = M.evaluate(y, p, "sentiment")
    out: dict[str, Any] = {
        "scored_rows": int(s.sum()),
        "accuracy": float((y == p).mean()),
        "macro_f1": overall["macro_f1"],
        "per_class_f1": {c: v["f1"] for c, v in overall["per_class"].items()},
        "by_category": {},
    }
    for cat, g in df[s].groupby("category", sort=True):
        idx = g.index.to_numpy()
        out["by_category"][cat] = {
            "n": len(idx),
            "accuracy": float((df.y.to_numpy()[idx].astype(int) == np.asarray(pred)[idx]).mean()),
        }
    return out


def negation_pairs(df: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    """Minimal pairs: both members right, and whether the prediction flips with the negation."""
    g = df[df.pair_id != ""].assign(pred=np.asarray(pred)[df.pair_id != ""])
    both = flips = 0
    for _, pair in g.groupby("pair_id"):
        both += int((pair.pred == pair.y.astype(int)).all())
        flips += int(pair.pred.nunique() == 2)
    n = g.pair_id.nunique()
    return {"pairs": n, "both_correct": both / n, "prediction_flips": flips / n}


def out_of_scope_confidence(df: pd.DataFrame, probs: np.ndarray) -> dict[str, Any]:
    """Descriptive only: the encoder has no abstain class, so ask how sure it is off-topic."""
    conf = np.asarray(probs).max(axis=1)
    out: dict[str, Any] = {}
    for cat in (OUT_OF_SCOPE, "objective_neutral"):
        m = (df.category == cat).to_numpy()
        c = conf[m]
        labels = label_names("sentiment")
        out[cat] = {
            "n": int(m.sum()),
            "max_prob_mean": float(c.mean()),
            "max_prob_median": float(np.median(c)),
            "share_above_0_9": float((c > 0.9).mean()),
            "predicted": {
                labels[k]: int(v)
                for k, v in zip(*np.unique(probs[m].argmax(1), return_counts=True), strict=True)
            },
        }
    return out


def h6(
    df: pd.DataFrame, pred_ce: np.ndarray, pred_aug: np.ndarray, seed: int = 42
) -> dict[str, Any]:
    """The declared serving rule (cycle2.yaml H6_serving_model)."""
    from vifeedback.evaluation import bootstrap as B

    s = df.scored.to_numpy()
    typed = s & df.category.isin(TYPED).to_numpy()
    other = s & ~typed
    y = df.y.to_numpy()
    k = len(label_names("sentiment"))
    pb = B.paired_bootstrap(
        y[typed].astype(int),
        np.asarray(pred_aug)[typed],
        np.asarray(pred_ce)[typed],
        k,
        n_resamples=10_000,
        seed=seed,
        metric=_accuracy,
    )
    acc = lambda p, m: float((np.asarray(p)[m] == y[m].astype(int)).mean())  # noqa: E731
    other_diff = acc(pred_aug, other) - acc(pred_ce, other)
    switch = bool(pb["ci_low"] > 0 and other_diff >= -H6_MAX_OTHER_LOSS)
    return {
        "typed_rows": int(typed.sum()),
        "typed_accuracy": {"ce": acc(pred_ce, typed), "augmented": acc(pred_aug, typed)},
        "typed_paired_aug_minus_ce": pb,
        "other_rows": int(other.sum()),
        "other_accuracy": {"ce": acc(pred_ce, other), "augmented": acc(pred_aug, other)},
        "other_diff": other_diff,
        "max_other_loss": H6_MAX_OTHER_LOSS,
        "switch": switch,
        "decision": "serve the augmented model" if switch else "keep the CE model",
    }


def _exact_mcnemar(b: int, c: int) -> float:
    from math import comb

    n, k = b + c, min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0


def seed_paired_drops(
    df: pd.DataFrame, ce: dict[str, np.ndarray], aug: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Cycle 3 V1 (cycle3.yaml): per category, in how many seeds the augmented model is less
    accurate than CE, and an exact McNemar test pooled over seeds (discordant pairs summed).

    A drop is confirmed when augmented < CE in at least 4 of 5 seeds and the pooled p < 0.05.
    """
    s = df.scored.to_numpy()
    y = df.y.to_numpy()
    seeds = sorted(set(ce) & set(aug), key=int)
    out: dict[str, Any] = {"seeds": seeds, "categories": {}}
    for cat in sorted(df[s].category.unique()):
        m = s & (df.category == cat).to_numpy()
        yc = y[m].astype(int)
        per_seed, lower, b, c = {}, 0, 0, 0
        for sd in seeds:
            right_ce, right_aug = ce[sd][m] == yc, aug[sd][m] == yc
            per_seed[sd] = {"ce": float(right_ce.mean()), "augmented": float(right_aug.mean())}
            lower += int(right_aug.mean() < right_ce.mean())
            b += int((right_ce & ~right_aug).sum())
            c += int((~right_ce & right_aug).sum())
        p = _exact_mcnemar(b, c)
        out["categories"][cat] = {
            "n": int(m.sum()),
            "mean_accuracy": {
                "ce": float(np.mean([v["ce"] for v in per_seed.values()])),
                "augmented": float(np.mean([v["augmented"] for v in per_seed.values()])),
            },
            "per_seed": per_seed,
            "seeds_augmented_lower": lower,
            "ce_only_right": b,
            "augmented_only_right": c,
            "pooled_exact_mcnemar_p": p,
            "confirmed_drop": bool(lower >= 4 and p < 0.05 and b > c),
        }
    return out
