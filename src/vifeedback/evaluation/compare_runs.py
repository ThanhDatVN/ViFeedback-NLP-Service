"""Result tables and paired comparisons, regenerated from the registry — never retyped.

Review task (P0): every number in the README and docs/EXPERIMENT_MATRIX.md should be the output of a
query over `results/registry.csv`, so a table cannot drift from the runs it summarizes.

Three hygiene steps happen before any aggregate is computed, and each is reported rather than
silently applied:

1. **Exact duplicate run ids** — the same run logged twice (a re-invoked script). Kept once.
2. **Re-runs of one condition and seed under different ids** — e.g. Phase 3 conditions re-trained in
   Phase 4. Counting both would double the seed count and shrink the standard deviation. Kept once;
   the disagreement between them is reported, since it measures run-to-run determinism.
3. **Grouping by condition, not phase** — the phase is bookkeeping; the condition
   (task, model, preprocessing, recipe, split) is what the number describes.

Seed-level comparisons pair runs by seed (docs/EVALUATION_PROTOCOL.md § 3): each seed fixes data order
and head initialization, so the within-seed difference removes that shared variance.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from vifeedback import paths

CONDITION = ("task", "model", "preprocessing", "recipe", "split")
METRICS = ("macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "mcc")


def load(registry: Any = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The registry with duplicates removed, plus a report of what was removed and why."""
    df = pd.read_csv(registry or paths.REGISTRY)
    hygiene: dict[str, Any] = {"rows_in_registry": len(df)}

    dup = df.run_id.duplicated(keep="last")
    hygiene["duplicate_run_ids"] = sorted(df.loc[dup, "run_id"].unique().tolist())
    df = df[~dup]

    key = [*CONDITION, "seed"]
    reruns = df[df.duplicated(key, keep=False)]
    spread = reruns.groupby(key)["macro_f1"].agg(lambda s: float(s.max() - s.min()))
    hygiene["reruns"] = {
        "n_condition_seeds": len(spread),
        "max_abs_macro_f1_disagreement": float(spread.max()) if len(spread) else None,
    }
    df = df.drop_duplicates(key, keep="last")
    hygiene["rows_used"] = len(df)
    return df.reset_index(drop=True), hygiene


def condition_table(df: pd.DataFrame, metrics: tuple[str, ...] = METRICS) -> pd.DataFrame:
    """One row per condition: seed count, mean, sample std (ddof=1), min and max per metric."""
    rows = []
    for cond, g in df.groupby(list(CONDITION), dropna=False):
        row: dict[str, Any] = dict(zip(CONDITION, cond, strict=True))
        row["n_seeds"] = len(g)
        row["seeds"] = ",".join(str(s) for s in sorted(g.seed))
        for m in metrics:
            v = g[m].astype(float)
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[f"{m}_min"], row[f"{m}_max"] = v.min(), v.max()
        rows.append(row)
    return pd.DataFrame(rows)


def _select(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    m = np.ones(len(df), dtype=bool)
    for k, v in spec.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


def paired_seed_comparison(
    df: pd.DataFrame, a: dict[str, Any], b: dict[str, Any], metric: str = "macro_f1"
) -> dict[str, Any]:
    """b - a on the seeds both conditions share: mean difference, t-based 95% CI, paired t-test.

    With five seeds the test has little power and the CI is wide; both are reported so that "no
    significant difference" is read as "not resolved at this budget", not as "equal".
    """
    from scipy import stats

    sa = _select(df, a).set_index("seed")[metric].astype(float)
    sb = _select(df, b).set_index("seed")[metric].astype(float)
    seeds = sorted(set(sa.index) & set(sb.index))
    out: dict[str, Any] = {"a": a, "b": b, "metric": metric, "seeds": seeds, "n": len(seeds)}
    # A seed fixes data order for every model, but head initialization only within one architecture
    # (review R7). Cross-architecture pairs share less, and the label says so.
    same_arch = a.get("model", "phobert-base") == b.get("model", "phobert-base")
    out["pairing"] = (
        "same architecture: data order and head initialization"
        if same_arch
        else "cross-architecture: data order only"
    )
    if len(seeds) < 2:
        out["note"] = "fewer than two shared seeds; no paired test"
        return out
    d = (sb.loc[seeds] - sa.loc[seeds]).to_numpy()
    se = d.std(ddof=1) / np.sqrt(len(d))
    half = float(stats.t.ppf(0.975, len(d) - 1) * se)
    t = stats.ttest_rel(sb.loc[seeds], sa.loc[seeds])
    out.update(
        {
            "mean_a": float(sa.loc[seeds].mean()),
            "mean_b": float(sb.loc[seeds].mean()),
            "mean_diff": float(d.mean()),
            "ci95": [float(d.mean() - half), float(d.mean() + half)],
            "wins_b": int((d > 0).sum()),
            "t": float(t.statistic),
            "p": float(t.pvalue),
        }
    )
    return out


def with_fdr(comparisons: list[dict[str, Any]], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Benjamini-Hochberg across one family of comparisons (docs/EVALUATION_PROTOCOL.md § 3)."""
    from vifeedback.evaluation.bootstrap import benjamini_hochberg

    tested = [c for c in comparisons if "p" in c]
    flags = benjamini_hochberg([c["p"] for c in tested], alpha=alpha)
    for c, f in zip(tested, flags, strict=True):
        c["significant_bh"] = bool(f)
    return comparisons


def to_markdown(table: pd.DataFrame, metric: str = "macro_f1") -> str:
    lines = [
        "| task | model | preprocessing | recipe | split | seeds | "
        f"{metric} mean ± std | min | max |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, r in table.iterrows():
        std = "" if pd.isna(r[f"{metric}_std"]) else f" ± {r[f'{metric}_std']:.4f}"
        lines.append(
            f"| {r.task} | {r.model} | {r.preprocessing} | {r.recipe} | {r.split} | "
            f"{r.n_seeds} | {r[f'{metric}_mean']:.4f}{std} | {r[f'{metric}_min']:.4f} | "
            f"{r[f'{metric}_max']:.4f} |"
        )
    return "\n".join(lines)


# The comparison family, declared here rather than chosen after seeing the table. Validation only:
# the test-set comparisons were run once each at Gate G4 and are not re-tested here.
_PB = {"model": "phobert-base", "recipe": "base", "split": "validation"}
COMPARISONS: tuple[tuple[str, dict[str, Any], dict[str, Any]], ...] = (
    (
        "sentiment: pyvi vs raw",
        {**_PB, "task": "sentiment", "preprocessing": "raw"},
        {**_PB, "task": "sentiment", "preprocessing": "seg_pyvi"},
    ),
    (
        "sentiment: vncorenlp vs raw",
        {**_PB, "task": "sentiment", "preprocessing": "raw"},
        {**_PB, "task": "sentiment", "preprocessing": "seg_vncorenlp"},
    ),
    (
        "sentiment: underthesea vs raw",
        {**_PB, "task": "sentiment", "preprocessing": "raw"},
        {**_PB, "task": "sentiment", "preprocessing": "seg_underthesea"},
    ),
    (
        "sentiment: vncorenlp vs pyvi",
        {**_PB, "task": "sentiment", "preprocessing": "seg_pyvi"},
        {**_PB, "task": "sentiment", "preprocessing": "seg_vncorenlp"},
    ),
    (
        "sentiment: phobert-large vs base (pyvi)",
        {**_PB, "task": "sentiment", "preprocessing": "seg_pyvi"},
        {**_PB, "task": "sentiment", "preprocessing": "seg_pyvi", "model": "phobert-large"},
    ),
    (
        # Confounded (ADR-019): XLM-R was never pretrained on segmented input.
        "sentiment: xlmr-base vs phobert-base (pyvi)",
        {**_PB, "task": "sentiment", "preprocessing": "seg_pyvi"},
        {**_PB, "task": "sentiment", "preprocessing": "seg_pyvi", "model": "xlmr-base"},
    ),
    (
        "topic: pyvi vs raw",
        {**_PB, "task": "topic", "preprocessing": "raw"},
        {**_PB, "task": "topic", "preprocessing": "seg_pyvi"},
    ),
)


def build(registry: Any = None) -> dict[str, Any]:
    df, hygiene = load(registry)
    table = condition_table(df)
    comps = []
    for name, a, b in COMPARISONS:
        c = paired_seed_comparison(df, a, b)
        c["name"] = name
        comps.append(c)
    return {"hygiene": hygiene, "table": table, "comparisons": with_fdr(comps)}
