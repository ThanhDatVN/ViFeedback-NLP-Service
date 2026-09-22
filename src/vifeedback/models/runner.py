"""Phase 1 baseline runner.

Executes the ladder for one task, evaluates every rung with the shared harness, and writes one run
directory plus one registry row per rung. Dev is used for all selection; test is evaluated only when
explicitly requested at a gate, and every such evaluation is logged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vifeedback.constants import n_classes
from vifeedback.data.loader import load
from vifeedback.evaluation import bootstrap as B
from vifeedback.evaluation import metrics as M
from vifeedback.evaluation import report as R
from vifeedback.models.baseline_tfidf import (
    LADDER,
    LADDER_BY_KEY,
    fit_with_dev_selection,
    predict_proba,
    tune_class_priors,
)


def run_ladder(
    task: str,
    *,
    seed: int = 42,
    include_test: bool = False,
    reason: str = "",
    keys: tuple[str, ...] | None = None,
    n_bootstrap: int = 2000,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    k = n_classes(task)
    tr, dv, te = load("train"), load("validation"), load("test")
    x_tr, y_tr = tr["sentence"].tolist(), tr[task].to_numpy()
    x_dv, y_dv = dv["sentence"].tolist(), dv[task].to_numpy()
    x_te, y_te = te["sentence"].tolist(), te[task].to_numpy()

    specs = [LADDER_BY_KEY[key] for key in keys] if keys else list(LADDER)
    results: list[dict[str, Any]] = []

    for spec in specs:
        fitted = fit_with_dev_selection(spec, x_tr, y_tr, x_dv, y_dv, task, seed=seed)
        model = fitted["model"]

        for split, x, y in (("validation", x_dv, y_dv), ("test", x_te, y_te)):
            if split == "test" and not include_test:
                continue

            y_pred = model.predict(x)
            y_prob = predict_proba(model, x, k)
            metrics = M.evaluate(y, y_pred, task, y_prob=y_prob)

            ci = B.bootstrap_ci(y, y_pred, k, n_resamples=n_bootstrap, seed=seed)
            metrics["macro_f1_ci"] = [ci["ci_low"], ci["ci_high"]]
            metrics["macro_f1_bootstrap_std"] = ci["std"]

            run_id = f"p1-{task[:4]}-{spec.key}-raw-base-s{seed}-{split[:3]}"
            config = {
                "phase": "P1",
                "task": task,
                "model": spec.key,
                "model_label": spec.label,
                "preprocessing": "raw",
                "recipe": "base",
                "seed": seed,
                "split": split,
                "best_params": fitted.get("best_params", {}),
                "fit_seconds": round(fitted["fit_seconds"], 2),
                "notes": spec.label,
                "reason": reason or "Gate G1 baseline ladder",
            }
            R.save_run(
                run_id,
                metrics,
                config=config,
                y_true=y,
                y_pred=y_pred,
                texts=x if split == "test" else None,
                y_prob=y_prob,
            )
            results.append({"spec": spec, "split": split, "metrics": metrics, "run_id": run_id})

            if verbose:
                print(M.format_report(metrics, f"\n{spec.label}  [{task} / {split}]"))

        # Decision-threshold tuning: fit the priors on dev, then apply them unchanged.
        if spec.search_space and spec.supports_proba:
            prob_dv = predict_proba(model, x_dv, k)
            if prob_dv is not None:
                w = tune_class_priors(prob_dv, y_dv, k, seed=seed)
                for split, x, y in (("validation", x_dv, y_dv), ("test", x_te, y_te)):
                    if split == "test" and not include_test:
                        continue
                    prob = predict_proba(model, x, k)
                    y_pred = (prob * w).argmax(axis=1)
                    metrics = M.evaluate(y, y_pred, task, y_prob=prob)
                    ci = B.bootstrap_ci(y, y_pred, k, n_resamples=n_bootstrap, seed=seed)
                    metrics["macro_f1_ci"] = [ci["ci_low"], ci["ci_high"]]
                    metrics["class_priors"] = [round(float(v), 4) for v in w]

                    run_id = f"p1-{task[:4]}-{spec.key}-raw-thresh-s{seed}-{split[:3]}"
                    config = {
                        "phase": "P1",
                        "task": task,
                        "model": spec.key,
                        "model_label": spec.label + " + tuned priors",
                        "preprocessing": "raw",
                        "recipe": "threshold-tuned",
                        "seed": seed,
                        "split": split,
                        "class_priors": [float(v) for v in w],
                        "fit_seconds": round(fitted["fit_seconds"], 2),
                        "notes": "per-class priors tuned on dev for macro-F1",
                        "reason": reason or "Gate G1 baseline ladder",
                    }
                    R.save_run(
                        run_id,
                        metrics,
                        config=config,
                        y_true=y,
                        y_pred=y_pred,
                        texts=x if split == "test" else None,
                        y_prob=prob,
                    )
                    results.append(
                        {
                            "spec": spec,
                            "split": split,
                            "metrics": metrics,
                            "run_id": run_id,
                            "tuned": True,
                        }
                    )
                    if verbose:
                        print(
                            M.format_report(
                                metrics, f"\n{spec.label} + tuned priors  [{task} / {split}]"
                            )
                        )

    return results


def summary_table(results: list[dict[str, Any]], split: str = "validation") -> str:
    """Compact ladder table, ordered as run. Per-class F1 columns come last: they are the point."""
    rows = [r for r in results if r["split"] == split]
    if not rows:
        return "(no results)"
    names = rows[0]["metrics"]["labels"]

    head = f"{'model':<38s} {'macroF1':>8s} {'wtdF1':>7s} {'acc':>6s} {'MCC':>6s}  " + "  ".join(
        f"F1:{n[:9]:>9s}" for n in names
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        m = r["metrics"]
        label = r["spec"].label + (" + priors" if r.get("tuned") else "")
        per = "  ".join(f"{m['per_class'][n]['f1']:>13.3f}" for n in names)
        lines.append(
            f"{label:<38s} {m['macro_f1']:>8.3f} {m['weighted_f1']:>7.3f} "
            f"{m['accuracy']:>6.3f} {m['mcc']:>6.3f}  {per}"
        )
    return "\n".join(lines)


def compare_best(
    results: list[dict[str, Any]], task: str, split: str = "validation"
) -> dict[str, Any]:
    """Paired bootstrap of the ladder's best rung against the conventional B1 word baseline."""
    rows = [r for r in results if r["split"] == split]
    b1 = next((r for r in rows if r["spec"].key == "b1" and not r.get("tuned")), None)
    best = max(rows, key=lambda r: r["metrics"]["macro_f1"])
    if b1 is None or best["run_id"] == b1["run_id"]:
        return {}

    import pandas as pd

    from vifeedback import paths

    def preds(run_id: str) -> np.ndarray:
        return pd.read_csv(paths.RUNS / run_id / "preds.csv")["y_pred"].to_numpy()

    y_true = pd.read_csv(paths.RUNS / b1["run_id"] / "preds.csv")["y_true"].to_numpy()
    out = B.paired_bootstrap(
        y_true, preds(best["run_id"]), preds(b1["run_id"]), n_classes(task), n_resamples=2000
    )
    out["best"] = best["spec"].label + (" + priors" if best.get("tuned") else "")
    out["baseline"] = b1["spec"].label
    return out
