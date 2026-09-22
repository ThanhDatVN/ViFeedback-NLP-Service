"""Run artifacts and the experiment registry.

Every run writes `results/runs/<run_id>/{config.yaml,metrics.json,preds.csv,confusion.png,env.json}`
and appends one row to `results/registry.csv`. The registry is append-only and committed: it *is* the
experiment log, which makes every table in docs/EXPERIMENT_MATRIX.md a query rather than a retyping
exercise, and every claim in the README traceable to a run_id (docs/ROADMAP.md § 4).

Test-set evaluations additionally append to `results/test_evaluations.log`, so the count of times the
test set has been touched stays visible (docs/EVALUATION_PROTOCOL.md § 4).
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from typing import Any

import numpy as np

from vifeedback import env, paths

REGISTRY_FIELDS = (
    "run_id",
    "date",
    "phase",
    "task",
    "model",
    "preprocessing",
    "recipe",
    "seed",
    "split",
    "macro_f1",
    "weighted_f1",
    "accuracy",
    "balanced_accuracy",
    "mcc",
    "macro_f1_ci_low",
    "macro_f1_ci_high",
    "p95_ms",
    "size_mb",
    "fit_seconds",
    "notes",
    "git_sha",
)


def yaml_safe(obj: Any) -> Any:
    """Coerce a config tree into something `yaml.safe_dump` accepts.

    Defensive by design. A run that has already spent five GPU-minutes must not be lost because one
    metadata value is an exotic type — `torch.__version__` is a `str` subclass that `safe_dump`
    refuses, and it cost exactly that once. Unknown types degrade to `str` rather than raising.
    """
    import numpy as _np

    if isinstance(obj, dict):
        return {str(k): yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [yaml_safe(v) for v in obj]
    # np.bool_ is NOT a subclass of Python bool, so it must be caught before the numeric
    # branches; otherwise it falls through to str() and a config records `fp16: 'True'`.
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return yaml_safe(obj.tolist())
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, int) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)  # drops str subclasses such as torch's TorchVersion
    return str(obj)


def save_run(
    run_id: str,
    metrics: dict[str, Any],
    *,
    config: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    texts: list[str] | None = None,
    y_prob: np.ndarray | None = None,
    figure: bool = True,
) -> dict[str, Any]:
    """Write one run's artifacts and append its registry row."""
    import yaml

    d = paths.run_dir(run_id)
    (d / "config.yaml").write_text(
        yaml.safe_dump(yaml_safe(config), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (d / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env_info = env.write(d / "env.json", extra={"run_id": run_id})

    with open(d / "preds.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["idx", "y_true", "y_pred"]
        if texts is not None:
            header.append("text")
        if y_prob is not None:
            header += [f"p_{n}" for n in metrics["labels"]]
        w.writerow(header)
        for i in range(len(y_true)):
            row: list[Any] = [i, int(y_true[i]), int(y_pred[i])]
            if texts is not None:
                row.append(texts[i])
            if y_prob is not None:
                row += [round(float(v), 6) for v in y_prob[i]]
            w.writerow(row)

    if figure:
        plot_confusion(metrics, d / "confusion.png", title=run_id)

    append_registry(run_id, metrics, config, env_info)

    if config.get("split") == "test":
        log_test_evaluation(run_id, config)

    return {"dir": str(d), "run_id": run_id}


def append_registry(
    run_id: str, metrics: dict[str, Any], config: dict[str, Any], env_info: dict[str, Any]
) -> None:
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    new = not paths.REGISTRY.exists()
    ci = metrics.get("macro_f1_ci")
    row = {
        "run_id": run_id,
        "date": datetime.now(UTC).strftime("%Y-%m-%d"),
        "phase": config.get("phase", ""),
        "task": config.get("task", ""),
        "model": config.get("model", ""),
        "preprocessing": config.get("preprocessing", ""),
        "recipe": config.get("recipe", ""),
        "seed": config.get("seed", ""),
        "split": config.get("split", ""),
        "macro_f1": round(metrics["macro_f1"], 4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
        "accuracy": round(metrics["accuracy"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "mcc": round(metrics["mcc"], 4),
        "macro_f1_ci_low": round(ci[0], 4) if ci else "",
        "macro_f1_ci_high": round(ci[1], 4) if ci else "",
        "p95_ms": config.get("p95_ms", ""),
        "size_mb": config.get("size_mb", ""),
        "fit_seconds": config.get("fit_seconds", ""),
        "notes": config.get("notes", ""),
        "git_sha": (env_info.get("git", {}) or {}).get("sha", "") or "",
    }
    with open(paths.REGISTRY, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def log_test_evaluation(run_id: str, config: dict[str, Any]) -> None:
    """Append to the visible test-set evaluation log (§ 4 test-set discipline)."""
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    line = (
        f"{datetime.now(UTC).strftime('%Y-%m-%d %H:%M')}\t{run_id}\t"
        f"{config.get('phase', '')}\t{config.get('reason', 'gate evaluation')}\n"
    )
    with open(paths.TEST_EVAL_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def test_evaluation_count() -> int:
    if not paths.TEST_EVAL_LOG.exists():
        return 0
    return sum(1 for _ in paths.TEST_EVAL_LOG.open(encoding="utf-8"))


def load_registry():
    import pandas as pd

    if not paths.REGISTRY.exists():
        return pd.DataFrame(columns=list(REGISTRY_FIELDS))
    return pd.read_csv(paths.REGISTRY)


# --- Figures ------------------------------------------------------------------------------------


def plot_confusion(metrics: dict[str, Any], out_path, title: str = "") -> None:
    """Counts and row-normalized rates side by side — counts show volume, rates show behaviour."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(metrics["confusion"])
    cmn = np.array(metrics["confusion_normalized"])
    names = metrics["labels"]

    fig, axes = plt.subplots(1, 2, figsize=(4.2 + 2.2 * len(names), 3.4 + 0.55 * len(names)))
    for ax, mat, label, fmt in (
        (axes[0], cm, "counts", "{:d}"),
        (axes[1], cmn, "row-normalized", "{:.2f}"),
    ):
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(label, fontsize=10)
        for r in range(len(names)):
            for c in range(len(names)):
                v = mat[r, c]
                ax.text(
                    c,
                    r,
                    fmt.format(int(v) if fmt == "{:d}" else v),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if cmn[r, c] > 0.55 else "black",
                )
    fig.colorbar(im, ax=axes, fraction=0.03, label="row-normalized rate")
    fig.suptitle(
        f"{title}\nmacro-F1 {metrics['macro_f1']:.3f} · weighted-F1 {metrics['weighted_f1']:.3f} "
        f"· acc {metrics['accuracy']:.3f}",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
