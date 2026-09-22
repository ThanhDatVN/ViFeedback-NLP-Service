"""Phase 2+ transformer run orchestration.

Wraps `trainer.train` with the project's measurement contract: evaluate with the shared harness,
write run artifacts, append to the registry, and aggregate across seeds as mean ± std — because a
single-seed number on a 458-example minority class is noise dressed as a result.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vifeedback import paths
from vifeedback.constants import SEEDS, n_classes
from vifeedback.evaluation import bootstrap as B
from vifeedback.evaluation import metrics as M
from vifeedback.evaluation import report as R
from vifeedback.training.trainer import TextDataset, TrainConfig, predict, softmax, train


def _splits(task: str, preprocessing: str = "raw"):
    """Load the three splits of one preprocessing variant.

    `preprocessing` names a materialized variant (Phase 3). `raw` is condition P0 and reads the
    corpus directly, so Phase 2's runs are already P0 and need no re-running.
    """
    from vifeedback.preprocess.variants import load_variant

    tr = load_variant(preprocessing, "train")
    dv = load_variant(preprocessing, "validation")
    te = load_variant(preprocessing, "test")
    col = "sentence"
    return (
        (tr[col].tolist(), tr[task].to_numpy()),
        (dv[col].tolist(), dv[task].to_numpy()),
        (te[col].tolist(), te[task].to_numpy()),
    )


def run_once(
    cfg: TrainConfig,
    *,
    include_test: bool = False,
    reason: str = "",
    n_bootstrap: int = 2000,
    verbose: bool = True,
    save: bool = True,
    save_checkpoint: bool = False,
) -> dict[str, Any]:
    """One seed, one configuration. Returns per-split metrics.

    `save_checkpoint` writes the selected weights to `models/<run_id>/`. Off by default because a
    135M-parameter checkpoint is ~540 MB and a 5-seed sweep would be 2.7 GB; on for anything that
    might become a champion, so a later test evaluation costs seconds rather than a retrain.
    """
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    (x_tr, y_tr), (x_dv, y_dv), (x_te, y_te) = _splits(cfg.task, cfg.preprocessing)
    k = n_classes(cfg.task)

    if verbose:
        print(f"\n[{cfg.run_id()}]  {cfg.model_key} / {cfg.task} / {cfg.recipe} / seed {cfg.seed}")

    out = train(cfg, x_tr, y_tr, x_dv, y_dv, verbose=verbose)
    model, tokenizer = out["model"], out["tokenizer"]
    collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

    evaluated: dict[str, Any] = {}
    targets = [("validation", x_dv, y_dv)] + ([("test", x_te, y_te)] if include_test else [])

    for split, x, y in targets:
        ds = TextDataset(x, y, tokenizer, cfg.max_length)
        loader = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collator)
        logits, labels = predict(model, loader, out["device"], cfg.fp16)
        prob = softmax(logits)
        y_pred = logits.argmax(axis=1)

        metrics = M.evaluate(labels, y_pred, cfg.task, y_prob=prob)
        ci = B.bootstrap_ci(labels, y_pred, k, n_resamples=n_bootstrap, seed=cfg.seed)
        metrics["macro_f1_ci"] = [ci["ci_low"], ci["ci_high"]]
        metrics["history"] = out["history"]
        metrics["best_epoch"] = out["best_epoch"]

        if save:
            config = dict(out["config"])
            config.update(
                {
                    "phase": f"P{cfg.extra.get('phase_num', 2)}",
                    "model": cfg.model_key,
                    "split": split,
                    "fit_seconds": out["train_seconds"],
                    "notes": cfg.notes or f"{cfg.model_key} {cfg.recipe}",
                    "reason": reason or "gate evaluation",
                    "determinism": out["determinism"],
                    "device": out["device"],
                    "best_epoch": out["best_epoch"],
                }
            )
            R.save_run(
                cfg.run_id(split),
                metrics,
                config=config,
                y_true=labels,
                y_pred=y_pred,
                texts=x if split == "test" else None,
                y_prob=prob,
            )

        evaluated[split] = metrics
        if verbose:
            print(M.format_report(metrics, f"  --> {split}"))

    if save_checkpoint:
        ckpt = paths.MODELS / cfg.run_id("ckpt")
        ckpt.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        if verbose:
            print(f"  checkpoint: {ckpt}")

    return {
        "config": cfg,
        "metrics": evaluated,
        "history": out["history"],
        "best_epoch": out["best_epoch"],
        "train_seconds": out["train_seconds"],
        "model": model,
        "tokenizer": tokenizer,
    }


def run_seeds(
    cfg: TrainConfig,
    seeds: tuple[int, ...] = SEEDS,
    *,
    include_test: bool = False,
    reason: str = "",
    verbose: bool = True,
    keep_models: bool = False,
    save_checkpoint: bool = False,
) -> dict[str, Any]:
    """Run the same configuration across seeds and aggregate as mean ± std."""
    import dataclasses

    runs = []
    for seed in seeds:
        seeded = dataclasses.replace(cfg, seed=seed)
        r = run_once(
            seeded,
            include_test=include_test,
            reason=reason,
            verbose=verbose,
            save_checkpoint=save_checkpoint,
        )
        if not keep_models:
            r.pop("model", None)
            r.pop("tokenizer", None)
            _free_cuda()
        runs.append(r)

    return {"config": cfg, "seeds": list(seeds), "runs": runs, "summary": aggregate(runs)}


def _free_cuda() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean ± std across seeds, per split, including per-class F1."""
    out: dict[str, Any] = {}
    for split in ("validation", "test"):
        present = [r for r in runs if split in r["metrics"]]
        if not present:
            continue
        ms = [r["metrics"][split] for r in present]
        entry: dict[str, Any] = {"n_seeds": len(ms)}
        for key in ("macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "mcc"):
            vals = np.array([m[key] for m in ms])
            entry[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        entry["per_class_f1"] = {}
        for name in ms[0]["per_class"]:
            vals = np.array([m["per_class"][name]["f1"] for m in ms])
            entry["per_class_f1"][name] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            }
        entry["best_epochs"] = [r["best_epoch"] for r in present]
        out[split] = entry
    return out


def format_summary(summary: dict[str, Any], title: str = "") -> str:
    lines = [title] if title else []
    for split, e in summary.items():
        lines.append(f"  [{split}]  n_seeds={e['n_seeds']}  best_epochs={e['best_epochs']}")
        for key in ("macro_f1", "weighted_f1", "accuracy"):
            s = e[key]
            lines.append(
                f"    {key:<14s} {s['mean']:.4f} ± {s['std']:.4f}   "
                f"[{s['min']:.4f}, {s['max']:.4f}]"
            )
        per = "  ".join(f"{n}={v['mean']:.3f}±{v['std']:.3f}" for n, v in e["per_class_f1"].items())
        lines.append(f"    per-class F1   {per}")
    return "\n".join(lines)
