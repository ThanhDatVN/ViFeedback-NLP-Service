"""`vifeedback train`: transformer fine-tuning, single-task and multi-task."""

from __future__ import annotations

import typer

from vifeedback.cli._apps import train_app


@train_app.command("run")
def train_run(
    task: str = typer.Option("sentiment", help="sentiment | topic"),
    model: str = typer.Option("phobert-base", help="key from constants.MODEL_IDS"),
    recipe: str = typer.Option("base", help="base | classweight | focal | logit-adjust | ..."),
    preprocessing: str = typer.Option("raw", help="variant name; raw = condition P0"),
    seeds: str = typer.Option("42", help="comma-separated, or 'all' for the 5 canonical seeds"),
    epochs: int = typer.Option(4),
    lr: float = typer.Option(2e-5),
    batch_size: int = typer.Option(32, help="per-step batch; see --grad-accum"),
    grad_accum: int = typer.Option(
        1,
        help="gradient accumulation steps. EFFECTIVE batch = batch_size * grad_accum. "
        "A model compared against another at a different effective batch is not a "
        "controlled comparison.",
    ),
    eval_batch_size: int = typer.Option(128),
    warmup_ratio: float = typer.Option(0.1),
    weight_decay: float = typer.Option(0.01),
    weight_scheme: str = typer.Option("balanced", help="balanced | sqrt | effective"),
    focal_gamma: float = typer.Option(2.0),
    tau: float = typer.Option(1.0, help="logit-adjustment tau"),
    freeze_embeddings: bool = typer.Option(
        False,
        help="freeze the embedding matrix. Makes xlm-roberta-base fit a 4GB GPU "
        "(192M of its 277M params are embeddings).",
    ),
    save_checkpoint: bool = typer.Option(False, help="write weights to models/<run_id>/"),
    max_length: int = typer.Option(96, help="Gate G0 decision: PhoBERT subword p99.9 = 87"),
    llrd: float = typer.Option(0.0, help="layer-wise lr decay, e.g. 0.9; 0 disables"),
    rdrop: float = typer.Option(0.0, help="R-Drop alpha; 0 disables"),
    fgm: float = typer.Option(0.0, help="FGM epsilon; 0 disables"),
    label_smoothing: float = typer.Option(0.0),
    phase: int = typer.Option(2, help="phase number, used in the run id"),
    include_test: bool = typer.Option(False, help="Also evaluate on test (logged; gates only)"),
    reason: str = typer.Option("", help="Why the test set is being touched"),
    augment: str = typer.Option(
        "", help="augmentation recipe (training/augment.py), e.g. diac-teen"
    ),
    augment_p: float = typer.Option(
        0.0, help="share of train sentences replaced by a perturbed copy"
    ),
    crt_epochs: int = typer.Option(0, help=">0: balanced classifier re-training after CE (cRT)"),
    crt_lr: float = typer.Option(1e-3, help="cRT head learning rate"),
    robustness: bool = typer.Option(False, help="evaluate validation under the Cycle 1 suites"),
) -> None:
    """Fine-tune a transformer encoder across one or more seeds."""
    from vifeedback.constants import SEEDS
    from vifeedback.training import TrainConfig, format_summary, run_seeds

    if include_test and not reason:
        raise typer.BadParameter("--reason is required when evaluating on test")

    seed_tuple = SEEDS if seeds == "all" else tuple(int(s) for s in seeds.split(","))

    loss_for_recipe = {
        "base": "ce",
        "classweight": "classweight",
        "focal": "focal",
        "focal-weighted": "focal-weighted",
        "logit-adjust": "logit-adjust",
    }

    cfg = TrainConfig(
        task=task,
        model_key=model,
        recipe=recipe,
        preprocessing=preprocessing,
        loss=loss_for_recipe.get(recipe, "ce"),
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        eval_batch_size=eval_batch_size,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        weight_scheme=weight_scheme,
        focal_gamma=focal_gamma,
        logit_adjust_tau=tau,
        freeze_embeddings=freeze_embeddings,
        max_length=max_length,
        llrd=llrd or None,
        rdrop_alpha=rdrop,
        fgm_epsilon=fgm,
        label_smoothing=label_smoothing,
        augment=augment,
        augment_p=augment_p,
        crt_epochs=crt_epochs,
        crt_lr=crt_lr,
        extra={"phase_num": phase},
    )
    typer.echo(f"  effective batch = {batch_size} x {grad_accum} = {batch_size * grad_accum}")
    res = run_seeds(
        cfg,
        seed_tuple,
        include_test=include_test,
        reason=reason,
        save_checkpoint=save_checkpoint,
        robustness=robustness,
    )
    typer.echo("")
    typer.echo(
        format_summary(
            res["summary"], f"=== {task.upper()} / {model} / {recipe} / {len(seed_tuple)} seeds ==="
        )
    )


@train_app.command("multitask")
def train_multitask_cmd(
    lam: float = typer.Option(
        1.0, "--lambda", help="topic loss weight: L = L_sent + lambda * L_topic"
    ),
    model: str = typer.Option("phobert-base"),
    preprocessing: str = typer.Option("seg_pyvi"),
    seeds: str = typer.Option("42,1337,2024"),
    phase: int = typer.Option(8),
) -> None:
    """Cycle 1 H4: one shared encoder, sentiment + topic heads (training/multitask.py).

    Writes one registry row per task, so each task is compared to its single-task control with the
    same seed-paired tooling as every other run.
    """
    import dataclasses

    from vifeedback.evaluation import bootstrap as B
    from vifeedback.evaluation import report as R
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.multitask import TASKS, evaluate_joint, predict, train_multitask
    from vifeedback.training.runner import _free_cuda
    from vifeedback.training.trainer import TrainConfig

    tr, dv = load_variant(preprocessing, "train"), load_variant(preprocessing, "validation")
    train = (tr["sentence"].tolist(), tr["sentiment"].to_numpy(), tr["topic"].to_numpy())
    dev = (dv["sentence"].tolist(), dv["sentiment"].to_numpy(), dv["topic"].to_numpy())
    recipe = f"mtl-l{lam:g}"

    for seed in (int(s) for s in seeds.split(",")):
        base = TrainConfig(
            task="sentiment",
            model_key=model,
            preprocessing=preprocessing,
            recipe=recipe,
            seed=seed,
            extra={"phase_num": phase},
        )
        typer.echo(f"\n[{recipe} seed {seed}]  lambda={lam}")
        out = train_multitask(base, lam, train, dev)
        logits = predict(out["model"], out["loader"], out["device"], base.fp16)
        y = {"sentiment": dev[1], "topic": dev[2]}
        joint = evaluate_joint(y, logits)
        for task in TASKS:
            m = joint["per_task"][task]
            k = len(m["labels"])
            ci = B.bootstrap_ci(y[task], logits[task].argmax(1), k, n_resamples=2000, seed=seed)
            m["macro_f1_ci"] = [ci["ci_low"], ci["ci_high"]]
            m["history"] = out["history"]
            m["best_epoch"] = out["best_epoch"]
            m["multitask"] = {
                "lambda_topic": lam,
                "joint_exact_match": joint["joint_exact_match"],
                "selection": "mean of sentiment and topic dev macro-F1",
            }
            cfg_t = dataclasses.replace(base, task=task)
            config = dict(out["config"]) | {
                "task": task,
                "phase": f"P{phase}",
                "model": model,
                "split": "validation",
                "fit_seconds": out["train_seconds"],
                "notes": f"shared encoder, lambda={lam}",
                "reason": "cycle 1 H4",
                "determinism": out["determinism"],
                "device": out["device"],
                "best_epoch": out["best_epoch"],
            }
            R.save_run(
                cfg_t.run_id("validation"),
                m,
                config=config,
                y_true=y[task],
                y_pred=logits[task].argmax(1),
                y_prob=softmax_np(logits[task]),
            )
            typer.echo(f"  {task:9s} macro-F1 {m['macro_f1']:.4f}")
        typer.echo(f"  joint exact match {joint['joint_exact_match']:.4f}")
        del out
        _free_cuda()


def softmax_np(z):
    import numpy as np

    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
