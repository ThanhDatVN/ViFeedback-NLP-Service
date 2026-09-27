"""Command-line entrypoint.

Every experiment in this project is a CLI call, so the laptop and Colab run identical code paths and
notebooks stay logic-free (docs/ROADMAP.md § 4).
"""

from __future__ import annotations

import json

import typer

from vifeedback import paths

app = typer.Typer(add_completion=False, help="ViFeedback — Vietnamese feedback classification")

data_app = typer.Typer(help="Phase 0: acquisition, integrity, profiling")
baseline_app = typer.Typer(help="Phase 1: classical baselines")
app.add_typer(data_app, name="data")
train_app = typer.Typer(help="Phase 2+: transformer fine-tuning")
app.add_typer(baseline_app, name="baseline")
serve_app = typer.Typer(help="Phase 6-7: export, benchmark, serve")
app.add_typer(train_app, name="train")
app.add_typer(serve_app, name="serve")
study_app = typer.Typer(help="Research studies (docs/REVIEW_AND_RESEARCH_PLAN.md § 7)")
app.add_typer(study_app, name="study")


# --- Phase 0 ------------------------------------------------------------------------------------


@data_app.command("fetch")
def data_fetch(
    force: bool = typer.Option(False, help="Re-download and overwrite data/raw"),
) -> None:
    """Download UIT-VSFC and write data/raw/*.parquet + manifest.json."""
    from vifeedback.data.loader import fetch

    manifest = fetch(force=force)
    for split, meta in manifest["splits"].items():
        typer.echo(f"  {split:11s} {meta['rows']:>6,} rows  sha256={meta['sha256'][:16]}...")
    typer.echo(f"  source: {manifest.get('source')}")


@data_app.command("report")
def data_report(
    tokenizer: bool = typer.Option(
        True, help="Include subword-length profiling (downloads a model)"
    ),
    figures: bool = typer.Option(True, help="Also regenerate EDA figures"),
) -> None:
    """Run the full Phase 0 integrity + profiling report -> results/data_report.json."""
    from vifeedback.data.profile import build_report, make_figures, write_report

    report = build_report(with_tokenizer=tokenizer)
    write_report(report)

    integ = report["integrity"]
    head = integ["leakage"]["headline"]
    typer.echo(f"  structural checks: {'PASS' if integ['structural_passed'] else 'FAIL'}")
    typer.echo(
        f"  leakage (test seen in train): exact {head['test_rows_seen_in_train_exact']} "
        f"({head['share_of_test_exact']:.2%}) | "
        f"normalized {head['test_rows_seen_in_train_normalized']} "
        f"({head['share_of_test_normalized']:.2%})"
    )
    if "max_length_decision" in report:
        d = report["max_length_decision"]
        typer.echo(
            f"  max_length: {d['max_length']} ({d['vs_model_default_256']}x shorter than 256)"
        )
    if figures:
        for name in make_figures():
            typer.echo(f"  figure: results/figures/{name}")
    typer.echo(f"  written: {paths.RESULTS / 'data_report.json'}")


@data_app.command("variants")
def data_variants(
    name: str = typer.Option("all", help="variant name, or 'all'"),
    force: bool = typer.Option(False, help="Rebuild even if already materialized"),
) -> None:
    """Materialize preprocessing variants into data/processed/ (Phase 3)."""
    from vifeedback.preprocess.variants import VARIANTS, build_all, summary_table

    names = tuple(VARIANTS) if name == "all" else (name,)
    metas = build_all(names, force=force)
    typer.echo(summary_table(metas))


@data_app.command("segmenters")
def data_segmenters() -> None:
    """Report which segmentation backends this machine can run, and why not."""
    from vifeedback.preprocess.segment import available

    for backend, info in available().items():
        mark = "OK " if info["available"] else "-- "
        typer.echo(f"  {mark} {backend:<14s} {info['reason']}")


@data_app.command("env")
def data_env() -> None:
    """Print the captured environment (CPU flags, packages, git)."""
    from vifeedback import env

    typer.echo(json.dumps(env.capture(), indent=2, ensure_ascii=False))


# --- Phase 1 ------------------------------------------------------------------------------------


@baseline_app.command("run")
def baseline_run(
    task: str = typer.Option("sentiment", help="sentiment | topic"),
    seed: int = typer.Option(42),
    include_test: bool = typer.Option(False, help="Also evaluate on test (logged; gates only)"),
    reason: str = typer.Option("", help="Why the test set is being touched"),
    quiet: bool = typer.Option(False, help="Suppress the per-model reports"),
) -> None:
    """Run the B0-B5 baseline ladder for one task."""
    from vifeedback.models.runner import compare_best, run_ladder, summary_table

    if include_test and not reason:
        raise typer.BadParameter("--reason is required when evaluating on test")

    results = run_ladder(
        task, seed=seed, include_test=include_test, reason=reason, verbose=not quiet
    )

    for split in ("validation", "test"):
        if not any(r["split"] == split for r in results):
            continue
        typer.echo("")
        typer.echo(f"=== {task.upper()} / {split.upper()} ===")
        typer.echo(summary_table(results, split))

        cmp_ = compare_best(results, task, split)
        if cmp_:
            typer.echo("")
            typer.echo(
                f"  paired bootstrap: {cmp_['best']} vs {cmp_['baseline']}"
                f"  diff {cmp_['observed_diff']:+.4f}"
                f"  [{cmp_['ci_low']:+.4f}, {cmp_['ci_high']:+.4f}]"
                f"  p={cmp_['p_value']:.4f}"
                f"  {'SIGNIFICANT' if cmp_['significant'] else 'not significant'}"
            )


@baseline_app.command("registry")
def baseline_registry(
    split: str = typer.Option("validation", help="validation | test | all"),
) -> None:
    """Print the experiment registry and the test-set evaluation count."""
    import pandas as pd

    from vifeedback.evaluation.report import load_registry, test_evaluation_count

    df = load_registry()
    if len(df) == 0:
        typer.echo("  registry is empty")
        return
    if split != "all":
        df = df[df["split"] == split]
    cols = ["run_id", "task", "model", "recipe", "macro_f1", "weighted_f1", "accuracy"]
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        typer.echo(df[cols].to_string(index=False))
    typer.echo("")
    typer.echo(f"  test-set evaluations to date: {test_evaluation_count()}")


# --- Phase 2+ ------------------------------------------------------------------------------------


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


# --- Phase 6-7 -----------------------------------------------------------------------------------


@serve_app.command("export")
def serve_export(
    checkpoint: str = typer.Option(..., help="path to a saved HF checkpoint"),
    task: str = typer.Option("sentiment"),
    preprocessing: str = typer.Option(
        "", help="the checkpoint's training preprocessing; inferred from its name if omitted"
    ),
    max_length: int = typer.Option(96),
    quantize: str = typer.Option("dynamic", help="none | dynamic | static"),
    out: str = typer.Option("", help="defaults to models/serve/<task>"),
    calib_size: int = typer.Option(300, help="stratified train sentences for static INT8"),
) -> None:
    """Export, verify against the quality contract, and release (review R3).

    Built in a staging directory; the served directory is replaced only if FP32 logits match, or
    INT8 stays within 0.005 macro-F1 of PyTorch on the full validation set.
    """
    from pathlib import Path

    from vifeedback.inference.release import release, stratified_subset
    from vifeedback.preprocess.variants import VARIANTS
    from vifeedback.training.runner import _raw_and_pipeline

    if not preprocessing:
        name = Path(checkpoint).name
        found = [v for v in sorted(VARIANTS, key=len, reverse=True) if v != "raw" and v in name]
        preprocessing = found[0] if found else ""
        if not preprocessing:
            raise typer.BadParameter(
                "cannot infer the checkpoint's preprocessing from its name; pass --preprocessing"
            )
    typer.echo(f"  preprocessing {preprocessing}")

    from vifeedback.preprocess.variants import load_variant

    raw_tr, pipeline = _raw_and_pipeline(preprocessing, "train")
    y_tr = load_variant(preprocessing, "train")[task].to_numpy()
    raw_dv, _ = _raw_and_pipeline(preprocessing, "validation")
    y_dv = load_variant(preprocessing, "validation")[task].to_numpy()
    calib_idx = stratified_subset(y_tr, calib_size)

    manifest = release(
        checkpoint=checkpoint,
        task=task,
        preprocessing=preprocessing,
        quantize=quantize,
        out_dir=Path(out) if out else paths.MODELS / "serve" / task,
        pipeline=pipeline,
        calib_raw=[raw_tr[i] for i in calib_idx],
        accept_raw=raw_dv,
        accept_y=y_dv,
        max_length=max_length,
        log=typer.echo,
    )
    typer.echo(f"  manifest: {manifest['model_file']}  sha256 {manifest['sha256'][:12]}...")


@serve_app.command("bench")
def serve_bench(
    task: str = typer.Option("sentiment"),
    model_dir: str = typer.Option("", help="defaults to models/serve/<task>"),
    threads: int = typer.Option(0, help="0 = ORT default"),
    timed: int = typer.Option(1000),
) -> None:
    """Benchmark CPU latency on the reference machine. Never run this on a cloud VM."""
    import json
    from pathlib import Path

    from vifeedback import paths
    from vifeedback.data.loader import load
    from vifeedback.inference import benchmark as BM
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import VARIANTS

    d = Path(model_dir) if model_dir else paths.MODELS / "serve" / task
    manifest = (
        json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if (d / "manifest.json").exists()
        else {}
    )
    clf = OnnxClassifier(d, threads or None, manifest.get("max_length", 96))
    # Time the preprocessing the artifact was verified with, not an assumed one (review R3/R10).
    backend = VARIANTS[manifest["preprocessing"]][0] if manifest.get("preprocessing") else "pyvi"
    seg = get_segmenter(backend)
    texts = load("test")["sentence"].tolist()

    rep = BM.benchmark_pipeline(
        model_fn=clf.predict,
        preprocess_fn=seg,
        texts=texts,
        label=f"{clf.path.name} threads={threads or 'default'}",
        size_mb=clf.size_mb,
        timed=timed,
    )
    out = paths.RESULTS / f"bench_{task}_{clf.path.stem}.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    m, e = rep["model_only"], rep["end_to_end"]
    typer.echo(f"  size          {rep['size_mb']} MB")
    typer.echo(f"  model p50/p95 {m['p50_ms']:.2f} / {m['p95_ms']:.2f} ms")
    typer.echo(f"  e2e   p50/p95 {e['p50_ms']:.2f} / {e['p95_ms']:.2f} ms")
    typer.echo(f"  preprocessing {rep['preprocess_share_of_p95']:.1%} of end-to-end p95")
    typer.echo(f"  throughput    {rep['throughput']}")
    if m["throttling_suspected"]:
        typer.secho(f"  WARNING: {m['warning']}", fg="red")
    typer.echo(f"  written       {out}")


@serve_app.command("run")
def serve_run(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False),
) -> None:
    """Run the API locally."""
    import uvicorn

    uvicorn.run("vifeedback.serving.app:app", host=host, port=port, reload=reload)


# --- Studies -------------------------------------------------------------------------------------


@study_app.command("neutral-audit")
def study_neutral_audit(
    checkpoint: str = typer.Option(
        "models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp",
        help="saved sentiment checkpoint used for the dev view",
    ),
    preprocessing: str = typer.Option(
        "seg_pyvi", help="must match the checkpoint's training input"
    ),
    oof: bool = typer.Option(True, help="also run k-fold OOF fine-tuning over train (~20 GPU-min)"),
    k: int = typer.Option(5, help="OOF folds"),
    seed: int = typer.Option(42),
) -> None:
    """Study A, steps 1 and 3: dev/OOF predictions, uncertainty, and a stratified audit sheet.

    Text-free prediction files and the summary are written to results/studies/study_a/ and are
    committed. Anything containing corpus text goes to its local/ subfolder, which is gitignored:
    the raw data is not redistributed (docs/DATA_CARD.md § 11).
    """
    import pandas as pd

    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.preprocess.variants import variant_dir

    out = paths.RESULTS / "studies" / "study_a"
    local = out / "local"
    local.mkdir(parents=True, exist_ok=True)
    task = "sentiment"

    def _load(split: str) -> pd.DataFrame:
        return pd.read_parquet(variant_dir(preprocessing) / f"{split}.parquet")

    tr, dv = _load("train"), _load("validation")
    summary: dict = {
        "task": task,
        "preprocessing": preprocessing,
        "checkpoint": checkpoint,
        "caveat_dev": "the checkpoint selected its epoch on validation, so the dev view is mildly "
        "optimistic; it describes errors, it does not estimate generalization",
    }

    typer.echo(f"dev predictions from {checkpoint}")
    dev_p = EA.predict_proba(checkpoint, dv["sentence"].tolist())
    dev_t = EA.prediction_table(
        dv["sentence_raw"].tolist(),
        dv["sentence"].tolist(),
        dv[task].to_numpy(),
        dev_p,
        task,
        split="validation",
        source="checkpoint",
    )
    summary["dev_metrics"] = M.evaluate(dev_t.gold_id, dev_t.pred_id, task, y_prob=dev_p)

    tables = {"validation": dev_t}
    if oof:
        from vifeedback.training.trainer import TrainConfig

        cfg = TrainConfig(
            task=task, model_key="phobert-base", preprocessing=preprocessing, seed=seed
        )
        typer.echo(f"OOF: {k} folds over train ({len(tr)} rows)")
        r = EA.oof_predictions(
            cfg,
            tr["sentence"].tolist(),
            tr[task].to_numpy(),
            dv["sentence"].tolist(),
            dv[task].to_numpy(),
            k=k,
            fold_seed=seed,
        )
        tr_t = EA.prediction_table(
            tr["sentence_raw"].tolist(),
            tr["sentence"].tolist(),
            tr[task].to_numpy(),
            r["oof_probs"],
            task,
            split="train",
            source="oof",
        )
        tr_t["fold"] = r["fold_id"]
        tables["train"] = tr_t
        dis = EA.fold_disagreement(r["dev_probs"])
        dev_t["fold_vote_agreement"] = dis["vote_agreement"]
        dev_t["fold_prob_std_max"] = dis["prob_std_max"]
        summary["oof"] = {
            "k": k,
            "config_hash": cfg.config_hash(),
            "folds": r["folds"],
            "gpu_seconds": r["total_seconds"],
            "metrics": M.evaluate(tr_t.gold_id, tr_t.pred_id, task, y_prob=r["oof_probs"]),
            "dev_fold_ensemble_metrics": M.evaluate(
                dev_t.gold_id,
                r["dev_probs"].mean(axis=0).argmax(axis=1),
                task,
                y_prob=r["dev_probs"].mean(axis=0),
            ),
        }
        issues = EA.suspected_label_issues(tr_t, top_k=200)
        summary["suspected_issues_top200_by_gold_pred"] = (
            issues.groupby(["gold", "pred"]).size().rename("n").reset_index().to_dict("records")
        )

    samples = []
    for split, t in tables.items():
        t.to_csv(
            local / f"{split}_predictions_with_text.csv", index_label="row", encoding="utf-8-sig"
        )
        # 6 significant digits, not fixed decimals: a probability of 3e-6 rounded to 5 decimals
        # becomes 0, and NLL / temperature fitting on the saved file would then be wrong.
        text_free = t.drop(columns=["text", "model_input"])
        text_free.to_csv(out / f"{split}_predictions.csv", index_label="row", float_format="%.6g")
        summary[f"error_rate_by_flag_{split}"] = EA.error_rate_by_flag(t).to_dict("records")
        summary[f"error_rate_by_flag_{split}_neutral"] = EA.error_rate_by_flag(
            t, gold_class="neutral"
        ).to_dict("records")
        samples.append(
            EA.stratified_audit_sample(
                t, per_error_cell=8, per_correct_class=5, n_random=20, seed=seed
            )
        )

    sample = pd.concat(samples, ignore_index=True)
    sheet = EA.export_annotation_sheet(sample, local / "audit_sheet.csv")
    sample[["split", "example_index", "stratum", "gold", "pred"]].to_csv(
        out / "audit_sample_index.csv", index=False
    )
    summary["audit_sample"] = {
        "n": len(sample),
        "by_split_stratum": sample.groupby(["split", "stratum"]).size().to_dict(),
    }
    summary["audit_sample"]["by_split_stratum"] = {
        f"{a}|{b}": int(v) for (a, b), v in summary["audit_sample"]["by_split_stratum"].items()
    }

    from vifeedback.evaluation.report import yaml_safe

    (out / "summary.json").write_text(
        json.dumps(yaml_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"  dev macro-F1 {summary['dev_metrics']['macro_f1']:.4f}")
    if oof:
        typer.echo(f"  OOF macro-F1 {summary['oof']['metrics']['macro_f1']:.4f}")
    typer.echo(f"  audit sheet ({len(sample)} rows): {sheet}")
    typer.echo(f"  summary: {out / 'summary.json'}")


@study_app.command("calibration")
def study_calibration(
    n_bins: int = typer.Option(15, help="ECE bins; disclosed in the output"),
    seed: int = typer.Option(42),
) -> None:
    """E09: calibration of saved predictions, temperature cross-fitted on disjoint halves.

    Reads Study A's prediction files, so it needs no GPU. Two views: the deployed checkpoint on
    validation, and the fold models on their held-out train folds (OOF, if present).
    """
    import pandas as pd

    from vifeedback.constants import label_names
    from vifeedback.evaluation import calibration as C
    from vifeedback.evaluation.report import yaml_safe

    src = paths.RESULTS / "studies" / "study_a"
    out = paths.RESULTS / "studies" / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"p_{n}" for n in label_names("sentiment")]
    report: dict = {"n_bins": n_bins, "protocol": "cross-fit on stratified halves", "views": {}}

    for view, fname in (("checkpoint_on_validation", "validation"), ("oof_on_train", "train")):
        f = src / f"{fname}_predictions.csv"
        if not f.exists():
            typer.echo(f"  skip {view}: {f.name} not found")
            continue
        df = pd.read_csv(f)
        probs, y = df[cols].to_numpy(), df["gold_id"].to_numpy()
        r = C.cross_fit_temperature(C.probs_to_logits(probs), y, seed=seed, n_bins=n_bins)
        r.pop("calibrated_probs")
        r["risk_coverage_uncalibrated"] = C.risk_coverage(probs, y)
        report["views"][view] = r
        u, t = r["uncalibrated"], r["temperature_scaled"]
        typer.echo(
            f"  {view:26s} n={u['n']:5d}  T={r['temperatures']}  "
            f"NLL {u['nll']:.4f}->{t['nll']:.4f}  Brier {u['brier']:.4f}->{t['brier']:.4f}  "
            f"ECE(w) {u['ece_equal_width']:.4f}->{t['ece_equal_width']:.4f}"
        )

    (out / "summary.json").write_text(
        json.dumps(yaml_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"  written {out / 'summary.json'}")


@study_app.command("robustness")
def study_robustness(
    checkpoint: str = typer.Option("models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp"),
    segmenter: str = typer.Option("pyvi", help="the checkpoint's training segmenter"),
    split: str = typer.Option("validation", help="test is reserved for final confirmation"),
    seed: int = typer.Option(42),
    n_boot: int = typer.Option(2000),
) -> None:
    """Study B: perturbation suites and predefined slices, paired against clean predictions.

    Perturbations are applied to raw text and then segmented, as the deployed pipeline would.
    """

    import numpy as np

    from vifeedback.constants import label_names
    from vifeedback.data.loader import load
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import robustness as R
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.segment import get_segmenter

    if split == "test":
        raise typer.BadParameter("test is reserved for the final confirmation run")

    task = "sentiment"
    k = len(label_names(task))
    df = load(split)
    raw, y = df["sentence"].tolist(), df[task].to_numpy()
    seg = get_segmenter(segmenter)
    out = paths.RESULTS / "studies" / "robustness"
    out.mkdir(parents=True, exist_ok=True)

    clean = EA.predict_proba(checkpoint, seg(raw)).argmax(1)
    report: dict = {
        "suite_version": R.SUITE_VERSION,
        "slices_version": R.SLICES_VERSION,
        "checkpoint": checkpoint,
        "segmenter": segmenter,
        "split": split,
        "seed": seed,
        "suites": {},
        "slices": R.slice_report(y, clean, R.slice_masks(raw), k),
    }
    for suite in R.SUITES:
        pert, changed = R.perturb(raw, suite, seed=seed)
        pred = EA.predict_proba(checkpoint, seg(pert)).argmax(1)
        entry = R.paired_delta(y, clean, pred, k, n_boot=n_boot, seed=seed)
        entry["changed_share"] = float(changed.mean())
        if changed.sum() >= 30:
            entry["changed_only"] = R.paired_delta(
                y[changed], clean[changed], pred[changed], k, n_boot=n_boot, seed=seed
            )
        # Where predictions go under shift. A rising minority-class recall can mean the model is
        # dumping unreadable input into that class, which the aggregate delta alone would hide.
        entry["pred_share_clean"] = (np.bincount(clean, minlength=k) / len(clean)).tolist()
        entry["pred_share_shifted"] = (np.bincount(pred, minlength=k) / len(pred)).tolist()
        entry["neutral_recall_clean"] = float((clean[y == 1] == 1).mean())
        entry["neutral_recall_shifted"] = float((pred[y == 1] == 1).mean())
        report["suites"][suite] = entry
        typer.echo(
            f"  {suite:16s} changed {entry['changed_share']:6.1%}  macro-F1 "
            f"{entry['macro_f1_a']:.4f} -> {entry['macro_f1_b']:.4f}  "
            f"delta {entry['delta']:+.4f} [{entry['delta_ci'][0]:+.4f}, {entry['delta_ci'][1]:+.4f}]"
        )
    probe = paths.DATA / "probes" / "negation_v1.csv"
    if probe.exists():
        import pandas as pd

        report["negation_probe"] = R.negation_probe(
            pd.read_csv(probe),
            lambda xs: EA.predict_proba(checkpoint, seg(xs)).argmax(1),
        )
        for name, r in report["negation_probe"].items():
            typer.echo(
                f"  negation {name:11s} pairs {r['n_pairs']:2d}  pair-acc {r['pair_accuracy']:.3f}  "
                f"flip {r['flip_rate']:.3f}  base-acc {r['base_accuracy']:.3f}"
            )
    if hasattr(seg, "close"):
        seg.close()

    (out / f"{split}.json").write_text(
        json.dumps(yaml_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(f"  written {out / f'{split}.json'}")


@study_app.command("tables")
def study_tables() -> None:
    """Regenerate result tables and the declared paired comparisons from the registry."""
    from vifeedback.evaluation import compare_runs as CR
    from vifeedback.evaluation.report import yaml_safe

    out = paths.RESULTS / "studies" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    r = CR.build()
    r["table"].to_csv(out / "conditions.csv", index=False, float_format="%.6g")
    (out / "conditions.md").write_text(
        "<!-- generated by `vifeedback study tables`; do not edit by hand -->\n\n"
        + CR.to_markdown(r["table"])
        + "\n",
        encoding="utf-8",
    )
    (out / "comparisons.json").write_text(
        json.dumps(yaml_safe({"hygiene": r["hygiene"], "comparisons": r["comparisons"]}), indent=2),
        encoding="utf-8",
    )
    h = r["hygiene"]
    typer.echo(
        f"  registry rows {h['rows_in_registry']} -> used {h['rows_used']}  "
        f"(duplicate ids {len(h['duplicate_run_ids'])}, re-run seeds {h['reruns']['n_condition_seeds']}, "
        f"max re-run disagreement {h['reruns']['max_abs_macro_f1_disagreement']})"
    )
    for c in r["comparisons"]:
        if "p" in c:
            typer.echo(
                f"  {c['name']:32s} diff {c['mean_diff']:+.4f} "
                f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]  wins {c['wins_b']}/{c['n']}  "
                f"p {c['p']:.4f}  BH {'yes' if c['significant_bh'] else 'no'}"
            )
    typer.echo(f"  written {out}")


@study_app.command("neutral-diagnosis")
def study_neutral_diagnosis() -> None:
    """Study A analysis: neutral error structure and a post-hoc decision-boundary test.

    CPU only; reads the outputs of `study neutral-audit`. The boundary bias is tuned on the OOF
    train predictions and evaluated on validation, so it is never scored on its own tuning data.
    """
    import pandas as pd

    from vifeedback.constants import label_names
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe

    task = "sentiment"
    names = label_names(task)
    k, neutral = len(names), names.index("neutral")
    src = paths.RESULTS / "studies" / "study_a"
    cols = [f"p_{n}" for n in names]
    out: dict = {}

    tables = {}
    for split in ("validation", "train"):
        f = src / f"{split}_predictions.csv"
        if f.exists():
            tables[split] = pd.read_csv(f)
            out[f"{split}_neutral"] = EA.neutral_diagnosis(tables[split])

    if "train" in tables:
        tr, dv = tables["train"], tables["validation"]
        tuned = EA.tune_class_bias(tr[cols].to_numpy(), tr.gold_id.to_numpy(), neutral, k)
        before = M.evaluate(dv.gold_id, dv.pred_id, task)
        after_pred = EA.apply_class_bias(dv[cols].to_numpy(), neutral, tuned["bias"])
        after = M.evaluate(dv.gold_id, after_pred, task)
        out["boundary_test"] = {
            "tuned_on": "OOF train predictions (fold models)",
            "evaluated_on": "validation, deployed checkpoint",
            "bias_on_neutral_logprob": tuned["bias"],
            "oof_macro_f1_at_zero": tuned["at_zero"],
            "oof_macro_f1_tuned": tuned["macro_f1"],
            "dev_macro_f1_before": before["macro_f1"],
            "dev_macro_f1_after": after["macro_f1"],
            "dev_neutral_f1_before": before["per_class"]["neutral"]["f1"],
            "dev_neutral_f1_after": after["per_class"]["neutral"]["f1"],
            "dev_neutral_recall_before": before["per_class"]["neutral"]["recall"],
            "dev_neutral_recall_after": after["per_class"]["neutral"]["recall"],
            "dev_neutral_precision_before": before["per_class"]["neutral"]["precision"],
            "dev_neutral_precision_after": after["per_class"]["neutral"]["precision"],
        }
        ng = tr[tr.gold == "neutral"]
        out["train_neutral_oof"] = {
            "n": len(ng),
            "confidently_other_share": float(((~ng.correct) & (ng.p_pred >= 0.9)).mean()),
            "p_gold_below_0.1_share": float((ng.p_gold < 0.1).mean()),
        }
        if "fold_vote_agreement" in dv.columns:
            dn = dv[dv.gold == "neutral"]
            out["dev_neutral_fold_agreement"] = {
                "all_folds_agree_share": float((dn.fold_vote_agreement == 1.0).mean()),
                "all_folds_agree_and_wrong_share": float(
                    ((dn.fold_vote_agreement == 1.0) & ~dn.correct).mean()
                ),
            }

    (src / "diagnosis.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(yaml_safe(out), indent=2, ensure_ascii=False)[:4000])


@study_app.command("cycle1")
def study_cycle1() -> None:
    """Apply the decision rules declared in configs/experiments/cycle1.yaml to the p8 runs."""
    import pandas as pd

    from vifeedback.evaluation import decisions as D
    from vifeedback.evaluation.report import yaml_safe

    runs = D.load_runs(phase=8)
    reg = pd.read_csv(paths.REGISTRY)
    ce = reg[
        (reg.model == "phobert-base")
        & (reg.preprocessing == "seg_pyvi")
        & (reg.recipe == "base")
        & (reg.task == "sentiment")
        & (reg.split == "validation")
    ].drop_duplicates("seed", keep="last")
    registry_ce = {int(s): float(v) for s, v in zip(ce.seed, ce.macro_f1, strict=True)}

    out = {
        "runs_found": {k: sorted(v) for k, v in runs.items()},
        "H1": D.h1(runs, registry_ce),
        "H2": D.h2(runs),
    }
    dst = paths.RESULTS / "studies" / "cycle1"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "decisions.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(yaml_safe(out), indent=2, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    app()
