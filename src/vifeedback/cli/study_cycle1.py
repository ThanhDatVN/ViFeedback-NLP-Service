"""`vifeedback study`, Cycle 1 and the standing studies it set up: the neutral audit (Study A),
robustness (Study B), calibration, CPU latency, result tables and the closing gate."""

from __future__ import annotations

import json

import typer

from vifeedback import paths
from vifeedback.cli._apps import study_app


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


@study_app.command("audit-sheet")
def study_audit_sheet() -> None:
    """Rebuild the local neutral-audit sheet from committed, text-free files. No GPU.

    The sheet holds corpus text, so it lives in the gitignored local/ folder and is lost with it.
    Everything needed to rebuild it is committed: which rows were sampled (audit_sample_index.csv)
    and their predictions (split prediction files, keyed by row). Text is re-joined from data/raw.
    """
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import error_analysis as EA

    src = paths.RESULTS / "studies" / "study_a"
    idx = pd.read_csv(src / "audit_sample_index.csv")
    parts = []
    for split, g in idx.groupby("split", sort=False):
        preds = pd.read_csv(src / f"{split}_predictions.csv").set_index("row")
        text = load(split)["sentence"]
        rows = preds.loc[g.example_index].copy()
        rows["text"] = text.iloc[g.example_index.to_numpy()].to_numpy()
        rows["stratum"] = g.stratum.to_numpy()
        rows.index.name = "example_index"
        parts.append(rows.reset_index())
    sample = pd.concat(parts, ignore_index=True)
    assert (sample.gold.to_numpy() == idx.gold.to_numpy()).all(), "row alignment check failed"
    sheet = EA.export_annotation_sheet(sample, src / "local" / "audit_sheet.csv")
    typer.echo(f"  rebuilt {len(sample)} rows -> {sheet}")


@study_app.command("audit-report")
def study_audit_report(
    sheet: str = typer.Option(
        "results/studies/study_a/local/audit_sheet.csv", help="the filled annotation sheet"
    ),
    second: str = typer.Option("", help="a second pass over >= 50 of the same rows, for kappa"),
    kind: str = typer.Option(
        "intra", help="inter (two people) | intra (one person, >= 24 h apart)"
    ),
) -> None:
    """Analyse the filled neutral-audit sheet and apply the decision tree frozen in cycle2.yaml.

    Refuses an incomplete sheet. Writes counts only (no text) to study_a/audit_report.json.
    """
    import pandas as pd

    from vifeedback.evaluation import audit as AU
    from vifeedback.evaluation.report import yaml_safe

    first = pd.read_csv(sheet, encoding="utf-8-sig", keep_default_na=False)
    other = pd.read_csv(second, encoding="utf-8-sig", keep_default_na=False) if second else None
    try:
        r = AU.report(first, other, kind)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from None
    out = paths.RESULTS / "studies" / "study_a" / "audit_report.json"
    out.write_text(json.dumps(yaml_safe(r), indent=2), encoding="utf-8")
    rs = r["random_stratum"]
    typer.echo(
        f"  random stratum (n={rs['n']}): incorrect gold {rs['incorrect']['rate']:.2f} "
        f"{tuple(round(x, 2) for x in rs['incorrect']['wilson_95'])}, "
        f"ambiguous {rs['ambiguous']['rate']:.2f}"
    )
    if "agreement" in r:
        a = r["agreement"]
        typer.echo(
            f"  {a['kind']}-annotator kappa {a['kappa']:.2f} on {a['n']} rows {a['warnings']}"
        )
    d = r["decision"]
    typer.echo(
        f"  scope n={d['n']}: incorrect {d['share_incorrect']:.2f}, ambiguous "
        f"{d['share_ambiguous']:.2f} -> {d['branch']}: {d['next']}"
    )
    typer.echo(f"  -> {out}")


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
        r["aurc"] = C.selective_scores(probs, y)  # E10: which confidence signal ranks errors best
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


@study_app.command("latency")
def study_latency(
    checkpoint: str = typer.Option("models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp"),
    onnx_dir: str = typer.Option("models/serve/sentiment", help="a released artifact (manifest)"),
    timed: int = typer.Option(300, help="timed single-sentence calls per repeat"),
    repeats: int = typer.Option(5),
    extra_onnx: str = typer.Option(
        "", help="comma-separated extra ONNX files to time, e.g. an INT8 graph that failed its gate"
    ),
    out_file: str = typer.Option(
        "", "--out", help="output JSON; default results/studies/latency/reference_cpu.json"
    ),
    with_torch: bool = typer.Option(
        True, "--torch/--no-torch", help="time the PyTorch rungs L0 and L1 (the slow part)"
    ),
) -> None:
    """Steady-state CPU latency ladder on the reference machine (review R10, § 8.4).

    The L rungs are model-only timing on inputs segmented once up front, so they run the same
    workload. The S rungs take raw test text, as the API receives it, through the service's own
    code (`serving.pipeline`): S0 is the pipeline before ADR-031 (lowercase, segment, logits), S1
    the served one (also the diacritic restorer, the features output and the out-of-scope score);
    S2 and S3 are the same on text with its diacritics stripped, where the restorer does its work
    (NEXT_PLAN v5, A1). Configurations run in two passes, forward then reversed, so a machine
    that warms up over the session shows as a pass disagreement instead of favouring whichever ran
    first. Throughput is texts/s. Peak RSS is recorded. Never run this while training.
    """
    import os
    import subprocess
    from pathlib import Path

    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback import env
    from vifeedback.data.loader import load
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.inference import benchmark as BM
    from vifeedback.inference.onnx_export import OnnxClassifier
    from vifeedback.preprocess.segment import get_segmenter

    torch.set_grad_enabled(False)
    try:  # nvidia-smi rather than torch.cuda.utilization(), which needs nvidia-ml-py
        util = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.split()
        gpu_busy = any(int(u) > 5 for u in util)
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu_busy = False
        typer.secho("  could not read GPU utilization; make sure nothing is training", fg="yellow")
    if gpu_busy:
        raise typer.BadParameter("the GPU is busy: a latency benchmark during training is invalid")

    from vifeedback.preprocess.normalize import strip_diacritics
    from vifeedback.serving import pipeline as SP

    seg = get_segmenter("pyvi")
    raw = load("test")["sentence"].tolist()  # timing inputs only; no labels are read
    raw_stripped = [strip_diacritics(t) for t in raw]
    texts = seg(raw)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).eval()
    onnx = OnnxClassifier(Path(onnx_dir))
    manifest = json.loads((Path(onnx_dir) / "manifest.json").read_text(encoding="utf-8"))
    restorer = (
        SP.load_restorer(Path(onnx_dir), manifest["restorer"]) if manifest.get("restorer") else None
    )
    ood = SP.load_ood(Path(onnx_dir), manifest["ood"], onnx) if manifest.get("ood") else None
    detector = SP.load_scope(Path(onnx_dir), manifest["scope"]) if manifest.get("scope") else None

    def before_adr031(batch):
        return onnx.logits(SP.prepare(batch, None, seg))

    def served(batch):  # as the API: the scope detector (ADR-034) when present, else Mahalanobis
        return SP.score(onnx, SP.prepare(batch, restorer, seg), None if detector else ood, detector)

    def torch_dynamic(batch):
        return model(
            **tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=96)
        ).logits

    def torch_padmax(batch):
        return model(
            **tok(batch, return_tensors="pt", padding="max_length", truncation=True, max_length=96)
        ).logits

    # name -> (callable, its inputs): L rungs take segmented text, S rungs raw text.
    configs = {}
    if with_torch:
        configs["L0 torch fp32, pad to 96"] = (torch_padmax, texts)
        configs["L1 torch fp32, dynamic padding"] = (torch_dynamic, texts)
    configs[f"L3 onnx fp32 ({onnx.path.name}), dynamic padding"] = (onnx.logits, texts)
    configs["S0 raw text: lowercase, pyvi, logits (before ADR-031)"] = (before_adr031, raw)
    configs["S1 raw text: served pipeline"] = (served, raw)
    configs["S2 unaccented text: before ADR-031"] = (before_adr031, raw_stripped)
    configs["S3 unaccented text: served pipeline"] = (served, raw_stripped)
    for i, f in enumerate(p for p in extra_onnx.split(",") if p):
        fp = Path(f)
        clf = OnnxClassifier(fp.parent, model_file=fp.name)
        # Timed only: an extra graph here need not have passed its release gate (it is labelled).
        configs[f"X{i} {fp.parent.parent.name}/{fp.name} (not released)"] = (clf.logits, texts)
    proc = psutil.Process(os.getpid())
    passes = []
    for order in (list(configs), list(reversed(configs))):
        res = {}
        for name in order:
            fn, inputs = configs[name]
            r = BM.time_callable(fn, inputs, timed=timed, repeats=repeats)
            r["rss_mb_after"] = round(proc.memory_info().rss / 1e6, 1)
            r["texts_per_s_b32"] = BM.throughput(fn, inputs, batch_sizes=(32,))[
                "batch32_texts_per_s"
            ]
            res[name] = r
            typer.echo(
                f"  {name:45s} p50 {r['p50_ms']:7.2f}  p95 {r['p95_ms']:7.2f} ms  "
                f"spread {r['p95_spread_across_repeats']:.1%}  {r['texts_per_s_b32']} texts/s"
            )
        passes.append(res)

    baseline = next(iter(configs))
    summary = BM.summarize_passes(passes, baseline=baseline)
    report = {
        "protocol": "L rungs: model-only on pre-segmented test inputs; S rungs: raw test text "
        "through serving.pipeline (S2/S3 with diacritics stripped); 200 warmup, median-of-repeats "
        "p95, two passes in rotated order; texts/s at batch 32; RSS after each config",
        "manifest_sha256": manifest.get("sha256"),
        "summary": summary,
        "passes": passes,
        "peak_rss_mb": round(proc.memory_info().rss / 1e6, 1),
        "environment": env.capture(),
    }
    out = (
        Path(out_file) if out_file else paths.RESULTS / "studies" / "latency" / "reference_cpu.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(yaml_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")
    for k, v in summary.items():
        if v["reportable"]:
            typer.echo(
                f"  {k:45s} p50 {v['p50_ms']:7.2f}  p95 {v['p95_ms']:7.2f} ms  [{v['basis']}]"
            )
        else:
            typer.secho(f"  {k:45s} NOT REPORTABLE: no steady pass", fg="red")
    typer.echo(f"  written {out}")


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


@study_app.command("cycle1")
def study_cycle1() -> None:
    """Apply the decision rules declared in configs/experiments/cycle1.yaml to the p8 runs."""
    import pandas as pd

    from vifeedback.evaluation import decisions as D
    from vifeedback.evaluation.report import yaml_safe

    runs = D.load_runs(phase=8)
    reg = pd.read_csv(paths.REGISTRY)

    def single_task(task: str) -> dict[int, float]:
        q = reg[
            (reg.model == "phobert-base")
            & (reg.preprocessing == "seg_pyvi")
            & (reg.recipe == "base")
            & (reg.task == task)
            & (reg.split == "validation")
            # The declared controls are the P4 laptop runs (cycle1.yaml). Not "the last row for
            # the seed": a later Kaggle re-run of the same config on another GPU differs by 0.001.
            & reg.run_id.astype(str).str.startswith("p4-")
        ].drop_duplicates("seed", keep="first")
        return {int(s): float(v) for s, v in zip(q.seed, q.macro_f1, strict=True)}

    registry_ce = single_task("sentiment")
    single = {"sentiment": registry_ce, "topic": single_task("topic")}

    out = {
        "runs_found": {k: sorted(v) for k, v in runs.items()},
        "H1": D.h1(runs, registry_ce),
        "H2": D.h2(runs),
        "H3": D.h3(reg),
        "H4": D.h4(D.load_multitask(8), single),
    }
    dst = paths.RESULTS / "studies" / "cycle1"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "decisions.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    typer.echo(json.dumps(yaml_safe(out), indent=2, ensure_ascii=False)[:6000])


@study_app.command("closing-gate")
def study_closing_gate(
    ce_checkpoint: str = typer.Option("models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp"),
    aug_glob: str = typer.Option("p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-*-ckp"),
    reason: str = typer.Option("Cycle 1 closing gate (cycle1.yaml): finalists on test, once"),
) -> None:
    """Cycle 1 closing gate: the finalists on test, once, with everything frozen beforehand.

    Per checkpoint (deployed CE model; H2-augmented model, seed 42): test metrics on the official split
    and on the overlap-excluded slice (configs/data/eval_slices_v1.json), calibration with the
    temperature fitted on validation and applied to test, and the robustness suites on test. Plus the
    5-seed paired test comparison from the registry. Every test touch is logged.
    """
    from pathlib import Path

    import pandas as pd

    from vifeedback.evaluation import closing_gate as CG
    from vifeedback.evaluation import decisions as D
    from vifeedback.evaluation.report import yaml_safe

    aug = sorted(paths.MODELS.glob(aug_glob))
    if not aug:
        raise typer.BadParameter(f"no augmented checkpoint matching {aug_glob} under models/")
    checkpoints = {"ce_deployed": Path(ce_checkpoint), "h2_augmented": aug[-1]}

    gi = CG.inputs()
    out: dict = {"reason": reason, "slice_version": gi["slice_version"], "checkpoints": {}}
    for name, ck in checkpoints.items():
        r = CG.row(ck, gi)
        out["checkpoints"][name] = r
        CG.log_test_use(ck, "P9", reason)
        typer.echo(
            f"  {name:13s} test macro-F1 {r['test']['macro_f1']:.4f}  (excl. overlap "
            f"{r['test_excluding_train_overlap']['macro_f1']:.4f})  "
            f"T={r['calibration']['temperature_fit_on_validation']:.2f}  no-diacritic "
            f"{r['robustness_test']['nodiacritic']['macro_f1']:.3f}"
        )

    reg = pd.read_csv(paths.REGISTRY)
    tr = reg[reg.split == "test"]

    def by_seed(q):
        return {int(s): float(m) for s, m in zip(q.seed, q.macro_f1, strict=True)}

    ce5 = by_seed(
        tr[tr.run_id.astype(str).str.match(r"p4-sent-phobert-base-seg_pyvi-base-s\d+-tes")]
    )
    aug5 = by_seed(
        tr[tr.run_id.astype(str).str.startswith("p9-sent-phobert-base-seg_pyvi-aug-diac-teen")]
    )
    out["five_seed_test"] = {
        "ce": ce5,
        "augmented": aug5,
        "paired_aug_minus_ce": D.paired(ce5, aug5),
    }
    dst = paths.RESULTS / "studies" / "closing_gate"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "summary.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    p5 = out["five_seed_test"]["paired_aug_minus_ce"]
    if p5.get("n"):
        typer.echo(
            f"  5-seed test, augmented - CE: {p5['mean_delta']:+.4f} {p5.get('ci95')} "
            f"wins {p5['wins']}/{p5['n']}"
        )
    typer.echo(f"  written {dst / 'summary.json'}")
