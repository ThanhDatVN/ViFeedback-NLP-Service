"""`vifeedback study`, Cycle 2: H5 topic stacking, H6 on the challenge set, H7 LLM reference."""

from __future__ import annotations

import json

import typer

from vifeedback import paths
from vifeedback.cli._apps import study_app


def _print_stacking(result: dict) -> None:
    for name in ("dense", "sparse", "stacked"):
        pc = result[name]["per_class_f1"]
        typer.echo(
            f"  {name:8s} macro-F1 {result[name]['macro_f1']:.4f}  "
            f"facility {pc['facility']:.3f}  others {pc['others']:.3f}"
        )
    pb = result["paired_stacked_minus_dense"]
    typer.echo(
        f"  stacked - dense: {pb['observed_diff']:+.4f} [{pb['ci_low']:+.4f}, {pb['ci_high']:+.4f}]"
        f"  supported={result['supported']}"
    )
    fc = result.get("descriptive_facility_sparse_minus_dense")
    if fc:
        typer.echo(
            f"  facility F1, sparse - dense (descriptive): {fc['observed_diff']:+.4f} "
            f"[{fc['ci_low']:+.4f}, {fc['ci_high']:+.4f}]"
        )


@study_app.command("topic-stacking")
def study_topic_stacking(
    seed: int = typer.Option(42),
    k: int = typer.Option(5),
    reuse: bool = typer.Option(
        False, help="re-score from the saved validation features (no training, no refit)"
    ),
) -> None:
    """Cycle 2 H5: out-of-fold PhoBERT + TF-IDF stacking for topic (5 fold fine-tunes, ~20 GPU-min)."""
    import numpy as np
    import pandas as pd

    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation import stacking as ST
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.preprocess.variants import load_variant
    from vifeedback.training.trainer import TrainConfig

    task = "topic"
    out = paths.RESULTS / "studies" / "topic_stacking"
    if reuse:
        v = pd.read_csv(out / "validation_features.csv")
        prev = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        cols_of = lambda p: v.filter(regex=rf"^{p}_\d+$").to_numpy()  # noqa: E731
        result = ST.compare(
            v.gold.to_numpy(),
            {
                "dense": cols_of("dense").argmax(1),
                "sparse": cols_of("sparse").argmax(1),
                "stacked": v.stacked_pred.to_numpy(),
            },
            task,
        )
        result["folds"], result["gpu_seconds"] = prev["folds"], prev["gpu_seconds"]
        (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
        _print_stacking(result)
        return

    tr, dv = load_variant("seg_pyvi", "train"), load_variant("seg_pyvi", "validation")
    y_tr, y_dv = tr[task].to_numpy(), dv[task].to_numpy()
    cfg = TrainConfig(task=task, model_key="phobert-base", preprocessing="seg_pyvi", seed=seed)

    typer.echo(f"dense OOF: {k} folds of phobert-base / topic")
    r = EA.oof_predictions(
        cfg, tr.sentence.tolist(), y_tr, dv.sentence.tolist(), y_dv, k=k, fold_seed=seed
    )
    dense_oof, fold_id = r["oof_probs"], r["fold_id"]
    dense_dv = r["dev_probs"].mean(axis=0)

    typer.echo("sparse OOF: TF-IDF B4, same folds, C fixed")
    sparse_oof = ST.sparse_oof_scores(tr.sentence_raw.tolist(), y_tr, fold_id, seed)
    sparse_full = ST.sparse_model(seed).fit(np.asarray(tr.sentence_raw, dtype=object), y_tr)
    sparse_dv = sparse_full.decision_function(np.asarray(dv.sentence_raw, dtype=object))

    meta = ST.fit_meta(ST.features(dense_oof, sparse_oof), y_tr, seed)
    stacked_dv = meta.predict(ST.features(dense_dv, sparse_dv))
    result = ST.compare(
        y_dv,
        {"dense": dense_dv.argmax(1), "sparse": sparse_dv.argmax(1), "stacked": stacked_dv},
        task,
    )
    result["folds"] = r["folds"]
    result["gpu_seconds"] = r["total_seconds"]

    out.mkdir(parents=True, exist_ok=True)
    cols = lambda p, a: {f"{p}_{i}": a[:, i] for i in range(a.shape[1])}  # noqa: E731
    pd.DataFrame(
        {"gold": y_tr, "fold": fold_id, **cols("dense", dense_oof), **cols("sparse", sparse_oof)}
    ).to_csv(out / "train_oof_features.csv", index_label="row", float_format="%.6g")
    pd.DataFrame(
        {
            "gold": y_dv,
            **cols("dense", dense_dv),
            **cols("sparse", sparse_dv),
            "stacked_pred": stacked_dv,
        }
    ).to_csv(out / "validation_features.csv", index_label="row", float_format="%.6g")
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")
    _print_stacking(result)


@study_app.command("challenge")
def study_challenge(
    ce: str = typer.Option(
        "models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp", help="seed-42 CE checkpoint (served)"
    ),
    aug: str = typer.Option(
        "models/p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-599cf21f-ckp",
        help="seed-42 H2-augmented checkpoint",
    ),
) -> None:
    """Cycle 2 H6 on the frozen challenge set: per-category results and the declared serving rule."""
    from vifeedback.constants import label_names
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.evaluation.report import yaml_safe

    df = CH.load()  # raises if the file differs from the hash frozen in cycle2.yaml
    x = CH.pipeline("seg_pyvi")(df.text.tolist())
    probs = {name: EA.predict_proba(ckp, x) for name, ckp in (("ce", ce), ("augmented", aug))}
    pred = {name: p.argmax(1) for name, p in probs.items()}

    result: dict = {
        "challenge_sha256": CH.declared_sha256(),
        "checkpoints": {"ce": ce, "augmented": aug},
        "h6": CH.h6(df, pred["ce"], pred["augmented"]),
    }
    for name in probs:
        result[name] = {
            **CH.category_report(df, pred[name]),
            "negation_pairs": CH.negation_pairs(df, pred[name]),
            "out_of_scope_confidence": CH.out_of_scope_confidence(df, probs[name]),
        }

    out = paths.RESULTS / "studies" / "challenge"
    out.mkdir(parents=True, exist_ok=True)
    names = label_names("sentiment")
    table = df[["id", "category", "pair_id", "text", "sentiment"]].copy()
    for name, p in probs.items():
        table[f"{name}_pred"] = [names[i] for i in pred[name]]
        table[f"{name}_conf"] = p.max(1).round(4)
    table.to_csv(out / "predictions.csv", index=False)  # constructed text only; no corpus rows
    (out / "summary.json").write_text(json.dumps(yaml_safe(result), indent=2), encoding="utf-8")

    cats = sorted(result["ce"]["by_category"])
    typer.echo(f"{'category':22s} {'n':>4s} {'CE':>6s} {'aug':>6s}")
    for c in cats:
        a, b = result["ce"]["by_category"][c], result["augmented"]["by_category"][c]
        typer.echo(f"{c:22s} {a['n']:4d} {a['accuracy']:6.3f} {b['accuracy']:6.3f}")
    for name in probs:
        r = result[name]
        typer.echo(
            f"{name:10s} macro-F1 {r['macro_f1']:.4f}  neutral F1 {r['per_class_f1']['neutral']:.3f}  "
            f"negation pairs both-correct {r['negation_pairs']['both_correct']:.2f}"
        )
    h = result["h6"]
    pb = h["typed_paired_aug_minus_ce"]
    typer.echo(
        f"H6 typed rows ({h['typed_rows']}): aug - CE {pb['observed_diff']:+.3f} "
        f"[{pb['ci_low']:+.3f}, {pb['ci_high']:+.3f}]; other rows {h['other_diff']:+.3f} "
        f"-> {h['decision']}"
    )


@study_app.command("llm-prompt-dev")
def study_llm_prompt_dev(
    model: str = typer.Option("Qwen/Qwen3-1.7B", help="the pilot; never an API model"),
    force: bool = typer.Option(False, help="overwrite a frozen prompt_dev.json"),
) -> None:
    """Cycle 2 H7 prompt development: score the prompt variants on a fixed train subset, freeze the best."""
    import datetime as dt

    from vifeedback.data.loader import load
    from vifeedback.evaluation import llm_reference as L
    from vifeedback.evaluation import metrics as M

    if L.PROMPT_DEV_FILE.exists() and not force:
        raise typer.BadParameter(
            f"{L.PROMPT_DEV_FILE} exists: the prompt is frozen (--force to redo)"
        )
    tr = load("train")
    y = tr.sentiment.to_numpy()
    idx = L.prompt_dev_subset(y)
    texts = tr.sentence.iloc[idx].tolist()
    scorer = L.HFScorer(model)
    rows = {}
    for variant in L.VARIANTS:
        r = L.run(scorer, texts, variant)
        ev = M.evaluate(y[idx], r["probs"].argmax(1), "sentiment")
        rows[variant] = {
            "macro_f1": ev["macro_f1"],
            "per_class_f1": {c: v["f1"] for c, v in ev["per_class"].items()},
            "seconds": r["seconds"],
        }
        typer.echo(
            f"  {variant:30s} macro-F1 {ev['macro_f1']:.4f}  "
            f"neutral F1 {rows[variant]['per_class_f1']['neutral']:.3f}"
        )
    best = max(L.VARIANTS, key=lambda v: rows[v]["macro_f1"])  # ties keep the earlier, simpler one
    per_class = {L.LABELS[c]: int((y[idx] == c).sum()) for c in range(len(L.LABELS))}
    L.write(
        L.PROMPT_DEV_FILE,
        {
            "model": model,
            "revision": scorer.revision,
            "subset": {"split": "train", "indices": idx.tolist(), "per_class": per_class},
            "variants": rows,
            "frozen_variant": best,
            "frozen_on": dt.date.today().isoformat(),
            "prompts": {v: "\n\n".join(parts) for v, parts in L.VARIANTS.items()},
        },
    )
    typer.echo(f"frozen: {best} -> {L.PROMPT_DEV_FILE}")


@study_app.command("llm-reference")
def study_llm_reference(
    model: str = typer.Option("Qwen/Qwen3-1.7B"),
    backend: str = typer.Option("hf", help="hf | openai"),
    data: str = typer.Option("challenge", help="challenge | validation | neu_esc"),
    shots: int = typer.Option(0, help="0, or 6 demonstrations (2 per class)"),
    demo_seed: int = typer.Option(1, help="demonstration draw (declared: 1 and 2)"),
    configs: str = typer.Option(
        "",
        help="several configurations as data:shots:seed,... in one process: the model loads once "
        "and configurations that already have a summary are skipped (replaces --data/--shots/--demo-seed)",
    ),
    batch_size: int = typer.Option(8),
    dtype: str = typer.Option("float16", help="HF backend: float16 | bfloat16 | float32"),
    max_tokens: int = typer.Option(8192, help="HF backend: padded tokens per batch"),
    prefix_cache: bool = typer.Option(
        True, help="HF backend: reuse the shared prompt prefix's KV cache (self-checked, else off)"
    ),
    licence_confirmed: bool = typer.Option(
        False, help="owner confirmed the data may be sent to the API (cycle2.yaml data_egress)"
    ),
) -> None:
    """H7: score LLM configurations with the frozen prompt; compare with the encoders.

    neu_esc (cycle3.yaml v5): zero-shot only, since demonstrations are UIT-VSFC text."""
    import gc

    from vifeedback.data.loader import load
    from vifeedback.evaluation import llm_reference as L

    multi = bool(configs)
    jobs = (
        [(d, int(k), int(s)) for d, k, s in (c.split(":") for c in configs.split(","))]
        if multi
        else [(data, shots, demo_seed)]
    )
    for d, k, _ in jobs:
        if d not in ("challenge", "validation", "neu_esc"):
            raise typer.BadParameter(f"unknown data {d!r}: challenge | validation | neu_esc")
        if backend == "openai" and (d != "challenge" or k) and not licence_confirmed:
            raise typer.BadParameter(
                "corpus text (validation, NEU-ESC, or train demonstrations) goes to the API only "
                "after the owner confirms the licence allows it; pass --licence-confirmed once that "
                "is settled"
            )
        if d == "neu_esc" and k:
            raise typer.BadParameter(
                "neu_esc is zero-shot only: demonstrations would be UIT-VSFC text"
            )

    variant = L.frozen_variant()
    frozen = json.loads(L.PROMPT_DEV_FILE.read_text(encoding="utf-8"))
    tr = load("train")
    tr_texts, tr_y = tr.sentence.tolist(), tr.sentiment.to_numpy()

    # Everything is prepared before the LLM loads: the validation baseline may need the encoders,
    # whose weights must leave a 4 GB GPU first.
    prepared = []
    for d, k, s in jobs:
        run_name = f"{d}-{variant}-k{k}" + (f"-s{s}" if k else "")
        if multi and (L.OUT / L.slug(model) / run_name / "summary.json").exists():
            typer.echo(f"{model} {run_name}: already finished, skipped")
            continue
        demo_idx = (
            L.draw_demo_indices(tr_texts, tr_y, s, k // 3, exclude=frozen["subset"]["indices"])
            if k
            else []
        )
        demos = [(tr_texts[i], L.LABELS[int(tr_y[i])]) for i in demo_idx]
        prepared.append((d, k, s, run_name, demo_idx, demos, _llm_reference_data(d)))
    if not prepared:
        return

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    scorer = (
        L.OpenAIScorer(model)
        if backend == "openai"
        else L.HFScorer(model, dtype=dtype, prefix_cache=prefix_cache)
    )
    for d, k, s, run_name, demo_idx, demos, ds in prepared:
        _llm_reference_one(
            scorer,
            model,
            backend,
            variant,
            d,
            k,
            s,
            run_name,
            demo_idx,
            demos,
            ds,
            batch_size=batch_size,
            max_tokens=max_tokens,
            dtype=dtype,
        )


def _llm_reference_data(data: str) -> dict:
    """Texts, gold labels, the scored mask and the seed-42 encoder predictions for one dataset."""
    import numpy as np
    import pandas as pd

    from vifeedback.data.loader import load
    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import llm_reference as L

    if data == "challenge":
        df = CH.load()
        enc = pd.read_csv(
            paths.RESULTS / "studies" / "challenge" / "predictions.csv", keep_default_na=False
        )
        return {
            "df": df,
            "texts": df.text.tolist(),
            "scored": df.scored.to_numpy(),
            "y": df.y.to_numpy(),
            "enc_pred": {
                n: enc[f"{n}_pred"].map(L.LABELS.index).to_numpy() for n in ("ce", "augmented")
            },
        }
    if data == "neu_esc":
        from vifeedback.evaluation import external as X

        ne = X.load_neu_esc("test")
        y = ne.sentiment.map({c: i for i, c in enumerate(L.LABELS)}).to_numpy().astype(float)
        enc = pd.read_csv(paths.RESULTS / "studies" / "external" / "neu_esc" / "predictions.csv")
        return {
            "texts": ne.text.tolist(),
            "scored": np.ones(len(y), dtype=bool),
            "y": y,
            "enc_pred": {
                "ce": enc["lowercased:ce-s42"].to_numpy(),
                "augmented": enc["lowercased:aug-s42"].to_numpy(),
            },
        }
    from vifeedback.evaluation import error_analysis as EA
    from vifeedback.preprocess.variants import load_variant

    dv = load("validation")
    y = dv.sentiment.to_numpy().astype(float)
    # Computed once from the local checkpoints and committed (labels only, no text), so a Kaggle
    # session without the checkpoints compares against the same predictions.
    cache = L.OUT / "encoder_validation_preds.csv"
    if not cache.exists():
        x = load_variant("seg_pyvi", "validation").sentence.tolist()
        ckp = {
            "ce": "models/p6-sent-phobert-base-seg_pyvi-base-s42-ckp",
            "augmented": "models/p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-599cf21f-ckp",
        }
        pd.DataFrame(
            {f"{n}_pred": EA.predict_proba(p, x).argmax(1) for n, p in ckp.items()}
        ).to_csv(cache, index_label="row")
    cached = pd.read_csv(cache)
    return {
        "texts": dv.sentence.tolist(),
        "scored": np.ones(len(y), dtype=bool),
        "y": y,
        "enc_pred": {n: cached[f"{n}_pred"].to_numpy() for n in ("ce", "augmented")},
    }


def _llm_reference_one(
    scorer,
    model,
    backend,
    variant,
    data,
    shots,
    demo_seed,
    run_name,
    demo_idx,
    demos,
    ds,
    batch_size: int,
    max_tokens: int,
    dtype: str,
) -> None:
    """Score one configuration, write its summary and predictions, print the comparison."""
    import pandas as pd

    from vifeedback.evaluation import challenge as CH
    from vifeedback.evaluation import llm_reference as L
    from vifeedback.evaluation import metrics as M

    texts, scored, y, enc_pred = ds["texts"], ds["scored"], ds["y"], ds["enc_pred"]
    r = L.run(scorer, texts, variant, demos, batch_size=batch_size, max_tokens=max_tokens)
    pred = r["probs"].argmax(1)
    ys, ps = y[scored].astype(int), pred[scored]
    ev = M.evaluate(ys, ps, "sentiment", y_prob=r["probs"][scored])
    summary: dict = {
        "model": model,
        "revision": scorer.revision,
        "backend": backend,
        "data": data,
        "variant": variant,
        "shots": shots,
        "demo_seed": demo_seed if shots else None,
        # Train row indices and labels only: corpus text is never written to a result file.
        "demos": [
            {"train_index": i, "label": lab} for i, (_, lab) in zip(demo_idx, demos, strict=True)
        ],
        "n_scored": int(scored.sum()),
        "macro_f1": ev["macro_f1"],
        "per_class": ev["per_class"],
        "seconds": r["seconds"],
        "seconds_per_1k": r["seconds_per_1k"],
        "vs_encoder": {
            name: L.compare_to_encoder(ys, ps, p[scored]) for name, p in enc_pred.items()
        },
    }
    if isinstance(scorer, L.OpenAIScorer):
        summary["api"] = {
            **scorer.usage,
            "served_models": sorted(scorer.served_models),
            "price_per_m_tokens_usd": L.API_PRICE_PER_M,
            "cost_usd": scorer.cost_usd(),
            "usd_per_1k": 1000 * scorer.cost_usd() / len(texts),
            "note": "latency includes the network round trip",
        }
    else:
        summary["gpu_seconds_per_1k"] = r["seconds_per_1k"]
        summary["batch_size"] = batch_size
        summary["dtype"] = dtype
        summary["max_tokens"] = max_tokens
        summary["prefix_cache"] = dict(scorer.prefix_cache_check)
    if data == "challenge":
        df = ds["df"]
        summary["challenge"] = {
            **CH.category_report(df, pred),
            "negation_pairs": CH.negation_pairs(df, pred),
            "out_of_scope_confidence": CH.out_of_scope_confidence(df, r["probs"]),
        }

    out = L.OUT / L.slug(model) / run_name
    L.write(out / "summary.json", summary)
    table = pd.DataFrame(r["probs"].round(5), columns=[f"p_{c}" for c in L.LABELS])
    table.insert(0, "pred", [L.LABELS[i] for i in pred])
    if data == "challenge":  # constructed text ids only; corpus text is never written here
        table.insert(0, "id", ds["df"].id)
    table.to_csv(out / "predictions.csv", index_label="row")

    pc = ev["per_class"]
    cache_note = ""
    if "prefix_cache" in summary:
        pcc = summary["prefix_cache"]
        cache_note = f"  prefix cache {'on' if pcc.get('used') else 'off'}" + (
            f" (check diff {pcc['check_max_abs_diff']:.1e})" if "check_max_abs_diff" in pcc else ""
        )
    typer.echo(
        f"{model} {run_name}: macro-F1 {ev['macro_f1']:.4f}  neutral F1 {pc['neutral']['f1']:.3f}"
        f"  ({r['seconds_per_1k']:.1f} s per 1k){cache_note}"
    )
    for name, c in summary["vs_encoder"].items():
        n = c["neutral_f1"]
        typer.echo(
            f"  vs {name:9s} neutral F1 {n['observed_diff']:+.3f} [{n['ci_low']:+.3f}, "
            f"{n['ci_high']:+.3f}] p={n['p_value']:.3f}; macro-F1 {c['macro_f1']['observed_diff']:+.3f}"
        )
    if isinstance(scorer, L.OpenAIScorer):
        typer.echo(f"  API: {scorer.usage['calls']} calls, USD {scorer.cost_usd():.4f}")


@study_app.command("h7-decide")
def study_h7_decide() -> None:
    """Apply the H7 rule once both arms exist (cycle2.yaml H7; cycle3.yaml v6 h7_decision).

    Challenge v1: Holm over the two zero-shot arms' neutral-F1 p-values against CE. NEU-ESC
    (descriptive, h7_neu_esc_local): each arm against the encoders, and Qwen3-4B against gpt-4o-mini
    on the same posts. Labels and scores only are written.
    """
    import pandas as pd

    from vifeedback.evaluation import llm_reference as L

    variant = L.frozen_variant()

    def run_dir(model: str, data: str):
        return L.OUT / L.slug(model) / f"{data}-{variant}-k0"

    def summary(model: str, data: str) -> dict | None:
        f = run_dir(model, data) / "summary.json"
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None

    decision = L.h7_decide({name: summary(m, "challenge") for name, m in L.H7_ARMS.items()})

    neu: dict = {}
    for name, m in L.H7_ARMS.items():
        s = summary(m, "neu_esc")
        neu[name] = (
            None
            if s is None
            else {
                "macro_f1": s["macro_f1"],
                "neutral_f1": s["per_class"]["neutral"]["f1"],
                "vs_encoder": {
                    enc: {k: c[k]["observed_diff"] for k in ("macro_f1", "neutral_f1")}
                    for enc, c in s["vs_encoder"].items()
                },
            }
        )
    if all(v is not None for v in neu.values()):
        from vifeedback.evaluation import external as X

        ne = X.load_neu_esc("test")
        y = ne.sentiment.map({c: i for i, c in enumerate(L.LABELS)}).to_numpy().astype(int)
        pred = {
            name: pd.read_csv(run_dir(m, "neu_esc") / "predictions.csv")
            .pred.map(L.LABELS.index)
            .to_numpy()
            for name, m in L.H7_ARMS.items()
        }
        neu["qwen3-4b_minus_gpt-4o-mini"] = L.compare_to_encoder(
            y, pred["qwen3-4b"], pred["gpt-4o-mini"]
        )

    out = {
        "declared_in": "cycle2.yaml H7_llm_reference; cycle3.yaml v6 h7_decision, h7_neu_esc_local",
        "prompt_variant": variant,
        "challenge_v1": decision,
        "neu_esc": neu,
        "caveat": "challenge v1 is development data since ADR-028; routing is measured on real text",
    }
    L.write(L.OUT / "h7_decision.json", out)

    for name, r in decision["arms"].items():
        if r is None:
            typer.echo(f"  {name:12s} challenge v1: not run yet")
            continue
        extra = f"  Holm p {r['p_holm']:.4f}" if "p_holm" in r else ""
        typer.echo(
            f"  {name:12s} challenge v1 neutral F1 {r['neutral_f1']:.3f}, minus CE "
            f"{r['neutral_f1_minus_ce']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
            f"p {r['p']:.4f}{extra}"
        )
    typer.echo(
        f"  H7: {decision['outcome']} (better on neutral: {decision['better_on_neutral']})"
        if decision["decided"]
        else f"  H7: not decided, waiting for {decision['missing']}"
    )
    for name in L.H7_ARMS:
        r = neu[name]
        typer.echo(
            f"  {name:12s} NEU-ESC: not run yet"
            if r is None
            else f"  {name:12s} NEU-ESC macro-F1 {r['macro_f1']:.3f}  neutral F1 {r['neutral_f1']:.3f}"
        )
    typer.echo(f"  written: {L.OUT / 'h7_decision.json'}")
