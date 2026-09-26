"""Error analysis and label-audit export — Study A of docs/REVIEW_AND_RESEARCH_PLAN.md.

The study's premise, from the review: *diagnose neutral before optimizing it*. Four competing
explanations for weak neutral performance — imbalance, ambiguous annotation, weak representation, a
biased decision boundary — call for different interventions, and optimizing blindly would test all
four at once and learn nothing about which one mattered.

This module produces the evidence, not the verdict:

* predictions and uncertainty from a saved checkpoint, on the checkpoint's own preprocessing;
* out-of-fold (OOF) predictions over `train`, so that suspected label issues can be ranked by a
  model that never saw the example it is scoring;
* measurable linguistic flags, so errors can be described by what the text contains;
* a stratified audit sample that mixes model-selected hard cases with a **random** component, so
  the audit does not only describe the examples a model already found difficult.

What it deliberately does not do is decide which gold labels are wrong. Model disagreement is
evidence for inspection, not proof of a labelling error (Northcutt et al., Confident Learning); the
adjudication is a human step, recorded in the exported sheet.

scikit-learn is not imported anywhere here. On the reference machine a host security policy blocks
its native extension (ADR-017), and every function below needs only numpy and pandas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vifeedback.constants import label_names

# --- Prediction -----------------------------------------------------------------------------------


def predict_proba(
    checkpoint: str | Path,
    texts: list[str],
    *,
    max_length: int = 96,
    batch_size: int = 128,
    device: str = "auto",
    fp16: bool = True,
) -> np.ndarray:
    """Class probabilities from a saved checkpoint.

    `texts` must already carry the checkpoint's preprocessing. Feeding raw text into a model trained
    on segmented input measures a pipeline that was never trained, which is the mismatch review R3
    found in the export path.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    from vifeedback.training.trainer import TextDataset, predict, softmax

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(dev).eval()

    ds = TextDataset(texts, np.zeros(len(texts), dtype=np.int64), tok, max_length)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tok, padding="longest", return_tensors="pt"),
    )
    logits, _ = predict(model, loader, dev, fp16 and dev != "cpu")
    return softmax(logits)


def uncertainty(probs: np.ndarray) -> dict[str, np.ndarray]:
    """Three standard uncertainty scores. Higher means *less* certain for all three.

    They disagree often enough to be worth keeping separately: max-probability ignores how the
    remaining mass is spread, margin looks only at the top two classes, and entropy uses all of it.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    top2 = np.sort(p, axis=1)[:, -2:]
    return {
        "one_minus_max_prob": 1.0 - top2[:, 1],
        "inverse_margin": 1.0 - (top2[:, 1] - top2[:, 0]),
        "entropy": -(p * np.log(p)).sum(axis=1) / np.log(p.shape[1]),  # normalized to [0, 1]
    }


# --- Stratified folds without scikit-learn -----------------------------------------------------------


def stratified_folds(
    y: np.ndarray, k: int = 5, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold indices, preserving class proportions in every fold.

    A plain random split can leave a fold with almost no neutral examples — there are only 458 in
    `train` — and an OOF model trained without the minority class says nothing useful about it.
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)) for f in range(k)]


def _probs_from_model(model: Any, tok: Any, device: str, cfg: Any, texts: list[str]) -> np.ndarray:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    from vifeedback.training.trainer import TextDataset, predict, softmax

    ds = TextDataset(texts, np.zeros(len(texts), dtype=np.int64), tok, cfg.max_length)
    collate = DataCollatorWithPadding(tok, padding="longest", return_tensors="pt")
    dl = DataLoader(ds, batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate)
    return softmax(predict(model, dl, device, cfg.fp16)[0])


def oof_predictions(
    cfg: Any,
    texts: list[str],
    labels: np.ndarray,
    dev_texts: list[str],
    dev_labels: np.ndarray,
    *,
    k: int = 5,
    fold_seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Out-of-fold class probabilities for every training example, plus each fold model's dev view.

    Each fold model trains on the other k-1 folds and selects its checkpoint on `dev` exactly as the
    main runs do, so the held-out fold is touched only at prediction time. The per-fold dev
    probabilities come for free and give a k-model disagreement signal on `dev` that one checkpoint
    cannot.

    Cost: k full fine-tunes on (k-1)/k of the data. The review asks for this cost to be budgeted
    explicitly; the caller records it.
    """
    import time

    from vifeedback.training.trainer import train

    labels = np.asarray(labels, dtype=np.int64)
    n_cls = int(labels.max()) + 1
    oof = np.full((len(texts), n_cls), np.nan)
    fold_id = np.full(len(texts), -1, dtype=np.int64)
    dev_probs: list[np.ndarray] = []
    folds_meta: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for f, (tr, ho) in enumerate(stratified_folds(labels, k=k, seed=fold_seed)):
        if verbose:
            print(f"fold {f + 1}/{k}: train {len(tr)}  held-out {len(ho)}")
        res = train(
            cfg,
            [texts[i] for i in tr],
            labels[tr],
            dev_texts,
            np.asarray(dev_labels),
            verbose=verbose,
        )
        model, tok, device = res["model"], res["tokenizer"], res["device"]
        ho_texts = [texts[i] for i in ho]
        oof[ho] = _probs_from_model(model, tok, device, cfg, ho_texts)
        fold_id[ho] = f
        dev_probs.append(_probs_from_model(model, tok, device, cfg, dev_texts))
        folds_meta.append(
            {
                "fold": f,
                "n_train": len(tr),
                "n_heldout": len(ho),
                "best_epoch": res["best_epoch"],
                "best_dev_macro_f1": round(float(res["best_dev_macro_f1"]), 4),
                "train_seconds": res["train_seconds"],
            }
        )
        del model, res
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert not np.isnan(oof).any(), "every training example must be held out exactly once"
    return {
        "oof_probs": oof,
        "fold_id": fold_id,
        "dev_probs": np.stack(dev_probs),  # (k, n_dev, n_cls)
        "folds": folds_meta,
        "total_seconds": round(time.perf_counter() - t0, 1),
    }


def fold_disagreement(dev_probs: np.ndarray) -> dict[str, np.ndarray]:
    """Across k fold models on the same examples: vote agreement and mean-probability spread."""
    preds = dev_probs.argmax(axis=2)  # (k, n)
    k = preds.shape[0]
    mode_share = np.array(
        [
            np.bincount(preds[:, i], minlength=dev_probs.shape[2]).max() / k
            for i in range(preds.shape[1])
        ]
    )
    return {
        "vote_agreement": mode_share,  # 1.0 = all folds agree
        "prob_std_max": dev_probs.std(axis=0).max(axis=1),
    }


# --- Linguistic flags ---------------------------------------------------------------------------------

# Markers measured in the EDA (notebooks/01_eda.ipynb § 8). Whole-token matches on the raw,
# already-lowercased corpus text. These describe what a sentence contains; they are not labels.
_MARKERS: dict[str, frozenset[str]] = {
    "negation": frozenset({"không", "ko", "chưa", "chẳng", "chả", "đừng", "khỏi"}),
    "contrast": frozenset({"nhưng", "tuy", "mặc", "song"}),
    "suggestion": frozenset({"nên", "cần", "mong", "đề", "nghị"}),
    "intensifier": frozenset({"rất", "quá", "lắm", "hơi", "khá", "cực"}),
    "politeness": frozenset({"cảm", "cám", "ơn", "ạ", "xin"}),
    "hedge": frozenset({"tạm", "bình", "thường", "được", "ổn"}),
}


def linguistic_flags(raw_texts: list[str]) -> pd.DataFrame:
    rows = []
    for t in raw_texts:
        toks = set(t.split())
        row: dict[str, Any] = {name: bool(toks & words) for name, words in _MARKERS.items()}
        n = len(t.split())
        row["n_syllables"] = n
        row["short_lt5"] = n < 5
        rows.append(row)
    return pd.DataFrame(rows)


# --- Prediction table and audit sample ----------------------------------------------------------------


def prediction_table(
    raw_texts: list[str],
    model_texts: list[str],
    y_true: np.ndarray,
    probs: np.ndarray,
    task: str,
    *,
    split: str,
    source: str,
) -> pd.DataFrame:
    """One row per example: text, gold, prediction, probabilities, uncertainty and flags."""
    names = label_names(task)
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = probs.argmax(axis=1)

    df = pd.DataFrame(
        {
            "split": split,
            "prediction_source": source,  # "checkpoint" (held-out) or "oof"
            "text": raw_texts,
            "model_input": model_texts,
            "gold": [names[i] for i in y_true],
            "pred": [names[i] for i in y_pred],
            "gold_id": y_true,
            "pred_id": y_pred,
            "correct": y_true == y_pred,
            "p_gold": probs[np.arange(len(y_true)), y_true],
            "p_pred": probs.max(axis=1),
        }
    )
    for i, n in enumerate(names):
        df[f"p_{n}"] = probs[:, i]
    for k, v in uncertainty(probs).items():
        df[k] = v
    return pd.concat([df, linguistic_flags(raw_texts)], axis=1)


def stratified_audit_sample(
    table: pd.DataFrame,
    *,
    per_error_cell: int = 12,
    per_correct_class: int = 8,
    n_random: int = 40,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the audit sample the review asks for (Study A, step 1).

    Three components, recorded in the `stratum` column so that every rate reported later can be
    computed *within* its stratum rather than extrapolated from a targeted sample:

    * **error cells** — up to `per_error_cell` examples from each off-diagonal confusion cell,
      taking the most confident errors first, since those are the ones most likely to be gold
      problems rather than model hesitation;
    * **correct, low-confidence** — up to `per_correct_class` correct predictions per class with the
      highest entropy. Without these the audit only sees failures and cannot tell an ambiguous
      *class* from an ambiguous *model*;
    * **random** — `n_random` examples drawn uniformly. The control: it keeps the audit from only
      describing what a model already found hard.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []

    errs = table[~table.correct]
    for (g, p), cell in errs.groupby(["gold", "pred"]):
        take = cell.sort_values("p_pred", ascending=False).head(per_error_cell).copy()
        take["stratum"] = f"error:{g}->{p}"
        parts.append(take)

    for g, cls in table[table.correct].groupby("gold"):
        take = cls.sort_values("entropy", ascending=False).head(per_correct_class).copy()
        take["stratum"] = f"correct_uncertain:{g}"
        parts.append(take)

    chosen = pd.concat(parts).index if parts else pd.Index([])
    pool = table.drop(index=chosen, errors="ignore")
    if len(pool):
        idx = rng.choice(pool.index.to_numpy(), size=min(n_random, len(pool)), replace=False)
        rand = pool.loc[idx].copy()
        rand["stratum"] = "random"
        parts.append(rand)

    sample = pd.concat(parts)
    return sample.sample(frac=1.0, random_state=seed).reset_index(names="example_index")


ANNOTATION_COLUMNS = (
    "annotator_label",  # one of the task labels, or "ambiguous"
    "neutral_subtype",  # objective_fact | no_opinion | request_suggestion | mixed | insufficient_context | n/a
    "gold_assessment",  # agree | ambiguous | incorrect
    "notes",
)


def export_annotation_sheet(sample: pd.DataFrame, path: Path) -> Path:
    """Write the sheet an annotator fills in. Model columns are kept but moved to the end.

    Annotators should label *before* looking at the model columns; putting them last makes that
    the path of least resistance. See docs/ANNOTATION_GUIDE.md for the definitions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    front = ["example_index", "split", "stratum", "text", "gold"]
    blank = {c: "" for c in ANNOTATION_COLUMNS}
    out = sample.assign(**blank)
    model_cols = ["pred", "p_pred", "p_gold", "entropy", "prediction_source"]
    flag_cols = [c for c in _MARKERS] + ["n_syllables"]
    cols = front + list(ANNOTATION_COLUMNS) + flag_cols + model_cols
    out[[c for c in cols if c in out.columns]].to_csv(path, index=False, encoding="utf-8-sig")
    return path


# --- Descriptive summaries (no labels invented) ------------------------------------------------------


def error_rate_by_flag(table: pd.DataFrame, gold_class: str | None = None) -> pd.DataFrame:
    """Error rate conditional on each linguistic flag, with support.

    Descriptive only. A higher error rate among negated sentences says negation co-occurs with
    errors; it does not say negation *causes* them, because length, topic and label mix all vary
    with it too.
    """
    t = table if gold_class is None else table[table.gold == gold_class]
    rows = []
    for flag in [*_MARKERS, "short_lt5"]:
        for val in (True, False):
            s = t[t[flag] == val]
            if len(s):
                rows.append(
                    {
                        "flag": flag,
                        "present": val,
                        "n": len(s),
                        "error_rate": round(float(1 - s.correct.mean()), 4),
                    }
                )
    return pd.DataFrame(rows)


def suspected_label_issues(oof: pd.DataFrame, top_k: int = 100) -> pd.DataFrame:
    """Rank training examples by how strongly an out-of-fold model disagrees with the gold label.

    Score = the OOF model's confidence in its own prediction when that prediction disagrees with
    gold. High scores are *candidates for inspection*, nothing more: a confident, wrong model and a
    wrong gold label look identical from here.
    """
    d = oof[~oof.correct].copy()
    d["issue_score"] = d["p_pred"] - d["p_gold"]
    return d.sort_values("issue_score", ascending=False).head(top_k)


# --- Neutral diagnosis ----------------------------------------------------------------------------


def neutral_diagnosis(table: pd.DataFrame, confident: float = 0.9) -> dict[str, Any]:
    """Where neutral errors go, and how confident the model is when it makes them.

    Confidence separates two explanations. Low-confidence neutral errors point to a genuinely
    ambiguous boundary; confident ones point to a systematic disagreement between the model and the
    labelling policy (possibly label noise, possibly a policy the model cannot infer from text).
    """
    names = sorted(table.gold.unique())
    cm = pd.crosstab(table.gold, table.pred).reindex(index=names, columns=names, fill_value=0)
    ng = table[table.gold == "neutral"]
    np_ = table[table.pred == "neutral"]
    ng_err = ng[~ng.correct]
    return {
        "confusion_gold_to_pred": cm.to_dict(orient="index"),
        "neutral_support": len(ng),
        "neutral_recall": float(ng.correct.mean()) if len(ng) else None,
        "neutral_precision": float(np_.correct.mean()) if len(np_) else None,
        "neutral_errors_to": ng_err.pred.value_counts().to_dict(),
        "false_neutral_from": np_[~np_.correct].gold.value_counts().to_dict(),
        "neutral_errors_confident_share": float((ng_err.p_pred >= confident).mean())
        if len(ng_err)
        else None,
        "neutral_errors_mean_p_gold": float(ng_err.p_gold.mean()) if len(ng_err) else None,
        "neutral_correct_mean_entropy": float(ng[ng.correct].entropy.mean())
        if ng.correct.any()
        else None,
        "neutral_errors_mean_entropy": float(ng_err.entropy.mean()) if len(ng_err) else None,
        "confident_threshold": confident,
    }


def tune_class_bias(
    probs: np.ndarray,
    y: np.ndarray,
    cls: int,
    k: int,
    grid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Additive bias on one class's log-probability, chosen to maximize macro-F1.

    A post-hoc test of the *decision-boundary* explanation: if shifting the boundary alone recovers
    much of the minority-class gap, the representation already separates the class and the loss or
    prior is misplacing the threshold. Must be tuned on one set and evaluated on another.
    """
    from vifeedback.evaluation.metrics import macro_f1

    grid = np.round(np.arange(-2.0, 4.0001, 0.05), 4) if grid is None else grid
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    scores = []
    for b in grid:
        adj = logp.copy()
        adj[:, cls] += b
        scores.append(macro_f1(np.asarray(y), adj.argmax(1), k))
    best = int(np.argmax(scores))
    return {
        "bias": float(grid[best]),
        "macro_f1": float(scores[best]),
        "at_zero": float(scores[int(np.argmin(np.abs(grid)))]),
    }


def apply_class_bias(probs: np.ndarray, cls: int, bias: float) -> np.ndarray:
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    logp[:, cls] += bias
    return logp.argmax(1)
