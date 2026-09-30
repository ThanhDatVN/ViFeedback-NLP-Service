"""Cycle 5, H10: consistency training on real typing (configs/experiments/cycle5.yaml v1).

The served recipe (PhoBERT-base, seg_pyvi, diac-teen augmentation p = 0.3) plus, at every step, a
batch of ViLexNorm pairs of the same size: a real social-media comment and its human
normalization. The pairs carry no sentiment label; the loss asks only that the two forms get the
same prediction (UDA's consistency loss, with real pairs as the perturbation):

* ``onesided``: KL(p(normalized) || p(original)), no gradient through the normalized form, which is
  closer to UIT-VSFC's text and so serves as the target (stability training);
* ``symmetric``: the bidirectional KL of R-Drop, gradients through both forms.

Both forms are computed with the model in training mode (dropout on), as UDA computes its target.
The weight is 1.0. Every text outside UIT-VSFC goes through the service's own transform
(`domain.serving_transform`: lowercase NFC, the released restorer, pyvi), in training and in
evaluation. The epoch is selected by UIT-VSFC validation macro-F1, as in the served recipe.

The flip rate is the share of pairs whose predicted labels differ between the two forms.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths
from vifeedback.constants import SEEDS, label_names
from vifeedback.training import domain as D

RECIPES = ("onesided", "symmetric")
ANCHORED = ("anchored_orig", "anchored_both")  # H10b (cycle5.yaml v4), a frozen teacher target
PHASE_NUM_B = 15
WEIGHT = 1.0
PHASE_NUM = 13
N_VILEXNORM_TRAIN = 8372
N_DEV = 837
OUT = paths.RESULTS / "studies" / "cycle5" / "h10"
DEV_INDEX = paths.RESULTS / "studies" / "cycle5" / "vilexnorm_dev_index.csv"
TEST_USES = paths.RESULTS / "studies" / "cycle5" / "vilexnorm_test_uses.log"
LABELS = label_names("sentiment")


# --- data ---------------------------------------------------------------------------------------


def split_indices(n: int = N_VILEXNORM_TRAIN) -> tuple[np.ndarray, np.ndarray]:
    """cycle5.yaml data.vilexnorm_train: rows in default_rng(42).permutation(8372) order; the
    first 837 are the development split, the rest the consistency-training pairs."""
    if n != N_VILEXNORM_TRAIN:
        raise ValueError(f"ViLexNorm train has {n} pairs; cycle5.yaml declared {N_VILEXNORM_TRAIN}")
    perm = np.random.default_rng(42).permutation(n)
    return perm[:N_DEV], perm[N_DEV:]


def write_dev_index() -> Path:
    """The development rows, indices only (no text)."""
    import pandas as pd

    dev, train = split_indices()
    DEV_INDEX.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": np.sort(dev)}).to_csv(DEV_INDEX, index=False)
    assert len(set(dev) & set(train)) == 0
    return DEV_INDEX


def vilexnorm_pairs(split: str, transform) -> dict[str, Any]:
    """ViLexNorm pairs through the serving transform. `split` is train_pairs, dev or test."""
    from vifeedback.evaluation import external as X

    df = X.load_vilexnorm("test" if split == "test" else "train")
    if split != "test":
        dev, train = split_indices(len(df))
        df = df.iloc[dev if split == "dev" else train]
    orig = transform(df.original.astype(str).tolist())
    norm = transform(df.normalized.astype(str).tolist())
    return {
        "orig": orig,
        "norm": norm,
        "changed": np.array([a != b for a, b in zip(orig, norm, strict=True)]),
    }


def flip_rate(pred_orig: np.ndarray, pred_norm: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred_orig) != np.asarray(pred_norm)))


# --- loss and training --------------------------------------------------------------------------


def consistency_loss(logits_orig, logits_norm, recipe: str, q=None):
    """The consistency term for one batch of pairs (batch-mean KL, in float32).

    H10: `onesided`, `symmetric`. H10b: `anchored_orig` / `anchored_both`, KL(q || p) with q the
    frozen teacher's probabilities on the normalized form (on the original; also on the normalized).
    """
    import torch.nn.functional as F

    from vifeedback.training.losses import rdrop_kl

    if recipe in ANCHORED:
        target = q.float()

        def kl(logits):
            return F.kl_div(F.log_softmax(logits.float(), dim=-1), target, reduction="batchmean")

        if recipe == "anchored_orig":
            return kl(logits_orig)
        return 0.5 * (kl(logits_orig) + kl(logits_norm))

    if recipe == "onesided":  # KL(p_norm || p_orig), the normalized form as a fixed target
        target = F.log_softmax(logits_norm.detach().float(), dim=-1)
        return F.kl_div(
            F.log_softmax(logits_orig.float(), dim=-1),
            target,
            log_target=True,
            reduction="batchmean",
        )
    if recipe == "symmetric":
        return rdrop_kl(logits_orig.float(), logits_norm.float())
    raise ValueError(f"recipe must be one of {RECIPES + ANCHORED}")


def config(recipe: str, seed: int):
    """The served recipe's TrainConfig; only the recipe name (and so the run id) changes."""
    from vifeedback.training.trainer import TrainConfig

    if recipe not in RECIPES + ANCHORED:
        raise ValueError(f"recipe must be one of {RECIPES + ANCHORED}")
    b = recipe in ANCHORED
    return TrainConfig(
        task="sentiment",
        model_key="phobert-base",
        preprocessing="seg_pyvi",
        recipe=f"h10b-{recipe}" if b else f"h10-{recipe}",
        augment="diac-teen",
        augment_p=0.3,
        seed=seed,
        extra={"phase_num": PHASE_NUM_B if b else PHASE_NUM, "consistency_weight": WEIGHT},
        notes=f"cycle5.yaml {'v4 H10b' if b else 'H10'} {recipe}, weight {WEIGHT}",
    )


def train_consistency(
    cfg,
    recipe: str,
    uit_train: tuple[list[str], np.ndarray],
    pairs: dict[str, Any],
    uit_dev: tuple[list[str], np.ndarray],
    dev_pairs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Cross-entropy on UIT-VSFC plus WEIGHT x the consistency term on a same-size batch of pairs.

    Optimizer, schedule, AMP, clipping and epoch selection as `trainer.train`. The two losses are
    backpropagated one after the other within a step, so the GPU never holds the activations of
    both batches at once; their gradients add up before the single optimizer step.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    from vifeedback.constants import LABELS as LABEL_MAPS
    from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS
    from vifeedback.evaluation import metrics as M
    from vifeedback.training.losses import build_loss
    from vifeedback.training.seeding import describe_determinism, seed_everything, worker_init_fn
    from vifeedback.training.trainer import (
        TextDataset,
        build_optimizer,
        linear_warmup_schedule,
        predict,
        softmax,
    )

    if cfg.grad_accum != 1:
        raise ValueError("H10 uses the served recipe's grad_accum = 1")
    seed_everything(cfg.seed)
    device = cfg.resolved_device()
    model_id, revision = MODEL_IDS[cfg.model_key], MODEL_REVISIONS.get(cfg.model_key)
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        revision=revision,
        num_labels=len(LABELS),
        id2label=dict(LABEL_MAPS["sentiment"]),
        label2id={n: i for i, n in LABEL_MAPS["sentiment"].items()},
    ).to(device)
    pad = DataCollatorWithPadding(tok, padding="longest", return_tensors="pt")

    q_all = pairs.get("q")  # H10b: the frozen teacher's probabilities on the normalized form

    class _Pairs(Dataset):
        def __init__(self, orig: list[str], norm: list[str]) -> None:
            kw = {"truncation": True, "max_length": cfg.max_length, "padding": False}
            self.o, self.n = tok(list(orig), **kw), tok(list(norm), **kw)

        def __len__(self) -> int:
            return len(self.o["input_ids"])

        def __getitem__(self, i: int) -> tuple[dict[str, Any], dict[str, Any], int]:
            return {k: v[i] for k, v in self.o.items()}, {k: v[i] for k, v in self.n.items()}, i

    def collate_pairs(items):
        q = None if q_all is None else torch.as_tensor(q_all[[i for _, _, i in items]])
        return pad([a for a, _, _ in items]), pad([b for _, b, _ in items]), q

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    g_pairs = torch.Generator()
    g_pairs.manual_seed(cfg.seed + 1)
    train_loader = DataLoader(
        TextDataset(uit_train[0], uit_train[1], tok, cfg.max_length),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=pad,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    pair_loader = DataLoader(
        _Pairs(pairs["orig"], pairs["norm"]),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        generator=g_pairs,
    )
    dev_loader = DataLoader(
        TextDataset(uit_dev[0], uit_dev[1], tok, cfg.max_length),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=pad,
    )

    opt = build_optimizer(model, cfg)
    sched = linear_warmup_schedule(opt, len(train_loader) * cfg.epochs, cfg.warmup_ratio)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device != "cpu")
    ce = build_loss(  # exactly the served recipe's criterion
        cfg.loss,
        np.bincount(np.asarray(uit_train[1]), minlength=len(LABELS)),
        gamma=cfg.focal_gamma,
        tau=cfg.logit_adjust_tau,
        label_smoothing=cfg.label_smoothing,
        weight_scheme=cfg.weight_scheme,
        device=device,
    )
    ac: dict[str, Any] = dict(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=cfg.fp16 and device != "cpu"
    )

    def pair_batches():
        while True:  # the pairs are fewer than UIT-VSFC's batches: reshuffle and go round again
            yield from pair_loader

    pair_iter = pair_batches()
    best: dict[str, Any] = {"score": -1.0, "epoch": -1, "state": None}
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_ce = run_cons = 0.0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            y = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(**ac):
                loss_ce = ce(model(**batch).logits, y)
            scaler.scale(loss_ce).backward()

            orig, norm, q = next(pair_iter)
            orig = {k: v.to(device, non_blocking=True) for k, v in orig.items()}
            norm = {k: v.to(device, non_blocking=True) for k, v in norm.items()}
            q = None if q is None else q.to(device, non_blocking=True)
            with torch.autocast(**ac):
                logits_o = model(**orig).logits
                if recipe == "anchored_orig":
                    logits_n = None  # the normalized form enters only through the teacher's q
                elif recipe == "onesided":
                    with torch.no_grad():
                        logits_n = model(**norm).logits
                else:
                    logits_n = model(**norm).logits
            loss_cons = WEIGHT * consistency_loss(logits_o, logits_n, recipe, q)
            scaler.scale(loss_cons).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_ce += loss_ce.item()
            run_cons += loss_cons.item()

        logits, labels = predict(model, dev_loader, device, cfg.fp16)
        dev = M.evaluate(labels, logits.argmax(1), "sentiment", y_prob=softmax(logits))
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_ce": run_ce / len(train_loader),
            "train_consistency": run_cons / len(train_loader),
            "dev_macro_f1": dev["macro_f1"],
            "dev_neutral_f1": dev["per_class"]["neutral"]["f1"],
            "seconds": round(time.perf_counter() - t0, 1),
        }
        if dev_pairs is not None:  # reported, not used for the epoch (cycle5.yaml epoch_selection)
            po = predict_labels(model, tok, dev_pairs["orig"])
            pn = predict_labels(model, tok, dev_pairs["norm"])
            row["vilexnorm_dev_flip_rate"] = flip_rate(po, pn)
            model.train()
        history.append(row)
        if verbose:
            print(
                f"  epoch {epoch}/{cfg.epochs}  CE {row['train_ce']:.4f}  consistency "
                f"{row['train_consistency']:.4f}  dev macro-F1 {row['dev_macro_f1']:.4f}  neutral "
                f"{row['dev_neutral_f1']:.3f}"
                + (
                    f"  ViLexNorm dev flips {row['vilexnorm_dev_flip_rate']:.3f}"
                    if dev_pairs is not None
                    else ""
                )
                + f"  ({row['seconds']:.0f}s)"
            )
        if dev["macro_f1"] > best["score"]:
            best = {
                "score": dev["macro_f1"],
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= cfg.early_stopping_patience:
            break

    model.load_state_dict(best["state"])
    return {
        "model": model.eval(),
        "tokenizer": tok,
        "config": asdict(cfg),
        "history": history,
        "best_epoch": best["epoch"],
        "selection": "UIT-VSFC dev macro-F1",
        "train_seconds": round(time.perf_counter() - t0, 1),
        "determinism": describe_determinism(),
        "device": device,
        "steps_per_epoch": len(train_loader),
        "pair_passes_per_epoch": round(len(train_loader) / max(1, len(pair_loader)), 2),
        "optimizer_steps": len(train_loader) * len(history),
    }


# --- evaluation ---------------------------------------------------------------------------------


def predict_labels(model, tokenizer, texts: list[str]) -> np.ndarray:
    return D.predict_proba(model, tokenizer, list(texts)).argmax(1)


def score_pairs(model, tokenizer, pairs: dict[str, Any]) -> dict[str, Any]:
    po = predict_labels(model, tokenizer, pairs["orig"])
    pn = predict_labels(model, tokenizer, pairs["norm"])
    ch = pairs["changed"]
    return {
        "pairs": len(po),
        "changed_pairs": int(ch.sum()),
        "flip_rate": flip_rate(po, pn),
        "flip_rate_changed_pairs": flip_rate(po[ch], pn[ch]) if ch.any() else float("nan"),
        "pred_orig": po,
        "pred_norm": pn,
    }


def evaluation_sets(transform) -> dict[str, tuple[list[str], np.ndarray]]:
    """The labelled sets every H10 model is scored on (never a test split)."""
    from vifeedback.training.runner import _splits

    _, uit_dv, _ = _splits("sentiment", "seg_pyvi")
    neu_in = D.neu_esc("validation", transform)
    neu_all = D.neu_esc("validation", transform, in_scope=False)
    return {
        "uit_validation": uit_dv,
        "uit_validation_stripped": D.uit_stripped_validation(transform),
        "neu_validation": (neu_in["x"], neu_in["y"]),
        "neu_validation_all": (neu_all["x"], neu_all["y"]),
    }


def _write(
    name: str, summary: dict[str, Any], scored: dict[str, Any], flips: dict[str, Any]
) -> Path:
    import pandas as pd

    from vifeedback.evaluation.report import yaml_safe

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    body = {
        **summary,
        "sets": {
            s: {k: v for k, v in r.items() if k not in ("pred", "prob")} for s, r in scored.items()
        },
        "vilexnorm_dev": {k: v for k, v in flips.items() if not k.startswith("pred_")},
    }
    (d / "summary.json").write_text(
        json.dumps(yaml_safe(body), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for s, r in scored.items():  # labels and probabilities only; no text
        table = pd.DataFrame(r["prob"].round(5), columns=[f"p_{c}" for c in LABELS])
        table.insert(0, "pred", [LABELS[i] for i in r["pred"]])
        table.to_csv(d / f"predictions_{s}.csv", index_label="row")
    pd.DataFrame(
        {
            "orig": [LABELS[i] for i in flips["pred_orig"]],
            "norm": [LABELS[i] for i in flips["pred_norm"]],
        }
    ).to_csv(d / "predictions_vilexnorm_dev.csv", index_label="dev_position")
    return d


def run(recipe: str, seed: int, verbose: bool = True, smoke: bool = False) -> dict[str, Any]:
    """Train one H10 recipe at one seed, save the checkpoint, score it on the development sets.

    `smoke` checks the wiring in a minute: 256 UIT-VSFC sentences and 256 pairs, one epoch, 128 per
    evaluation set, and nothing is written.
    """
    import dataclasses

    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import report as R
    from vifeedback.training.runner import _augment_train, _splits

    cfg = config(recipe, seed)
    transform = D.serving_transform()
    (x_tr, y_tr), uit_dv, _ = _splits("sentiment", "seg_pyvi")
    x_tr, augmentation = _augment_train(cfg, x_tr)
    pairs = vilexnorm_pairs("train_pairs", transform)
    dev_pairs = vilexnorm_pairs("dev", transform)
    sets = evaluation_sets(transform)
    if smoke:
        cfg = dataclasses.replace(cfg, epochs=1)
        x_tr, y_tr = list(x_tr)[:256], np.asarray(y_tr)[:256]
        pairs = {k: v[:256] for k, v in pairs.items()}
        dev_pairs = {k: v[:128] for k, v in dev_pairs.items()}
        uit_dv = (list(uit_dv[0])[:128], np.asarray(uit_dv[1])[:128])
        sets = {k: (list(x)[:128], np.asarray(y)[:128]) for k, (x, y) in sets.items()}
    if verbose:
        print(
            f"[{cfg.run_id()}] H10 {recipe} seed {seed}: UIT-VSFC train {len(x_tr)}, ViLexNorm "
            f"pairs {len(pairs['orig'])} (dev {len(dev_pairs['orig'])})"
        )
    out = train_consistency(
        cfg, recipe, (list(x_tr), np.asarray(y_tr)), pairs, uit_dv, dev_pairs, verbose
    )
    model, tok = out["model"], out["tokenizer"]
    scored = D.score_sets(model, tok, sets)
    flips = score_pairs(model, tok, dev_pairs)
    summary = {
        "recipe": recipe,
        "seed": seed,
        "history": out["history"],
        "best_epoch": out["best_epoch"],
        "consistency_weight": WEIGHT,
    }
    if smoke:
        return {"summary": {**summary, "smoke": True}, "scored": scored, "flips": flips}

    ckpt = paths.MODELS / cfg.run_id("ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    uv = scored["uit_validation"]
    metrics = M.evaluate(np.asarray(uit_dv[1]), uv["pred"], "sentiment", y_prob=uv["prob"])
    metrics.update(
        {"history": out["history"], "best_epoch": out["best_epoch"], "augmentation": augmentation}
    )
    R.save_run(
        cfg.run_id("validation"),
        metrics,
        config={
            **out["config"],
            "phase": f"P{PHASE_NUM}",
            "model": cfg.model_key,
            "split": "validation",
            "fit_seconds": out["train_seconds"],
            "notes": cfg.notes,
            "reason": "cycle5.yaml H10",
            "determinism": out["determinism"],
            "device": out["device"],
            "best_epoch": out["best_epoch"],
            "selection": out["selection"],
            "vilexnorm_pairs": len(pairs["orig"]),
        },
        y_true=np.asarray(uit_dv[1]),
        y_pred=uv["pred"],
        y_prob=uv["prob"],
    )
    summary.update(
        {
            "run_id": cfg.run_id("validation"),
            "checkpoint": str(ckpt.relative_to(paths.ROOT)),
            "train_seconds": out["train_seconds"],
            "augmentation": augmentation,
            "pair_passes_per_epoch": out["pair_passes_per_epoch"],
        }
    )
    _write(f"{recipe}-s{seed}", summary, scored, flips)
    return {"summary": summary, "scored": scored, "flips": flips}


def evaluate_control(seed: int) -> dict[str, Any]:
    """The served recipe's checkpoint of one seed, scored with the candidates' code."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = D.control_checkpoint(seed)
    transform = D.serving_transform()
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    scored = D.score_sets(model, tok, evaluation_sets(transform))
    flips = score_pairs(model, tok, vilexnorm_pairs("dev", transform))
    summary = {"recipe": "control", "seed": seed, "checkpoint": str(ckpt.relative_to(paths.ROOT))}
    _write(f"control-s{seed}", summary, scored, flips)
    return {"summary": summary, "scored": scored, "flips": flips}


def _summary(name: str) -> dict[str, Any]:
    path = OUT / name / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"{name} has not run: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select() -> dict[str, Any]:
    """cycle5.yaml H10 recipe_selection, from the seed-42 summaries."""
    from vifeedback.evaluation.report import yaml_safe

    ctrl = _summary("control-s42")
    ctrl_uit = ctrl["sets"]["uit_validation"]["macro_f1"]
    ctrl_flip = ctrl["vilexnorm_dev"]["flip_rate"]
    rows = {}
    for r in RECIPES:
        s = _summary(f"{r}-s42")
        uit = s["sets"]["uit_validation"]["macro_f1"]
        rows[r] = {
            "uit_validation": uit,
            "uit_minus_control": uit - ctrl_uit,
            "vilexnorm_dev_flip_rate": s["vilexnorm_dev"]["flip_rate"],
            "eligible": uit >= ctrl_uit - 0.01,
        }
    eligible = {r: v for r, v in rows.items() if v["eligible"]}
    chosen = (
        min(eligible, key=lambda r: eligible[r]["vilexnorm_dev_flip_rate"]) if eligible else None
    )
    if chosen is None:
        outcome = "not supported: no recipe is eligible"
    elif rows[chosen]["vilexnorm_dev_flip_rate"] >= ctrl_flip:
        outcome, chosen = "not supported: no eligible recipe lowers the dev flip rate", None
    else:
        outcome = "confirm"
    out = {
        "declared_in": "configs/experiments/cycle5.yaml H10 recipe_selection",
        "control_s42": {"uit_validation": ctrl_uit, "vilexnorm_dev_flip_rate": ctrl_flip},
        "recipes": rows,
        "chosen": chosen,
        "outcome": outcome,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def flip_bootstrap(
    cand: list[np.ndarray], ctrl: list[np.ndarray], n_resamples: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Seed-averaged paired difference in flip rate (candidate minus control), pairs resampled.

    A flip rate is a mean over pairs, so the seed-averaged difference is the mean over pairs of each
    pair's seed-averaged difference, and resampling pairs resamples those per-pair values.
    """
    per_pair = np.mean(
        [np.asarray(c, float) - np.asarray(b, float) for c, b in zip(cand, ctrl, strict=True)],
        axis=0,
    )
    rng = np.random.default_rng(seed)
    n = len(per_pair)
    draws = per_pair[rng.integers(0, n, (n_resamples, n))].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "observed_diff": float(per_pair.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_one_sided": float((np.sum(draws >= 0) + 1) / (n_resamples + 1)),
        "n_resamples": n_resamples,
        "seeds": len(cand),
        "pairs": n,
    }


def apply_rule(flip: dict[str, Any], diffs: dict[str, float]) -> dict[str, Any]:
    """cycle5.yaml H10 rule (1) to (5): the flip interval below 0, and four non-inferiority guards."""
    limits = {
        "2_uit_validation_macro_f1": -0.005,
        "3_uit_validation_neutral_f1": -0.02,
        "4_uit_validation_stripped_macro_f1": -0.01,
        "5_neu_validation_macro_f1": -0.01,
    }
    rules: dict[str, Any] = {
        "1_vilexnorm_test_flip_rate_down": {**flip, "passed": flip["ci_high"] < 0}
    }
    for name, limit in limits.items():
        rules[name] = {"mean_diff": diffs[name], "limit": limit, "passed": diffs[name] >= limit}
    return {"rules": rules, "passed": all(r["passed"] for r in rules.values())}


def log_test_use(rule: str, models: list[str]) -> None:
    """Every ViLexNorm test evaluation in Cycle 5 is recorded (cycle5.yaml data.vilexnorm_test)."""
    from datetime import UTC, datetime

    TEST_USES.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_USES, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\t{rule}\t{','.join(models)}\n")


def confirm(recipe: str) -> dict[str, Any]:
    """cycle5.yaml H10 rule, five seeds each side. The only function that reads ViLexNorm test;
    every call is logged."""
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.serving import pipeline as SP

    seeds = list(SEEDS)
    cand = {s: paths.MODELS / config(recipe, s).run_id("ckpt") for s in seeds}
    ctrl = {s: D.control_checkpoint(s) for s in seeds}
    missing = [str(p) for p in cand.values() if not p.exists()]
    missing += [
        f"{n}-s{s}" for n in (recipe, "control") for s in seeds if not (OUT / f"{n}-s{s}").exists()
    ]
    if missing:
        raise FileNotFoundError(f"run h10-run / h10-control at every seed first: {missing}")

    transform = D.serving_transform()
    te = vilexnorm_pairs("test", transform)
    log_test_use(
        "cycle5.yaml H10 rule (1) and reported flips",
        [p.name for p in [*cand.values(), *ctrl.values()]],
    )

    def preds(ckpt: Path) -> dict[str, np.ndarray]:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        return {
            "orig": predict_labels(model, tok, te["orig"]),
            "norm": predict_labels(model, tok, te["norm"]),
        }

    p_cand = {s: preds(cand[s]) for s in seeds}
    p_ctrl = {s: preds(ctrl[s]) for s in seeds}
    flips_c = [p_cand[s]["orig"] != p_cand[s]["norm"] for s in seeds]
    flips_b = [p_ctrl[s]["orig"] != p_ctrl[s]["norm"] for s in seeds]
    flip = flip_bootstrap(flips_c, flips_b)

    def val(name: str, set_name: str, key: str) -> float:
        r = _summary(name)["sets"][set_name]
        return r["macro_f1"] if key == "macro_f1" else r["per_class"]["neutral"]["f1"]

    def mean_diff(set_name: str, key: str = "macro_f1") -> float:
        return float(
            np.mean(
                [
                    val(f"{recipe}-s{s}", set_name, key) - val(f"control-s{s}", set_name, key)
                    for s in seeds
                ]
            )
        )

    decision = apply_rule(
        flip,
        {
            "2_uit_validation_macro_f1": mean_diff("uit_validation"),
            "3_uit_validation_neutral_f1": mean_diff("uit_validation", "neutral_f1"),
            "4_uit_validation_stripped_macro_f1": mean_diff("uit_validation_stripped"),
            "5_neu_validation_macro_f1": mean_diff("neu_validation"),
        },
    )

    # Reported, not tested (cycle5.yaml reported_not_tested).
    ch = te["changed"]
    d_man = paths.MODELS / "serve" / "sentiment"
    manifest = json.loads((d_man / "manifest.json").read_text(encoding="utf-8"))
    scope = SP.load_scope(d_man, manifest["scope"])

    def directions(p: dict[str, np.ndarray]) -> dict[str, int]:
        f = p["orig"] != p["norm"]
        return {
            f"{LABELS[a]}->{LABELS[b]}": int(((p["orig"] == a) & (p["norm"] == b) & f).sum())
            for a in range(len(LABELS))
            for b in range(len(LABELS))
            if a != b
        }

    reported = {
        "flip_rate_per_seed": {
            s: {"candidate": float(flips_c[i].mean()), "control": float(flips_b[i].mean())}
            for i, s in enumerate(seeds)
        },
        "changed_pairs": int(ch.sum()),
        "flip_rate_changed_pairs_mean": {
            "candidate": float(np.mean([f[ch].mean() for f in flips_c])),
            "control": float(np.mean([f[ch].mean() for f in flips_b])),
        },
        "flip_directions_seed42": {
            "candidate": directions(p_cand[42]),
            "control": directions(p_ctrl[42]),
        },
        "neu_validation_all_macro_f1_mean_diff": mean_diff("neu_validation_all"),
        "scope_flagged_vilexnorm_test": {
            "original": float((scope.decision(te["orig"]) < scope.threshold).mean()),
            "normalized": float((scope.decision(te["norm"]) < scope.threshold).mean()),
        },
    }
    out = {
        "declared_in": "configs/experiments/cycle5.yaml H10 rule",
        "recipe": recipe,
        "seeds": seeds,
        **decision,
        "reported": reported,
        "vilexnorm_test_pairs": len(ch),
    }
    d = OUT / f"confirm-{recipe}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cols: dict[str, list[str]] = {}
    for s in seeds:  # labels only
        for side, p in (("candidate", p_cand[s]), ("control", p_ctrl[s])):
            cols[f"{side}-s{s}-orig"] = [LABELS[i] for i in p["orig"]]
            cols[f"{side}-s{s}-norm"] = [LABELS[i] for i in p["norm"]]
    pd.DataFrame(cols).to_csv(d / "predictions_vilexnorm_test.csv", index_label="row")
    return out


__all__ = [
    "ANCHORED",
    "N_DEV",
    "OUT",
    "PHASE_NUM",
    "RECIPES",
    "apply_rule",
    "config",
    "confirm",
    "consistency_loss",
    "evaluate_control",
    "flip_bootstrap",
    "flip_rate",
    "run",
    "select",
    "split_indices",
    "train_consistency",
    "vilexnorm_pairs",
    "write_dev_index",
]
