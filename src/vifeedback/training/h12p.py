"""Cycle 5 H12': held-out in-scope NEU-ESC train posts (cycle5.yaml v6).

The held-out set is drawn once, stratified by label, and written as row indices into the NEU-ESC
train split (no text), so every later step reads the same 3,000 posts.

The recipe, ``two_heads_anchored``, is H8's two heads (ADR-033: a UIT-VSFC head that is served and a
NEU-ESC head trained on NEU-ESC, one shared encoder) trained with H10b's anchored consistency on the
UIT-VSFC head (ADR-042). The control is the served recipe, H10b anchored_orig, at its five seeds.
"""

from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vifeedback import paths
from vifeedback.constants import SEEDS

N_HOLDOUT = 3000
SEED = 44
OUT = paths.RESULTS / "studies" / "cycle5" / "h12p"
INDEX = paths.RESULTS / "studies" / "cycle5" / "h12p_holdout_index.csv"
RECIPE = "two_heads_anchored"
PHASE_NUM = 16
WEIGHT = 1.0


def holdout_split(frame: pd.DataFrame, off_topic: Collection[str]) -> tuple[np.ndarray, np.ndarray]:
    """Row positions (held-out, training) of the in-scope posts, as cycle5.yaml v6 declares.

    Within each label, its in-scope rows in loader order are ordered by
    default_rng(44).permutation(n_label); the first round(3000 * n_label / n_in_scope) are held out.
    """
    in_scope = np.flatnonzero(~frame.topic.isin(off_topic).to_numpy())
    labels = frame.sentiment.to_numpy()[in_scope]
    held: list[np.ndarray] = []
    for label in sorted(set(labels)):
        rows = in_scope[labels == label]
        k = round(N_HOLDOUT * len(rows) / len(in_scope))
        held.append(rows[np.random.default_rng(SEED).permutation(len(rows))[:k]])
    holdout = np.sort(np.concatenate(held))
    return holdout, np.setdiff1d(in_scope, holdout)


def write_holdout_index() -> Path:
    """Draw the split from NEU-ESC train and write the held-out row positions with their labels."""
    from vifeedback.evaluation import external as X
    from vifeedback.training.domain import OFF_TOPIC

    frame = X.load_neu_esc("train")
    holdout, _ = holdout_split(frame, OFF_TOPIC)
    if INDEX.exists():
        old = pd.read_csv(INDEX)["row"].to_numpy()
        if not np.array_equal(old, holdout):
            raise RuntimeError(f"{INDEX} exists with a different split; it is drawn once")
        return INDEX
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row": holdout, "sentiment": frame.sentiment.to_numpy()[holdout]}).to_csv(
        INDEX, index=False
    )
    return INDEX


# --- data -----------------------------------------------------------------------------------------


def neu_train_and_holdout(transform) -> dict[str, dict[str, Any]]:
    """The in-scope NEU-ESC train split as declared: 18,113 training posts and the 3,000 held out,
    through the serving transform, with UIT-VSFC label ids (Toxic -> negative) and topics."""
    from vifeedback.evaluation import external as X
    from vifeedback.training.domain import LABELS, OFF_TOPIC

    frame = X.load_neu_esc("train")
    holdout, train = holdout_split(frame, OFF_TOPIC)
    if not np.array_equal(pd.read_csv(INDEX)["row"].to_numpy(), holdout):
        raise RuntimeError(f"the held-out split differs from {INDEX}")
    ids = {c: i for i, c in enumerate(LABELS)}
    out: dict[str, dict[str, Any]] = {}
    for name, rows in (("train", train), ("holdout", holdout)):
        part = frame.iloc[rows]
        raw = part.text.tolist()
        out[name] = {
            "raw": raw,
            "x": transform(raw),
            "y": part.sentiment.map(ids).to_numpy().astype(int),
            "topic": part.topic.to_numpy(),
        }
    return out


def config(seed: int):
    """The served recipe's TrainConfig; only the recipe name and phase (so the run id) change."""
    from vifeedback.training.trainer import TrainConfig

    return TrainConfig(
        task="sentiment",
        model_key="phobert-base",
        preprocessing="seg_pyvi",
        recipe=f"h12p-{RECIPE}",
        augment="diac-teen",
        augment_p=0.3,
        seed=seed,
        extra={"phase_num": PHASE_NUM, "consistency_weight": WEIGHT},
        notes=f"cycle5.yaml v6 H12' {RECIPE}, weight {WEIGHT}",
    )


def control_checkpoint(seed: int) -> Path:
    """The control of H12': the served recipe, H10b anchored_orig, at one seed."""
    from vifeedback.training import consistency as C

    ck = paths.MODELS / C.config("anchored_orig", seed).run_id("ckpt")
    if not ck.exists():
        raise FileNotFoundError(ck)
    return ck


# --- training -------------------------------------------------------------------------------------


def train_two_heads_anchored(
    cfg,
    uit_train: tuple[list[str], np.ndarray],
    neu_train: tuple[list[str], np.ndarray],
    pairs: dict[str, Any],
    uit_dev: tuple[list[str], np.ndarray],
    neu_dev: tuple[list[str], np.ndarray],
    verbose: bool = True,
) -> dict[str, Any]:
    """H8's masked two-head fine-tuning plus, at every step, a same-size batch of ViLexNorm pairs
    with WEIGHT x KL(q(normalized) || p_uit(original)) on the UIT-VSFC head (H10b's anchored_orig).

    The two losses are backpropagated one after the other within a step, as in H10b. The epoch is
    selected by the mean of UIT-VSFC and in-scope NEU-ESC validation macro-F1 of the UIT-VSFC head.
    """
    import math
    import time
    from dataclasses import asdict

    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS
    from vifeedback.evaluation import metrics as M
    from vifeedback.training import domain as D
    from vifeedback.training.seeding import describe_determinism, seed_everything, worker_init_fn
    from vifeedback.training.trainer import build_optimizer, linear_warmup_schedule

    if cfg.grad_accum != 1:
        raise ValueError("H12' uses the served recipe's grad_accum = 1")
    k = len(D.LABELS)
    seed_everything(cfg.seed)
    device = cfg.resolved_device()
    tok = AutoTokenizer.from_pretrained(
        MODEL_IDS[cfg.model_key], revision=MODEL_REVISIONS.get(cfg.model_key)
    )
    model = D._two_head_model(cfg.model_key).to(device)
    kw: dict[str, Any] = {"truncation": True, "max_length": cfg.max_length, "padding": False}

    class _Mixed(Dataset):
        def __init__(self, parts: list[tuple[list[str], np.ndarray, int]]) -> None:
            self.enc = tok([t for x, _, _ in parts for t in x], **kw)
            self.y = np.concatenate([np.asarray(y) for _, y, _ in parts])
            self.d = np.concatenate([np.full(len(x), d) for x, _, d in parts])

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i: int) -> dict[str, Any]:
            item = {key: v[i] for key, v in self.enc.items()}
            item["labels"] = int(self.y[i])
            item["domain"] = int(self.d[i])
            return item

    q_all = np.asarray(pairs["q"])

    class _Pairs(Dataset):
        def __init__(self, orig: list[str]) -> None:
            self.o = tok(list(orig), **kw)

        def __len__(self) -> int:
            return len(self.o["input_ids"])

        def __getitem__(self, i: int) -> tuple[dict[str, Any], int]:
            return {key: v[i] for key, v in self.o.items()}, i

    pad = DataCollatorWithPadding(tok, padding="longest", return_tensors="pt")

    def collate(items):
        y = torch.tensor([it.pop("labels") for it in items])
        d = torch.tensor([it.pop("domain") for it in items])
        return pad(items), y, d

    def collate_pairs(items):
        return pad([a for a, _ in items]), torch.as_tensor(q_all[[i for _, i in items]])

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    g_pairs = torch.Generator()
    g_pairs.manual_seed(cfg.seed + 1)
    train_loader = DataLoader(
        _Mixed([(uit_train[0], uit_train[1], 0), (neu_train[0], neu_train[1], 1)]),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=cfg.num_workers,
    )
    pair_loader = DataLoader(
        _Pairs(list(pairs["orig"])),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_pairs,
        generator=g_pairs,
    )
    dev_loaders = {
        name: DataLoader(
            _Mixed([(x, y, 0)]), batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate
        )
        for name, (x, y) in (("uit", uit_dev), ("neu", neu_dev))
    }

    opt = build_optimizer(model, cfg)
    total = math.ceil(len(train_loader) / cfg.grad_accum) * cfg.epochs
    sched = linear_warmup_schedule(opt, total, cfg.warmup_ratio)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device != "cpu")
    ce = nn.CrossEntropyLoss(reduction="none")
    ac: dict[str, Any] = dict(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=cfg.fp16 and device != "cpu"
    )

    def pair_batches():
        while True:  # fewer pair batches than training batches: reshuffle and go round again
            yield from pair_loader

    pair_iter = pair_batches()

    @torch.no_grad()
    def dev_logits(loader) -> dict[str, np.ndarray]:
        model.eval()
        out: dict[str, list[np.ndarray]] = {"uit": [], "neu": []}
        for batch, _, _ in loader:
            batch = {key: v.to(device) for key, v in batch.items()}
            with torch.autocast(**ac):
                logits = model(batch["input_ids"], batch["attention_mask"])
            for h in out:
                out[h].append(logits[h].float().cpu().numpy())
        return {h: np.concatenate(v) for h, v in out.items()}

    best: dict[str, Any] = {"score": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_ce = run_cons = 0.0
        for batch, y, d in train_loader:
            opt.zero_grad(set_to_none=True)
            batch = {key: v.to(device, non_blocking=True) for key, v in batch.items()}
            y, d = y.to(device), d.to(device)
            with torch.autocast(**ac):
                logits = model(batch["input_ids"], batch["attention_mask"])
                # each example's loss from its own dataset's head (H8)
                per = torch.where(d == 0, ce(logits["uit"], y), ce(logits["neu"], y))
                loss_ce = per.mean()
            scaler.scale(loss_ce).backward()

            orig, q = next(pair_iter)
            orig = {key: v.to(device, non_blocking=True) for key, v in orig.items()}
            q = q.to(device, non_blocking=True).float()
            with torch.autocast(**ac):
                logits_o = model(orig["input_ids"], orig["attention_mask"])["uit"]
            loss_cons = WEIGHT * F.kl_div(
                F.log_softmax(logits_o.float(), dim=-1), q, reduction="batchmean"
            )
            scaler.scale(loss_cons).backward()

            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_ce += loss_ce.item()
            run_cons += loss_cons.item()

        uit_l = dev_logits(dev_loaders["uit"])
        neu_l = dev_logits(dev_loaders["neu"])
        f1 = {
            "dev_uit_head_uit": M.macro_f1(uit_dev[1], uit_l["uit"].argmax(1), k),
            "dev_neu_head_uit": M.macro_f1(neu_dev[1], neu_l["uit"].argmax(1), k),
            "dev_neu_head_neu": M.macro_f1(neu_dev[1], neu_l["neu"].argmax(1), k),
        }
        score = (f1["dev_uit_head_uit"] + f1["dev_neu_head_uit"]) / 2  # the served head
        history.append(
            {
                "epoch": epoch,
                "train_ce": run_ce / len(train_loader),
                "train_consistency": run_cons / len(train_loader),
                **f1,
                "selection_score": score,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        if verbose:
            h = history[-1]
            print(
                f"  epoch {epoch}/{cfg.epochs}  CE {h['train_ce']:.4f}  consistency "
                f"{h['train_consistency']:.4f}  UIT dev {f1['dev_uit_head_uit']:.4f}  NEU dev "
                f"(UIT head) {f1['dev_neu_head_uit']:.4f} (NEU head) {f1['dev_neu_head_neu']:.4f}"
                f"  ({h['seconds']:.0f}s)"
            )
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {key: v.detach().cpu().clone() for key, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= cfg.early_stopping_patience:
            break

    model.load_state_dict(best["state"])
    model = model.cpu()
    return {
        "served": D.as_sequence_classifier(model, "uit", cfg.model_key),
        "neu_head": D.as_sequence_classifier(model, "neu", cfg.model_key),
        "tokenizer": tok,
        "config": asdict(cfg),
        "history": history,
        "best_epoch": best["epoch"],
        "selection": "mean(UIT-VSFC dev, NEU-ESC dev) macro-F1 of the UIT-VSFC head",
        "train_seconds": round(time.perf_counter() - t0, 1),
        "determinism": describe_determinism(),
        "device": device,
    }


# --- orchestration --------------------------------------------------------------------------------


def _write(name: str, summary: dict[str, Any], scored: dict[str, Any], dev: dict[str, Any]) -> Path:
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.training.domain import LABELS

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    body = {
        **summary,
        "sets": {
            s: {k: v for k, v in r.items() if k not in ("pred", "prob")} for s, r in scored.items()
        },
        "vilexnorm_dev": {k: v for k, v in dev.items() if not k.startswith("pred_")},
    }
    (d / "summary.json").write_text(
        json.dumps(yaml_safe(body), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for s, r in scored.items():  # labels and probabilities only; no text
        table = pd.DataFrame(r["prob"].round(5), columns=[f"p_{c}" for c in LABELS])
        table.insert(0, "pred", [LABELS[i] for i in r["pred"]])
        table.to_csv(d / f"predictions_{s}.csv", index_label="row")
    return d


def run(seed: int, verbose: bool = True, smoke: bool = False) -> dict[str, Any]:
    """Train H12' at one seed; score it on the development sets (never the held-out posts)."""
    import dataclasses

    import torch

    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import report as R
    from vifeedback.training import anchored as A
    from vifeedback.training import consistency as C
    from vifeedback.training import domain as D
    from vifeedback.training.runner import _augment_train, _splits

    cfg = config(seed)
    transform = D.serving_transform()
    (x_tr, y_tr), uit_dv, _ = _splits("sentiment", "seg_pyvi")
    x_tr, augmentation = _augment_train(cfg, x_tr)
    neu = neu_train_and_holdout(transform)["train"]
    neu_x = D.augment_neu(neu["raw"], neu["x"], cfg.augment, cfg.augment_p, seed, transform)
    neu_dv = D.neu_esc("validation", transform)
    tr, dv = A.pairs("train", transform), A.pairs("dev", transform)
    sets = C.evaluation_sets(transform)
    if smoke:
        cfg = dataclasses.replace(cfg, epochs=1)
        x_tr, y_tr = list(x_tr)[:256], np.asarray(y_tr)[:256]
        neu_x, neu = neu_x[:256], {**neu, "y": neu["y"][:256]}
        tr = {k: v[:256] for k, v in tr.items()}
        dv = {k: v[:128] for k, v in dv.items()}
        uit_dv = (list(uit_dv[0])[:128], np.asarray(uit_dv[1])[:128])
        neu_dv = {**neu_dv, "x": neu_dv["x"][:128], "y": neu_dv["y"][:128]}
        sets = {k: (list(x)[:128], np.asarray(y)[:128]) for k, (x, y) in sets.items()}
    tr["q"] = A.teacher_q(tr["norm"])
    q_dev = A.teacher_q(dv["norm"])
    if verbose:
        print(
            f"[{cfg.run_id()}] H12' seed {seed}: UIT-VSFC train {len(x_tr)}, NEU-ESC train "
            f"{len(neu_x)} (held-out excluded), pairs {len(tr['orig'])}"
        )
    out = train_two_heads_anchored(
        cfg,
        (list(x_tr), np.asarray(y_tr)),
        (neu_x, neu["y"]),
        tr,
        uit_dv,
        (neu_dv["x"], neu_dv["y"]),
        verbose,
    )
    model, tok = out["served"], out["tokenizer"]
    scored = D.score_sets(model, tok, sets)
    dev = A.score_pairs(model, tok, dv, q_dev)
    summary = {"recipe": RECIPE, "seed": seed, "history": out["history"]}
    summary["best_epoch"] = out["best_epoch"]
    if smoke:
        return {"summary": {**summary, "smoke": True}, "scored": scored, "dev": dev}

    ckpt = paths.MODELS / cfg.run_id("ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    torch.save(out["neu_head"].classifier.state_dict(), ckpt / "neu_head.pt")  # reported only
    uv = scored["uit_validation"]
    metrics = M.evaluate(
        np.asarray(sets["uit_validation"][1]), uv["pred"], "sentiment", y_prob=uv["prob"]
    )
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
            "reason": "cycle5.yaml v6 H12'",
            "determinism": out["determinism"],
            "device": out["device"],
            "best_epoch": out["best_epoch"],
            "selection": out["selection"],
            "neu_esc_train_posts": len(neu_x),
        },
        y_true=np.asarray(sets["uit_validation"][1]),
        y_pred=uv["pred"],
        y_prob=uv["prob"],
    )
    summary.update(
        {
            "run_id": cfg.run_id("validation"),
            "checkpoint": str(ckpt.relative_to(paths.ROOT)),
            "train_seconds": out["train_seconds"],
            "augmentation": augmentation,
            "neu_esc_train_posts": len(neu_x),
        }
    )
    _write(f"{RECIPE}-s{seed}", summary, scored, dev)
    return {"summary": summary, "scored": scored, "dev": dev}


def evaluate_control(seed: int) -> dict[str, Any]:
    """The control (H10b anchored_orig) at one seed, scored with the candidates' code."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.training import anchored as A
    from vifeedback.training import consistency as C
    from vifeedback.training import domain as D

    ck = control_checkpoint(seed)
    transform = D.serving_transform()
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForSequenceClassification.from_pretrained(ck)
    dv = A.pairs("dev", transform)
    scored = D.score_sets(model, tok, C.evaluation_sets(transform))
    dev = A.score_pairs(model, tok, dv, A.teacher_q(dv["norm"]))
    summary = {"recipe": "control", "seed": seed, "checkpoint": str(ck.relative_to(paths.ROOT))}
    _write(f"control-s{seed}", summary, scored, dev)
    return {"summary": summary, "scored": scored, "dev": dev}


def _summary(name: str) -> dict[str, Any]:
    f = OUT / name / "summary.json"
    if not f.exists():
        raise FileNotFoundError(f"{name} has not run: {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def eligibility() -> dict[str, Any]:
    """cycle5.yaml v6 H12' eligibility at seed 42."""
    from vifeedback.evaluation.report import yaml_safe

    c, r = _summary("control-s42")["sets"], _summary(f"{RECIPE}-s42")["sets"]
    uit, c_uit = r["uit_validation"]["macro_f1"], c["uit_validation"]["macro_f1"]
    neu, c_neu = r["neu_validation"]["macro_f1"], c["neu_validation"]["macro_f1"]
    eligible = uit >= c_uit - 0.01 and neu > c_neu
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v6 H12' eligibility",
        "control_s42": {"uit_validation": c_uit, "neu_validation": c_neu},
        RECIPE: {"uit_validation": uit, "neu_validation": neu},
        "uit_minus_control": uit - c_uit,
        "neu_minus_control": neu - c_neu,
        "eligible": eligible,
        "outcome": "confirm at the other four seeds" if eligible else "not supported: stops",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eligibility.json").write_text(json.dumps(yaml_safe(out), indent=2), encoding="utf-8")
    return out


def apply_rule(
    held: dict[str, Any], diffs: dict[str, float], agree_diff: float, tv: float
) -> dict[str, Any]:
    """cycle5.yaml v6 H12' rule (1) to (5)."""
    limits = {
        "2_uit_validation_macro_f1": -0.005,
        "3_uit_validation_neutral_f1": -0.02,
        "4_uit_validation_stripped_macro_f1": -0.01,
    }
    rules: dict[str, dict[str, Any]] = {
        "1_holdout_macro_f1_up": {**held, "passed": held["ci_low"] > 0}
    }
    for name, limit in limits.items():
        rules[name] = {"mean_diff": diffs[name], "limit": limit, "passed": diffs[name] >= limit}
    rules["5_vilexnorm_confirm_pairs"] = {
        "agreement_mean_diff": agree_diff,
        "agreement_limit": -0.02,
        "label_tv_mean": tv,
        "label_tv_limit": 0.10,
        "passed": agree_diff >= -0.02 and tv <= 0.10,
    }
    return {"rules": rules, "passed": all(r["passed"] for r in rules.values())}


def confirm() -> dict[str, Any]:
    """cycle5.yaml v6 H12' rule, five seeds each side; the only function that reads the held-out
    posts' labels."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.serving import pipeline as SP
    from vifeedback.training import anchored as A
    from vifeedback.training import domain as D

    if not (OUT / "eligibility.json").exists():
        raise FileNotFoundError("run h12p-eligibility after seed 42 first")
    elig = json.loads((OUT / "eligibility.json").read_text(encoding="utf-8"))
    if not elig["eligible"]:
        raise RuntimeError("H12' was not eligible at seed 42; the declared rule stops it")
    seeds = list(SEEDS)
    cand = {s: paths.MODELS / config(s).run_id("ckpt") for s in seeds}
    ctrl = {s: control_checkpoint(s) for s in seeds}
    for s in seeds:
        _summary(f"{RECIPE}-s{s}")
        _summary(f"control-s{s}")
    transform = D.serving_transform()
    held = neu_train_and_holdout(transform)["holdout"]
    cf = A.pairs("confirm", transform)
    teacher = A.teacher_q(cf["norm"]).argmax(1)
    k = len(D.LABELS)

    def preds(ck: Path) -> dict[str, np.ndarray]:
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForSequenceClassification.from_pretrained(ck)
        return {
            "held": D.predict_proba(model, tok, held["x"]).argmax(1),
            "orig": D.predict_proba(model, tok, cf["orig"]).argmax(1),
            "norm": D.predict_proba(model, tok, cf["norm"]).argmax(1),
        }

    pc = {s: preds(cand[s]) for s in seeds}
    pb = {s: preds(ctrl[s]) for s in seeds}
    rule1 = D.seed_paired_bootstrap(
        held["y"], [pc[s]["held"] for s in seeds], [pb[s]["held"] for s in seeds], 10_000
    )

    def val(name: str, set_name: str, key: str) -> float:
        r = _summary(name)["sets"][set_name]
        return r["macro_f1"] if key == "macro_f1" else r["per_class"]["neutral"]["f1"]

    def mean_diff(set_name: str, key: str = "macro_f1") -> float:
        return float(
            np.mean(
                [
                    val(f"{RECIPE}-s{s}", set_name, key) - val(f"control-s{s}", set_name, key)
                    for s in seeds
                ]
            )
        )

    agree_c = [float(np.mean(pc[s]["orig"] == teacher)) for s in seeds]
    agree_b = [float(np.mean(pb[s]["orig"] == teacher)) for s in seeds]
    tv = float(np.mean([A.label_tv(pc[s]["orig"], teacher) for s in seeds]))
    decision = apply_rule(
        rule1,
        {
            "2_uit_validation_macro_f1": mean_diff("uit_validation"),
            "3_uit_validation_neutral_f1": mean_diff("uit_validation", "neutral_f1"),
            "4_uit_validation_stripped_macro_f1": mean_diff("uit_validation_stripped"),
        },
        float(np.mean(agree_c) - np.mean(agree_b)),
        tv,
    )

    # Reported, not tested.
    def per_class(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
        ev = M.evaluate(y, p, "sentiment")
        return {c: ev["per_class"][c]["f1"] for c in D.LABELS}

    topics = sorted(set(held["topic"]))
    neu_head = {}
    import torch

    tok42 = AutoTokenizer.from_pretrained(cand[42])
    m42 = AutoModelForSequenceClassification.from_pretrained(cand[42])
    m42.classifier.load_state_dict(torch.load(cand[42] / "neu_head.pt"))
    p_neu = D.predict_proba(m42, tok42, held["x"]).argmax(1)
    neu_head = {
        "macro_f1": M.macro_f1(held["y"], p_neu, k),
        "per_class": per_class(held["y"], p_neu),
    }
    manifest = json.loads(
        (paths.MODELS / "serve" / "sentiment" / "manifest.json").read_text(encoding="utf-8")
    )
    scope = SP.load_scope(paths.MODELS / "serve" / "sentiment", manifest["scope"])
    reported = {
        "per_seed_holdout_macro_f1": {
            s: {
                "candidate": M.macro_f1(held["y"], pc[s]["held"], k),
                "control": M.macro_f1(held["y"], pb[s]["held"], k),
            }
            for s in seeds
        },
        "holdout_per_class_seed42": {
            "candidate": per_class(held["y"], pc[42]["held"]),
            "control": per_class(held["y"], pb[42]["held"]),
        },
        "holdout_per_topic_seed42": {
            t: {
                "n": int((held["topic"] == t).sum()),
                "candidate": M.macro_f1(
                    held["y"][held["topic"] == t], pc[42]["held"][held["topic"] == t], k
                ),
                "control": M.macro_f1(
                    held["y"][held["topic"] == t], pb[42]["held"][held["topic"] == t], k
                ),
            }
            for t in topics
        },
        "neu_head_seed42_on_holdout": neu_head,
        "neu_validation_macro_f1_mean_diff": mean_diff("neu_validation"),
        "neu_validation_all_macro_f1_mean_diff": mean_diff("neu_validation_all"),
        "confirm_pairs": {
            "agreement_candidate": agree_c,
            "agreement_control": agree_b,
            "flip_rate_candidate": [float(np.mean(pc[s]["orig"] != pc[s]["norm"])) for s in seeds],
            "flip_rate_control": [float(np.mean(pb[s]["orig"] != pb[s]["norm"])) for s in seeds],
        },
        "scope_flagged_holdout": float((scope.decision(held["x"]) < scope.threshold).mean()),
    }
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v6 H12' rule",
        "recipe": RECIPE,
        "seeds": seeds,
        **decision,
        "reported": reported,
        "holdout_posts": len(held["y"]),
    }
    d = OUT / "confirm"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    cols: dict[str, list[str]] = {}
    for s in seeds:  # labels only
        cols[f"candidate-s{s}"] = [D.LABELS[i] for i in pc[s]["held"]]
        cols[f"control-s{s}"] = [D.LABELS[i] for i in pb[s]["held"]]
    pd.DataFrame(cols).to_csv(d / "predictions_holdout.csv", index_label="holdout_position")
    return out
