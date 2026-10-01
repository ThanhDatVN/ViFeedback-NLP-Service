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
TEACHER_Q = paths.RESULTS / "studies" / "cycle5" / "h12p_teacher_q.csv"
RESTORER_RECORD = paths.RESULTS / "studies" / "export" / "laptop_fp16_h10b_v17500_restorer.json"
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


def neu_train_and_holdout(
    transform, parts: tuple[str, ...] = ("train", "holdout")
) -> dict[str, dict[str, Any]]:
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
        if name not in parts:
            continue
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


# --- inputs that must be identical on every machine --------------------------------------------


def pair_rows(split: str) -> np.ndarray:
    """ViLexNorm train row indices of `anchored.pairs(split)`, in the order it returns them."""
    from vifeedback.training import anchored as A
    from vifeedback.training import consistency as C

    dev, train_pairs = C.split_indices()
    confirm, train = A.confirm_split(len(train_pairs))
    # A machine whose numpy draws other permutations would silently train on other pairs.
    for rows, f in ((dev, C.DEV_INDEX), (train_pairs[confirm], A.CONFIRM_INDEX)):
        if set(rows.tolist()) != set(pd.read_csv(f)["row"].tolist()):
            raise RuntimeError(f"this machine's ViLexNorm split differs from {f.name}")
    if split == "dev":
        return dev
    return train_pairs[confirm if split == "confirm" else train]


def write_teacher_q() -> Path:
    """The frozen teacher's probabilities on every ViLexNorm train normalized form, by row.

    Read from the H10b cache (models/distill, keyed by the SHA-1 of the transformed text) so that a
    machine without the teacher's checkpoints (Kaggle) trains on exactly the same targets. Indices
    and probabilities only, no text.
    """
    from vifeedback.evaluation import external as X
    from vifeedback.training import domain as D
    from vifeedback.training.distill import SoftLabelCache

    cache = SoftLabelCache(paths.MODELS / "distill" / "soft_labels_served_T1.npz")
    norm = D.serving_transform()(X.load_vilexnorm("train").normalized.astype(str).tolist())
    rows = np.sort(np.concatenate([pair_rows(s) for s in ("train", "confirm", "dev")]))
    q = cache.get([norm[i] for i in rows])
    table = pd.DataFrame(q, columns=[f"p_{c}" for c in D.LABELS])
    table.insert(0, "row", rows)
    TEACHER_Q.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TEACHER_Q, index=False, float_format="%.9g")
    return TEACHER_Q


def teacher_q_rows(split: str) -> np.ndarray:
    """The teacher's probabilities for `anchored.pairs(split)`, from the committed table."""
    table = pd.read_csv(TEACHER_Q).set_index("row")
    return table.loc[pair_rows(split)].to_numpy(dtype=np.float32)


def prepare_serving_assets() -> Path:
    """On a machine without the release (Kaggle): the restorer rebuilt from UIT-VSFC train and a
    manifest naming it, refused unless its SHA-256 is the released one. The serving transform reads
    nothing else."""
    from vifeedback.data.loader import load
    from vifeedback.preprocess.diacritics import Restorer
    from vifeedback.serving.pipeline import _checked

    d = paths.MODELS / "serve" / "sentiment"
    if (d / "manifest.json").exists():
        return d
    spec = json.loads(RESTORER_RECORD.read_text(encoding="utf-8"))["restorer"]
    d.mkdir(parents=True, exist_ok=True)
    Restorer.fit(load("train").sentence.tolist()).save(d / spec["file"])
    _checked(d / spec["file"], spec["sha256"])  # raises unless byte-identical to the release
    manifest = {"preprocessing": "seg_pyvi", "restorer": spec, "built_for": "H12' on Kaggle"}
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return d


# --- predictions on the confirmation data (labels only) -----------------------------------------


def write_predictions(name: str, model, tok, neu_head=None) -> Path:
    """One model's labels on the held-out posts and the confirmation pairs, for `confirm`.

    No label of the held-out posts is read here; `confirm` is the only reader.
    """
    from vifeedback.training import anchored as A
    from vifeedback.training import domain as D

    transform = D.serving_transform()
    held = neu_train_and_holdout(transform, ("holdout",))["holdout"]
    cf = A.pairs("confirm", transform)
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    out = {"pred": D.predict_proba(model, tok, held["x"]).argmax(1)}
    if neu_head is not None:
        out["pred_neu_head"] = D.predict_proba(neu_head, tok, held["x"]).argmax(1)
    pd.DataFrame({k: [D.LABELS[i] for i in v] for k, v in out.items()}).to_csv(
        d / "predictions_holdout.csv", index_label="holdout_position"
    )
    pd.DataFrame(
        {
            "orig": [D.LABELS[i] for i in D.predict_proba(model, tok, cf["orig"]).argmax(1)],
            "norm": [D.LABELS[i] for i in D.predict_proba(model, tok, cf["norm"]).argmax(1)],
        }
    ).to_csv(d / "predictions_confirm.csv", index_label="confirm_position")
    return d


def _neu_head_model(ck: Path):
    """The candidate's NEU-ESC head on its encoder (reported only)."""
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(ck)
    model.classifier.load_state_dict(torch.load(ck / "neu_head.pt", map_location="cpu"))
    return model.eval()


def predict_candidate(seed: int) -> Path:
    """Write a trained candidate's confirmation predictions (e.g. seed 42, trained before
    `run` wrote them)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ck = paths.MODELS / config(seed).run_id("ckpt")
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForSequenceClassification.from_pretrained(ck)
    return write_predictions(f"{RECIPE}-s{seed}", model, tok, _neu_head_model(ck))


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
    """Train H12' at one seed, score it on the development sets, and write its labels on the
    confirmation data (labels only; no held-out label is read)."""
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
    neu = neu_train_and_holdout(transform, ("train",))["train"]
    neu_x = D.augment_neu(neu["raw"], neu["x"], cfg.augment, cfg.augment_p, seed, transform)
    neu_dv = D.neu_esc("validation", transform)
    tr, dv = A.pairs("train", transform), A.pairs("dev", transform)
    tr["q"] = teacher_q_rows("train")
    q_dev = teacher_q_rows("dev")
    sets = C.evaluation_sets(transform)
    if smoke:
        cfg = dataclasses.replace(cfg, epochs=1)
        x_tr, y_tr = list(x_tr)[:256], np.asarray(y_tr)[:256]
        neu_x, neu = neu_x[:256], {**neu, "y": neu["y"][:256]}
        tr = {k: v[:256] for k, v in tr.items()}
        dv, q_dev = {k: v[:128] for k, v in dv.items()}, q_dev[:128]
        uit_dv = (list(uit_dv[0])[:128], np.asarray(uit_dv[1])[:128])
        neu_dv = {**neu_dv, "x": neu_dv["x"][:128], "y": neu_dv["y"][:128]}
        sets = {k: (list(x)[:128], np.asarray(y)[:128]) for k, (x, y) in sets.items()}
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
    write_predictions(f"{RECIPE}-s{seed}", model, tok, out["neu_head"])
    return {"summary": summary, "scored": scored, "dev": dev}


def evaluate_control(seed: int) -> dict[str, Any]:
    """The control (H10b anchored_orig) at one seed, scored with the candidates' code, and its
    labels on the confirmation data."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.training import anchored as A
    from vifeedback.training import consistency as C
    from vifeedback.training import domain as D

    ck = control_checkpoint(seed)
    tok = AutoTokenizer.from_pretrained(ck)
    model = AutoModelForSequenceClassification.from_pretrained(ck)
    name = f"control-s{seed}"
    if (OUT / name / "summary.json").exists():
        # Recorded already (perhaps on another machine): add only the confirmation labels, so a
        # record is never rewritten with another GPU's rounding.
        write_predictions(name, model, tok)
        return {"summary": _summary(name), "scored": None, "dev": None}
    transform = D.serving_transform()
    dv = A.pairs("dev", transform)
    scored = D.score_sets(model, tok, C.evaluation_sets(transform))
    dev = A.score_pairs(model, tok, dv, teacher_q_rows("dev"))
    summary = {"recipe": "control", "seed": seed, "checkpoint": ck.name}
    _write(f"control-s{seed}", summary, scored, dev)
    write_predictions(f"control-s{seed}", model, tok)
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


def missing_for_confirm() -> list[str]:
    """What `confirm` still needs: summaries and prediction files of ten models."""
    out = []
    for s in SEEDS:
        for name in (f"{RECIPE}-s{s}", f"control-s{s}"):
            for f in ("summary.json", "predictions_holdout.csv", "predictions_confirm.csv"):
                if not (OUT / name / f).exists():
                    out.append(f"{name}/{f}")
    return out


def confirm() -> dict[str, Any]:
    """cycle5.yaml v6 H12' rule, five seeds each side, from the prediction files alone (no model
    is loaded). The only function that reads the held-out posts' labels."""
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.training import anchored as A
    from vifeedback.training import domain as D

    if not (OUT / "eligibility.json").exists():
        raise FileNotFoundError("run h12p-eligibility after seed 42 first")
    if not json.loads((OUT / "eligibility.json").read_text(encoding="utf-8"))["eligible"]:
        raise RuntimeError("H12' was not eligible at seed 42; the declared rule stops it")
    missing = missing_for_confirm()
    if missing:
        raise FileNotFoundError(f"missing before confirmation: {missing}")
    seeds = list(SEEDS)
    ids = {c: i for i, c in enumerate(D.LABELS)}
    frame = X.load_neu_esc("train")
    holdout, _ = holdout_split(frame, D.OFF_TOPIC)
    index = pd.read_csv(INDEX)
    if not np.array_equal(index["row"].to_numpy(), holdout):
        raise RuntimeError(f"the held-out split differs from {INDEX}")
    y = index["sentiment"].map(ids).to_numpy()
    topic = frame.topic.to_numpy()[holdout]
    teacher = teacher_q_rows("confirm").argmax(1)
    k = len(D.LABELS)

    def labels(name: str, file: str, col: str) -> np.ndarray:
        return pd.read_csv(OUT / name / file)[col].map(ids).to_numpy()

    pc = {s: labels(f"{RECIPE}-s{s}", "predictions_holdout.csv", "pred") for s in seeds}
    pb = {s: labels(f"control-s{s}", "predictions_holdout.csv", "pred") for s in seeds}
    co = {s: labels(f"{RECIPE}-s{s}", "predictions_confirm.csv", "orig") for s in seeds}
    cn = {s: labels(f"{RECIPE}-s{s}", "predictions_confirm.csv", "norm") for s in seeds}
    bo = {s: labels(f"control-s{s}", "predictions_confirm.csv", "orig") for s in seeds}
    bn = {s: labels(f"control-s{s}", "predictions_confirm.csv", "norm") for s in seeds}
    if any(len(v) != len(y) for v in (*pc.values(), *pb.values())):
        raise ValueError("a prediction file does not cover the 3,000 held-out posts")
    rule1 = D.seed_paired_bootstrap(y, [pc[s] for s in seeds], [pb[s] for s in seeds], 10_000)

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

    agree_c = [float(np.mean(co[s] == teacher)) for s in seeds]
    agree_b = [float(np.mean(bo[s] == teacher)) for s in seeds]
    tv = float(np.mean([A.label_tv(co[s], teacher) for s in seeds]))
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
    def per_class(p: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
        m = np.ones(len(y), bool) if mask is None else mask
        ev = M.evaluate(y[m], p[m], "sentiment")
        return {c: ev["per_class"][c]["f1"] for c in D.LABELS}

    head42 = pd.read_csv(OUT / f"{RECIPE}-s42" / "predictions_holdout.csv")
    neu_head = None
    if "pred_neu_head" in head42:
        p_neu = head42["pred_neu_head"].map(ids).to_numpy()
        neu_head = {"macro_f1": M.macro_f1(y, p_neu, k), "per_class": per_class(p_neu)}
    scope_flagged = None
    serve = paths.MODELS / "serve" / "sentiment"
    mf = serve / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
    if manifest.get("scope"):  # the released detector; absent on a machine that only trained
        from vifeedback.serving import pipeline as SP

        scope = SP.load_scope(serve, manifest["scope"])
        x = D.serving_transform()(frame.text.to_numpy()[holdout].tolist())
        scope_flagged = float((scope.decision(x) < scope.threshold).mean())
    reported = {
        "per_seed_holdout_macro_f1": {
            s: {"candidate": M.macro_f1(y, pc[s], k), "control": M.macro_f1(y, pb[s], k)}
            for s in seeds
        },
        "holdout_per_class_seed42": {"candidate": per_class(pc[42]), "control": per_class(pb[42])},
        "holdout_per_topic_seed42": {
            t: {
                "n": int((topic == t).sum()),
                "candidate": M.macro_f1(y[topic == t], pc[42][topic == t], k),
                "control": M.macro_f1(y[topic == t], pb[42][topic == t], k),
            }
            for t in sorted(set(topic))
        },
        "neu_head_seed42_on_holdout": neu_head,
        "neu_validation_macro_f1_mean_diff": mean_diff("neu_validation"),
        "neu_validation_all_macro_f1_mean_diff": mean_diff("neu_validation_all"),
        "confirm_pairs": {
            "agreement_candidate": agree_c,
            "agreement_control": agree_b,
            "flip_rate_candidate": [float(np.mean(co[s] != cn[s])) for s in seeds],
            "flip_rate_control": [float(np.mean(bo[s] != bn[s])) for s in seeds],
        },
        "scope_flagged_holdout": scope_flagged,
    }
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v6 H12' rule",
        "recipe": RECIPE,
        "seeds": seeds,
        **decision,
        "reported": reported,
        "holdout_posts": len(y),
    }
    d = OUT / "confirm"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def import_results(zip_path: Path) -> list[str]:
    """Merge a Kaggle results zip (made by notebooks/kaggle_h12p.ipynb) into the repository.

    Only H12' result folders, run folders and registry rows are accepted; a file that would
    replace a different existing file is refused.
    """
    import zipfile

    allowed = ("results/studies/cycle5/h12p/", "results/runs/p16-")
    written = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
        bad = [n for n in names if not n.startswith(allowed) and n != "registry_rows.csv"]
        if bad or any(".." in n or n.startswith("/") for n in names):
            raise ValueError(f"unexpected entries in {zip_path}: {bad[:5]}")
        for n in names:
            if n == "registry_rows.csv":
                continue
            dst = paths.ROOT / n
            data = z.read(n)
            if dst.exists() and dst.read_bytes() != data:
                raise FileExistsError(f"{n} exists with different content")
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            written.append(n)
        if "registry_rows.csv" in names:
            import io

            rows = pd.read_csv(io.BytesIO(z.read("registry_rows.csv")))
            reg = pd.read_csv(paths.REGISTRY)
            new = rows[~rows.run_id.isin(reg.run_id)]
            if len(new):
                new[reg.columns].to_csv(paths.REGISTRY, mode="a", header=False, index=False)
                written.append(f"registry.csv (+{len(new)} rows)")
    return written
