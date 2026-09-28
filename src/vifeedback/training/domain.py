"""Cycle 4, H8: training with student text from another institution (configs/experiments/cycle4.yaml).

Three recipes against the served recipe, all fed NEU-ESC text through the service's own transform
(`serving.pipeline.prepare`: lowercase NFC, the released diacritic restorer, pyvi):

* ``mixed``: UIT-VSFC train plus in-scope NEU-ESC train, one head (`trainer.train`);
* ``two-heads``: one shared encoder, a UIT-VSFC head and a NEU-ESC head, each trained only on its
  own dataset; the UIT-VSFC head is the one served, so the service's labels keep UIT-VSFC's policy;
* ``sequential``: the control checkpoint of the same seed, fine-tuned on in-scope NEU-ESC train.

Every recipe selects its epoch by the mean of UIT-VSFC validation macro-F1 and in-scope NEU-ESC
validation macro-F1, from the head that would be served. Off-topic NEU-ESC posts never enter
training, so the out-of-scope score keeps its meaning.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths
from vifeedback.constants import SEEDS, label_names

OFF_TOPIC = ("Spam", "News", "Jobs & Recruitment", "Club & Events")
RECIPES = ("mixed", "two-heads", "sequential")
PHASE_NUM = 12
OUT = paths.RESULTS / "studies" / "cycle4" / "h8"
TEST_USES = paths.RESULTS / "studies" / "cycle4" / "neu_esc_test_uses.log"
LABELS = label_names("sentiment")


def serving_transform():
    """The served release's text transform: what NEU-ESC text goes through at serving time."""
    import json

    from vifeedback.preprocess.segment import get_segmenter
    from vifeedback.preprocess.variants import VARIANTS
    from vifeedback.serving import pipeline as SP

    d = paths.MODELS / "serve" / "sentiment"
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    restorer = SP.load_restorer(d, manifest["restorer"]) if manifest.get("restorer") else None
    seg = get_segmenter(VARIANTS[manifest["preprocessing"]][0])

    def transform(texts: list[str]) -> list[str]:
        return SP.prepare(texts, restorer, seg)

    return transform


def neu_esc(split: str, transform, in_scope: bool = True) -> dict[str, Any]:
    """NEU-ESC posts, transformed as served, with UIT-VSFC label ids (Toxic -> negative)."""
    from vifeedback.evaluation import external as X

    ne = X.load_neu_esc(split)
    if in_scope:
        ne = ne[~ne.topic.isin(OFF_TOPIC)]
    raw = ne.text.tolist()
    return {
        "raw": raw,
        "x": transform(raw),
        "y": ne.sentiment.map({c: i for i, c in enumerate(LABELS)}).to_numpy().astype(int),
        "topic": ne.topic.to_numpy(),
    }


def augment_neu(
    raw: list[str], x: list[str], recipe: str, p: float, seed: int, transform
) -> list[str]:
    """The served augmentation applied to NEU-ESC raw text, then the serving transform."""
    from vifeedback.training.augment import augment

    aug_raw, changed = augment(raw, recipe, p, seed)
    idx = np.flatnonzero(changed)
    out = list(x)
    for i, t in zip(idx, transform([aug_raw[i] for i in idx]), strict=True):
        out[i] = t
    return out


def uit_stripped_validation(transform) -> tuple[list[str], np.ndarray]:
    """UIT-VSFC validation with every diacritic removed, then the serving transform (restorer)."""
    from vifeedback.data.loader import load
    from vifeedback.preprocess.normalize import strip_diacritics

    dv = load("validation")
    return transform([strip_diacritics(t) for t in dv.sentence.tolist()]), dv.sentiment.to_numpy()


def control_checkpoint(seed: int) -> Path:
    """The served recipe's checkpoint for a seed: p9 at 42, p10 (cycle3.yaml V1) otherwise."""
    hits = sorted(paths.MODELS.glob(f"p*-sent-phobert-base-seg_pyvi-aug-diac-teen-s{seed}-*-ckp"))
    hits = [h for h in hits if "-vln-" not in h.name]
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one control checkpoint for seed {seed}, found {hits}")
    return hits[0]


# --- two heads ------------------------------------------------------------------------------------


def _two_head_model(model_key: str):
    import torch
    from torch import nn
    from transformers import AutoModel

    from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS
    from vifeedback.training.multitask import _Head

    class TwoHeads(nn.Module):
        """Shared encoder; `classifiers.uit` and `classifiers.neu` mirror RobertaClassificationHead
        (the name contains "classifier", so trainer.build_optimizer gives them the head rate)."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                MODEL_IDS[model_key],
                revision=MODEL_REVISIONS.get(model_key),
                add_pooling_layer=False,
            )
            h = self.encoder.config.hidden_size
            drop = self.encoder.config.hidden_dropout_prob
            self.classifiers = nn.ModuleDict(
                {d: _Head(h, len(LABELS), drop) for d in ("uit", "neu")}
            )
            std = self.encoder.config.initializer_range
            for head in self.classifiers.values():  # as transformers initializes a new head
                for lin in head.modules():
                    if isinstance(lin, nn.Linear):
                        nn.init.normal_(lin.weight, mean=0.0, std=std)
                        nn.init.zeros_(lin.bias)

        def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
            cls = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0]
            return {d: head(cls) for d, head in self.classifiers.items()}

    return TwoHeads()


def as_sequence_classifier(two_heads, head: str, model_key: str):
    """A standard RobertaForSequenceClassification with the shared encoder and one head, so export,
    evaluation and serving treat it like every other checkpoint."""
    from transformers import AutoModelForSequenceClassification

    from vifeedback.constants import LABELS as LABEL_MAPS
    from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS

    hf = AutoModelForSequenceClassification.from_pretrained(
        MODEL_IDS[model_key],
        revision=MODEL_REVISIONS.get(model_key),
        num_labels=len(LABELS),
        id2label=dict(LABEL_MAPS["sentiment"]),
        label2id={n: i for i, n in LABEL_MAPS["sentiment"].items()},
    )
    base = getattr(hf, hf.base_model_prefix)
    missing, unexpected = base.load_state_dict(two_heads.encoder.state_dict(), strict=False)
    if unexpected or any(not k.startswith("pooler.") for k in missing):
        raise RuntimeError(f"encoder mismatch: missing {missing}, unexpected {unexpected}")
    src = two_heads.classifiers[head]
    hf.classifier.dense.load_state_dict(src.dense.state_dict())
    hf.classifier.out_proj.load_state_dict(src.out_proj.state_dict())
    return hf.eval()


def train_two_heads(
    cfg,
    uit_train: tuple[list[str], np.ndarray],
    neu_train: tuple[list[str], np.ndarray],
    uit_dev: tuple[list[str], np.ndarray],
    neu_dev: tuple[list[str], np.ndarray],
    verbose: bool = True,
) -> dict[str, Any]:
    """Masked two-head fine-tuning, optimizer, schedule, AMP and clipping as `trainer.train`."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS
    from vifeedback.evaluation import metrics as M
    from vifeedback.training.seeding import describe_determinism, seed_everything, worker_init_fn
    from vifeedback.training.trainer import build_optimizer, linear_warmup_schedule

    seed_everything(cfg.seed)
    device = cfg.resolved_device()
    tok = AutoTokenizer.from_pretrained(
        MODEL_IDS[cfg.model_key], revision=MODEL_REVISIONS.get(cfg.model_key)
    )
    model = _two_head_model(cfg.model_key).to(device)

    class _Mixed(Dataset):
        def __init__(self, parts: list[tuple[list[str], np.ndarray, int]]) -> None:
            texts = [t for x, _, _ in parts for t in x]
            self.enc = tok(texts, truncation=True, max_length=cfg.max_length, padding=False)
            self.y = np.concatenate([np.asarray(y) for _, y, _ in parts])
            self.d = np.concatenate([np.full(len(x), d) for x, _, d in parts])

        def __len__(self) -> int:
            return len(self.y)

        def __getitem__(self, i: int) -> dict[str, Any]:
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = int(self.y[i])
            item["domain"] = int(self.d[i])
            return item

    pad = DataCollatorWithPadding(tok, padding="longest", return_tensors="pt")

    def collate(items):
        y = torch.tensor([it.pop("labels") for it in items])
        d = torch.tensor([it.pop("domain") for it in items])
        return pad(items), y, d

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        _Mixed([(uit_train[0], uit_train[1], 0), (neu_train[0], neu_train[1], 1)]),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=cfg.num_workers,
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

    @torch.no_grad()
    def dev_logits(loader) -> dict[str, np.ndarray]:
        model.eval()
        out: dict[str, list[np.ndarray]] = {"uit": [], "neu": []}
        for batch, _, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
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
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for step, (batch, y, d) in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            y, d = y.to(device), d.to(device)
            with torch.autocast(**ac):
                logits = model(batch["input_ids"], batch["attention_mask"])
                # Each example's loss comes from its own dataset's head; the batch mean over examples.
                per = torch.where(d == 0, ce(logits["uit"], y), ce(logits["neu"], y))
                loss = per.mean() / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum
            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        uit_l = dev_logits(dev_loaders["uit"])
        neu_l = dev_logits(dev_loaders["neu"])
        f1 = {
            "dev_uit_head_uit": M.macro_f1(uit_dev[1], uit_l["uit"].argmax(1), len(LABELS)),
            "dev_neu_head_uit": M.macro_f1(neu_dev[1], neu_l["uit"].argmax(1), len(LABELS)),
            "dev_neu_head_neu": M.macro_f1(neu_dev[1], neu_l["neu"].argmax(1), len(LABELS)),
        }
        score = (f1["dev_uit_head_uit"] + f1["dev_neu_head_uit"]) / 2  # the served head
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / len(train_loader),
                **f1,
                "selection_score": score,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        if verbose:
            print(
                f"  epoch {epoch}/{cfg.epochs}  loss {history[-1]['train_loss']:.4f}  "
                f"UIT dev {f1['dev_uit_head_uit']:.4f}  NEU dev (UIT head) {f1['dev_neu_head_uit']:.4f}  "
                f"(NEU head) {f1['dev_neu_head_neu']:.4f}  ({history[-1]['seconds']:.0f}s)"
            )
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= cfg.early_stopping_patience:
            break

    model.load_state_dict(best["state"])
    model = model.cpu()
    return {
        "served": as_sequence_classifier(model, "uit", cfg.model_key),
        "neu_head": as_sequence_classifier(model, "neu", cfg.model_key),
        "tokenizer": tok,
        "config": asdict(cfg),
        "history": history,
        "best_epoch": best["epoch"],
        "selection": "mean(UIT-VSFC dev, NEU-ESC dev) macro-F1 of the UIT-VSFC head",
        "train_seconds": round(time.perf_counter() - t0, 1),
        "determinism": describe_determinism(),
        "device": device,
    }


# --- evaluation -----------------------------------------------------------------------------------


def predict_proba(
    model, tokenizer, texts: list[str], max_length: int = 96, batch_size: int = 128
) -> np.ndarray:
    """Softmax probabilities from a sequence classifier, fp16 on a GPU, dynamic padding."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[s : s + batch_size],
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**enc).logits
            out.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def score_sets(model, tokenizer, sets: dict[str, tuple[list[str], np.ndarray]]) -> dict[str, Any]:
    """Metrics and label predictions (no text) for each named evaluation set."""
    from vifeedback.evaluation import metrics as M

    res: dict[str, Any] = {}
    for name, (x, y) in sets.items():
        prob = predict_proba(model, tokenizer, x)
        pred = prob.argmax(1)
        ev = M.evaluate(np.asarray(y), pred, "sentiment", y_prob=prob)
        res[name] = {
            "macro_f1": ev["macro_f1"],
            "accuracy": ev["accuracy"],
            "per_class": ev["per_class"],
            "pred": pred,
            "prob": prob,
        }
    return res


def seed_paired_bootstrap(
    y: np.ndarray,
    cand: list[np.ndarray],
    ctrl: list[np.ndarray],
    n_resamples: int = 5000,
    seed: int = 0,
) -> dict[str, Any]:
    """Seed-averaged paired difference in macro-F1 (candidate minus control), posts resampled once
    per draw and the seed pairs averaged within it (cycle4.yaml H8 rule 1)."""
    from vifeedback.evaluation import metrics as M

    k = len(LABELS)
    y = np.asarray(y)

    def diff(idx: np.ndarray) -> float:
        return float(
            np.mean(
                [
                    M.macro_f1(y[idx], c[idx], k) - M.macro_f1(y[idx], b[idx], k)
                    for c, b in zip(cand, ctrl, strict=True)
                ]
            )
        )

    rng = np.random.default_rng(seed)
    n = len(y)
    observed = diff(np.arange(n))
    draws = np.array([diff(rng.integers(0, n, n)) for _ in range(n_resamples)])
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {
        "observed_diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_one_sided": float((np.sum(draws <= 0) + 1) / (n_resamples + 1)),
        "n_resamples": n_resamples,
        "seeds": len(cand),
    }


def log_test_use(rule: str, models: list[str]) -> None:
    """Every NEU-ESC test evaluation is recorded (cycle4.yaml data.neu_esc_test)."""
    from datetime import UTC, datetime

    TEST_USES.parent.mkdir(parents=True, exist_ok=True)
    with open(TEST_USES, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat(timespec='seconds')}\t{rule}\t{','.join(models)}\n")


# --- orchestration (the CLI commands are thin wrappers) ------------------------------------------


def config(recipe: str, seed: int):
    """The served recipe's TrainConfig, varied only where cycle4.yaml says (sequential: 2 epochs at 1e-5)."""
    from vifeedback.training.trainer import TrainConfig

    kw: dict[str, Any] = {}
    if recipe == "sequential":
        kw = {"epochs": 2, "lr": 1e-5}
    return TrainConfig(
        task="sentiment",
        model_key="phobert-base",
        preprocessing="seg_pyvi",
        recipe=f"h8-{recipe}",
        augment="diac-teen",
        augment_p=0.3,
        seed=seed,
        extra={"phase_num": PHASE_NUM},
        notes=f"cycle4.yaml H8 {recipe}",
        **kw,
    )


def _evaluation_sets(transform, neu_dv: dict[str, Any], uit_dv: tuple[list[str], np.ndarray]):
    return {
        "uit_validation": uit_dv,
        "uit_validation_stripped": uit_stripped_validation(transform),
        "neu_validation": (neu_dv["x"], neu_dv["y"]),
    }


def _write(name: str, summary: dict[str, Any], scored: dict[str, Any]) -> Path:
    import json

    import pandas as pd

    from vifeedback.evaluation.report import yaml_safe

    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    summary = {
        **summary,
        "sets": {
            s: {k: v for k, v in r.items() if k not in ("pred", "prob")} for s, r in scored.items()
        },
    }
    (d / "summary.json").write_text(
        json.dumps(yaml_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for s, r in scored.items():  # labels and probabilities only; no text
        table = pd.DataFrame(r["prob"].round(5), columns=[f"p_{c}" for c in LABELS])
        table.insert(0, "pred", [LABELS[i] for i in r["pred"]])
        table.to_csv(d / f"predictions_{s}.csv", index_label="row")
    return d


def run(recipe: str, seed: int, verbose: bool = True, smoke: bool = False) -> dict[str, Any]:
    """Train one H8 recipe at one seed, save its checkpoint, evaluate it on the selection sets.

    `smoke` checks the wiring in a minute: 256 examples per dataset, one epoch, 128 per evaluation
    set, and nothing is written (no checkpoint, registry row or result folder).
    """
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import report as R
    from vifeedback.training.runner import _augment_train, _splits
    from vifeedback.training.trainer import train

    if recipe not in RECIPES:
        raise ValueError(f"recipe must be one of {RECIPES}")
    cfg = config(recipe, seed)
    transform = serving_transform()
    (x_tr, y_tr), uit_dv, _ = _splits("sentiment", "seg_pyvi")
    x_tr, augmentation = _augment_train(cfg, x_tr)
    neu_tr = neu_esc("train", transform)
    neu_x = augment_neu(neu_tr["raw"], neu_tr["x"], cfg.augment, cfg.augment_p, seed, transform)
    neu_dv = neu_esc("validation", transform)
    if smoke:
        import dataclasses

        cfg = dataclasses.replace(cfg, epochs=1)
        x_tr, y_tr = list(x_tr)[:256], np.asarray(y_tr)[:256]
        neu_x, neu_tr = neu_x[:256], {**neu_tr, "y": neu_tr["y"][:256]}
        uit_dv = (list(uit_dv[0])[:128], np.asarray(uit_dv[1])[:128])
        neu_dv = {**neu_dv, "x": neu_dv["x"][:128], "y": neu_dv["y"][:128]}
    if verbose:
        print(
            f"[{cfg.run_id()}] H8 {recipe} seed {seed}: UIT-VSFC train {len(x_tr)}, "
            f"NEU-ESC train {len(neu_x)} (in scope), NEU-ESC validation {len(neu_dv['y'])}"
        )

    extra_head = None
    if recipe == "mixed":
        out = train(
            cfg,
            list(x_tr) + neu_x,
            np.concatenate([y_tr, neu_tr["y"]]),
            uit_dv[0],
            uit_dv[1],
            verbose=verbose,
            extra_dev=(neu_dv["x"], neu_dv["y"]),
        )
        model, tok = out["model"], out["tokenizer"]
    elif recipe == "sequential":
        out = train(
            cfg,
            neu_x,
            neu_tr["y"],
            uit_dv[0],
            uit_dv[1],
            verbose=verbose,
            extra_dev=(neu_dv["x"], neu_dv["y"]),
            init_from=str(control_checkpoint(seed)),
        )
        model, tok = out["model"], out["tokenizer"]
    else:
        out = train_two_heads(
            cfg,
            (list(x_tr), y_tr),
            (neu_x, neu_tr["y"]),
            uit_dv,
            (neu_dv["x"], neu_dv["y"]),
            verbose=verbose,
        )
        model, tok, extra_head = out["served"], out["tokenizer"], out["neu_head"]

    sets = _evaluation_sets(transform, neu_dv, uit_dv)
    if smoke:
        sets["uit_validation_stripped"] = tuple(v[:128] for v in sets["uit_validation_stripped"])
        scored = score_sets(model, tok, sets)
        return {
            "summary": {"recipe": recipe, "seed": seed, "smoke": True, "history": out["history"]},
            "scored": scored,
        }

    ckpt = paths.MODELS / cfg.run_id("ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    scored = score_sets(model, tok, sets)
    if extra_head is not None:
        scored["neu_validation_neu_head"] = score_sets(
            extra_head, tok, {"x": (neu_dv["x"], neu_dv["y"])}
        )["x"]

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
            "reason": "cycle4.yaml H8",
            "determinism": out["determinism"],
            "device": out["device"],
            "best_epoch": out["best_epoch"],
            "selection": out["selection"],
            "init_from": str(control_checkpoint(seed)) if recipe == "sequential" else None,
        },
        y_true=np.asarray(uit_dv[1]),
        y_pred=uv["pred"],
        y_prob=uv["prob"],
    )
    summary = {
        "recipe": recipe,
        "seed": seed,
        "run_id": cfg.run_id("validation"),
        "checkpoint": str(ckpt.relative_to(paths.ROOT)),
        "history": out["history"],
        "best_epoch": out["best_epoch"],
        "selection": out["selection"],
        "train_seconds": out["train_seconds"],
        "neu_esc_train_posts": len(neu_x),
        "augmentation": augmentation,
    }
    _write(f"{recipe}-s{seed}", summary, scored)
    return {"summary": summary, "scored": scored}


def evaluate_control(seed: int) -> dict[str, Any]:
    """The control checkpoint of one seed on the selection sets, with the candidates' code."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.training.runner import _splits

    ckpt = control_checkpoint(seed)
    transform = serving_transform()
    _, uit_dv, _ = _splits("sentiment", "seg_pyvi")
    neu_dv = neu_esc("validation", transform)
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    scored = score_sets(model, tok, _evaluation_sets(transform, neu_dv, uit_dv))
    summary = {"recipe": "control", "seed": seed, "checkpoint": str(ckpt.relative_to(paths.ROOT))}
    _write(f"control-s{seed}", summary, scored)
    return {"summary": summary, "scored": scored}


def select() -> dict[str, Any]:
    """cycle4.yaml recipe_selection, from the seed-42 summaries."""
    import json

    def macro(name: str, s: str) -> float:
        return json.loads((OUT / name / "summary.json").read_text(encoding="utf-8"))["sets"][s][
            "macro_f1"
        ]

    ctrl_uit = macro("control-s42", "uit_validation")
    rows = {}
    for r in RECIPES:
        if not (OUT / f"{r}-s42" / "summary.json").exists():
            raise FileNotFoundError(f"{r}-s42 has not run")
        uit, neu = macro(f"{r}-s42", "uit_validation"), macro(f"{r}-s42", "neu_validation")
        rows[r] = {
            "uit_validation": uit,
            "neu_validation": neu,
            "uit_minus_control": uit - ctrl_uit,
            "eligible": uit >= ctrl_uit - 0.01,
        }
    eligible = {r: v for r, v in rows.items() if v["eligible"]}
    chosen = max(eligible, key=lambda r: eligible[r]["neu_validation"]) if eligible else None
    out = {
        "declared_in": "configs/experiments/cycle4.yaml H8 recipe_selection",
        "control_s42": {
            "uit_validation": ctrl_uit,
            "neu_validation": macro("control-s42", "neu_validation"),
        },
        "recipes": rows,
        "chosen": chosen,
        "outcome": "confirm" if chosen else "not supported: no recipe is eligible",
    }
    from vifeedback.evaluation.report import yaml_safe

    (OUT / "selection.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def confirm(recipe: str) -> dict[str, Any]:
    """cycle4.yaml H8 rule, five seeds each side. The only function that reads NEU-ESC test; every
    call is logged."""
    import json

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.evaluation import ood as OOD
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.training.runner import _splits

    seeds = list(SEEDS)
    cand = {s: paths.MODELS / config(recipe, s).run_id("ckpt") for s in seeds}
    ctrl = {s: control_checkpoint(s) for s in seeds}
    missing = [str(p) for p in cand.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"run the chosen recipe at every seed first: {missing}")

    def summary(name: str) -> dict[str, Any]:
        return json.loads((OUT / name / "summary.json").read_text(encoding="utf-8"))["sets"]

    transform = serving_transform()
    te = neu_esc("test", transform)  # in scope
    te_all = neu_esc("test", transform, in_scope=False)
    log_test_use(
        "cycle4.yaml H8 rule (1) and reported sets",
        [p.name for p in [*cand.values(), *ctrl.values()]],
    )

    def preds(ckpt: Path) -> dict[str, np.ndarray]:
        tok = AutoTokenizer.from_pretrained(ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt)
        return {
            "in_scope": predict_proba(model, tok, te["x"]).argmax(1),
            "all": predict_proba(model, tok, te_all["x"]).argmax(1),
        }

    p_cand = {s: preds(cand[s]) for s in seeds}
    p_ctrl = {s: preds(ctrl[s]) for s in seeds}

    from vifeedback.evaluation import metrics as M

    k = len(LABELS)
    rule1 = seed_paired_bootstrap(
        te["y"], [p_cand[s]["in_scope"] for s in seeds], [p_ctrl[s]["in_scope"] for s in seeds]
    )
    per_seed = {
        s: {
            "neu_test_in_scope": {
                "candidate": M.macro_f1(te["y"], p_cand[s]["in_scope"], k),
                "control": M.macro_f1(te["y"], p_ctrl[s]["in_scope"], k),
            },
            "neu_test_all": {
                "candidate": M.macro_f1(te_all["y"], p_cand[s]["all"], k),
                "control": M.macro_f1(te_all["y"], p_ctrl[s]["all"], k),
            },
        }
        for s in seeds
    }

    def mean_diff(set_name: str, key: str) -> float:
        def val(sets: dict[str, Any]) -> float:
            r = sets[set_name]
            return r["macro_f1"] if key == "macro_f1" else r["per_class"]["neutral"]["f1"]

        return float(
            np.mean([val(summary(f"{recipe}-s{s}")) - val(summary(f"control-s{s}")) for s in seeds])
        )

    rule2 = mean_diff("uit_validation", "macro_f1")
    rule3 = mean_diff("uit_validation", "neutral_f1")
    rule4 = mean_diff("uit_validation_stripped", "macro_f1")

    # (5) the out-of-scope score refitted for the seed-42 model (B4): UIT-VSFC train + in-scope
    # NEU-ESC train features; threshold keeps 95% of UIT-VSFC validation.
    (x_tr, y_tr), (x_dv, _), _ = _splits("sentiment", "seg_pyvi")
    neu_tr = neu_esc("train", transform)
    neu_dv = neu_esc("validation", transform)
    off = te_all["x"]
    off = [t for t, topic in zip(off, te_all["topic"], strict=True) if topic in OFF_TOPIC]
    ck = str(cand[42])
    f_fit, _ = OOD.encode(ck, list(x_tr) + neu_tr["x"])
    maha = OOD.fit_mahalanobis(f_fit, np.concatenate([np.asarray(y_tr), neu_tr["y"]]))
    l_dv, f_dv = OOD.encode(ck, list(x_dv))
    l_off, f_off = OOD.encode(ck, off)
    l_in, f_in = OOD.encode(ck, neu_dv["x"])
    s_dv = OOD.scores(l_dv, f_dv, maha)["neg_mahalanobis"]
    check = OOD.evaluate(s_dv, OOD.scores(l_off, f_off, maha)["neg_mahalanobis"])
    in_scope_flagged = float(
        (OOD.scores(l_in, f_in, maha)["neg_mahalanobis"] < check["threshold"]).mean()
    )
    # Reported (v2, ADR-032): the refitted score within NEU-ESC test, in-scope against off-topic.
    from sklearn.metrics import roc_auc_score

    l_te, f_te = OOD.encode(ck, te["x"])
    s_te_in = OOD.scores(l_te, f_te, maha)["neg_mahalanobis"]
    s_te_off = OOD.scores(l_off, f_off, maha)["neg_mahalanobis"]
    within = float(
        roc_auc_score(
            np.r_[np.ones(len(s_te_in)), np.zeros(len(s_te_off))], np.r_[s_te_in, s_te_off]
        )
    )

    rules = {
        "1_neu_test_in_scope_macro_f1_up": {**rule1, "passed": rule1["ci_low"] > 0},
        "2_uit_validation_macro_f1": {"mean_diff": rule2, "passed": rule2 >= -0.005},
        "3_uit_validation_neutral_f1": {"mean_diff": rule3, "passed": rule3 >= -0.02},
        "4_uit_validation_stripped_macro_f1": {"mean_diff": rule4, "passed": rule4 >= -0.01},
        "5_out_of_scope_seed42": {
            **check,
            "neu_validation_in_scope_flagged": in_scope_flagged,
            "off_topic_posts": len(off),
            "within_neu_test_auroc_reported": within,
            "passed": check["auroc"] >= 0.90 and in_scope_flagged <= 0.10,
        },
    }
    out = {
        "declared_in": "configs/experiments/cycle4.yaml H8 rule",
        "recipe": recipe,
        "seeds": seeds,
        "rules": rules,
        "passed": all(r["passed"] for r in rules.values()),
        "per_seed": per_seed,
        "neu_test_posts": {"in_scope": len(te["y"]), "all": len(te_all["y"])},
    }
    d = OUT / f"confirm-{recipe}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    import pandas as pd

    cols = {}
    for s in seeds:
        cols[f"candidate-s{s}"] = [LABELS[i] for i in p_cand[s]["all"]]
        cols[f"control-s{s}"] = [LABELS[i] for i in p_ctrl[s]["all"]]
    table = pd.DataFrame(cols)
    table.insert(0, "in_scope", ~np.isin(te_all["topic"], OFF_TOPIC))
    table.to_csv(d / "predictions_neu_test.csv", index_label="row")  # labels only
    return out


__all__ = [
    "OFF_TOPIC",
    "OUT",
    "PHASE_NUM",
    "RECIPES",
    "SEEDS",
    "as_sequence_classifier",
    "augment_neu",
    "config",
    "confirm",
    "control_checkpoint",
    "evaluate_control",
    "log_test_use",
    "neu_esc",
    "predict_proba",
    "run",
    "score_sets",
    "seed_paired_bootstrap",
    "select",
    "serving_transform",
    "train_two_heads",
    "uit_stripped_validation",
]
