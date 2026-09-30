"""Cycle 5, H11: a 6-layer student distilled from the teacher (NEXT_PLAN v6 § 3; S7 <= 200 MB).

* **Student.** PhoBERT's architecture with 6 layers. Two initializations, the selection axis:
  ``teacher-alternate`` copies the teacher's embeddings, layers 1, 3, 5, 7, 9, 11 (the last one is
  the layer the head reads) and head; ``pretrained-first6`` takes pretrained PhoBERT's first six
  layers and a new head (the Patient-KD setup).
* **Teacher.** An ensemble: the mean of the teacher recipe's five seeds' probabilities at
  temperature T (Hinton et al., 2015).
* **Transfer set.** UIT-VSFC train with its gold labels (the served recipe's augmentation applied),
  plus unlabeled text the teacher labels: in-scope NEU-ESC train posts and the ViLexNorm
  consistency-training originals, through the serving transform. No validation or test text.
* **Loss.** Labelled: ALPHA x T^2 x KL(teacher || student at T) + (1 - ALPHA) x CE. Unlabeled: the
  KL term alone.

Teacher probabilities are cached under models/ keyed by the SHA-1 of each text (no text stored), so
the five teachers score each distinct input once across all seeds.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths
from vifeedback.constants import label_names

INITS = ("teacher-alternate", "pretrained-first6")
ALTERNATE = (1, 3, 5, 7, 9, 11)
N_LAYERS = 6
TEMPERATURE = 2.0
ALPHA = 0.5
PHASE_NUM = 14
LABELS = label_names("sentiment")


# --- students -----------------------------------------------------------------------------------


def student_from_teacher(teacher, layers: tuple[int, ...] = ALTERNATE):
    """A student with the teacher's embeddings, the chosen layers in order, and its head."""
    import copy

    from transformers import AutoModelForSequenceClassification

    cfg = copy.deepcopy(teacher.config)
    cfg.num_hidden_layers = len(layers)
    student = AutoModelForSequenceClassification.from_config(cfg)
    t_base = getattr(teacher, teacher.base_model_prefix)
    s_base = getattr(student, student.base_model_prefix)
    s_base.embeddings.load_state_dict(t_base.embeddings.state_dict())
    for j, i in enumerate(layers):
        s_base.encoder.layer[j].load_state_dict(t_base.encoder.layer[i].state_dict())
    student.classifier.load_state_dict(teacher.classifier.state_dict())
    return student


def student_from_pretrained(model_id: str, revision: str | None = None, n_layers: int = N_LAYERS):
    """Pretrained weights for the first `n_layers` layers and the embeddings; a new 3-class head."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from vifeedback.constants import LABELS as LABEL_MAPS

    cfg = AutoConfig.from_pretrained(
        model_id,
        revision=revision,
        num_hidden_layers=n_layers,
        num_labels=len(LABELS),
        id2label=dict(LABEL_MAPS["sentiment"]),
        label2id={n: i for i, n in LABEL_MAPS["sentiment"].items()},
    )
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, config=cfg
    )


# --- teacher probabilities ----------------------------------------------------------------------


def tempered_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / temperature
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


class SoftLabelCache:
    """Teacher-ensemble probabilities at temperature T, keyed by the SHA-1 of the input text."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.table: dict[str, np.ndarray] = {}
        if self.path.exists():
            with np.load(self.path) as z:
                self.table = dict(zip(z["keys"].tolist(), z["probs"], strict=True))

    def missing(self, texts: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for t in texts:
            k = _key(t)
            if k not in self.table and k not in seen:
                seen.add(k)
                out.append(t)
        return out

    def add(self, texts: list[str], probs: np.ndarray) -> None:
        for t, p in zip(texts, probs, strict=True):
            self.table[_key(t)] = np.asarray(p, dtype=np.float32)

    def get(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.table[_key(t)] for t in texts])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self.table), dtype="U40")
        probs = np.stack(list(self.table.values())) if self.table else np.zeros((0, len(LABELS)))
        np.savez(self.path, keys=keys, probs=probs)


def teacher_logits(model, tokenizer, texts: list[str], batch_size: int = 128) -> np.ndarray:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[s : s + batch_size],
                truncation=True,
                max_length=96,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                out.append(model(**enc).logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(LABELS)))


def _load(checkpoint) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    return (
        AutoModelForSequenceClassification.from_pretrained(checkpoint),
        AutoTokenizer.from_pretrained(checkpoint),
    )


def ensemble_soft_labels(
    checkpoints: list[Path],
    texts: list[str],
    cache: SoftLabelCache,
    verbose: bool = True,
    load=_load,
) -> np.ndarray:
    """Mean tempered probability over the teachers, computed only for texts the cache lacks.
    `load(checkpoint)` returns (model, tokenizer)."""
    todo = cache.missing(texts)
    if todo:
        acc = np.zeros((len(todo), len(LABELS)))
        for ck in checkpoints:
            t0 = time.perf_counter()
            model, tok = load(ck)
            acc += tempered_probs(teacher_logits(model, tok, todo), TEMPERATURE)
            del model
            if verbose:
                print(
                    f"  teacher {Path(ck).name}: {len(todo)} texts ({time.perf_counter() - t0:.0f}s)"
                )
        cache.add(todo, acc / len(checkpoints))
        cache.save()
    return cache.get(texts)


# --- loss and training --------------------------------------------------------------------------


def distill_loss(
    student_logits, teacher_probs, labels, temperature: float = TEMPERATURE, alpha: float = ALPHA
):
    """Per-batch mean of alpha*T^2*KL + (1-alpha)*CE on labelled rows (label >= 0), T^2*KL alone
    on unlabeled rows (label -1). KL is KL(teacher || student at T), in float32."""
    import torch
    import torch.nn.functional as F

    logits = student_logits.float()
    log_p = F.log_softmax(logits / temperature, dim=-1)
    q = teacher_probs.float()
    kl = (q * (q.clamp_min(1e-12).log() - log_p)).sum(-1) * temperature**2
    ce = F.cross_entropy(logits, labels.clamp_min(0), reduction="none")
    labelled = labels >= 0
    per = torch.where(labelled, alpha * kl + (1 - alpha) * ce, kl)
    return per.mean()


def config(init: str, seed: int, teacher: str):
    """The served recipe's TrainConfig; the student's init and teacher are part of its identity."""
    from vifeedback.training.trainer import TrainConfig

    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}")
    return TrainConfig(
        task="sentiment",
        model_key="phobert-base",
        preprocessing="seg_pyvi",
        recipe=f"h11-{init}-from-{teacher}",
        augment="diac-teen",
        augment_p=0.3,
        seed=seed,
        extra={
            "phase_num": PHASE_NUM,
            "student_layers": N_LAYERS,
            "temperature": TEMPERATURE,
            "alpha": ALPHA,
        },
        notes=f"cycle5.yaml H11 {init}, teacher {teacher}",
    )


def train_student(
    cfg,
    student,
    tokenizer,
    transfer: tuple[list[str], np.ndarray, np.ndarray],
    uit_dev: tuple[list[str], np.ndarray],
    verbose: bool = True,
) -> dict[str, Any]:
    """Distillation with the served recipe's optimizer, schedule, AMP and clipping; the epoch is
    selected by UIT-VSFC validation macro-F1. `transfer` is (texts, labels with -1 for unlabeled,
    teacher probabilities)."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import DataCollatorWithPadding

    from vifeedback.evaluation import metrics as M
    from vifeedback.training.seeding import describe_determinism, seed_everything, worker_init_fn
    from vifeedback.training.trainer import (
        TextDataset,
        build_optimizer,
        linear_warmup_schedule,
        predict,
        softmax,
    )

    seed_everything(cfg.seed)
    device = cfg.resolved_device()
    model = student.to(device)
    pad = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    texts, y, q = transfer

    class _Transfer(Dataset):
        def __init__(self) -> None:
            self.enc = tokenizer(
                list(texts), truncation=True, max_length=cfg.max_length, padding=False
            )

        def __len__(self) -> int:
            return len(y)

        def __getitem__(self, i: int) -> dict[str, Any]:
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = int(y[i])
            item["idx"] = i
            return item

    def collate(items):
        idx = torch.tensor([it.pop("idx") for it in items])
        batch = pad(items)
        return batch, torch.as_tensor(q[idx.numpy()])

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    loader = DataLoader(
        _Transfer(),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,
        generator=g,
    )
    dev_loader = DataLoader(
        TextDataset(uit_dev[0], uit_dev[1], tokenizer, cfg.max_length),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=pad,
    )
    opt = build_optimizer(model, cfg)
    sched = linear_warmup_schedule(opt, len(loader) * cfg.epochs, cfg.warmup_ratio)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device != "cpu")
    ac: dict[str, Any] = dict(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=cfg.fp16 and device != "cpu"
    )
    best: dict[str, Any] = {"score": -1.0, "epoch": -1, "state": None}
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch, probs in loader:
            opt.zero_grad(set_to_none=True)
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(**ac):
                logits = model(**batch).logits
            loss = distill_loss(logits, probs.to(device), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        logits, labels_np = predict(model, dev_loader, device, cfg.fp16)
        dev = M.evaluate(labels_np, logits.argmax(1), "sentiment", y_prob=softmax(logits))
        row = {
            "epoch": epoch,
            "train_loss": running / len(loader),
            "dev_macro_f1": dev["macro_f1"],
            "dev_neutral_f1": dev["per_class"]["neutral"]["f1"],
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(row)
        if verbose:
            print(
                f"  epoch {epoch}/{cfg.epochs}  loss {row['train_loss']:.4f}  dev macro-F1 "
                f"{row['dev_macro_f1']:.4f}  neutral {row['dev_neutral_f1']:.3f}  ({row['seconds']:.0f}s)"
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
        "config": asdict(cfg),
        "history": history,
        "best_epoch": best["epoch"],
        "selection": "UIT-VSFC dev macro-F1",
        "train_seconds": round(time.perf_counter() - t0, 1),
        "determinism": describe_determinism(),
        "device": device,
        "transfer_rows": {"labelled": int((y >= 0).sum()), "unlabeled": int((y < 0).sum())},
    }


def cache_path(teacher: str) -> Path:
    return paths.MODELS / "distill" / f"soft_labels_{teacher}_T{TEMPERATURE:g}.npz"


# --- orchestration: cycle5.yaml v2 H11 (the CLI commands are thin wrappers) ----------------------

OUT = paths.RESULTS / "studies" / "cycle5" / "h11"
POOLED_SETS = ("uit_validation", "neu_validation_all")  # the S5' acceptance set, 4,888 rows


def resolve_teacher() -> tuple[str, dict[int, Path], Path]:
    """cycle5.yaml v2 H11 teacher: H10's recipe if H10 passed, otherwise the served recipe.

    Returns the teacher's name, its checkpoint per seed, and the H10 folder holding its per-seed
    summaries and validation predictions (written by the same scoring code)."""
    import json

    from vifeedback.constants import SEEDS
    from vifeedback.training import consistency as C
    from vifeedback.training import domain as D

    passed = None
    for f in sorted(C.OUT.glob("confirm-*/decision.json")):
        if json.loads(f.read_text(encoding="utf-8"))["passed"]:
            passed = f.parent.name.removeprefix("confirm-")
    sel = C.OUT / "selection.json"
    decided = passed is not None or (
        sel.exists() and json.loads(sel.read_text(encoding="utf-8"))["outcome"] != "confirm"
    )
    decided = decided or any(C.OUT.glob("confirm-*/decision.json"))
    if not decided:
        raise RuntimeError("H10 has not decided yet; H11's teacher depends on it (cycle5.yaml v2)")
    if passed:
        return (
            f"h10-{passed}",
            {s: paths.MODELS / C.config(passed, s).run_id("ckpt") for s in SEEDS},
            C.OUT,
        )
    return "served", {s: D.control_checkpoint(s) for s in SEEDS}, C.OUT


def _teacher_dir(teacher: str, h10_out: Path, seed: int) -> Path:
    return h10_out / (f"control-s{seed}" if teacher == "served" else f"{teacher[4:]}-s{seed}")


def _write(
    name: str, summary: dict[str, Any], scored: dict[str, Any], flips: dict[str, Any]
) -> Path:
    import json

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
    return d


def run(init: str, seed: int, verbose: bool = True, smoke: bool = False) -> dict[str, Any]:
    """Distil one student at one seed from the declared teacher; score it on the development sets.

    `smoke` checks the wiring: 256 labelled and 256 unlabeled rows, one epoch, 128 per evaluation set,
    and nothing is written except teacher probabilities in the local cache.
    """
    import dataclasses

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation import report as R
    from vifeedback.training import consistency as C
    from vifeedback.training import domain as D
    from vifeedback.training.runner import _augment_train, _splits
    from vifeedback.training.seeding import seed_everything

    teacher, ckpts, _ = resolve_teacher()
    cfg = config(init, seed, teacher)
    transform = D.serving_transform()
    (x_tr, y_tr), uit_dv, _ = _splits("sentiment", "seg_pyvi")
    x_tr, augmentation = _augment_train(cfg, x_tr)
    unlabeled = (
        D.neu_esc("train", transform)["x"] + C.vilexnorm_pairs("train_pairs", transform)["orig"]
    )
    sets = C.evaluation_sets(transform)
    dev_pairs = C.vilexnorm_pairs("dev", transform)
    if smoke:
        cfg = dataclasses.replace(cfg, epochs=1)
        x_tr, y_tr, unlabeled = list(x_tr)[:256], np.asarray(y_tr)[:256], unlabeled[:256]
        uit_dv = (list(uit_dv[0])[:128], np.asarray(uit_dv[1])[:128])
        sets = {k: (list(x)[:128], np.asarray(y)[:128]) for k, (x, y) in sets.items()}
        dev_pairs = {k: v[:128] for k, v in dev_pairs.items()}
    texts = list(x_tr) + list(unlabeled)
    labels = np.concatenate([np.asarray(y_tr), -np.ones(len(unlabeled), dtype=int)])
    if verbose:
        print(
            f"[{cfg.run_id()}] H11 {init} seed {seed}, teacher {teacher}: {len(x_tr)} labelled, "
            f"{len(unlabeled)} unlabeled"
        )
    q = ensemble_soft_labels(
        [ckpts[s] for s in sorted(ckpts)], texts, SoftLabelCache(cache_path(teacher)), verbose
    )

    seed_everything(cfg.seed)  # the new head of pretrained-first6 is initialized reproducibly
    if init == "teacher-alternate":
        tok = AutoTokenizer.from_pretrained(ckpts[seed])
        student = student_from_teacher(
            AutoModelForSequenceClassification.from_pretrained(ckpts[seed])
        )
    else:
        mid, rev = MODEL_IDS[cfg.model_key], MODEL_REVISIONS.get(cfg.model_key)
        tok = AutoTokenizer.from_pretrained(mid, revision=rev)
        student = student_from_pretrained(mid, rev)
    out = train_student(cfg, student, tok, (texts, labels, q), uit_dv, verbose)
    model = out["model"]
    scored = D.score_sets(model, tok, sets)
    flips = C.score_pairs(model, tok, dev_pairs)
    n_params = sum(p.numel() for p in model.parameters())
    summary = {
        "init": init,
        "seed": seed,
        "teacher": teacher,
        "teacher_checkpoints": [p.name for p in ckpts.values()],
        "history": out["history"],
        "best_epoch": out["best_epoch"],
        "parameters": n_params,
        "fp16_weight_bytes": 2 * n_params,
        "transfer_rows": out["transfer_rows"],
    }
    if smoke:
        return {"summary": {**summary, "smoke": True}, "scored": scored, "flips": flips}

    ckpt = paths.MODELS / cfg.run_id("ckpt")
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
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
            "model": f"{cfg.model_key}-{N_LAYERS}layer-student",
            "split": "validation",
            "fit_seconds": out["train_seconds"],
            "notes": cfg.notes,
            "reason": "cycle5.yaml v2 H11",
            "determinism": out["determinism"],
            "device": out["device"],
            "best_epoch": out["best_epoch"],
            "selection": out["selection"],
            "teacher": teacher,
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
        }
    )
    _write(f"{init}-s{seed}", summary, scored, flips)
    return {"summary": summary, "scored": scored, "flips": flips}


def _summary(path: Path) -> dict[str, Any]:
    import json

    f = path / "summary.json"
    if not f.exists():
        raise FileNotFoundError(f"{path.name} has not run: {f}")
    return json.loads(f.read_text(encoding="utf-8"))


def select() -> dict[str, Any]:
    """cycle5.yaml v2 H11 selection: the student with the higher mean of UIT-VSFC and NEU-ESC
    validation macro-F1 at seed 42."""
    import json

    from vifeedback.evaluation.report import yaml_safe

    rows = {}
    for init in INITS:
        s = _summary(OUT / f"{init}-s42")["sets"]
        uit, neu = s["uit_validation"]["macro_f1"], s["neu_validation_all"]["macro_f1"]
        rows[init] = {"uit_validation": uit, "neu_validation_all": neu, "mean": (uit + neu) / 2}
    chosen = max(rows, key=lambda r: rows[r]["mean"])
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v2 H11 selection",
        "students": rows,
        "chosen": chosen,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "selection.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


def seed_averaged_drop_bound(
    y: np.ndarray,
    teacher: list[np.ndarray],
    student: list[np.ndarray],
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """One-sided 95% upper bound of the seed-averaged macro-F1 drop (teacher minus student), rows
    resampled once per draw and the seed pairs averaged within it (the S5' procedure, 5 seeds)."""
    from vifeedback.evaluation.int8_power import _macro_f1_batch

    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    draws = []
    for s in range(0, n_resamples, 200):
        idx = rng.integers(0, n, (min(200, n_resamples - s), n))
        draws.append(
            np.mean(
                [
                    _macro_f1_batch(y, np.asarray(t), idx) - _macro_f1_batch(y, np.asarray(u), idx)
                    for t, u in zip(teacher, student, strict=True)
                ],
                axis=0,
            )
        )
    d = np.concatenate(draws)
    full = np.arange(n)[None, :]
    observed = float(
        np.mean(
            [
                _macro_f1_batch(y, np.asarray(t), full)[0]
                - _macro_f1_batch(y, np.asarray(u), full)[0]
                for t, u in zip(teacher, student, strict=True)
            ]
        )
    )
    return {
        "observed_drop": observed,
        "upper_95_one_sided": float(np.quantile(d, 0.95)),
        "n_resamples": n_resamples,
        "rows": n,
        "seeds": len(teacher),
    }


def apply_rule(
    size_bytes: int, pooled: dict[str, float], drops: dict[str, float], parity: dict[str, float]
) -> dict[str, Any]:
    """cycle5.yaml v2 H11 rule (1) to (4)."""
    rules: dict[str, dict[str, Any]] = {
        "1_fp16_graph_at_most_200_mb": {"bytes": size_bytes, "passed": size_bytes <= 200 * 10**6},
        "2_pooled_drop_bound": {
            **pooled,
            "limit": 0.01,
            "passed": pooled["upper_95_one_sided"] <= 0.01,
        },
        "3a_uit_validation_macro_f1_drop": {
            "drop": drops["uit"],
            "limit": 0.01,
            "passed": drops["uit"] <= 0.01,
        },
        "3b_uit_validation_neutral_f1_drop": {
            "drop": drops["neutral"],
            "limit": 0.03,
            "passed": drops["neutral"] <= 0.03,
        },
        "3c_uit_stripped_macro_f1_drop": {
            "drop": drops["stripped"],
            "limit": 0.015,
            "passed": drops["stripped"] <= 0.015,
        },
        "4_graph_parity_and_batch_independence": {
            **parity,
            "passed": parity["max_abs_logit_diff"] <= 1e-4
            and parity["label_agreement"] == 1.0
            and parity["batch_vs_single_max_diff"] <= 1e-4,
        },
    }
    return {"rules": rules, "passed": all(r["passed"] for r in rules.values())}


def export_and_check(ckpt: Path, texts: list[str], out_dir: Path) -> tuple[Path, dict[str, float]]:
    """Rule (1) and (4): the FP16-storage graph of a student and its parity with the PyTorch student
    whose weights went through the same FP16 rounding, on `texts` (CPU)."""
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from vifeedback.inference import onnx_export as OX

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).eval()
    dummy = tok(texts[:2], return_tensors="pt", padding=True, truncation=True, max_length=96)
    path = OX.export_fp16_storage(
        model, (dummy["input_ids"], dummy["attention_mask"]), out_dir / "model.fp16.onnx"
    )
    tok.save_pretrained(out_dir)
    ref_model = OX.fp16_rounded(model)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    def onnx_logits(enc) -> np.ndarray:
        return sess.run(
            None,
            {
                "input_ids": enc["input_ids"].numpy(),
                "attention_mask": enc["attention_mask"].numpy(),
            },
        )[0]

    diffs, agree = [], []
    with torch.inference_mode():
        for s in range(0, len(texts), 64):
            enc = tok(
                texts[s : s + 64], return_tensors="pt", padding=True, truncation=True, max_length=96
            )
            ref = ref_model(**enc).logits.numpy()
            got = onnx_logits(enc)
            diffs.append(np.abs(got - ref).max())
            agree.append(got.argmax(1) == ref.argmax(1))
    batch = tok(texts[:32], return_tensors="pt", padding=True, truncation=True, max_length=96)
    together = onnx_logits(batch)
    alone = np.concatenate(
        [
            onnx_logits(tok([t], return_tensors="pt", truncation=True, max_length=96))
            for t in texts[:32]
        ]
    )
    return path, {
        "max_abs_logit_diff": float(max(diffs)),
        "label_agreement": float(np.concatenate(agree).mean()),
        "rows_checked": len(texts),
        "batch_vs_single_max_diff": float(np.abs(together - alone).max()),
    }


def confirm(init: str) -> dict[str, Any]:
    """cycle5.yaml v2 H11 rule, five seeds each side (validation sets only; no test split)."""
    import json

    import pandas as pd

    from vifeedback.constants import SEEDS
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.training import consistency as C
    from vifeedback.training import domain as D

    teacher, _, h10_out = resolve_teacher()
    seeds = list(SEEDS)
    stud_dirs = {s: OUT / f"{init}-s{s}" for s in seeds}
    teach_dirs = {s: _teacher_dir(teacher, h10_out, s) for s in seeds}
    for d in [*stud_dirs.values(), *teach_dirs.values()]:
        _summary(d)  # every side has run

    transform = D.serving_transform()
    sets = C.evaluation_sets(transform)
    y = np.concatenate([np.asarray(sets[s][1]) for s in POOLED_SETS])
    ids = {c: i for i, c in enumerate(LABELS)}

    def preds(d: Path) -> np.ndarray:
        return np.concatenate(
            [
                pd.read_csv(d / f"predictions_{s}.csv")["pred"].map(ids).to_numpy()
                for s in POOLED_SETS
            ]
        )

    pooled = seed_averaged_drop_bound(
        y, [preds(teach_dirs[s]) for s in seeds], [preds(stud_dirs[s]) for s in seeds]
    )

    def mean_drop(set_name: str, key: str = "macro_f1") -> float:
        def val(d: Path) -> float:
            r = _summary(d)["sets"][set_name]
            return r["macro_f1"] if key == "macro_f1" else r["per_class"]["neutral"]["f1"]

        return float(np.mean([val(teach_dirs[s]) - val(stud_dirs[s]) for s in seeds]))

    drops = {
        "uit": mean_drop("uit_validation"),
        "neutral": mean_drop("uit_validation", "neutral_f1"),
        "stripped": mean_drop("uit_validation_stripped"),
    }
    ck42 = paths.ROOT / _summary(stud_dirs[42])["checkpoint"]
    graph, parity = export_and_check(
        ck42, list(sets["uit_validation"][0]), paths.MODELS / "distill" / f"{init}-s42-fp16"
    )
    decision = apply_rule(graph.stat().st_size, pooled, drops, parity)
    out = {
        "declared_in": "configs/experiments/cycle5.yaml v2 H11 rule",
        "init": init,
        "teacher": teacher,
        "seeds": seeds,
        **decision,
        "graph": {"path": str(graph.relative_to(paths.ROOT)), "bytes": graph.stat().st_size},
        "reported": {
            "neu_validation_in_scope_drop": mean_drop("neu_validation"),
            "parameters": _summary(stud_dirs[42])["parameters"],
        },
    }
    d = OUT / f"confirm-{init}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "decision.json").write_text(
        json.dumps(yaml_safe(out), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out


__all__ = [
    "ALPHA",
    "ALTERNATE",
    "INITS",
    "N_LAYERS",
    "OUT",
    "PHASE_NUM",
    "POOLED_SETS",
    "TEMPERATURE",
    "SoftLabelCache",
    "apply_rule",
    "cache_path",
    "config",
    "confirm",
    "distill_loss",
    "ensemble_soft_labels",
    "export_and_check",
    "resolve_teacher",
    "run",
    "seed_averaged_drop_bound",
    "select",
    "student_from_pretrained",
    "student_from_teacher",
    "teacher_logits",
    "tempered_probs",
    "train_student",
]
