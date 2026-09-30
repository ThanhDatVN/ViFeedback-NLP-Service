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


__all__ = [
    "ALPHA",
    "ALTERNATE",
    "INITS",
    "N_LAYERS",
    "PHASE_NUM",
    "TEMPERATURE",
    "SoftLabelCache",
    "cache_path",
    "config",
    "distill_loss",
    "ensemble_soft_labels",
    "student_from_pretrained",
    "student_from_teacher",
    "teacher_logits",
    "tempered_probs",
    "train_student",
]
