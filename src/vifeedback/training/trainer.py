"""Fine-tuning loop for transformer encoders.

A hand-written loop rather than `transformers.Trainer`, for two reasons: the Phase 4 ladder needs
R-Drop, FGM and layer-wise learning-rate decay, which are awkward to bolt onto Trainer; and a
transparent loop is easier to defend in an interview than a framework call.

Non-negotiables enforced here (docs/EVALUATION_PROTOCOL.md § 4):
* checkpoint selection is on **dev macro-F1**, never on dev loss or accuracy — with a 4% minority
  class, selecting on loss selects the model that has learned to ignore neutral;
* dynamic padding, so sequence length tracks the batch rather than `max_length`;
* every run is seeded, and what determinism was actually achieved is recorded.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from vifeedback.constants import n_classes
from vifeedback.evaluation import metrics as M
from vifeedback.training.losses import FGM, build_loss, rdrop_kl
from vifeedback.training.seeding import describe_determinism, seed_everything, worker_init_fn


@dataclass
class TrainConfig:
    task: str = "sentiment"
    model_key: str = "phobert-base"
    preprocessing: str = "raw"
    recipe: str = "base"

    max_length: int = 96  # from Gate G0: PhoBERT subword p99.9 = 87
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 4
    lr: float = 2e-5
    head_lr: float | None = None
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    grad_accum: int = 1

    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    logit_adjust_tau: float = 1.0
    weight_scheme: str = "balanced"

    freeze_embeddings: bool = False  # makes xlm-roberta-base fit a 4 GB GPU
    llrd: float | None = None  # e.g. 0.9 -> layer-wise decay downward
    rdrop_alpha: float = 0.0
    fgm_epsilon: float = 0.0

    seed: int = 42
    fp16: bool = True
    early_stopping_patience: int = 3
    num_workers: int = 0  # Windows: workers add process-spawn overhead on a dataset this small
    device: str = "auto"

    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    # Fields that change the result. Anything listed here is part of the run's identity; anything
    # omitted (notes, device, num_workers) is not.
    _IDENTITY_FIELDS = (
        "task",
        "model_key",
        "preprocessing",
        "recipe",
        "loss",
        "max_length",
        "batch_size",
        "grad_accum",
        "epochs",
        "lr",
        "head_lr",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "label_smoothing",
        "focal_gamma",
        "logit_adjust_tau",
        "weight_scheme",
        "llrd",
        "rdrop_alpha",
        "fgm_epsilon",
        "freeze_embeddings",
        "seed",
        "fp16",
        "early_stopping_patience",
    )

    def config_hash(self) -> str:
        """Short digest of every field that can change the result.

        Exists because `run_id()` did not include the learning rate, epoch count or max length: two
        genuinely different configurations produced the same id, and `save_run()` writes to
        `results/runs/<run_id>/`, so the second silently overwrote the first. Verified during review
        (R11) — the snapshot had no duplicates, but nothing prevented one.
        """
        import hashlib

        payload = "|".join(f"{k}={getattr(self, k)!r}" for k in self._IDENTITY_FIELDS)
        return hashlib.sha256(payload.encode()).hexdigest()[:8]

    def run_id(self, split: str = "val") -> str:
        return (
            f"p{self.extra.get('phase_num', 2)}-{self.task[:4]}-{self.model_key}-"
            f"{self.preprocessing}-{self.recipe}-s{self.seed}-{self.config_hash()}-{split[:3]}"
        )


class TextDataset(Dataset):
    """Pre-tokenized without padding; the collator pads per batch."""

    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer, max_length: int):
        self.encodings = tokenizer(
            list(texts), truncation=True, max_length=max_length, padding=False
        )
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[i])
        return item


def build_optimizer(model, cfg: TrainConfig):
    """AdamW with no weight decay on biases/LayerNorm, and optional layer-wise lr decay.

    LLRD gives the top layers a higher learning rate than the bottom ones, on the reasoning that
    lower layers hold general language features worth disturbing less. It is one of the few
    stability tricks that reliably helps small-dataset BERT fine-tuning.
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    head_lr = cfg.head_lr or cfg.lr * 5

    def wd(name: str) -> float:
        return 0.0 if any(n in name for n in no_decay) else cfg.weight_decay

    groups: list[dict[str, Any]] = []

    if cfg.llrd is None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            is_head = "classifier" in name or name.startswith("classifier")
            groups.append(
                {"params": [p], "lr": head_lr if is_head else cfg.lr, "weight_decay": wd(name)}
            )
    else:
        encoder = getattr(model, model.base_model_prefix)
        n_layers = len(encoder.encoder.layer)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier" in name:
                lr = head_lr
            elif "encoder.layer." in name:
                idx = int(name.split("encoder.layer.")[1].split(".")[0])
                lr = cfg.lr * (cfg.llrd ** (n_layers - 1 - idx))
            else:  # embeddings — the bottom of the stack
                lr = cfg.lr * (cfg.llrd**n_layers)
            groups.append({"params": [p], "lr": lr, "weight_decay": wd(name)})

    return torch.optim.AdamW(groups, lr=cfg.lr, eps=1e-8)


def linear_warmup_schedule(optimizer, num_training_steps: int, warmup_ratio: float):
    warmup = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (num_training_steps - step) / max(1, num_training_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def predict(model, loader, device: str, fp16: bool) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, labels_all = [], []
    autocast = torch.autocast(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=fp16 and device != "cpu"
    )
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast:
            out = model(**batch).logits
        logits_all.append(out.float().cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def train(
    cfg: TrainConfig,
    train_texts: list[str],
    train_labels: np.ndarray,
    dev_texts: list[str],
    dev_labels: np.ndarray,
    verbose: bool = True,
) -> dict[str, Any]:
    """Fine-tune, selecting the checkpoint by dev macro-F1. Returns model, tokenizer and history."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
    )

    from vifeedback.constants import LABELS, MODEL_IDS

    seed_everything(cfg.seed)
    device = cfg.resolved_device()
    k = n_classes(cfg.task)
    model_id = MODEL_IDS[cfg.model_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=k,
        id2label={i: n for i, n in LABELS[cfg.task].items()},
        label2id={n: i for i, n in LABELS[cfg.task].items()},
    ).to(device)

    if cfg.freeze_embeddings:
        # XLM-R carries 192M of its 277M parameters in the embedding matrix, so freezing it
        # drops optimizer state from ~4.4 GB to ~1.4 GB. Defensible on 11k training examples:
        # a 250k-row embedding table cannot be meaningfully updated from that much data.
        frozen = 0
        for name, param in model.named_parameters():
            if "embeddings" in name:
                param.requires_grad = False
                frozen += param.numel()
        if verbose:
            print(f"  froze {frozen / 1e6:.0f}M embedding parameters")

    collator = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    g = torch.Generator()
    g.manual_seed(cfg.seed)

    train_ds = TextDataset(train_texts, train_labels, tokenizer, cfg.max_length)
    dev_ds = TextDataset(dev_texts, dev_labels, tokenizer, cfg.max_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=cfg.num_workers,
    )

    counts = np.bincount(np.asarray(train_labels), minlength=k)
    criterion = build_loss(
        cfg.loss,
        counts,
        gamma=cfg.focal_gamma,
        tau=cfg.logit_adjust_tau,
        label_smoothing=cfg.label_smoothing,
        weight_scheme=cfg.weight_scheme,
        device=device,
    )

    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_steps = steps_per_epoch * cfg.epochs
    scheduler = linear_warmup_schedule(optimizer, total_steps, cfg.warmup_ratio)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device != "cpu")
    fgm = FGM(model, cfg.fgm_epsilon) if cfg.fgm_epsilon > 0 else None

    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    autocast_kw = dict(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=cfg.fp16 and device != "cpu"
    )

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, n_batches = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device, non_blocking=True)
            batch = {k_: v.to(device, non_blocking=True) for k_, v in batch.items()}

            with torch.autocast(**autocast_kw):
                logits = model(**batch).logits
                loss = criterion(logits, labels)
                if cfg.rdrop_alpha > 0:
                    logits2 = model(**batch).logits
                    loss = 0.5 * (loss + criterion(logits2, labels))
                    loss = loss + cfg.rdrop_alpha * rdrop_kl(logits, logits2)
                loss = loss / cfg.grad_accum

            scaler.scale(loss).backward()

            if fgm is not None:
                # Adversarial step: perturb the embeddings along the gradient, accumulate, restore.
                #
                # Deliberately NO unscale_ here. FGM uses only the gradient *direction*
                # (it divides by the gradient norm), and direction is invariant to the uniform
                # factor AMP applies — so the scaled gradient gives the identical perturbation.
                #
                # Unscaling first, as this did until it was caught by review, leaves the clean
                # gradient divided by `scale` while the adversarial gradient arrives multiplied by
                # it. Their sum is then wrong by a factor of `scale` on the adversarial term
                # (measured: 258 where 4 was correct at scale=128), and with grad_accum > 1 the
                # next microbatch raises "unscale_() has already been called".
                fgm.attack()
                with torch.autocast(**autocast_kw):
                    adv_loss = criterion(model(**batch).logits, labels) / cfg.grad_accum
                scaler.scale(adv_loss).backward()
                fgm.restore()

            running += loss.item() * cfg.grad_accum
            n_batches += 1

            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(train_loader):
                # Exactly once per optimizer step, with every accumulated gradient — clean and
                # adversarial — still on the same scale.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        logits, labels_np = predict(model, dev_loader, device, cfg.fp16)
        dev = M.evaluate(labels_np, logits.argmax(axis=1), cfg.task, y_prob=softmax(logits))
        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "dev_macro_f1": dev["macro_f1"],
            "dev_weighted_f1": dev["weighted_f1"],
            "dev_accuracy": dev["accuracy"],
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.perf_counter() - t0, 1),
        }
        minority = min(dev["per_class"].items(), key=lambda kv: kv[1]["support"])
        row["dev_minority_f1"] = minority[1]["f1"]
        history.append(row)

        if verbose:
            print(
                f"  epoch {epoch}/{cfg.epochs}  loss {row['train_loss']:.4f}  "
                f"dev macro-F1 {dev['macro_f1']:.4f}  wtd {dev['weighted_f1']:.4f}  "
                f"acc {dev['accuracy']:.4f}  {minority[0]} F1 {minority[1]['f1']:.3f}  "
                f"({row['seconds']:.0f}s)"
            )

        # Selection on dev macro-F1 — the whole point.
        if dev["macro_f1"] > best["macro_f1"]:
            best = {
                "macro_f1": dev["macro_f1"],
                "epoch": epoch,
                "state": {kk: v.detach().cpu().clone() for kk, v in model.state_dict().items()},
            }
        elif epoch - best["epoch"] >= cfg.early_stopping_patience:
            if verbose:
                print(f"  early stopping at epoch {epoch} (best was {best['epoch']})")
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": asdict(cfg),
        "history": history,
        "best_epoch": best["epoch"],
        "best_dev_macro_f1": best["macro_f1"],
        "train_seconds": round(time.perf_counter() - t0, 1),
        "determinism": describe_determinism(),
        "device": device,
    }
