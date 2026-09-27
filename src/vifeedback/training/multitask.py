"""Shared encoder, two heads — experiment E07 / Study D, Cycle 1 H4.

One PhoBERT encoder feeds a sentiment head and a topic head, trained on
`L = L_sentiment + lambda * L_topic`. The question (review Q4) is whether the two tasks share useful
structure or interfere. The data card measured real dependence between them: Cramér's V = 0.344, and
`facility` is 95.6% negative.

Built to be a *controlled* comparison with the single-task runs:

* each head is the same architecture as RobertaForSequenceClassification's (dense → tanh → out), so the
  only change is sharing the encoder;
* same optimizer groups (heads at 5x the encoder learning rate, no decay on bias/LayerNorm), schedule,
  AMP, clipping, batch size, epochs and seeding as `trainer.train`;
* the checkpoint epoch is selected by the **mean** of the two dev macro-F1 scores, declared in
  `configs/experiments/cycle1.yaml` before any run, so neither task is favoured after the fact.

Gold topic is never an input, so the joint model is evaluated exactly as it would be served.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vifeedback.constants import MODEL_IDS, MODEL_REVISIONS, label_names
from vifeedback.evaluation import metrics as M
from vifeedback.training.seeding import describe_determinism, seed_everything, worker_init_fn
from vifeedback.training.trainer import TrainConfig, linear_warmup_schedule, softmax

TASKS = ("sentiment", "topic")


class _Head(nn.Module):
    """Mirrors transformers' RobertaClassificationHead: dropout → dense → tanh → dropout → out."""

    def __init__(self, hidden: int, n_labels: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, n_labels)

    def forward(self, cls: torch.Tensor) -> torch.Tensor:
        x = self.dropout(cls)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))


class MultiTaskModel(nn.Module):
    def __init__(self, model_key: str) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            MODEL_IDS[model_key], revision=MODEL_REVISIONS.get(model_key), add_pooling_layer=False
        )
        h = self.encoder.config.hidden_size
        p = self.encoder.config.hidden_dropout_prob
        self.heads = nn.ModuleDict({t: _Head(h, len(label_names(t)), p) for t in TASKS})

    def forward(self, input_ids, attention_mask) -> dict[str, torch.Tensor]:
        cls = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[
            :, 0
        ]
        return {t: head(cls) for t, head in self.heads.items()}


class _TwoLabelDataset(Dataset):
    def __init__(self, texts, y_sent, y_topic, tokenizer, max_length: int) -> None:
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
        self.y = {"sentiment": np.asarray(y_sent), "topic": np.asarray(y_topic)}

    def __len__(self) -> int:
        return len(self.y["sentiment"])

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels_sentiment"] = int(self.y["sentiment"][i])
        item["labels_topic"] = int(self.y["topic"][i])
        return item


def _collate(tokenizer):
    from transformers import DataCollatorWithPadding

    pad = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")

    def fn(items):
        ys = {t: torch.tensor([it.pop(f"labels_{t}") for it in items]) for t in TASKS}
        batch = pad(items)
        return batch, ys

    return fn


def _optimizer(model: MultiTaskModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight", "layer_norm")
    head_lr = cfg.head_lr or cfg.lr * 5  # same rule as trainer.build_optimizer
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups.append(
            {
                "params": [p],
                "lr": head_lr if name.startswith("heads.") else cfg.lr,
                "weight_decay": 0.0 if any(n in name for n in no_decay) else cfg.weight_decay,
            }
        )
    return torch.optim.AdamW(groups, lr=cfg.lr, eps=1e-8)


@torch.no_grad()
def predict(model: MultiTaskModel, loader, device: str, fp16: bool) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {t: [] for t in TASKS}
    ac = torch.autocast(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=fp16 and device != "cpu"
    )
    for batch, _ in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with ac:
            logits = model(batch["input_ids"], batch["attention_mask"])
        for t in TASKS:
            out[t].append(logits[t].float().cpu().numpy())
    return {t: np.concatenate(v) for t, v in out.items()}


def train_multitask(
    cfg: TrainConfig,
    lam: float,
    train: tuple[list[str], np.ndarray, np.ndarray],
    dev: tuple[list[str], np.ndarray, np.ndarray],
    verbose: bool = True,
) -> dict[str, Any]:
    """Fine-tune the shared model; select the epoch by mean(dev sentiment F1, dev topic F1)."""
    from transformers import AutoTokenizer

    seed_everything(cfg.seed)
    device = cfg.resolved_device()
    tok = AutoTokenizer.from_pretrained(
        MODEL_IDS[cfg.model_key], revision=MODEL_REVISIONS.get(cfg.model_key)
    )
    model = MultiTaskModel(cfg.model_key).to(device)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    collate = _collate(tok)
    tr_loader = DataLoader(
        _TwoLabelDataset(*train, tok, cfg.max_length),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=cfg.num_workers,
    )
    dv_loader = DataLoader(
        _TwoLabelDataset(*dev, tok, cfg.max_length),
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    opt = _optimizer(model, cfg)
    total = math.ceil(len(tr_loader) / cfg.grad_accum) * cfg.epochs
    sched = linear_warmup_schedule(opt, total, cfg.warmup_ratio)
    scaler = torch.amp.GradScaler(enabled=cfg.fp16 and device != "cpu")
    ce = nn.CrossEntropyLoss()
    ac: dict[str, Any] = dict(
        device_type=device.split(":")[0], dtype=torch.float16, enabled=cfg.fp16 and device != "cpu"
    )

    best: dict[str, Any] = {"score": -1.0, "epoch": -1, "state": None}
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for step, (batch, ys) in enumerate(tr_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(**ac):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = ce(logits["sentiment"], ys["sentiment"].to(device)) + lam * ce(
                    logits["topic"], ys["topic"].to(device)
                )
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * cfg.grad_accum
            if (step + 1) % cfg.grad_accum == 0 or (step + 1) == len(tr_loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        logits = predict(model, dv_loader, device, cfg.fp16)
        f1 = {
            t: M.macro_f1(np.asarray(dev[1 + i]), logits[t].argmax(1), len(label_names(t)))
            for i, t in enumerate(TASKS)
        }
        score = float(np.mean(list(f1.values())))
        history.append(
            {
                "epoch": epoch,
                "loss": running / len(tr_loader),
                **{f"dev_{t}": v for t, v in f1.items()},
                "selection_score": score,
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        if verbose:
            print(
                f"  epoch {epoch}/{cfg.epochs}  loss {running / len(tr_loader):.4f}  dev sentiment "
                f"{f1['sentiment']:.4f}  topic {f1['topic']:.4f}  mean {score:.4f}  "
                f"({history[-1]['seconds']:.0f}s)"
            )
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    model.load_state_dict(best["state"])
    return {
        "model": model,
        "tokenizer": tok,
        "loader": dv_loader,
        "device": device,
        "config": asdict(cfg) | {"lambda_topic": lam},
        "history": history,
        "best_epoch": best["epoch"],
        "train_seconds": round(time.perf_counter() - t0, 1),
        "determinism": describe_determinism(),
    }


def evaluate_joint(y: dict[str, np.ndarray], logits: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-task metric suites plus joint exact match (both predictions right)."""
    preds = {t: logits[t].argmax(1) for t in TASKS}
    per_task = {t: M.evaluate(y[t], preds[t], t, y_prob=softmax(logits[t])) for t in TASKS}
    both = (preds["sentiment"] == y["sentiment"]) & (preds["topic"] == y["topic"])
    return {"per_task": per_task, "joint_exact_match": float(both.mean())}
