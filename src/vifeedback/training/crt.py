"""Classifier re-training (cRT) — Study A step 4, "balanced head retraining".

Kang et al. (ICLR 2020, *Decoupling Representation and Classifier for Long-Tailed Recognition*)
separate two things a standard fine-tune entangles: learning the representation, and placing the
classifier on top of it. cRT keeps the encoder from an ordinary cross-entropy run, re-initializes the
classification head, and retrains only the head with **class-balanced sampling**.

Why it is the right control here: it tests whether the minority-class gap is in the *classifier*
(fixable without touching the representation) or in the *representation* (not fixable this way).
Study A's post-hoc bias test found the boundary nearly optimal already. cRT asks a stronger version
of the same question, because it can re-shape the head rather than just shift one logit.

The encoder is frozen and in eval mode, so its [CLS] features are deterministic. They are computed
once and the head is trained on them directly: seconds, not minutes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


@torch.no_grad()
def encoder_features(
    model, tokenizer, texts: list[str], max_length: int, device: str, batch_size: int = 128
) -> torch.Tensor:
    """Final-layer hidden states at position 0 — exactly what RobertaClassificationHead reads."""
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    from vifeedback.training.trainer import TextDataset

    encoder = getattr(model, model.base_model_prefix)
    encoder.eval()
    ds = TextDataset(texts, np.zeros(len(texts), dtype=np.int64), tokenizer, max_length)
    collate = DataCollatorWithPadding(tokenizer, padding="longest", return_tensors="pt")
    feats = []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate):
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        feats.append(encoder(**batch).last_hidden_state[:, 0, :].float().cpu())
    return torch.cat(feats)


def retrain_head(
    model,
    feats_tr: torch.Tensor,
    y_tr: np.ndarray,
    feats_dv: torch.Tensor,
    y_dv: np.ndarray,
    *,
    task: str,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Re-initialize `model.classifier` and train it on frozen features with balanced sampling.

    The epoch is chosen by dev macro-F1, the same selection rule as every other run. The model is
    modified in place and returned inside the result.
    """
    from vifeedback.evaluation.metrics import macro_f1

    torch.manual_seed(seed)
    k = int(np.max(y_tr)) + 1
    head = type(model.classifier)(model.config).to(device)
    head.train()
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    y_tr_t = torch.as_tensor(np.asarray(y_tr), dtype=torch.long)
    counts = np.bincount(np.asarray(y_tr), minlength=k)
    # Balanced sampling: each class is drawn with equal probability, examples uniformly within it.
    weights = torch.as_tensor(1.0 / counts[np.asarray(y_tr)], dtype=torch.double)
    g = torch.Generator().manual_seed(seed)
    loss_fn = torch.nn.CrossEntropyLoss()

    def dev_f1() -> float:
        head.eval()
        with torch.no_grad():
            logits = head(feats_dv.to(device)[:, None, :]).float().cpu().numpy()
        head.train()
        return macro_f1(np.asarray(y_dv), logits.argmax(1), k)

    best: dict[str, Any] = {"f1": -1.0, "epoch": 0, "state": None}
    history = []
    for epoch in range(1, epochs + 1):
        idx = torch.multinomial(weights, num_samples=len(y_tr_t), replacement=True, generator=g)
        for s in range(0, len(idx), batch_size):
            b = idx[s : s + batch_size]
            x = feats_tr[b].to(device)[:, None, :]  # the head reads features[:, 0, :]
            loss = loss_fn(head(x), y_tr_t[b].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        f1 = dev_f1()
        history.append({"epoch": epoch, "dev_macro_f1": f1})
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "epoch": epoch,
                "state": {k_: v.detach().clone() for k_, v in head.state_dict().items()},
            }

    head.load_state_dict(best["state"])
    head.eval()
    model.classifier = head
    return {
        "model": model,
        "best_epoch": best["epoch"],
        "best_dev_macro_f1": best["f1"],
        "history": history,
        "task": task,
    }
