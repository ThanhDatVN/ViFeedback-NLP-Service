"""Classification losses for the Phase 4 improvement ladder.

All of Tier A lives here. They are written now, in Phase 2, so that the Phase 4 sweep is a config
change rather than a code change — which is what keeps a one-axis-at-a-time ablation honest.

Gate G1 measured the motivation: tuning per-class decision priors lifted the baseline's neutral F1
from 0.353 to 0.468 with no change to the model, so the neutral failure is substantially a
decision-rule problem. These losses move that correction into training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def class_weights_from_counts(counts: np.ndarray, scheme: str = "balanced") -> torch.Tensor:
    """Per-class loss weights.

    `balanced`   — sklearn's n / (k * n_c); aggressive, and on a 4% class it is very aggressive.
    `effective`  — Cui et al. effective-number reweighting, which saturates and is gentler.
    `sqrt`       — square-root inverse frequency, the usual middle ground.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n, k = counts.sum(), len(counts)

    if scheme == "balanced":
        w = n / (k * np.maximum(counts, 1))
    elif scheme == "sqrt":
        w = np.sqrt(n / (k * np.maximum(counts, 1)))
    elif scheme == "effective":
        beta = 1.0 - 1.0 / n
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = (1.0 / np.maximum(eff, 1e-12)) * k / np.sum(1.0 / np.maximum(eff, 1e-12))
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    return torch.tensor(w / w.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss: down-weights already-easy examples by (1 - p_t)^gamma.

    Orthogonal to class weighting — it reweights by *difficulty* rather than by *frequency* — so the
    two are swept as separate axes rather than bundled.
    """

    def __init__(
        self, gamma: float = 2.0, weight: torch.Tensor | None = None, label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            target,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LogitAdjustedLoss(nn.Module):
    """Menon et al. logit adjustment: subtract tau * log(prior) from the logits during training.

    Principled for long-tailed data — it targets the balanced error rate directly rather than
    heuristically reweighting — and costs nothing at inference, unlike an ensemble.
    """

    def __init__(self, priors: np.ndarray, tau: float = 1.0, label_smoothing: float = 0.0):
        super().__init__()
        p = np.asarray(priors, dtype=np.float64)
        adj = tau * np.log(np.maximum(p / p.sum(), 1e-12))
        self.register_buffer("adjustment", torch.tensor(adj, dtype=torch.float32))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits + self.adjustment, target, label_smoothing=self.label_smoothing
        )


def rdrop_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL between two dropout passes (R-Drop).

    A consistency regularizer that needs no extra labels, which is exactly the right shape for a
    458-example minority class.
    """
    p = F.log_softmax(logits_a, dim=-1)
    q = F.log_softmax(logits_b, dim=-1)
    return 0.5 * (
        F.kl_div(p, q, log_target=True, reduction="batchmean")
        + F.kl_div(q, p, log_target=True, reduction="batchmean")
    )


def build_loss(
    name: str,
    counts: np.ndarray,
    *,
    gamma: float = 2.0,
    tau: float = 1.0,
    label_smoothing: float = 0.0,
    weight_scheme: str = "balanced",
    device: str = "cpu",
) -> nn.Module:
    """Factory keyed by the recipe name used in run ids and the registry."""
    if name in ("ce", "base"):
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "classweight":
        w = class_weights_from_counts(counts, weight_scheme).to(device)
        return nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(gamma=gamma, label_smoothing=label_smoothing).to(device)
    if name == "focal-weighted":
        w = class_weights_from_counts(counts, weight_scheme).to(device)
        return FocalLoss(gamma=gamma, weight=w, label_smoothing=label_smoothing).to(device)
    if name == "logit-adjust":
        return LogitAdjustedLoss(counts, tau=tau, label_smoothing=label_smoothing).to(device)
    raise ValueError(f"unknown loss {name!r}")


class FGM:
    """Fast Gradient Method adversarial training on the embedding layer.

    One extra forward/backward per step. Perturbing embeddings rather than tokens is what makes it
    applicable to discrete text at all, and it targets the same fragility the Phase 5 perturbation
    suites will measure.
    """

    def __init__(self, model: nn.Module, epsilon: float = 1.0, param_name: str = "word_embeddings"):
        self.model = model
        self.epsilon = epsilon
        self.param_name = param_name
        self._backup: dict[str, torch.Tensor] = {}

    def attack(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name and param.grad is not None:
                self._backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup.clear()
