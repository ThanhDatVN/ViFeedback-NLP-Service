"""Deterministic seeding.

Covers Python, NumPy, torch (CPU and CUDA) and DataLoader workers. Full determinism on GPU is not
always attainable — some cuDNN kernels have no deterministic implementation — so
`describe_determinism()` reports what was actually achieved rather than letting the code imply more
than it delivers (docs/EVALUATION_PROTOCOL.md § 2).
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a distinct but reproducible seed."""
    import torch

    seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(seed)
    random.seed(seed)


def describe_determinism() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch": None}
    return {
        # str() matters: torch.__version__ is a TorchVersion (a str subclass) that
        # yaml.safe_dump refuses to represent, which killed a whole Phase 2 sweep.
        "torch": str(torch.__version__),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda": torch.cuda.is_available(),
        "note": (
            "Non-deterministic cuDNN kernels may remain; run-to-run variation is reported as seed "
            "std over 5 seeds rather than claimed away."
        ),
    }
