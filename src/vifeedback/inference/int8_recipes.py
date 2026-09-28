"""Cycle 3 S5: careful INT8, chosen on fidelity to FP32, accepted once on validation (cycle3.yaml).

Plain dynamic INT8 cost 0.088 neutral F1 (ADR-022). The recipes here keep the parts most sensitive to
quantization in FP32: the classifier head, the last encoder layers, or the FFN output projections,
whose activations carry the outliers that per-tensor INT8 handles worst (Bondarenko et al., 2021).
Weights are quantized per channel.

The recipe is chosen by **fidelity to the FP32 graph on a stratified train subset**: how often the
INT8 graph gives the same label, especially where FP32 says neutral. That uses no gold label and no
validation data, so the single acceptance on validation stays an independent test.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

HEAD = "/classifier/"
MAX_MB = 200.0  # ROADMAP S7 minimum


def _in_layers(name: str, layers: tuple[int, ...]) -> bool:
    return any(f"/layer.{i}/" in name for i in layers)


def _ffn_output(name: str) -> bool:
    return "/output/dense/" in name and "/attention/" not in name


# recipe -> which MatMul/Gemm nodes stay FP32 (a function of the node names)
RECIPES: dict[str, Callable[[list[str]], list[str]]] = {
    "pc": lambda names: [],
    "pc-head": lambda names: [n for n in names if n.startswith(HEAD)],
    "pc-head-last2": lambda names: [
        n for n in names if n.startswith(HEAD) or _in_layers(n, (10, 11))
    ],
    "pc-head-ffnout": lambda names: [n for n in names if n.startswith(HEAD) or _ffn_output(n)],
}


def matmul_nodes(model_path: Path) -> list[str]:
    import onnx

    graph = onnx.load(str(model_path), load_external_data=False).graph
    return [n.name for n in graph.node if n.op_type in ("MatMul", "Gemm")]


def build(pre: Path, tokenizer_dir: Path, out_dir: Path, recipe: str) -> dict[str, Any]:
    """Quantize the pre-processed plain export with one recipe into out_dir/model.int8.onnx."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer_dir.resolve() != out_dir.resolve():
        for f in tokenizer_dir.iterdir():
            if f.is_file() and f.suffix in (".txt", ".json", ".codes") and f.name != "config.json":
                shutil.copy2(f, out_dir / f.name)
    exclude = RECIPES[recipe](matmul_nodes(pre))
    dst = out_dir / "model.int8.onnx"
    quantize_dynamic(
        str(pre), str(dst), weight_type=QuantType.QInt8, per_channel=True, nodes_to_exclude=exclude
    )
    return {
        "recipe": recipe,
        "path": str(dst),
        "fp32_nodes": len(exclude),
        "size_mb": dst.stat().st_size / 1e6,
    }


def fidelity(fp32_logits: np.ndarray, int8_logits: np.ndarray, neutral: int = 1) -> dict[str, Any]:
    """Agreement with the FP32 graph: overall, on FP32-neutral rows, and the logit error."""
    a, b = fp32_logits.argmax(1), int8_logits.argmax(1)
    neu = a == neutral
    diff = np.abs(fp32_logits - int8_logits)
    return {
        "label_agreement": float((a == b).mean()),
        "neutral_agreement": float((b[neu] == neutral).mean()) if neu.any() else float("nan"),
        "fp32_neutral_rows": int(neu.sum()),
        "mean_abs_logit_diff": float(diff.mean()),
        "max_abs_logit_diff": float(diff.max()),
    }


def select(candidates: dict[str, dict[str, Any]]) -> str | None:
    """Eligible: artifact <= MAX_MB. Best: neutral agreement, then overall agreement, then logit error."""
    ok = {k: v for k, v in candidates.items() if v["size_mb"] <= MAX_MB}
    if not ok:
        return None
    return max(
        ok,
        key=lambda k: (
            ok[k]["neutral_agreement"],
            ok[k]["label_agreement"],
            -ok[k]["mean_abs_logit_diff"],
        ),
    )


def acceptance(y: np.ndarray, fp32_pred: np.ndarray, int8_pred: np.ndarray) -> dict[str, Any]:
    """The declared rule's quality half: the one-sided 95% upper bound of the macro-F1 drop <= 0.005
    (paired bootstrap), and neutral F1 loss <= 0.02. Latency and size are checked separately."""
    from vifeedback.evaluation import bootstrap as B
    from vifeedback.evaluation import metrics as M

    pb = B.paired_bootstrap(y, fp32_pred, int8_pred, 3, n_resamples=10_000, seed=42, alpha=0.10)
    f_fp32 = M.evaluate(y, fp32_pred, "sentiment")["per_class"]["neutral"]["f1"]
    f_int8 = M.evaluate(y, int8_pred, "sentiment")["per_class"]["neutral"]["f1"]
    return {
        "macro_f1_drop": pb["observed_diff"],
        "macro_f1_drop_upper_95_one_sided": pb["ci_high"],
        "neutral_f1_fp32": f_fp32,
        "neutral_f1_int8": f_int8,
        "neutral_f1_loss": f_fp32 - f_int8,
        "quality_passed": bool(pb["ci_high"] <= 0.005 and f_fp32 - f_int8 <= 0.02),
    }
