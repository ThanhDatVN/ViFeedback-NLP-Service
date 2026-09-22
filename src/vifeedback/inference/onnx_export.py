"""ONNX export and quantization — Phase 6.

The ladder in docs/ROADMAP.md, with one correction already forced by measurement: two
configuration-only changes (`max_length` 96 from Gate G0, plus dynamic padding) already deliver
**3.49x** on CPU p95, from 177.5 ms to 50.8 ms. So Phase 6's question is no longer "can we reach
60 ms" — that is already met — but "does ONNX or INT8 add anything on top, on a CPU with no
AVX512-VNNI?" The answer may legitimately be no, and that is still a result.

Every exported artifact is checked against the PyTorch model before it is allowed to ship:
logit parity for FP32, label agreement plus a macro-F1 budget for INT8.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

OPSET = 17


def export_fp32(model, tokenizer, out_dir: Path, max_length: int = 96) -> Path:
    """Export to ONNX with dynamic axes on batch *and* sequence.

    Both axes must be dynamic or the graph silently re-introduces fixed-length padding, which is
    the very thing that cost 3.49x in the first place.
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "model.onnx"

    model.eval()
    dummy = tokenizer(
        "giảng_viên nhiệt_tình với sinh_viên .",
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = tuple(dummy[k] for k in ("input_ids", "attention_mask"))

    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
            },
            opset_version=OPSET,
            do_constant_folding=True,
        )
    tokenizer.save_pretrained(out_dir)
    return path


def optimize_graph(src: Path, dst: Path) -> Path:
    """ORT graph-level optimization (node fusion, constant folding). Lossless."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(dst)
    ort.InferenceSession(str(src), opts, providers=["CPUExecutionProvider"])
    return dst


def quantize_dynamic_int8(src: Path, dst: Path) -> Path:
    """Dynamic INT8. **May be slower than FP32** on a CPU without AVX512-VNNI — hypothesis H3.

    The reference machine (AMD Ryzen 5 6600H) has AVX2 and no VNNI, so a regression here is the
    predicted outcome rather than a failure. Measure it and report it either way.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def quantize_static_int8(
    src: Path, dst: Path, calibration_texts: list[str], tokenizer, max_length: int = 96
) -> Path:
    """Static INT8 with real calibration data.

    Calibration texts come from **dev**, never test. The ADR-015 standing rule applies: a parameter
    fitted on the evaluation set inflates the evaluation.
    """
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize_static

    class _Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.data = iter(
                [
                    {
                        "input_ids": enc["input_ids"].astype(np.int64),
                        "attention_mask": enc["attention_mask"].astype(np.int64),
                    }
                    for enc in (
                        tokenizer(t, return_tensors="np", truncation=True, max_length=max_length)
                        for t in calibration_texts
                    )
                ]
            )

        def get_next(self):
            return next(self.data, None)

    quantize_static(str(src), str(dst), _Reader(), weight_type=QuantType.QInt8)
    return dst


class OnnxClassifier:
    """Thread-configurable ORT session with dynamic padding.

    Dynamic padding is not a detail: the median UIT-VSFC sentence is 14 subwords against a 96-token
    `max_length`, so padding to `max_length` does roughly 7x the arithmetic for no benefit.
    """

    def __init__(self, model_dir: Path, intra_op_threads: int | None = None, max_length: int = 96):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        if intra_op_threads:
            opts.intra_op_num_threads = intra_op_threads
            opts.inter_op_num_threads = 1
        candidates = ["model.quant.onnx", "model.opt.onnx", "model.onnx"]
        self.path = next((model_dir / c for c in candidates if (model_dir / c).exists()), None)
        if self.path is None:
            raise FileNotFoundError(f"no .onnx found in {model_dir}")

        self.session = ort.InferenceSession(
            str(self.path), opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_length = max_length

    def logits(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, return_tensors="np", padding=True, truncation=True, max_length=self.max_length
        )
        out = self.session.run(
            ["logits"],
            {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            },
        )
        return out[0]

    def predict(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        z = self.logits(texts)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        return prob.argmax(axis=1), prob

    @property
    def size_mb(self) -> float:
        return round(self.path.stat().st_size / 1e6, 2)


def verify_parity(
    torch_model, onnx_clf: OnnxClassifier, texts: list[str], atol: float = 1e-3
) -> dict[str, Any]:
    """Assert the exported graph still computes the same function.

    FP32 is held to logit closeness; INT8 cannot be, so it is held to label agreement instead. An
    export that silently changed behaviour would otherwise reach production looking fine — this is
    the check that catches it.
    """
    import torch

    torch_model.eval()
    enc = onnx_clf.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=onnx_clf.max_length
    )
    with torch.no_grad():
        ref = torch_model(**enc).logits.numpy()
    got = onnx_clf.logits(texts)

    return {
        "max_abs_logit_diff": float(np.abs(ref - got).max()),
        "logits_close": bool(np.allclose(ref, got, atol=atol)),
        "label_agreement": float((ref.argmax(1) == got.argmax(1)).mean()),
        "n": len(texts),
    }
