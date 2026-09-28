"""NEXT_PLAN v5 F1: PhobertBPE gives transformers' PhobertTokenizer inputs exactly.

    python scripts/check_phobert_tokenizer.py [model_dir]

Encodes, in padded batches of 64 with max_length 96, every UIT-VSFC sentence as the model saw it
(the seg_pyvi variant, all splits), every NEU-ESC post as the service would feed it
(serving.pipeline.prepare), and a list of edge cases, with both tokenizers, and compares input_ids
and attention_mask exactly. Writes counts only (no text) to
results/studies/engineering/phobert_tokenizer.json. Exit code 1 on any difference.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EDGE = [
    "",
    " ",
    "a",
    "<s>",
    "thầy <mask> dạy",
    "giảng_viên</s>nhiệt_tình",
    "<unk> <pad> </s>",
    "dòng một\ndòng hai",
    "tab\tngăn\tcách",
    "  nhiều   khoảng   trắng  ",
    "😀 emoji và ký tự lạ ⚡",
    "rất_rất_rất " * 80,  # longer than max_length: truncation
    "COVID-19 , 2026 ; http://example.com @user #tag",
]


def main() -> int:
    from transformers import AutoTokenizer

    from vifeedback.inference.phobert_tokenizer import PhobertBPE
    from vifeedback.preprocess.variants import load_variant

    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "serve" / "sentiment"
    hf = AutoTokenizer.from_pretrained(model_dir)
    ours = PhobertBPE(model_dir)

    groups = {
        "uit_vsfc_seg_pyvi": [
            t for s in ("train", "validation", "test") for t in load_variant("seg_pyvi", s).sentence
        ],
        "edge_cases": EDGE,
    }
    try:
        from vifeedback.evaluation import external as X
        from vifeedback.training.domain import serving_transform

        transform = serving_transform()
        groups["neu_esc_served"] = transform(
            [t for s in ("train", "validation", "test") for t in X.load_neu_esc(s).text]
        )
    except FileNotFoundError:
        print("NEU-ESC is not fetched; checking UIT-VSFC and the edge cases only")

    out: dict = {
        "question": "does PhobertBPE reproduce PhobertTokenizer exactly? (NEXT_PLAN v5 F1)",
        "model_dir": model_dir.name,
    }
    bad_total = 0
    for name, texts in groups.items():
        bad = 0
        for i in range(0, len(texts), 64):
            batch = list(texts[i : i + 64])
            a = hf(batch, return_tensors="np", padding=True, truncation=True, max_length=96)
            b = ours(batch, return_tensors="np", padding=True, truncation=True, max_length=96)
            same = (
                a["input_ids"].shape == b["input_ids"].shape
                and np.array_equal(a["input_ids"], b["input_ids"])
                and np.array_equal(a["attention_mask"], b["attention_mask"])
            )
            if not same:
                bad += 1
                if name == "edge_cases":
                    for t in batch:
                        x = hf(
                            [t], return_tensors="np", padding=True, truncation=True, max_length=96
                        )["input_ids"]
                        y = ours(
                            [t], return_tensors="np", padding=True, truncation=True, max_length=96
                        )["input_ids"]
                        if not np.array_equal(x, y):
                            print(f"  edge case differs: {t[:40]!r}")
        out[name] = {"texts": len(texts), "batches": -(-len(texts) // 64), "batches_differing": bad}
        bad_total += bad
    out["identical"] = bad_total == 0
    dst = ROOT / "results" / "studies" / "engineering" / "phobert_tokenizer.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
