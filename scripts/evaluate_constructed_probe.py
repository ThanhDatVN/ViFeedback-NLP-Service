"""Exploratory score on authored course-forum examples; never evidence for H12 transfer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from vifeedback.evaluation import metrics
from vifeedback.inference.onnx_export import OnnxClassifier
from vifeedback.preprocess.segment import get_segmenter
from vifeedback.serving import pipeline

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "data" / "probes" / "constructed_course_forum_v1.csv"
LABELS = ("negative", "neutral", "positive")


def evaluate(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    classifier = OnnxClassifier(model_dir, max_length=manifest.get("max_length", 96))
    restorer = (
        pipeline.load_restorer(model_dir, manifest["restorer"])
        if manifest.get("restorer")
        else None
    )
    scope = pipeline.load_scope(model_dir, manifest["scope"]) if manifest.get("scope") else None
    with PROBE.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 60 and len({r["id"] for r in rows}) == 60
    model_input = pipeline.prepare([r["text"] for r in rows], restorer, get_segmenter("pyvi"))
    labels, probabilities, scores = pipeline.score(
        classifier,
        model_input,
        scope=scope,
        temperature=manifest.get("temperature", {}).get("value"),
    )
    for i, row in enumerate(rows):
        row["predicted_sentiment"] = LABELS[int(labels[i])]
        row["confidence"] = float(probabilities[i].max())
        row["scope_score"] = float(scores[i]) if scores is not None else None
        row["predicted_in_scope"] = bool(scores[i] >= scope.threshold) if scope else None
    used = [r for r in rows if r["expected_sentiment"] in LABELS]
    y = np.array([LABELS.index(r["expected_sentiment"]) for r in used])
    p = np.array([LABELS.index(r["predicted_sentiment"]) for r in used])
    evaluation = metrics.evaluate(y, p, "sentiment")
    groups = sorted({r["group"] for r in rows})
    group_results = {
        group: {
            "n": sum(r["group"] == group for r in rows),
            "label_accuracy": float(
                np.mean(
                    [
                        r["expected_sentiment"] == r["predicted_sentiment"]
                        for r in rows
                        if r["group"] == group
                    ]
                )
            )
            if group != "off_topic"
            else None,
        }
        for group in groups
    }
    if scope:
        for group in groups:
            group_results[group]["scope_correct"] = float(
                np.mean(
                    [
                        r["predicted_in_scope"] == (r["expected_in_scope"] == "1")
                        for r in rows
                        if r["group"] == group
                    ]
                )
            )
    result = {
        "probe": str(PROBE.relative_to(ROOT)).replace("\\", "/"),
        "probe_sha256": hashlib.sha256(PROBE.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((model_dir / "manifest.json").read_bytes()).hexdigest(),
        "model_sha256": manifest["sha256"],
        "restorer_sha256": manifest.get("restorer", {}).get("sha256"),
        "scope_sha256": manifest.get("scope", {}).get("sha256"),
        "temperature": manifest.get("temperature", {}).get("value"),
        "scope_threshold": scope.threshold if scope else None,
        "n": len(rows),
        "in_scope_labelled": len(used),
        "macro_f1": evaluation["macro_f1"],
        "per_class_f1": {name: evaluation["per_class"][name]["f1"] for name in LABELS},
        "groups": group_results,
        "rows": [{key: row[key] for key in row if key != "text"} for row in rows],
        "interpretation": "Exploratory authored examples; not a real third-institution sample or H12 confirmation.",
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "serve" / "sentiment")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.model_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"{result['in_scope_labelled']} labelled examples: macro-F1 {result['macro_f1']:.3f}; "
        f"12 off-topic scope probes, results in {args.out}"
    )
