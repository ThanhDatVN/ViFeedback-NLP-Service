"""The model card renders from committed results alone (no model files needed)."""

from __future__ import annotations

from vifeedback.inference import publish as P

MANIFEST = {
    "checkpoint": "models/p9-sent-phobert-base-seg_pyvi-aug-diac-teen-s42-599cf21f-ckp",
    "model_file": "model.opt.onnx",
    "max_length": 96,
    "labels": ["negative", "neutral", "positive"],
    "acceptance": {"max_abs_logit_diff": 1.7e-5},
    "_files": {"model.opt.onnx": "c" * 64},
}


def test_card_reads_every_number_from_results():
    ev = P.evidence(MANIFEST)
    card = P.model_card("someone/vifeedback-sentiment", MANIFEST, ev)
    assert card.startswith("---\nlanguage: vi\n")
    assert "0.8644" in card  # validation macro-F1 of the released checkpoint
    assert "0.8371" in card  # its single test evaluation (closing gate)
    assert "| `unaccented_typed` | 50 |" in card
    assert "{" not in card.split("```")[0]  # no unrendered placeholder before the code sample


def test_evidence_matches_the_checkpoint_not_the_other_finalist():
    ev = P.evidence(MANIFEST)
    assert ev["run_id"].startswith("p9-")
    assert ev["test"]["robustness_test"]["nodiacritic"]["macro_f1"] > 0.6  # CE scores 0.27
