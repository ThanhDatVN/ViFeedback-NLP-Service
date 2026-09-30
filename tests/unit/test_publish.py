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


def test_stale_hub_files_are_the_ones_the_new_bundle_lacks(tmp_path):
    """An update deletes what the release no longer ships (ood.npz after ADR-034), nothing else."""
    for name in ("model.opt.onnx", "scope.npz", "README.md", "pytorch/config.json"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"x")
    remote = [".gitattributes", "README.md", "model.opt.onnx", "ood.npz", "pytorch/config.json"]
    assert P.stale_files(remote, tmp_path) == ["ood.npz"]


def test_basename_splits_windows_paths_on_any_os():
    """The closing gate recorded a Windows path; on Linux, Path().name would not split it."""
    assert P._basename(r"D:\GitHub\repo\models\p9-x-ckp") == "p9-x-ckp"
    assert P._basename("models/p9-x-ckp") == "p9-x-ckp"


def test_student_card_reads_the_h11_record_and_states_the_test_gap():
    """ADR-039/040: the distilled student's card, from committed results only."""
    import json

    from vifeedback import paths

    record = (
        paths.RESULTS / "studies" / "export" / "laptop_fp16_student_restorer_scope_manifest.json"
    )
    manifest = json.loads(record.read_text(encoding="utf-8"))
    manifest["_files"] = {"model.fp16.onnx": "d" * 64}
    ev = P.evidence(manifest)
    card = P.model_card("someone/vifeedback-student", manifest, ev)
    assert "license: cc-by-nc-sa-4.0" in card and "185.1 MB" in card
    assert "0.8175" in card  # seed-42 test macro-F1 (H11 closing gate)
    assert "lower in 5 of 5 seeds" in card  # the test gap is stated, not hidden
    assert "as accurate as its teacher" not in card
    assert "{" not in card.split("```")[0]


def test_h10b_card_reads_the_h10b_record_and_the_five_seed_test_gap():
    """ADR-042/043: the served H10b model's card, from committed results only."""
    import json

    from vifeedback import paths

    record = paths.RESULTS / "studies" / "export" / "laptop_fp32_h10b_temperature_manifest.json"
    manifest = json.loads(record.read_text(encoding="utf-8"))
    manifest["_files"] = {"model.opt.onnx": "e" * 64}
    ev = P.evidence(manifest)
    card = P.model_card("someone/vifeedback-sentiment", manifest, ev)
    assert "license: cc-by-nc-sa-4.0" in card  # ViLexNorm pairs in training
    assert "0.8617" in card and "0.8208" in card  # validation; the single seed-42 test
    assert "-0.0059" in card and "not distinguishable" in card  # five seeds, stated
    assert "T = 1.35" in card
    assert "0.916" in card  # challenge v1 through the served pipeline
    assert "{" not in card.split("```")[0]
