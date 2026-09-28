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
