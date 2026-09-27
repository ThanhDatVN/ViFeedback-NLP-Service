"""Unit tests for the export quality contract (review R3).

Everything here runs without the `onnx` package, which the reference machine's security policy
blocks (ADR-017): the release decision is a pure function of logits, and file selection is plain
file handling. The end-to-end export is exercised on Kaggle (notebooks/kaggle_train.ipynb § 4d).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from vifeedback.inference import release as RL


def _logits(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.choice(3, size=n, p=[0.45, 0.1, 0.45])
    z = rng.normal(size=(n, 3)) + 4.0 * np.eye(3)[y]
    return z, y


class TestResolveModelFile:
    def test_manifest_file_wins_over_a_stale_quantized_file(self, tmp_path) -> None:
        """The exact R3 bug: re-exporting FP32 next to an old model.quant.onnx served the old one."""
        (tmp_path / "model.quant.onnx").write_bytes(b"stale int8")
        (tmp_path / "model.opt.onnx").write_bytes(b"fresh fp32")
        manifest = {
            "model_file": "model.opt.onnx",
            "sha256": RL.sha256(tmp_path / "model.opt.onnx"),
        }
        (tmp_path / RL.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        assert RL.resolve_model_file(tmp_path).name == "model.opt.onnx"

    def test_checksum_mismatch_is_refused(self, tmp_path) -> None:
        (tmp_path / "model.opt.onnx").write_bytes(b"swapped after release")
        manifest = {"model_file": "model.opt.onnx", "sha256": "0" * 64}
        (tmp_path / RL.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="SHA-256"):
            RL.resolve_model_file(tmp_path)

    def test_missing_named_file_is_refused(self, tmp_path) -> None:
        (tmp_path / RL.MANIFEST).write_text(
            json.dumps({"model_file": "model.opt.onnx", "sha256": "x"}), encoding="utf-8"
        )
        with pytest.raises(FileNotFoundError):
            RL.resolve_model_file(tmp_path)

    def test_legacy_directories_without_a_manifest_still_load(self, tmp_path) -> None:
        (tmp_path / "model.onnx").write_bytes(b"x")
        assert RL.resolve_model_file(tmp_path).name == "model.onnx"


class TestAcceptance:
    def test_identical_fp32_passes(self) -> None:
        z, y = _logits()
        r = RL.acceptance(z, z + 1e-5, y, 3, quantized=False, single_logits=(z + 1e-5)[:32])
        assert r["passed"] and r["logits_close"]

    def test_fp32_with_divergent_logits_is_blocked_even_if_labels_agree(self) -> None:
        """logits_close used to be computed and ignored."""
        z, y = _logits()
        r = RL.acceptance(z, z * 1.01, y, 3, quantized=False)
        assert r["label_agreement"] == 1.0
        assert not r["passed"] and "FP32 logits diverge" in r["failures"][0]

    def test_int8_within_budget_passes(self) -> None:
        z, y = _logits()
        noisy = z + np.random.default_rng(1).normal(scale=0.05, size=z.shape)
        assert RL.acceptance(z, noisy, y, 3, quantized=True)["passed"]

    def test_int8_that_loses_the_minority_class_is_blocked(self) -> None:
        z, y = _logits()
        broken = z.copy()
        broken[:, 1] -= 10.0  # never predicts class 1
        r = RL.acceptance(z, broken, y, 3, quantized=True)
        assert not r["passed"]
        assert any("macro-F1 drop" in f for f in r["failures"])

    def test_batch_versus_single_mismatch_is_blocked(self) -> None:
        """A dynamic-axis export bug shows up as padded-batch logits != batch-of-one logits."""
        z, y = _logits()
        r = RL.acceptance(z, z, y, 3, quantized=False, single_logits=z[:32] + 1.0)
        assert not r["passed"]


class TestCalibrationSubset:
    def test_every_class_is_represented(self) -> None:
        y = np.array([0] * 5325 + [1] * 458 + [2] * 5643)
        idx = RL.stratified_subset(y, 300)
        counts = np.bincount(y[idx], minlength=3)
        assert counts.min() >= 10
        assert abs(len(idx) - 300) <= 3

    def test_is_deterministic(self) -> None:
        y = np.array([0, 1, 2] * 200)
        assert np.array_equal(
            RL.stratified_subset(y, 60, seed=3), RL.stratified_subset(y, 60, seed=3)
        )


class TestQuantizedBatchDependence:
    def test_int8_logits_may_move_with_the_batch_but_labels_may_not(self) -> None:
        """Dynamic INT8 scales activations per batch; a logit shift alone must not block it."""
        z, y = _logits()
        shifted_single = z[:32] + 0.8  # same argmax, different logits
        assert RL.acceptance(z, z, y, 3, quantized=True, single_logits=shifted_single)["passed"]

    def test_int8_label_flips_between_batch_and_single_are_blocked(self) -> None:
        z, y = _logits()
        flipped = z[:32].copy()
        flipped[:4] = flipped[:4][:, ::-1] * 3  # 4 of 32 change label
        r = RL.acceptance(z, z, y, 3, quantized=True, single_logits=flipped)
        assert not r["passed"]
