"""Unit tests for run-artifact serialization.

Regression coverage for a bug that destroyed a finished Phase 2 training run: `torch.__version__`
is a `TorchVersion`, a `str` subclass, and `yaml.safe_dump` refuses to represent it. The run had
already spent five GPU-minutes when the *writer* failed.

The lesson generalizes, and these tests encode it: artifact writing must never be able to discard
completed work because one metadata value has an exotic type.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from vifeedback.evaluation.report import yaml_safe


class TestYamlSafe:
    def test_str_subclasses_are_coerced_to_plain_str(self) -> None:
        class VersionLike(str):
            """Stands in for torch.torch_version.TorchVersion."""

        value = VersionLike("2.14.0+cu126")
        out = yaml_safe({"torch": value})
        assert type(out["torch"]) is str
        assert yaml.safe_dump(out)  # would raise RepresenterError without the coercion

    def test_the_exact_failure_that_lost_a_run(self) -> None:
        torch = pytest.importorskip("torch")
        from vifeedback.training.seeding import describe_determinism

        config = {"determinism": describe_determinism(), "version": torch.__version__}
        with pytest.raises(yaml.representer.RepresenterError):
            yaml.safe_dump({"version": torch.__version__})
        assert yaml.safe_dump(yaml_safe(config))

    @pytest.mark.parametrize(
        ("value", "expected_type"),
        [
            (np.int64(3), int),
            (np.float32(1.5), float),
            (np.bool_(True), bool),
            (3, int),
            (1.5, float),
            ("x", str),
            (True, bool),
            (None, type(None)),
        ],
    )
    def test_scalars_become_plain_python(self, value, expected_type) -> None:
        assert type(yaml_safe(value)) is expected_type

    def test_arrays_and_tuples_become_lists(self) -> None:
        assert yaml_safe(np.array([1, 2, 3])) == [1, 2, 3]
        assert yaml_safe((1, 2)) == [1, 2]

    def test_nested_structures_are_walked(self) -> None:
        nested = {"a": {"b": [np.int64(1), {"c": np.float64(2.0)}]}}
        out = yaml_safe(nested)
        assert out == {"a": {"b": [1, {"c": 2.0}]}}
        assert yaml.safe_dump(out)

    def test_unknown_objects_degrade_to_str_instead_of_raising(self) -> None:
        """The point of the whole function: never lose a finished run over metadata."""

        class Exotic:
            def __str__(self) -> str:
                return "exotic"

        assert yaml_safe({"k": Exotic()}) == {"k": "exotic"}
        assert yaml.safe_dump(yaml_safe({"k": Exotic()}))

    def test_bool_is_not_downgraded_to_int(self) -> None:
        """bool is a subclass of int; mishandling it turns `fp16: true` into `fp16: 1`."""
        out = yaml_safe({"fp16": True, "epochs": 4})
        assert out["fp16"] is True
        assert type(out["epochs"]) is int

    def test_unicode_config_values_survive(self) -> None:
        out = yaml_safe({"note": "giảng viên nhiệt tình"})
        assert yaml.safe_load(yaml.safe_dump(out, allow_unicode=True))["note"] == (
            "giảng viên nhiệt tình"
        )


class TestRunIdentity:
    """Regression cover for R11: two different configs must not share a run id, and a run
    directory must not be silently overwritten by a different configuration."""

    def _cfg(self, **kw):
        from vifeedback.training.trainer import TrainConfig

        base = dict(
            task="sentiment",
            model_key="phobert-base",
            preprocessing="seg_pyvi",
            recipe="base",
            seed=42,
        )
        base.update(kw)
        return TrainConfig(**base)

    def test_lr_epochs_and_max_length_change_the_run_id(self):
        """The exact collision found in review: these three fields were absent from the id."""
        base = dict(lr=2e-5, epochs=4, max_length=96)
        a = self._cfg(**base)
        for changed in (dict(lr=5e-5), dict(epochs=10), dict(max_length=256)):
            b = self._cfg(**{**base, **changed})
            assert a.run_id("val") != b.run_id("val"), f"{changed} did not change the run id"

    def test_cosmetic_fields_do_not_change_the_run_id(self):
        """Identity must track what changes the result, not annotations."""
        a = self._cfg(notes="first attempt")
        b = self._cfg(notes="second attempt")
        assert a.config_hash() == b.config_hash()

    def test_config_hash_is_stable_across_processes(self):
        """A hash seeded by PYTHONHASHSEED would make run ids irreproducible."""
        import subprocess
        import sys

        script = (
            "from vifeedback.training.trainer import TrainConfig;"
            "print(TrainConfig(task='sentiment', seed=42, lr=2e-5).config_hash())"
        )
        outs = {
            subprocess.run(
                [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
            ).stdout.strip()
            for _ in range(2)
        }
        assert len(outs) == 1 and outs != {""}, f"config_hash is not stable: {outs}"

    def test_save_run_refuses_to_overwrite_a_different_config(self, tmp_path, monkeypatch):
        import numpy as np
        import pytest as _pytest

        from vifeedback import paths
        from vifeedback.evaluation import metrics as M
        from vifeedback.evaluation import report as R

        monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
        monkeypatch.setattr(paths, "RESULTS", tmp_path)
        monkeypatch.setattr(paths, "REGISTRY", tmp_path / "registry.csv")
        monkeypatch.setattr(paths, "TEST_EVAL_LOG", tmp_path / "test_evaluations.log")

        y = np.array([0, 1, 2, 0, 1, 2])
        m = M.evaluate(y, y, "sentiment")
        cfg = {"task": "sentiment", "split": "validation", "lr": 2e-5, "epochs": 4}

        R.save_run("demo", m, config=cfg, y_true=y, y_pred=y, figure=False)
        # Identical config: a legitimate repeat, must succeed.
        R.save_run("demo", m, config=dict(cfg), y_true=y, y_pred=y, figure=False)
        # Different config under the same id: must refuse.
        with _pytest.raises(FileExistsError, match="DIFFERENT configuration"):
            R.save_run("demo", m, config={**cfg, "lr": 5e-5}, y_true=y, y_pred=y, figure=False)
