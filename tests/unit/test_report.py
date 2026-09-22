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
