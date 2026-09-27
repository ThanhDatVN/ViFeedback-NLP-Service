"""`vifeedback results merge`: the safe path for bringing another machine's results home.

Written after the Kaggle output was extracted over the repository and results/, which held
uncommitted local runs, was lost. Pinned: it refuses to run without a local registry, appends only new
run ids, and never overwrites an existing run directory.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from vifeedback import cli, paths
from vifeedback.evaluation.report import REGISTRY_FIELDS


def _registry(path, run_ids):
    pd.DataFrame(
        [{k: "" for k in REGISTRY_FIELDS} | {"run_id": r, "macro_f1": 0.8} for r in run_ids]
    ).to_csv(path, index=False)


@pytest.fixture
def local(tmp_path, monkeypatch):
    res = tmp_path / "results"
    (res / "runs").mkdir(parents=True)
    monkeypatch.setattr(paths, "RESULTS", res)
    monkeypatch.setattr(paths, "RUNS", res / "runs")
    monkeypatch.setattr(paths, "REGISTRY", res / "registry.csv")
    return res


def _other(tmp_path, run_ids):
    src = tmp_path / "kaggle_results"
    (src / "runs").mkdir(parents=True)
    _registry(src / "registry.csv", run_ids)
    for r in run_ids:
        (src / "runs" / r).mkdir()
        (src / "runs" / r / "metrics.json").write_text('{"from": "kaggle"}', encoding="utf-8")
    return src


def test_refuses_without_a_local_registry(local, tmp_path) -> None:
    src = _other(tmp_path, ["k1"])
    r = CliRunner().invoke(cli.app, ["results", "merge", str(src)])
    assert r.exit_code != 0
    assert not paths.REGISTRY.exists()


def test_appends_new_ids_and_never_overwrites_run_dirs(local, tmp_path) -> None:
    _registry(paths.REGISTRY, ["a", "shared"])
    (paths.RUNS / "shared").mkdir()
    (paths.RUNS / "shared" / "metrics.json").write_text('{"from": "local"}', encoding="utf-8")
    src = _other(tmp_path, ["shared", "k1"])

    r = CliRunner().invoke(cli.app, ["results", "merge", str(src)])
    assert r.exit_code == 0, r.output
    reg = pd.read_csv(paths.REGISTRY)
    assert sorted(reg.run_id) == ["a", "k1", "shared"]
    assert (paths.RUNS / "shared" / "metrics.json").read_text(
        encoding="utf-8"
    ) == '{"from": "local"}'
    assert (paths.RUNS / "k1" / "metrics.json").exists()
