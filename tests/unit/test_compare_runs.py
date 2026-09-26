"""Unit tests for registry-derived tables.

The hygiene steps are what these pin: a duplicated row or a re-run under a new id must not inflate the
seed count, and seed pairing must match seeds rather than row order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vifeedback.evaluation import compare_runs as CR
from vifeedback.evaluation.report import REGISTRY_FIELDS


def _row(run_id: str, prep: str, seed: int, f1: float, phase: str = "P3") -> dict:
    r = {k: "" for k in REGISTRY_FIELDS}
    r.update(
        run_id=run_id,
        phase=phase,
        task="sentiment",
        model="phobert-base",
        preprocessing=prep,
        recipe="base",
        seed=seed,
        split="validation",
        macro_f1=f1,
        weighted_f1=f1,
        accuracy=f1,
        balanced_accuracy=f1,
        mcc=f1,
    )
    return r


@pytest.fixture
def registry(tmp_path):
    seeds = [42, 1337, 2024, 7, 31337]
    rows = [_row(f"raw-{s}", "raw", s, 0.84 + 0.001 * i) for i, s in enumerate(seeds)]
    rows += [_row(f"seg-{s}", "seg_pyvi", s, 0.86 + 0.001 * i) for i, s in enumerate(seeds)]
    rows.append(_row("seg-42", "seg_pyvi", 42, 0.86))  # exact duplicate id
    rows.append(_row("p4-seg-1337", "seg_pyvi", 1337, 0.861, phase="P4"))  # re-run, new id
    path = tmp_path / "registry.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestHygiene:
    def test_duplicates_and_reruns_do_not_inflate_seed_counts(self, registry) -> None:
        df, h = CR.load(registry)
        assert h["duplicate_run_ids"] == ["seg-42"]
        assert h["reruns"]["n_condition_seeds"] == 1
        assert h["reruns"]["max_abs_macro_f1_disagreement"] == pytest.approx(0.0)
        table = CR.condition_table(df)
        assert set(table.n_seeds) == {5}

    def test_phase_is_not_part_of_the_condition(self, registry) -> None:
        df, _ = CR.load(registry)
        assert len(CR.condition_table(df)) == 2


class TestPairedComparison:
    def test_pairs_by_seed_not_by_row_order(self, registry) -> None:
        df, _ = CR.load(registry)
        df = df.sample(frac=1.0, random_state=0)  # scramble row order
        c = CR.paired_seed_comparison(df, {"preprocessing": "raw"}, {"preprocessing": "seg_pyvi"})
        assert c["n"] == 5 and c["wins_b"] == 5
        assert c["mean_diff"] == pytest.approx(0.02)
        assert c["ci95"][0] == pytest.approx(0.02) and c["ci95"][1] == pytest.approx(0.02)

    def test_too_few_shared_seeds_is_reported_not_tested(self, registry) -> None:
        df, _ = CR.load(registry)
        c = CR.paired_seed_comparison(
            df[df.seed != 1337][df[df.seed != 1337].seed.isin([42])],
            {"preprocessing": "raw"},
            {"preprocessing": "seg_pyvi"},
        )
        assert "p" not in c and "note" in c

    def test_fdr_marks_only_tested_comparisons(self) -> None:
        comps = CR.with_fdr([{"p": 0.001}, {"p": 0.9}, {"note": "untested"}])
        assert comps[0]["significant_bh"] is True
        assert comps[1]["significant_bh"] is False
        assert "significant_bh" not in comps[2]


def test_markdown_has_one_row_per_condition(registry) -> None:
    df, _ = CR.load(registry)
    md = CR.to_markdown(CR.condition_table(df))
    assert len(md.splitlines()) == 2 + 2
    assert "±" in md and not np.isnan(CR.condition_table(df).macro_f1_std).any()
