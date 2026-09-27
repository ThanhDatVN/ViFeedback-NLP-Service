"""The rule that decides which latency numbers may be reported (review R10, § 8.4).

Pinned on the exact case the first reference-machine run produced: the configuration measured first
was not yet in a steady state (p95 126.6 ms, spread 10.3%), its reversed-order repeat was (84.1 ms,
1.7%). A speedup computed against the unsteady number would be inflated by 1.5x.
"""

from __future__ import annotations

import pytest

from vifeedback.inference.benchmark import summarize_passes


def _r(p95: float, spread: float, p50: float | None = None) -> dict:
    return {
        "p50_ms": p50 or p95 * 0.8,
        "p95_ms": p95,
        "p99_ms": p95 * 1.1,
        "p95_spread_across_repeats": spread,
        "throttling_suspected": spread > 0.10,
        "texts_per_s_b32": 10.0,
    }


def test_unsteady_pass_is_excluded_from_the_baseline() -> None:
    passes = [
        {"L0": _r(126.6, 0.103), "L3": _r(21.3, 0.163)},
        {"L0": _r(84.1, 0.017), "L3": _r(23.9, 0.044)},
    ]
    s = summarize_passes(passes, baseline="L0")
    assert s["L0"]["p95_ms"] == pytest.approx(84.1)
    assert s["L0"]["basis"].startswith("pass 2 only")
    assert s["L3"]["speedup_p95_vs_baseline"] == pytest.approx(84.1 / 23.9, rel=1e-2)


def test_two_agreeing_steady_passes_are_averaged() -> None:
    passes = [{"L1": _r(45.5, 0.05)}, {"L1": _r(45.8, 0.07)}]
    s = summarize_passes(passes, baseline="L1")
    assert s["L1"]["basis"] == "mean of both steady passes"
    assert s["L1"]["p95_ms"] == pytest.approx(45.65)


def test_no_steady_pass_is_not_reportable_and_gets_no_speedup() -> None:
    passes = [{"L0": _r(80, 0.01), "X": _r(30, 0.4)}, {"L0": _r(81, 0.01), "X": _r(40, 0.3)}]
    s = summarize_passes(passes, baseline="L0")
    assert s["X"]["reportable"] is False
    assert "speedup_p95_vs_baseline" not in s["X"]
