"""CPU latency harness — the spec in docs/EVALUATION_PROTOCOL.md § Latency harness, in code.

The rules that separate this from a `time.time()` call in a notebook:

* 200 warmup iterations, discarded — lazy init, allocator warmup and ORT arena growth dominate an
  unwarmed measurement;
* inputs sampled from the **real test-length distribution**, never a fixed-length dummy. Benchmarking
  on 96-token padded dummies measures a workload the service never sees, and it is precisely the
  configuration that flatters quantization;
* 5 repetitions, median of the five p95s, and a >10% disagreement between repetitions means the
  machine was thermally throttling — re-run rather than report it;
* **model-only and end-to-end reported separately.** Reporting only model-only is the standard way
  this number gets quietly inflated.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

import numpy as np

WARMUP = 200
TIMED = 1000
REPEATS = 5
THROTTLE_TOLERANCE = 0.10


def _percentiles(t: np.ndarray) -> dict[str, float]:
    return {
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "p99_ms": round(float(np.percentile(t, 99)), 3),
        "mean_ms": round(float(t.mean()), 3),
    }


def time_callable(
    fn: Callable[[list[str]], Any],
    texts: list[str],
    *,
    warmup: int = WARMUP,
    timed: int = TIMED,
    repeats: int = REPEATS,
    seed: int = 42,
) -> dict[str, Any]:
    """Time `fn` on single inputs sampled from `texts`. Returns the median-of-repeats percentiles."""
    rng = np.random.default_rng(seed)
    sample = [texts[i] for i in rng.integers(0, len(texts), size=warmup + timed)]

    for t in sample[:warmup]:
        fn([t])

    runs = []
    for _ in range(repeats):
        gc.collect()
        lat = np.empty(timed, dtype=np.float64)
        for i, t in enumerate(sample[warmup:]):
            t0 = time.perf_counter_ns()
            fn([t])
            lat[i] = (time.perf_counter_ns() - t0) / 1e6
        runs.append(lat)

    p95s = np.array([np.percentile(r, 95) for r in runs])
    spread = float((p95s.max() - p95s.min()) / max(p95s.min(), 1e-9))
    median_run = runs[int(np.argsort(p95s)[len(p95s) // 2])]

    out = _percentiles(median_run)
    out.update(
        {
            "repeats": repeats,
            "timed_per_repeat": timed,
            "p95_spread_across_repeats": round(spread, 4),
            "throttling_suspected": spread > THROTTLE_TOLERANCE,
        }
    )
    if out["throttling_suspected"]:
        out["warning"] = (
            f"p95 varied {spread:.1%} across repeats (>{THROTTLE_TOLERANCE:.0%}). The machine was "
            "probably throttling; this number measures thermal state, not the model. Re-run cool."
        )
    return out


def throughput(
    fn: Callable[[list[str]], Any],
    texts: list[str],
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    n_batches: int = 30,
    seed: int = 42,
) -> dict[str, float]:
    """Requests per second at several batch sizes. Batch throughput and single-request p95 are
    different questions and a serving decision needs both."""
    rng = np.random.default_rng(seed)
    out = {}
    for bs in batch_sizes:
        batches = [
            [texts[i] for i in rng.integers(0, len(texts), size=bs)] for _ in range(n_batches)
        ]
        for b in batches[:3]:
            fn(b)
        t0 = time.perf_counter()
        for b in batches:
            fn(b)
        elapsed = time.perf_counter() - t0
        out[f"batch{bs}_req_per_s"] = round(n_batches * bs / elapsed, 1)
    return out


def benchmark_pipeline(
    *,
    model_fn: Callable[[list[str]], Any],
    texts: list[str],
    preprocess_fn: Callable[[list[str]], list[str]] | None = None,
    label: str = "",
    size_mb: float | None = None,
    **kw,
) -> dict[str, Any]:
    """Full report: model-only, preprocessing-only, and end-to-end.

    Separating the three is what let Phase 3 conclude that segmentation costs 1.2% of end-to-end
    p95 rather than the majority share the hypothesis predicted.
    """
    from vifeedback import env

    report: dict[str, Any] = {
        "label": label,
        "size_mb": size_mb,
        "model_only": time_callable(model_fn, texts, **kw),
        "throughput": throughput(model_fn, texts),
        "environment": env.capture(),
    }

    if preprocess_fn is not None:
        report["preprocess_only"] = time_callable(preprocess_fn, texts, **kw)

        def end_to_end(batch: list[str]):
            return model_fn(preprocess_fn(batch))

        report["end_to_end"] = time_callable(end_to_end, texts, **kw)
        pre = report["preprocess_only"]["p95_ms"]
        e2e = report["end_to_end"]["p95_ms"]
        report["preprocess_share_of_p95"] = round(pre / e2e, 4) if e2e else None

    return report


def compare(reports: list[dict[str, Any]], baseline_label: str | None = None) -> Any:
    """Ladder table with speedups relative to the first (or named) configuration."""
    import pandas as pd

    rows = []
    for r in reports:
        e2e = r.get("end_to_end", r["model_only"])
        rows.append(
            {
                "configuration": r["label"],
                "size_MB": r.get("size_mb"),
                "p50_ms": r["model_only"]["p50_ms"],
                "p95_ms": r["model_only"]["p95_ms"],
                "p99_ms": r["model_only"]["p99_ms"],
                "e2e_p95_ms": e2e["p95_ms"],
                "req_per_s_b32": r["throughput"].get("batch32_req_per_s"),
                "throttled": r["model_only"]["throttling_suspected"],
            }
        )
    df = pd.DataFrame(rows)
    ref = (
        df[df.configuration == baseline_label].p95_ms.iloc[0]
        if baseline_label and (df.configuration == baseline_label).any()
        else df.p95_ms.iloc[0]
    )
    df["speedup_vs_baseline"] = (ref / df.p95_ms).round(2)
    return df
