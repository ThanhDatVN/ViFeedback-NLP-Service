"""Materialized preprocessing variants — Phase 3.

Each variant is written once to `data/processed/<name>/` and reused by every run that references it.
Three reasons this matters rather than segmenting on the fly:

* preprocessing cost never contaminates a training-time measurement;
* every seed of every condition sees byte-identical input, so a difference between conditions is
  attributable to the condition;
* VnCoreNLP's JVM is touched once at build time instead of inside a training loop.

Conditions map to docs/ROADMAP.md Phase 3. Note that `raw` is condition **P0**, which Phase 2 already
ran at 5 seeds — so Phase 3 only needs to add the segmented conditions.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd

from vifeedback import paths
from vifeedback.constants import SPLITS
from vifeedback.data.loader import load
from vifeedback.preprocess.normalize import basic_clean
from vifeedback.preprocess.segment import get_segmenter, segmentation_stats

# name -> (segmenter backend, apply basic_clean, phase-3 condition id)
VARIANTS: dict[str, tuple[str, bool, str]] = {
    "raw": ("none", False, "P0"),
    "seg_vncorenlp": ("vncorenlp", False, "P1"),
    "seg_underthesea": ("underthesea", False, "P2"),
    "seg_pyvi": ("pyvi", False, "P2b"),
    "norm_seg_vncorenlp": ("vncorenlp", True, "P3"),
}


def variant_dir(name: str) -> paths.Path:  # type: ignore[name-defined]
    return paths.DATA_PROCESSED / name


def build(name: str, force: bool = False) -> dict[str, Any]:
    """Materialize one variant and record what it actually changed.

    The `segmentation_stats` block is the guard against a silent no-op: a backend that returned its
    input unchanged would otherwise read as "segmentation makes no difference" in the ablation, which
    is a completely different claim from the one H2 is testing.
    """
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; expected one of {sorted(VARIANTS)}")

    backend, clean, condition = VARIANTS[name]
    out_dir = variant_dir(name)
    meta_path = out_dir / "meta.json"

    if meta_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    out_dir.mkdir(parents=True, exist_ok=True)
    seg = get_segmenter(backend)

    meta: dict[str, Any] = {
        "variant": name,
        "condition": condition,
        "backend": backend,
        "basic_clean": clean,
        "splits": {},
    }

    try:
        for split in SPLITS:
            df = load(split)
            raw = df["sentence"].tolist()
            source = [basic_clean(t) for t in raw] if clean else raw

            t0 = time.perf_counter()
            segmented = seg(source)
            elapsed = time.perf_counter() - t0

            out = df.copy()
            out["sentence"] = segmented
            out["sentence_raw"] = raw
            out.to_parquet(out_dir / f"{split}.parquet", index=False)

            meta["splits"][split] = {
                "rows": len(out),
                "build_seconds": round(elapsed, 2),
                "ms_per_sentence": round(1000 * elapsed / max(len(out), 1), 4),
                **segmentation_stats(raw, segmented),
            }
    finally:
        seg.close()

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


def build_all(names: tuple[str, ...] | None = None, force: bool = False) -> dict[str, Any]:
    return {n: build(n, force=force) for n in (names or tuple(VARIANTS))}


def load_variant(name: str, split: str) -> pd.DataFrame:
    """Load a materialized variant, falling back to the raw corpus for `raw`."""
    if name == "raw":
        return load(split)
    path = variant_dir(name) / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `vifeedback data variants --name {name}` first"
        )
    return pd.read_parquet(path)


def summary_table(metas: dict[str, Any]) -> str:
    """One row per variant: what it changed, and what it cost to build."""
    head = (
        f"{'variant':<22s} {'cond':>5s} {'changed%':>9s} {'underscores':>12s} "
        f"{'tok.reduction':>14s} {'ms/sentence':>12s}"
    )
    lines = [head, "-" * len(head)]
    for name, meta in metas.items():
        tr = meta["splits"]["train"]
        lines.append(
            f"{name:<22s} {meta['condition']:>5s} {tr['changed_share']:>8.1%} "
            f"{tr['underscores_added']:>12,} {tr['token_reduction']:>13.1%} "
            f"{tr['ms_per_sentence']:>12.3f}"
        )
    return "\n".join(lines)
