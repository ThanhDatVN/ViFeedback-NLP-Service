"""Vietnamese word segmentation backends — the Phase 3 / H2 experiment.

Vietnamese writes *syllables* separated by whitespace; a word may span several of them
(`sinh viên` = one word, two syllables). Word segmentation joins them with an underscore
(`sinh_viên`). PhoBERT's model card states input **must** be segmented, because its pretraining
corpus was processed with VnCoreNLP's RDRSegmenter. Published work
([arXiv:2301.00418](https://arxiv.org/abs/2301.00418)) finds segmentation unnecessary for Vietnamese
sentiment classification. Both cannot be operationally true, and ADR-004 resolves it by measurement.

This module exists to make that measurement **joint**: every backend reports not only its output but
its **cost**, because the operational question is accuracy *per millisecond*, not accuracy alone.
`py_vncorenlp` starts a JVM, so it may well cost more p95 than the transformer it feeds.

Backends degrade gracefully: `available()` reports what this machine can actually run, so the
ablation records "not measurable here, and why" rather than crashing or silently skipping.
"""

from __future__ import annotations

import pathlib
import shutil
import time
from typing import Any

import numpy as np

BACKENDS = ("none", "underthesea", "pyvi", "vncorenlp")


# --- Availability -------------------------------------------------------------------------------


def ensure_java() -> str | None:
    """Make a JVM reachable, preferring one already on PATH.

    The reference machine has no system Java, which would make RDRSegmenter — and therefore the
    canonical P1 condition at the heart of H2 — unmeasurable here (risk R1). `jdk4py` ships a JDK as
    an ordinary pip package, so the JVM can be provided without a system-level install and without
    making the project depend on the developer's machine configuration.

    Returns the JAVA_HOME that was set, or None if Java was already on PATH.
    """
    import os

    if shutil.which("java") is not None:
        return None
    try:
        import jdk4py
    except ImportError:
        return None

    java_home = str(jdk4py.JAVA_HOME)
    os.environ["JAVA_HOME"] = java_home
    bin_dir = str(pathlib.Path(java_home) / "bin")
    if bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return java_home


def java_available() -> bool:
    """VnCoreNLP is a Java toolkit called through a JVM bridge. No Java, no RDRSegmenter."""
    ensure_java()
    return shutil.which("java") is not None


def available() -> dict[str, dict[str, Any]]:
    """What this machine can actually run, and why not when it cannot."""
    out: dict[str, dict[str, Any]] = {"none": {"available": True, "reason": "identity"}}

    for name, module in (("underthesea", "underthesea"), ("pyvi", "pyvi")):
        try:
            __import__(module)
            out[name] = {"available": True, "reason": "importable"}
        except ImportError:
            out[name] = {"available": False, "reason": f"{module} not installed"}

    if not java_available():
        out["vncorenlp"] = {
            "available": False,
            "reason": "no JVM on PATH — py_vncorenlp cannot run (risk R1)",
        }
    else:
        try:
            __import__("py_vncorenlp")
            out["vncorenlp"] = {"available": True, "reason": "JVM + py_vncorenlp present"}
        except ImportError:
            out["vncorenlp"] = {"available": False, "reason": "py_vncorenlp not installed"}
    return out


# --- Backends -----------------------------------------------------------------------------------


_VNCORENLP_BASE = "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master"

# Only the word-segmentation assets. py_vncorenlp's own downloader pulls NER, POS and dependency
# models too (~200 MB) and shells out to `wget`, which does not exist on Windows. We need `wseg`
# alone, so we fetch three files with urllib and keep the download portable and reproducible.
_VNCORENLP_FILES = (
    ("VnCoreNLP-1.2.jar", "VnCoreNLP-1.2.jar"),
    ("models/wordsegmenter/vi-vocab", "models/wordsegmenter/vi-vocab"),
    ("models/wordsegmenter/wordsegmenter.rdr", "models/wordsegmenter/wordsegmenter.rdr"),
)


def vncorenlp_dir() -> pathlib.Path:
    """Where VnCoreNLP's models live — deliberately **outside** the repository.

    VnCoreNLP resolves its model directory from the jar's own URL and never URL-decodes the result,
    so a path containing a space becomes `.../ViFeedback%20NLP%20Service/...` and the segmenter dies
    with "wordsegmenter.rdr is not found". Since the repo path is not ours to constrain — and a
    Docker `WORKDIR` or a user's checkout may well contain a space — the models are kept under the
    user cache, which is space-free on both Windows and Linux.
    """
    return pathlib.Path.home() / ".cache" / "vifeedback" / "vncorenlp"


def download_vncorenlp(save_dir: str) -> str:
    """Fetch the VnCoreNLP jar and word-segmenter models. Idempotent."""
    import urllib.request

    root = pathlib.Path(save_dir)
    for remote, local in _VNCORENLP_FILES:
        dest = root / local
        if dest.exists() and dest.stat().st_size > 0:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{_VNCORENLP_BASE}/{remote}", dest)
    return str(root)


class Segmenter:
    """Common interface. `__call__` takes and returns a list of strings."""

    name = "base"
    requires_jvm = False

    def __call__(self, texts: list[str]) -> list[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NoSegmenter(Segmenter):
    """Identity — condition P0, raw syllable text straight into the tokenizer."""

    name = "none"

    def __call__(self, texts: list[str]) -> list[str]:
        return list(texts)


class UndertheseaSegmenter(Segmenter):
    """Pure-Python segmentation — condition P2, and the Docker fallback for risk R1."""

    name = "underthesea"

    def __init__(self) -> None:
        from underthesea import word_tokenize

        self._fn = word_tokenize

    def __call__(self, texts: list[str]) -> list[str]:
        # `format="text"` joins multi-syllable words with an underscore, matching PhoBERT's
        # pretraining convention.
        return [self._fn(t, format="text") for t in texts]


class PyviSegmenter(Segmenter):
    """The fastest of the common Python segmenters — a speed-oriented fallback."""

    name = "pyvi"

    def __init__(self) -> None:
        from pyvi import ViTokenizer

        self._fn = ViTokenizer.tokenize

    def __call__(self, texts: list[str]) -> list[str]:
        return [self._fn(t) for t in texts]


class VnCoreNLPSegmenter(Segmenter):
    """RDRSegmenter via py_vncorenlp — condition P1, PhoBERT's canonical pipeline.

    Requires a JVM. The model directory is downloaded once into `models/vncorenlp`.
    """

    name = "vncorenlp"
    requires_jvm = True

    def __init__(self, save_dir: str | None = None) -> None:
        if not java_available():
            raise RuntimeError(
                "VnCoreNLP needs a JVM and none is reachable. This is risk R1; see ADR-004. "
                "Install `jdk4py` (pip, no system change) or a system JDK, "
                "or use the `underthesea` backend."
            )
        import os

        import py_vncorenlp

        d = save_dir or str(vncorenlp_dir())
        if " " in d:
            raise RuntimeError(
                f"VnCoreNLP cannot load models from a path containing a space: {d!r}. "
                "See vncorenlp_dir() for why."
            )
        download_vncorenlp(d)

        # py_vncorenlp chdir()s into the model directory and never returns; starting the JVM also
        # mutates process state. Both are restored here so callers are not silently relocated.
        cwd = os.getcwd()
        try:
            self._model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=d)
        finally:
            os.chdir(cwd)

    def __call__(self, texts: list[str]) -> list[str]:
        out = []
        for t in texts:
            segmented = self._model.word_segment(t)
            out.append(" ".join(segmented) if segmented else t)
        return out


def get_segmenter(name: str) -> Segmenter:
    if name == "none":
        return NoSegmenter()
    if name == "underthesea":
        return UndertheseaSegmenter()
    if name == "pyvi":
        return PyviSegmenter()
    if name == "vncorenlp":
        return VnCoreNLPSegmenter()
    raise ValueError(f"unknown segmenter {name!r}; expected one of {BACKENDS}")


# --- Measurement --------------------------------------------------------------------------------


def benchmark(
    name: str,
    texts: list[str],
    n_warmup: int = 50,
    n_timed: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Per-sentence segmentation latency, measured the same way as the model benchmark.

    This is the number that decides H2. If segmentation p95 exceeds the quantized encoder's p95,
    then "no significant accuracy difference" is not a tie — it is a decisive win for dropping it,
    and the finding lands in Week 4 rather than Week 7.

    Timed one sentence at a time because that is the serving workload (batch = 1); batch throughput
    is reported separately.
    """
    seg = get_segmenter(name)
    rng = np.random.default_rng(seed)
    sample = [texts[i] for i in rng.integers(0, len(texts), size=n_warmup + n_timed)]

    for t in sample[:n_warmup]:
        seg([t])

    timings = np.empty(n_timed, dtype=np.float64)
    for i, t in enumerate(sample[n_warmup:]):
        t0 = time.perf_counter_ns()
        seg([t])
        timings[i] = (time.perf_counter_ns() - t0) / 1e6  # ms

    t0 = time.perf_counter_ns()
    seg(texts[:1000])
    batch_ms = (time.perf_counter_ns() - t0) / 1e6

    seg.close()
    return {
        "backend": name,
        "n_timed": n_timed,
        "p50_ms": round(float(np.percentile(timings, 50)), 4),
        "p95_ms": round(float(np.percentile(timings, 95)), 4),
        "p99_ms": round(float(np.percentile(timings, 99)), 4),
        "mean_ms": round(float(timings.mean()), 4),
        "batch1000_ms": round(batch_ms, 1),
        "throughput_per_s": round(1000 / (batch_ms / 1000), 1) if batch_ms > 0 else None,
    }


def segmentation_stats(raw: list[str], segmented: list[str]) -> dict[str, Any]:
    """How much the segmenter actually changed — the sanity check for a silent no-op.

    A backend that returns its input unchanged would otherwise show up in the ablation as
    "segmentation makes no difference", which is a very different claim.
    """
    joined = sum(s.count("_") for s in segmented)
    changed = sum(1 for r, s in zip(raw, segmented, strict=True) if r != s)
    raw_tokens = sum(len(r.split()) for r in raw)
    seg_tokens = sum(len(s.split()) for s in segmented)
    return {
        "sentences": len(raw),
        "sentences_changed": changed,
        "changed_share": round(changed / max(len(raw), 1), 4),
        "underscores_added": joined,
        "raw_whitespace_tokens": raw_tokens,
        "segmented_whitespace_tokens": seg_tokens,
        "token_reduction": round(1 - seg_tokens / max(raw_tokens, 1), 4),
    }
