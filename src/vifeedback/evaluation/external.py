"""External evaluation corpora: pinned, hash-checked, never committed (docs/EVALUATION_DATA.md).

* **ViLexNorm** (EACL 2024): real Vietnamese social-media comments paired with their human
  normalization. No sentiment labels, so it serves an *invariance* test: a model robust to real
  informal typing gives the same label to the original and to the normalized sentence.
* **NEU-ESC** (2025): real posts from Vietnamese university forums with human sentiment labels. A
  domain-shift and real-noise test after a declared label mapping.

Files live in `data/external/` (git-ignored). Their revisions and SHA-256 are committed in
`configs/data/external_reference.json`; a file that does not match is refused.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vifeedback import paths

EXT = paths.DATA / "external"
REF = paths.CONFIGS / "data" / "external_reference.json"

# NEU-ESC sentiment -> UIT-VSFC labels. `Toxic` is hostile language, a negative evaluation in the
# UIT-VSFC sense; results are reported with and without those rows (cycle3.yaml).
NEU_ESC_SENTIMENT = {
    "Neutral": "neutral",
    "Positive": "positive",
    "Negative": "negative",
    "Toxic": "negative",
}
# The topics closest to course feedback (the second, narrower view).
NEU_ESC_COURSE_TOPICS = ("Academic", "Service")


def reference() -> dict[str, Any]:
    return dict(json.loads(REF.read_text(encoding="utf-8")))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(name: str) -> dict[str, str]:
    """Download one corpus at its pinned revision into data/external/<name>/.

    A file whose hash is already recorded must match it. A hash not yet recorded (a gated corpus
    fetched for the first time) is written into the reference file, to be committed.
    """
    ref = reference()
    spec = ref[name]
    out = EXT / name
    out.mkdir(parents=True, exist_ok=True)
    got = {}
    for split, f in spec["files"].items():
        dst = out / Path(f["path"]).name
        if spec["source"] == "github":
            import urllib.request

            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}/{f['path']}"
            with urllib.request.urlopen(url, timeout=120) as r:
                dst.write_bytes(r.read())
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import GatedRepoError

            try:
                hf_hub_download(
                    spec["repo"],
                    f["path"],
                    repo_type="dataset",
                    revision=spec["revision"],
                    local_dir=out,
                )
            except GatedRepoError:
                raise PermissionError(
                    f"{spec['repo']} is gated: open https://huggingface.co/datasets/{spec['repo']} "
                    "while logged in with the account of your HF_TOKEN, accept the conditions, then "
                    "run this again"
                ) from None
        sha = _sha256(dst)
        if f["sha256"] and f["sha256"] != sha:
            raise ValueError(
                f"{dst} sha256 {sha[:12]} does not match the reference {f['sha256'][:12]}"
            )
        f["sha256"] = sha
        got[split] = sha
    REF.write_text(json.dumps(ref, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return got


def _checked(name: str, split: str) -> Path:
    f = reference()[name]["files"][split]
    path = EXT / name / Path(f["path"]).name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing: run `vifeedback data fetch-external --name {name}`"
        )
    if not f["sha256"] or _sha256(path) != f["sha256"]:
        raise ValueError(f"{path} does not match configs/data/external_reference.json")
    return path


def load_vilexnorm(split: str = "test") -> pd.DataFrame:
    df = pd.read_csv(_checked("vilexnorm", split))[["original", "normalized"]]
    return df.dropna().reset_index(drop=True)


def load_neu_esc(split: str = "test") -> pd.DataFrame:
    """NEU-ESC with UIT-VSFC sentiment labels added (`sentiment`), the original kept (`source_label`)."""
    df = pd.read_csv(_checked("neu_esc", split))
    text_col = next(c for c in df.columns if c.lower() == "text")
    sent_col = next(c for c in df.columns if c.lower() == "sentiment")
    topic_col = next(c for c in df.columns if c.lower() in ("classification", "topic"))
    out = pd.DataFrame(
        {
            "text": df[text_col].astype(str),
            "source_label": df[sent_col].astype(str).str.strip(),
            "topic": df[topic_col].astype(str).str.strip(),
        }
    )
    unknown = sorted(set(out.source_label) - set(NEU_ESC_SENTIMENT))
    if unknown:
        raise ValueError(f"unexpected NEU-ESC sentiment labels {unknown}")
    out["sentiment"] = out.source_label.map(NEU_ESC_SENTIMENT)
    return out


def case_variants(texts: list[str]) -> dict[str, list[str]]:
    """Inputs as a person types them: the corpus is lowercase, people capitalize the first letter."""
    return {
        "as_is": list(texts),
        "sentence_case": [t[:1].upper() + t[1:] for t in texts],
    }


def flip_rate(pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    """Invariance: the share of pairs whose predicted label changes, with a Wilson interval."""
    from vifeedback.evaluation.audit import wilson

    flips = int((np.asarray(pred_a) != np.asarray(pred_b)).sum())
    n = len(pred_a)
    return {
        "n": n,
        "flips": flips,
        "rate": flips / n if n else float("nan"),
        "wilson_95": wilson(flips, n),
    }


def paired_flip_test(flips_a: np.ndarray, flips_b: np.ndarray) -> dict[str, Any]:
    """Exact McNemar on per-pair flip indicators of two models over the same pairs."""
    from math import comb

    a, b = np.asarray(flips_a, bool), np.asarray(flips_b, bool)
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n) if n else 1.0
    return {"only_first_flips": only_a, "only_second_flips": only_b, "exact_mcnemar_p": p}
