"""UIT-VSFC acquisition and loading.

`fetch()` pulls the official splits from the Hugging Face Hub once and persists them as parquet with
a SHA256 manifest. Raw data is never committed (docs/DATA_CARD.md § 9); the manifest is what pins the
bytes, so a silent upstream change is detectable rather than invisible.

**Provenance chain.** The canonical release is the UIT NLP Group's distribution, mirrored on the Hub
as `uitnlp/vietnamese_students_feedback`. That repo ships a legacy loading *script* which downloads
nine plain-text files from Google Drive; `datasets>=3.0` refuses to execute dataset scripts, and
Google Drive is not a dependable programmatic source. We therefore read the Hub's own auto-converted
parquet branch (`refs/convert/parquet`), which is a byte-faithful conversion of the same script's
output, and pin it by SHA256 in the manifest. The split sizes are asserted against the published
figures in `tests/data/`, so a substituted or re-ordered upstream would fail the build.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from vifeedback import paths
from vifeedback.constants import HF_DATASET_ID, SPLITS
from vifeedback.preprocess.normalize import to_nfc

MANIFEST = paths.DATA_RAW / "manifest.json"

_COLUMNS = ("sentence", "sentiment", "topic")

# Hub-maintained parquet conversion of the script-based dataset. See the module docstring.
PARQUET_REVISION = "refs/convert/parquet"
PARQUET_FILES = {
    "train": "default/train/0000.parquet",
    "validation": "default/validation/0000.parquet",
    "test": "default/test/0000.parquet",
}


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool = False) -> dict[str, Any]:
    """Download the official splits and write `data/raw/{split}.parquet` + `manifest.json`.

    Text is NFC-composed exactly once, on ingest. Doing it here rather than in each consumer means
    no downstream code can accidentally compare an NFC string against an NFD one.
    """
    paths.ensure_dirs()

    if MANIFEST.exists() and not force:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if all(paths.raw_file(s).exists() for s in SPLITS):
            return manifest

    from huggingface_hub import hf_hub_download

    manifest: dict[str, Any] = {
        "dataset": HF_DATASET_ID,
        "source": f"{HF_DATASET_ID}@{PARQUET_REVISION}",
        "splits": {},
    }
    for split in SPLITS:
        local = hf_hub_download(
            HF_DATASET_ID,
            PARQUET_FILES[split],
            repo_type="dataset",
            revision=PARQUET_REVISION,
        )
        df = pd.read_parquet(local)
        missing = set(_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{split}: expected columns {_COLUMNS}, missing {sorted(missing)}")

        df = df[list(_COLUMNS)].copy()
        df["sentence"] = df["sentence"].astype(str).map(to_nfc)
        df["sentiment"] = df["sentiment"].astype("int8")
        df["topic"] = df["topic"].astype("int8")

        out = paths.raw_file(split)
        df.to_parquet(out, index=False)
        manifest["splits"][split] = {
            "file": out.name,
            "rows": len(df),
            "sha256": _sha256(out),
        }

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load(split: str) -> pd.DataFrame:
    """Load one split. Raises with an actionable message if `fetch()` has not been run."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    path = paths.raw_file(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `make data` (vifeedback data fetch) first")
    return pd.read_parquet(path)


def load_all() -> dict[str, pd.DataFrame]:
    return {s: load(s) for s in SPLITS}


def xy(split: str, task: str) -> tuple[pd.Series, pd.Series]:
    """(text, label) for one split and task — the interface every model trains against."""
    df = load(split)
    return df["sentence"], df[task]
