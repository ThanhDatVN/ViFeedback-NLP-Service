"""A teencode lexicon learned from real typing: ViLexNorm train pairs (Cycle 3, S2a).

The Cycle 1 augmentation used 14 hand-picked entries, and on real social-media text it changed nothing
(ViLexNorm invariance, cycle3.yaml). This module learns, from human-normalized comments, which
standard words people write differently, how (the variants and their frequencies), and how often
(the empirical rate at which each standard form appears in a non-standard spelling).

Only the **train** split is used; the test split stays unseen for the invariance test. ViLexNorm is
CC BY-NC-SA 4.0, research only: the lexicon is a derived work, so it is built locally into the
git-ignored `data/external/` and only its SHA-256 is recorded. A model trained with it inherits the
non-commercial condition.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths

LEXICON = paths.DATA / "external" / "teencode_vln.json"
_EDGE_PUNCT = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    toks = (_EDGE_PUNCT.sub("", t) for t in str(text).lower().split())
    return [t for t in toks if t]


def build(
    pairs: list[tuple[str, str]], min_count: int = 3, max_std: int = 3, max_noisy: int = 2
) -> dict[str, Any]:
    """Learn standard -> noisy variants from (original, normalized) pairs.

    For each aligned replacement (normalized span of <= `max_std` syllables written as <= `max_noisy`
    tokens), count the variant. A standard form's rate is its noisy count over all its occurrences
    in the normalized text. Variants seen fewer than `min_count` times are dropped.
    """
    variants: dict[str, Counter[str]] = defaultdict(Counter)
    occurrences: Counter[str] = Counter()
    for original, normalized in pairs:
        o, n = _tokens(original), _tokens(normalized)
        for k in range(1, max_std + 1):
            occurrences.update(" ".join(n[i : i + k]) for i in range(len(n) - k + 1))
        ops = SequenceMatcher(a=n, b=o, autojunk=False).get_opcodes()
        for tag, i1, i2, j1, j2 in ops:
            if tag == "replace" and i2 - i1 <= max_std and j2 - j1 <= max_noisy:
                std, noisy = " ".join(n[i1:i2]), " ".join(o[j1:j2])
                if noisy != std:
                    variants[std][noisy] += 1
    entries: dict[str, dict[str, Any]] = {}
    for std, counts in variants.items():
        kept = {v: c for v, c in counts.items() if c >= min_count}
        if not kept:
            continue
        total = sum(kept.values())
        entries[std] = {
            "rate": min(1.0, total / max(occurrences[std], total)),
            "variants": dict(sorted(kept.items(), key=lambda kv: -kv[1])),
        }
    return dict(sorted(entries.items(), key=lambda kv: -sum(kv[1]["variants"].values())))


def build_from_vilexnorm() -> dict[str, Any]:
    """Build from the pinned ViLexNorm train split and write data/external/teencode_vln.json."""
    from vifeedback.evaluation import external as X

    df = X.load_vilexnorm("train")
    entries = build(list(zip(df.original, df.normalized, strict=True)))
    blob = json.dumps(entries, ensure_ascii=False, sort_keys=True).encode("utf-8")
    LEXICON.parent.mkdir(parents=True, exist_ok=True)
    LEXICON.write_bytes(blob)
    return {
        "entries": len(entries),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "source": "ViLexNorm train split, pinned in configs/data/external_reference.json",
    }


_CACHE: dict[str, Any] = {}
_LENGTHS: dict[int, list[int]] = {}


def load(path: Path = LEXICON) -> dict[str, Any]:
    if str(path) not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing: run `vifeedback data build-lexicon` (needs ViLexNorm, "
                "`vifeedback data fetch-external --name vilexnorm`)"
            )
        _CACHE[str(path)] = json.loads(path.read_text(encoding="utf-8"))
    return dict(_CACHE[str(path)])


def perturb(text: str, rng: np.random.Generator, lexicon: dict[str, Any] | None = None) -> str:
    """Rewrite standard forms as people type them, each occurrence at its empirical rate."""
    lex = lexicon if lexicon is not None else load()
    lengths = _LENGTHS.get(id(lex))
    if lengths is None:
        lengths = _LENGTHS[id(lex)] = sorted({len(k.split()) for k in lex}, reverse=True)
    toks = text.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        for n in lengths:  # longest match first
            key = " ".join(toks[i : i + n])
            if len(toks) - i >= n and key in lex:
                entry = lex[key]
                if rng.random() < entry["rate"]:
                    forms = list(entry["variants"])
                    weights = np.array(list(entry["variants"].values()), dtype=float)
                    out.append(forms[rng.choice(len(forms), p=weights / weights.sum())])
                else:
                    out.extend(toks[i : i + n])
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)
