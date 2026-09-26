"""Robustness: versioned input perturbations and naturally occurring slices — Study B.

Two different questions, kept apart because they are answered with different evidence:

* **Shift** — how much does quality drop when the *same* sentences are written differently (no
  diacritics, teencode, typos)? Answered with paired perturbations of the evaluation set.
* **Slices** — how well does the model do on sentences that *already* have a property (negation,
  contrast, a suggestion, very short)? Answered by subsetting the clean evaluation set, with support
  reported next to every number.

Rules (docs/EVALUATION_PROTOCOL.md § 7):

* Perturbations change inputs only, never labels, and are applied to **raw** text *before*
  segmentation — the deployed pipeline segments whatever the user typed.
* Every transformation is deterministic in (`SUITE_VERSION`, seed, example index), independently of
  order, so a single example can be regenerated in isolation and results stay comparable across runs.
* Clean and perturbed predictions are compared **on the same examples**, and uncertainty comes from a
  bootstrap over *original* examples, so several perturbations of one sentence never count as
  independent evidence (review § 8, "paired/grouped uncertainty").
* Only examples a transformation actually changed are informative. The changed share is reported,
  and a suite that changes almost nothing is flagged rather than read as "robust".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vifeedback.evaluation.metrics import macro_f1
from vifeedback.preprocess.normalize import strip_diacritics

# Bump when any transformation below changes behaviour. Results are only comparable within a version.
SUITE_VERSION = "1"

# --- Transformations --------------------------------------------------------------------------------

# Standard form -> informal variants. The inverse of the probe dictionary used to size teencode in the
# corpus (data/profile.py). Multi-syllable entries are matched before single-syllable ones.
TEENCODE: dict[str, tuple[str, ...]] = {
    "như thế nào": ("ntn",),
    "bình thường": ("bt", "bthg"),
    "giảng viên": ("gv",),
    "sinh viên": ("sv",),
    "mọi người": ("mn", "mng"),
    "không": ("ko", "k", "hok", "kg"),
    "được": ("dc", "đc", "đk"),
    "vậy": ("z", "dz", "v"),
    "gì": ("j",),
    "với": ("vs",),
    "quá": ("wa", "wá", "qá"),
    "biết": ("bít", "bik"),
    "rồi": ("r", "rùi"),
    "thì": ("thy",),
}

_LETTERS = (
    "abcdeghiklmnopqrstuvxyàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
)


def _rng(seed: int, idx: int, suite: str) -> np.random.Generator:
    # Seeded from (version, suite, seed, index): order-independent and suite-independent.
    return np.random.default_rng([int(SUITE_VERSION), sum(map(ord, suite)), seed, idx])


def nodiacritic(text: str, rng: np.random.Generator, p: float = 1.0) -> str:
    """Strip diacritics from each syllable with probability `p` (p=1: the whole sentence)."""
    if p >= 1.0:
        return strip_diacritics(text)
    return " ".join(strip_diacritics(s) if rng.random() < p else s for s in text.split())


def teencode(text: str, rng: np.random.Generator, p: float = 0.3) -> str:
    """Replace dictionary words with an informal variant, each occurrence with probability `p`."""
    toks = text.split()
    out: list[str] = []
    i = 0
    keys = sorted(TEENCODE, key=lambda k: -len(k.split()))
    while i < len(toks):
        for key in keys:
            n = len(key.split())
            if " ".join(toks[i : i + n]) == key:
                if rng.random() < p:
                    variants = TEENCODE[key]
                    out.append(variants[rng.integers(len(variants))])
                else:
                    out.extend(toks[i : i + n])
                i += n
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def charnoise(text: str, rng: np.random.Generator, rate: float = 0.05) -> str:
    """Per-letter swap-with-neighbour, drop, duplicate or substitute, each letter with prob `rate`."""
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            op = rng.integers(4)
            if op == 0 and i + 1 < len(chars) and chars[i + 1].isalpha():  # swap
                out.extend([chars[i + 1], c])
                i += 2
                continue
            if op == 1:  # drop
                i += 1
                continue
            if op == 2:  # duplicate
                out.extend([c, c])
            else:  # substitute
                out.append(_LETTERS[rng.integers(len(_LETTERS))])
        else:
            out.append(c)
        i += 1
    return "".join(out)


SUITES: dict[str, Callable[[str, np.random.Generator], str]] = {
    "nodiacritic": lambda t, r: nodiacritic(t, r, 1.0),
    "nodiacritic-50": lambda t, r: nodiacritic(t, r, 0.5),
    "teencode-30": lambda t, r: teencode(t, r, 0.3),
    "teencode-100": lambda t, r: teencode(t, r, 1.0),
    "charnoise-5": lambda t, r: charnoise(t, r, 0.05),
    "charnoise-10": lambda t, r: charnoise(t, r, 0.10),
}


def perturb(texts: list[str], suite: str, seed: int = 42) -> tuple[list[str], np.ndarray]:
    """Apply one suite. Returns the perturbed texts and a mask of the examples it actually changed."""
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}; expected one of {sorted(SUITES)}")
    fn = SUITES[suite]
    out = [fn(t, _rng(seed, i, suite)) for i, t in enumerate(texts)]
    changed = np.array([a != b for a, b in zip(texts, out, strict=True)])
    return out, changed


# --- Naturally occurring slices ---------------------------------------------------------------------

# Predefined before looking at any slice result (review, first-five task 4: "freeze ... the initial
# robustness-slice definitions"). Defined on the raw, lowercased text. Changing one means bumping SLICES_VERSION.
SLICES_VERSION = "1"


def _has(words: set[str]) -> Callable[[str], bool]:
    return lambda t: bool(set(t.split()) & words)


SLICES: dict[str, Callable[[str], bool]] = {
    "negation": _has({"không", "ko", "chưa", "chẳng", "chả", "đừng"}),
    "contrast": _has({"nhưng", "tuy", "mặc", "song"}),
    "suggestion": _has({"nên", "cần", "mong"}),
    "short_lt5": lambda t: len(t.split()) < 5,
    "long_ge30": lambda t: len(t.split()) >= 30,
    "emoticon_token": lambda t: any(w.startswith("colon") for w in t.split()),
    "anonymized_name": lambda t: "wzjwz" in t,
}


def slice_masks(raw_texts: list[str]) -> dict[str, np.ndarray]:
    return {name: np.array([fn(t) for t in raw_texts]) for name, fn in SLICES.items()}


# --- Paired evaluation with grouped uncertainty ---------------------------------------------------


def paired_delta(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    k: int,
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """macro-F1(b) - macro-F1(a) on identical examples, with a bootstrap CI over groups.

    `groups` maps each row to its original example. When several perturbed variants of one sentence
    are stacked, resampling whole groups keeps them together; resampling rows would treat them as
    independent and shrink the interval. Defaults to one group per row.
    """
    y, pred_a, pred_b = map(np.asarray, (y, pred_a, pred_b))
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        pick = np.concatenate([members[g] for g in rng.integers(len(uniq), size=len(uniq))])
        deltas[b] = macro_f1(y[pick], pred_b[pick], k) - macro_f1(y[pick], pred_a[pick], k)

    a, b_ = macro_f1(y, pred_a, k), macro_f1(y, pred_b, k)
    return {
        "n": len(y),
        "n_groups": len(uniq),
        "macro_f1_a": a,
        "macro_f1_b": b_,
        "delta": b_ - a,
        "delta_ci": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "flip_rate": float((pred_a != pred_b).mean()),
    }


def slice_report(
    y: np.ndarray,
    pred: np.ndarray,
    masks: dict[str, np.ndarray],
    k: int,
    min_support: int = 30,
    min_class_support: int = 5,
) -> list[dict[str, Any]]:
    """Per-slice accuracy and macro-F1 with support.

    Two separate reliability checks. `reliable` needs `min_support` examples in the slice.
    `macro_f1_reliable` additionally needs every class to have `min_class_support` examples: a slice
    of 247 suggestions holding one neutral sentence has a large n and a meaningless macro-F1, since
    one third of it is decided by that single example.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    rows = []
    for name, m in masks.items():
        n = int(m.sum())
        row: dict[str, Any] = {"slice": name, "n": n, "share": float(m.mean())}
        if n:
            row["accuracy"] = float((y[m] == pred[m]).mean())
            row["macro_f1"] = macro_f1(y[m], pred[m], k)
            row["class_support"] = np.bincount(y[m], minlength=k).tolist()
        row["reliable"] = n >= min_support
        row["macro_f1_reliable"] = bool(
            n >= min_support and n and min(np.bincount(y[m], minlength=k)) >= min_class_support
        )
        rows.append(row)
    return rows


# --- Negation minimal pairs -----------------------------------------------------------------------


def negation_probe(pairs: Any, predict: Callable[[list[str]], np.ndarray]) -> dict[str, Any]:
    """Score `data/probes/negation_v*.csv`. `predict` maps raw texts to label ids.

    Sets are reported separately and never pooled: `neg_to_pos` expected labels are arguable
    (data/probes/README.md).
    """
    from vifeedback.constants import SENTIMENT_LABELS

    to_id = {v: k for k, v in SENTIMENT_LABELS.items()}
    out: dict[str, Any] = {}
    for name, g in pairs.groupby("set"):
        pb = np.asarray(predict(g["base"].tolist()))
        pn = np.asarray(predict(g["negated"].tolist()))
        yb = g["base_label"].map(to_id).to_numpy()
        yn = g["negated_label"].map(to_id).to_numpy()
        base_ok, neg_ok = pb == yb, pn == yn
        out[str(name)] = {
            "n_pairs": len(g),
            "pair_accuracy": float((base_ok & neg_ok).mean()),
            "pair_accuracy_given_correct_base": (
                float(neg_ok[base_ok].mean()) if base_ok.any() else None
            ),
            "flip_rate": float((pb != pn).mean()),
            "base_accuracy": float(base_ok.mean()),
            "failed_pair_ids": g["pair_id"][~(base_ok & neg_ok)].astype(int).tolist(),
            # The probe is constructed text, not corpus data, so predictions can be published.
            "predictions": [
                {
                    "pair_id": int(pid),
                    "base_pred": SENTIMENT_LABELS[int(a)],
                    "negated_pred": SENTIMENT_LABELS[int(b)],
                }
                for pid, a, b in zip(g["pair_id"], pb, pn, strict=True)
            ],
        }
    return out
