"""Lexical analysis — which words actually carry each class.

Raw frequency is the wrong tool for this: the most frequent word in every class is the same word.
Chi-square is better but unstable on rare terms, and on a 458-example class almost everything is
rare.

This module uses the **log-odds ratio with an informative Dirichlet prior** (Monroe, Colaresi &
Quinn, 2008), which shrinks estimates for rare terms toward the corpus prior and returns a
z-score, so terms from a 4% class and a 50% class are directly comparable. That property is what
makes it usable here at all.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from vifeedback.constants import LABELS, label_names


def _tokenize(texts: pd.Series, ngram: int = 1) -> list[Counter]:
    out = []
    for t in texts:
        toks = t.split()
        if ngram > 1:
            toks = [" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)]
        out.append(Counter(toks))
    return out


def log_odds_with_prior(
    df: pd.DataFrame,
    task: str,
    ngram: int = 1,
    min_count: int = 5,
    text_col: str = "sentence",
) -> pd.DataFrame:
    """Per-class z-scored log-odds against the rest of the corpus.

    A z-score of +2 means the term is significantly over-represented in that class relative to what
    the corpus prior would predict. Because the prior does the shrinking, a term appearing 8 times
    in the 458-example neutral class can outrank one appearing 400 times in the majority classes —
    which is exactly the comparison this project needs to be able to make.
    """
    names = label_names(task)
    counts_per_class: dict[str, Counter] = {n: Counter() for n in names}

    for label_id, name in LABELS[task].items():
        sub = df.loc[df[task] == label_id, text_col]
        for c in _tokenize(sub, ngram):
            counts_per_class[name].update(c)

    total = Counter()
    for c in counts_per_class.values():
        total.update(c)
    vocab = [w for w, n in total.items() if n >= min_count]
    if not vocab:
        return pd.DataFrame(columns=["term", "class", "z", "count_in_class", "count_total"])

    alpha0 = sum(total[w] for w in vocab)  # informative prior = corpus counts
    rows = []
    for name in names:
        ci = counts_per_class[name]
        ni = sum(ci[w] for w in vocab)
        rest = {w: total[w] - ci[w] for w in vocab}
        nj = sum(rest.values())

        for w in vocab:
            a0 = total[w]
            yi, yj = ci[w], rest[w]
            # log-odds of term w in class i versus the rest, both smoothed by the corpus prior
            li = np.log((yi + a0) / (ni + alpha0 - yi - a0))
            lj = np.log((yj + a0) / (nj + alpha0 - yj - a0))
            var = 1.0 / (yi + a0) + 1.0 / (yj + a0)
            rows.append(
                {
                    "term": w,
                    "class": name,
                    "z": (li - lj) / np.sqrt(var),
                    "count_in_class": yi,
                    "count_total": a0,
                }
            )
    return pd.DataFrame(rows)


def top_terms(
    df: pd.DataFrame, task: str, ngram: int = 1, top_k: int = 15, min_count: int = 5
) -> dict[str, pd.DataFrame]:
    """Top-k most distinctive terms per class, by z-score."""
    scored = log_odds_with_prior(df, task, ngram=ngram, min_count=min_count)
    return {
        name: g.nlargest(top_k, "z")[["term", "z", "count_in_class", "count_total"]].reset_index(
            drop=True
        )
        for name, g in scored.groupby("class")
    }


def class_conditional_length(
    df: pd.DataFrame, task: str, text_col: str = "sentence"
) -> pd.DataFrame:
    """Do the classes differ in length?

    If they do, length is a feature the model can exploit — and a confound the error analysis has to
    control for before attributing an error to meaning rather than to brevity.
    """
    from vifeedback.preprocess import syllable_count

    rows = []
    for label_id, name in LABELS[task].items():
        s = df.loc[df[task] == label_id, text_col]
        syl = s.map(syllable_count)
        rows.append(
            {
                "class": name,
                "n": len(s),
                "syllables_mean": round(float(syl.mean()), 2),
                "syllables_p50": int(syl.quantile(0.5)),
                "syllables_p95": int(syl.quantile(0.95)),
                "chars_mean": round(float(s.str.len().mean()), 1),
                "share_under_5_syllables": round(float((syl < 5).mean()), 4),
            }
        )
    return pd.DataFrame(rows)


def vocabulary_overlap(df: pd.DataFrame, task: str, text_col: str = "sentence") -> pd.DataFrame:
    """Jaccard overlap of the vocabularies of each class pair.

    High overlap means the classes are not separable by vocabulary alone, which is the case for
    *composition* or *negation scope* — precisely where a bag-of-words model must fail and a
    contextual one can win.
    """
    names = label_names(task)
    vocabs = {}
    for label_id, name in LABELS[task].items():
        toks = set()
        for t in df.loc[df[task] == label_id, text_col]:
            toks.update(t.split())
        vocabs[name] = toks

    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            inter = len(vocabs[a] & vocabs[b])
            union = len(vocabs[a] | vocabs[b])
            mat.loc[a, b] = round(inter / union, 3) if union else 0.0
    return mat


def subword_fertility(
    df: pd.DataFrame, model_key: str = "phobert-base", text_col: str = "sentence"
) -> dict[str, Any]:
    """Subwords per syllable — how hard the tokenizer finds this text.

    A fertility near 1.0 means the vocabulary covers the corpus well. High fertility means words are
    being shattered into pieces, which is what happens to out-of-vocabulary teencode and misspellings
    and is a direct measure of how far the text sits from the pretraining distribution.
    """
    from transformers import AutoTokenizer

    from vifeedback.constants import MODEL_IDS
    from vifeedback.preprocess import syllable_count

    tok = AutoTokenizer.from_pretrained(MODEL_IDS[model_key])
    texts = df[text_col].tolist()
    syl = np.array([syllable_count(t) for t in texts], dtype=np.float64)
    sub = np.array([len(tok.tokenize(t)) for t in texts], dtype=np.float64)
    fert = np.divide(sub, syl, out=np.ones_like(sub), where=syl > 0)

    worst_idx = np.argsort(-fert)[:10]
    return {
        "model": MODEL_IDS[model_key],
        "fertility_mean": round(float(fert.mean()), 3),
        "fertility_p50": round(float(np.percentile(fert, 50)), 3),
        "fertility_p95": round(float(np.percentile(fert, 95)), 3),
        "share_above_2": round(float((fert > 2).mean()), 4),
        "worst_examples": [
            {
                "text": texts[i][:80],
                "syllables": int(syl[i]),
                "subwords": int(sub[i]),
                "fertility": round(float(fert[i]), 2),
            }
            for i in worst_idx
        ],
    }
