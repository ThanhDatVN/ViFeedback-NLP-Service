"""Cycle 3 S2b: restore missing diacritics before classification (cycle3.yaml).

Trained on UIT-VSFC **train** text only (99.86% diacritized), so no external data enters the
pipeline. For every syllable written without diacritics, the candidates are the accented forms seen
in train for that key; the sequence is chosen by Viterbi over a word-bigram model interpolated with
unigrams. Syllables that already carry diacritics, and unknown keys, are kept as typed.

It runs only when an input is essentially unaccented (share of accented letters below a threshold
set on train), so normal input is never touched: the validation predictions stay identical.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vifeedback.preprocess.normalize import strip_diacritics

BOS = "<s>"


def accented_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(c != strip_diacritics(c) for c in letters) / len(letters) if letters else 0.0


class Restorer:
    def __init__(
        self,
        candidates: dict[str, list[str]],
        unigram: dict[str, int],
        bigram: dict[str, dict[str, int]],
        threshold: float,
        lam: float = 0.8,
        top_k: int = 5,
    ):
        self.candidates, self.unigram, self.bigram = candidates, unigram, bigram
        self.threshold, self.lam, self.top_k = threshold, lam, top_k
        self.total = sum(unigram.values())
        self.vocab = len(unigram) + 1

    # -- fitting -----------------------------------------------------------------------------------
    @classmethod
    def fit(cls, sentences: list[str], max_share_restored: float = 0.001) -> Restorer:
        """Counts from train; the threshold is the largest accented share below which at most
        `max_share_restored` of train sentences (of >= 5 letters) fall, i.e. would be rewritten."""
        forms: dict[str, Counter[str]] = defaultdict(Counter)
        unigram: Counter[str] = Counter()
        bigram: dict[str, Counter[str]] = defaultdict(Counter)
        shares = []
        for s in sentences:
            toks = s.split()
            prev = BOS
            for t in toks:
                forms[strip_diacritics(t)][t] += 1
                unigram[t] += 1
                bigram[prev][t] += 1
                prev = t
            if sum(c.isalpha() for c in s) >= 5:
                shares.append(accented_share(s))
        shares.sort()
        threshold = shares[int(max_share_restored * len(shares))] if shares else 0.0
        cands = {k: [w for w, _ in c.most_common()] for k, c in forms.items()}
        return cls(cands, dict(unigram), {k: dict(v) for k, v in bigram.items()}, threshold)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "candidates": self.candidates,
                    "unigram": self.unigram,
                    "bigram": self.bigram,
                    "threshold": self.threshold,
                    "lam": self.lam,
                    "top_k": self.top_k,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Restorer:
        d: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["candidates"], d["unigram"], d["bigram"], d["threshold"], d["lam"], d["top_k"])

    # -- inference ---------------------------------------------------------------------------------
    def _logp(self, prev: str, w: str) -> float:
        big = self.bigram.get(prev, {})
        n_prev = sum(big.values()) if big else 0
        p_bi = big.get(w, 0) / n_prev if n_prev else 0.0
        p_uni = (self.unigram.get(w, 0) + 1) / (self.total + self.vocab)
        return math.log(self.lam * p_bi + (1 - self.lam) * p_uni)

    def needs_restoring(self, text: str) -> bool:
        return sum(c.isalpha() for c in text) >= 5 and accented_share(text) < self.threshold

    def restore(self, text: str) -> str:
        """Viterbi over accented candidates for unaccented syllables; others are fixed."""
        toks = text.split()
        if not toks:
            return text
        options = []
        for t in toks:
            if t == strip_diacritics(t) and t in self.candidates:
                options.append(self.candidates[t][: self.top_k])
            else:
                options.append([t])
        best: dict[str, tuple[float, list[str]]] = {BOS: (0.0, [])}
        for opts in options:
            nxt: dict[str, tuple[float, list[str]]] = {}
            for w in opts:
                score, path = max(
                    ((s + self._logp(prev, w), p) for prev, (s, p) in best.items()),
                    key=lambda x: x[0],
                )
                nxt[w] = (score, [*path, w])
            best = nxt
        return " ".join(max(best.values(), key=lambda x: x[0])[1])

    def __call__(self, text: str) -> str:
        return self.restore(text) if self.needs_restoring(text) else text
