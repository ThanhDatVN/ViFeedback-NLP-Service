"""PhoBERT's BPE tokenizer without `transformers` (NEXT_PLAN v5 F1).

The runtime image carried `transformers` (118 MB, plus tokenizers, huggingface_hub and hf_xet) only
to turn segmented text into PhoBERT ids. This is the same computation as transformers'
`PhobertTokenizer` with `padding=True, truncation=True, max_length=...` on single sequences:

1. special tokens written in the text (`<s>`, `<pad>`, `</s>`, `<unk>`, `<mask>`) are kept whole;
2. the rest is split with `\\S+\\n?`, and each piece is merged by the ranks in `bpe.codes`, with
   `@@` marking a word that continues;
3. pieces map to ids through `vocab.txt` (after the four special ids), unknown pieces to `<unk>`;
4. ids are cut to `max_length - 2` and wrapped as `<s> ... </s>`; the batch is right-padded with
   `<pad>`, and the attention mask marks the real tokens.

It is verified id for id against `PhobertTokenizer` on every UIT-VSFC and NEU-ESC text
(`scripts/check_phobert_tokenizer.py`), so the ONNX graph receives exactly the same inputs.
"""

from __future__ import annotations

import json
import re
from itertools import pairwise
from pathlib import Path

import numpy as np

SPECIAL = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}


def _pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return set(pairwise(word))


class PhobertBPE:
    """Drop-in for the call `tokenizer(texts, return_tensors="np", padding=True, truncation=True,
    max_length=n)` on a PhoBERT artifact directory (vocab.txt, bpe.codes, added_tokens.json)."""

    def __init__(self, model_dir: Path) -> None:
        model_dir = Path(model_dir)
        self.encoder: dict[str, int] = dict(SPECIAL)
        with open(model_dir / "vocab.txt", encoding="utf-8") as f:
            for raw in f.readlines():
                line = raw.strip()
                idx = line.rfind(" ")
                if idx == -1:
                    raise ValueError("Incorrect dictionary format, expected '<token> <cnt>'")
                self.encoder[line[:idx]] = len(self.encoder)
        added = model_dir / "added_tokens.json"
        if added.exists():
            self.encoder.update(json.loads(added.read_text(encoding="utf-8")))
        with open(model_dir / "bpe.codes", encoding="utf-8") as f:
            merges = [tuple(m.split()[:-1]) for m in f.read().split("\n")[:-1]]
        self.ranks = {m: i for i, m in enumerate(merges)}
        # The tokens transformers never splits: the four specials and, if present, <mask>.
        self.special = set(SPECIAL) | ({"<mask>"} & set(self.encoder))
        # Longest first, as transformers' trie matches the longest added token at a position.
        self._split = re.compile(
            "(" + "|".join(re.escape(t) for t in sorted(self.special, key=len, reverse=True)) + ")"
        )
        self._cache: dict[str, str] = {}

    @staticmethod
    def is_phobert_dir(model_dir: Path) -> bool:
        return (Path(model_dir) / "vocab.txt").exists() and (Path(model_dir) / "bpe.codes").exists()

    def _bpe(self, token: str) -> str:
        if token in self._cache:
            return self._cache[token]
        word = (*token[:-1], token[-1] + "</w>")
        pairs = _pairs(word)
        if not pairs:
            return token
        while True:
            bigram = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if bigram not in self.ranks:
                break
            first, second = bigram
            new: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new.extend(word[i:])
                    break
                new.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new.append(first + second)
                    i += 2
                else:
                    new.append(word[i])
                    i += 1
            word = tuple(new)
            if len(word) == 1:
                break
            pairs = _pairs(word)
        out = "@@ ".join(word)[:-4]
        self._cache[token] = out
        return out

    def tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for chunk in self._split.split(text):
            if not chunk:
                continue
            if chunk in self.special:
                tokens.append(chunk)
                continue
            for piece in re.findall(r"\S+\n?", chunk):
                tokens.extend(self._bpe(piece).split(" "))
        return tokens

    def encode(self, text: str, max_length: int) -> list[int]:
        unk = self.encoder["<unk>"]
        ids = [self.encoder.get(t, unk) for t in self.tokenize(text)]
        return [SPECIAL["<s>"], *ids[: max(max_length - 2, 0)], SPECIAL["</s>"]]

    def __call__(
        self,
        texts: list[str],
        return_tensors: str = "np",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 96,
    ) -> dict[str, np.ndarray]:
        if return_tensors != "np" or not padding or not truncation:
            raise ValueError(
                "PhobertBPE supports only return_tensors='np', padding=True, truncation=True"
            )
        rows = [self.encode(t, max_length) for t in texts]
        width = max(len(r) for r in rows)
        ids = np.full((len(rows), width), SPECIAL["<pad>"], dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, : len(r)] = r
            mask[i, : len(r)] = 1
        return {"input_ids": ids, "attention_mask": mask}
