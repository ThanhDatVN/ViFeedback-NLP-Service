"""S7b (cycle5.yaml v7): the served model with its vocabulary cut to the pieces its input uses.

The word-embedding matrix is 64,001 x 768, about 98 MB in FP16, while the training and coverage
texts use about 15,600 of its pieces. Slicing the rows changes no other weight. A piece outside the
kept vocabulary becomes ``<unk>``, as any unknown piece does today, so the fidelity check compares
labels with the full graph on held-out text.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vifeedback import paths
from vifeedback.inference.phobert_tokenizer import SPECIAL, PhobertBPE

N_ENTRIES = 17_500  # the four specials, <mask> and the kept vocab.txt pieces
OUT = paths.RESULTS / "studies" / "cycle5" / "s7b"


def read_vocab(model_dir: Path) -> list[tuple[str, int]]:
    """vocab.txt as (piece, pretraining count), in file order (ids 4, 5, ...)."""
    rows = []
    with open(Path(model_dir) / "vocab.txt", encoding="utf-8") as f:
        for raw in f.read().splitlines():
            line = raw.strip()
            idx = line.rfind(" ")
            rows.append((line[:idx], int(line[idx + 1 :])))
    return rows


def used_pieces(tok: PhobertBPE, texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(tok.tokenize(t))
    return out


def choose(vocab: list[tuple[str, int]], used: set[str], n_entries: int = N_ENTRIES) -> np.ndarray:
    """Positions in vocab.txt to keep: every used piece, then the most frequent others."""
    budget = n_entries - len(SPECIAL) - 1  # minus <mask>
    keep = [i for i, (piece, _) in enumerate(vocab) if piece in used]
    if len(keep) > budget:
        raise ValueError(f"{len(keep)} used pieces exceed the budget of {budget}")
    others = sorted(
        (i for i, (piece, _) in enumerate(vocab) if piece not in used),
        key=lambda i: (-vocab[i][1], i),
    )
    return np.array(sorted(keep + others[: budget - len(keep)]), dtype=np.int64)


def old_ids(keep: np.ndarray, mask_id: int) -> np.ndarray:
    """The embedding rows of the trimmed model, in its id order: specials, kept pieces, <mask>."""
    return np.concatenate([np.arange(len(SPECIAL)), keep + len(SPECIAL), [mask_id]])


def write_trimmed(src: Path, dst: Path, keep: np.ndarray) -> dict[str, Any]:
    """A checkpoint whose word-embedding rows and tokenizer files are cut to ``keep``."""
    import torch
    from transformers import AutoModelForSequenceClassification

    src, dst = Path(src), Path(dst)
    added = json.loads((src / "added_tokens.json").read_text(encoding="utf-8"))
    rows = old_ids(keep, added["<mask>"])
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    emb = model.get_input_embeddings()
    new = torch.nn.Embedding(len(rows), emb.embedding_dim, padding_idx=emb.padding_idx)
    new.weight.data = emb.weight.data[torch.as_tensor(rows)].clone()
    model.set_input_embeddings(new)
    model.config.vocab_size = len(rows)
    if dst.exists():
        shutil.rmtree(dst)
    model.save_pretrained(dst)

    vocab = read_vocab(src)
    with open(dst / "vocab.txt", "w", encoding="utf-8", newline="\n") as f:
        for i in keep:
            f.write(f"{vocab[i][0]} {vocab[i][1]}\n")
    mask_new = len(rows) - 1
    (dst / "added_tokens.json").write_text(
        json.dumps({"<mask>": mask_new}, indent=2), encoding="utf-8"
    )
    shutil.copy2(src / "bpe.codes", dst / "bpe.codes")
    cfg = json.loads((src / "tokenizer_config.json").read_text(encoding="utf-8"))
    decoder = cfg.get("added_tokens_decoder", {})
    if str(added["<mask>"]) in decoder:
        decoder[str(mask_new)] = decoder.pop(str(added["<mask>"]))
    (dst / "tokenizer_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    record = {
        "source": src.name,
        "entries": len(rows),
        "kept_pieces": len(keep),
        "kept_sha256": hashlib.sha256(keep.tobytes()).hexdigest(),
    }
    (dst / "trim.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def coverage_texts(transform: Callable[[list[str]], list[str]]) -> dict[str, list[str]]:
    """The vocabulary source of cycle5.yaml v7, as model input."""
    from vifeedback.evaluation import external as X
    from vifeedback.evaluation import robustness as R
    from vifeedback.preprocess.normalize import strip_diacritics
    from vifeedback.preprocess.variants import load_variant

    uit = load_variant("seg_pyvi", "train")["sentence"].tolist()
    vl = X.load_vilexnorm("train")
    held = set(pd.read_csv(paths.RESULTS / "studies" / "cycle5" / "vilexnorm_dev_index.csv").row)
    held |= set(pd.read_csv(paths.RESULTS / "studies" / "cycle5" / "h10b_confirm_index.csv").row)
    train_rows = [i for i in range(len(vl)) if i not in held]  # H10b's 6,035 training pairs
    pairs = vl.iloc[train_rows]
    return {
        "uit_train": uit,
        "uit_train_stripped": [strip_diacritics(t) for t in uit],
        "uit_train_teencode": R.perturb(uit, "teencode-100")[0],
        "neu_esc_train": transform(X.load_neu_esc("train").text.tolist()),
        "vilexnorm_training_original": transform(pairs.original.tolist()),
        "vilexnorm_training_normalized": transform(pairs.normalized.tolist()),
    }


def removed_share(full: PhobertBPE, kept: set[str], texts: list[str]) -> dict[str, float]:
    """Of the pieces that the full vocabulary knows, the share the trimmed one does not."""
    n_known = n_removed = n_texts = 0
    for t in texts:
        pieces = [p for p in full.tokenize(t) if p in full.encoder]
        gone = sum(p not in kept for p in pieces)
        n_known += len(pieces)
        n_removed += gone
        n_texts += gone > 0
    return {
        "pieces_removed": n_removed / max(n_known, 1),
        "texts_with_a_removed_piece": n_texts / max(len(texts), 1),
    }


def logits(clf: Any, texts: list[str], batch: int = 64) -> np.ndarray:
    return np.concatenate([clf.logits(texts[i : i + batch]) for i in range(0, len(texts), batch)])


def closing_gate(served: Path, fp32: Path) -> dict[str, Any]:
    """cycle5.yaml v7 S7b serving: the served trimmed graph on UIT-VSFC test once, logged, for the
    card; the FP32 graph of the same model on the same text for label agreement."""
    from vifeedback.evaluation import closing_gate as CG
    from vifeedback.evaluation import metrics as M
    from vifeedback.evaluation.report import yaml_safe
    from vifeedback.inference.onnx_export import OnnxClassifier

    dst = OUT / "closing_gate.json"
    if dst.exists():
        raise FileExistsError(f"the S7b test evaluation has run already: {dst}")
    ms = json.loads((served / "manifest.json").read_text(encoding="utf-8"))
    mf = json.loads((fp32 / "manifest.json").read_text(encoding="utf-8"))
    if f"-v{N_ENTRIES}-" not in ms["checkpoint"]:
        raise ValueError(f"{served} does not serve the trimmed graph: {ms['checkpoint']}")
    gi = CG.inputs()
    preds = {}
    for name, d, m in (("trimmed", served, ms), ("fp32", fp32, mf)):
        clf = OnnxClassifier(d, max_length=m["max_length"], model_file=m["model_file"])
        preds[name] = logits(clf, gi["x_te"]).argmax(1)
    CG.log_test_use(
        Path(ms["checkpoint"]),
        "P15",
        "S7b (cycle5.yaml v7): the served trimmed graph on test once, for the card; the FP32 "
        "graph of the same model for label agreement",
    )
    y = gi["y_te"]
    out: dict[str, Any] = {"declared_in": "configs/experiments/cycle5.yaml v7 S7b serving"}
    for name, p in preds.items():
        ev = M.evaluate(y, p, "sentiment")
        out[name] = {
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev.get("weighted_f1"),
            "per_class_f1": {c: v["f1"] for c, v in ev["per_class"].items()},
        }
    out["label_agreement"] = float((preds["trimmed"] == preds["fp32"]).mean())
    out["n"] = len(y)
    OUT.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(yaml_safe(out), indent=2), encoding="utf-8")
    return out
