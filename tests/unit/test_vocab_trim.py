"""S7b's vocabulary cut on a tiny RoBERTa: the same logits for text whose pieces are kept."""

from __future__ import annotations

import json

import numpy as np
import pytest

from vifeedback.inference import vocab_trim as V

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

PIECES = list("abcdefghij")  # one-character words need no BPE merge


def _checkpoint(path):
    from transformers import RobertaConfig, RobertaForSequenceClassification

    torch.manual_seed(0)
    n = len(V.SPECIAL) + len(PIECES) + 1
    cfg = RobertaConfig(
        vocab_size=n,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=40,
        pad_token_id=1,
        num_labels=3,
    )
    RobertaForSequenceClassification(cfg).eval().save_pretrained(path)
    counts = range(1000, 1000 - len(PIECES), -1)
    (path / "vocab.txt").write_text(
        "".join(f"{p} {c}\n" for p, c in zip(PIECES, counts, strict=True)), encoding="utf-8"
    )
    (path / "bpe.codes").write_text("", encoding="utf-8")
    (path / "added_tokens.json").write_text(json.dumps({"<mask>": n - 1}), encoding="utf-8")
    decoder = {str(n - 1): {"content": "<mask>", "special": True}}
    (path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": decoder}), encoding="utf-8"
    )


def test_choose_keeps_used_pieces_then_the_most_frequent():
    vocab = [("a", 5), ("b", 50), ("c", 40), ("d", 1)]
    keep = V.choose(vocab, {"d"}, n_entries=len(V.SPECIAL) + 1 + 2)
    assert list(keep) == [1, 3]  # "d" (used), then "b" (most frequent), in vocab order
    with pytest.raises(ValueError, match="exceed"):
        V.choose(vocab, {"a", "b", "c"}, n_entries=len(V.SPECIAL) + 1 + 2)


def test_trimmed_model_gives_the_same_logits_on_kept_pieces(tmp_path):
    from vifeedback.inference.phobert_tokenizer import PhobertBPE

    src, dst = tmp_path / "src", tmp_path / "dst"
    _checkpoint(src)
    used = {"a", "b", "c", "f", "g"}
    keep = V.choose(V.read_vocab(src), used, n_entries=len(V.SPECIAL) + 1 + 6)
    V.write_trimmed(src, dst, keep)

    from transformers import AutoModelForSequenceClassification

    full = AutoModelForSequenceClassification.from_pretrained(src).eval()
    small = AutoModelForSequenceClassification.from_pretrained(dst).eval()
    assert small.config.vocab_size == len(V.SPECIAL) + 1 + 6
    texts = ["a b c", "f g c a"]
    for model, d in ((full, src), (small, dst)):
        enc = PhobertBPE(d)(texts, max_length=16)
        with torch.no_grad():
            out = model(
                input_ids=torch.as_tensor(enc["input_ids"]),
                attention_mask=torch.as_tensor(enc["attention_mask"]),
            ).logits.numpy()
        assert (enc["input_ids"] != V.SPECIAL["<unk>"]).all()  # every piece is a real token
        if model is full:
            ref = out
    np.testing.assert_allclose(out, ref, atol=1e-6)
    # a removed piece becomes <unk> in the trimmed tokenizer
    assert PhobertBPE(src).encode("e", 8)[1] != V.SPECIAL["<unk>"]
    assert PhobertBPE(dst).encode("d", 8)[1] != V.SPECIAL["<unk>"]  # filled by frequency
    assert PhobertBPE(dst).encode("e", 8)[1] == V.SPECIAL["<unk>"]
    assert json.loads((dst / "added_tokens.json").read_text())["<mask>"] == len(V.SPECIAL) + 6
    assert V.removed_share(PhobertBPE(src), used, ["a e"])["pieces_removed"] == 0.5
