"""H7 LLM reference: prompt construction, sampling and the API label mapping (no model, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from vifeedback.evaluation import llm_reference as L


class TestLabelMapping:
    def test_pools_variants_of_the_same_label(self):
        top = [
            {"token": "positive", "logprob": np.log(0.5)},
            {"token": " positive", "logprob": np.log(0.2)},
            {"token": "negative", "logprob": np.log(0.2)},
            {"token": "neutral", "logprob": np.log(0.1)},
        ]
        lp = L.label_logprobs_from_top(top)
        assert L.LABELS == ["negative", "neutral", "positive"]
        assert np.exp(lp) == pytest.approx([0.2, 0.1, 0.7])

    def test_ambiguous_prefix_is_ignored_and_missing_label_gets_the_floor(self):
        top = [
            {"token": "ne", "logprob": -0.1},  # prefix of negative and neutral: ignored
            {"token": "Pos", "logprob": -1.0},
            {"token": "neut", "logprob": -2.0},
        ]
        lp = L.label_logprobs_from_top(top)
        assert lp[L.LABELS.index("positive")] == pytest.approx(-1.0)
        assert lp[L.LABELS.index("neutral")] == pytest.approx(-2.0)
        assert lp[L.LABELS.index("negative")] == pytest.approx(-3.0)  # min(-0.1,-1,-2) - 1


class TestPrompts:
    def test_messages_put_demos_before_the_sentence(self):
        msgs = L.messages("câu hỏi", "v2_policy", [("a", "positive"), ("b", "neutral")])
        assert [m["role"] for m in msgs] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]
        assert msgs[-1]["content"] == "Feedback: câu hỏi"
        assert "suggestion" in msgs[0]["content"]

    def test_at_most_four_variants_all_ending_with_the_answer_format(self):
        assert len(L.VARIANTS) <= 4  # cycle2.yaml: prompt development budget
        for parts in L.VARIANTS.values():
            assert parts[-1].startswith("Answer with exactly one word")


class TestSampling:
    y = np.array([0] * 500 + [1] * 100 + [2] * 500)

    def test_prompt_dev_subset_has_a_minority_floor_and_is_fixed(self):
        idx = L.prompt_dev_subset(self.y)
        assert len(idx) == 200
        assert (self.y[idx] == 1).sum() == 40
        assert np.array_equal(idx, L.prompt_dev_subset(self.y))

    def test_demos_are_balanced_short_and_avoid_the_dev_subset(self):
        texts = [f"câu {i}" for i in range(len(self.y))]
        texts[0] = " ".join(["dài"] * 40)
        exclude = L.prompt_dev_subset(self.y)
        demos = L.draw_demos(texts, self.y, seed=1, per_class=2, exclude=exclude)
        assert sorted(lab for _, lab in demos) == sorted(L.LABELS * 2)
        used = {texts.index(t) for t, _ in demos}
        assert not used & set(exclude.tolist()) and 0 not in used
        assert demos != L.draw_demos(texts, self.y, seed=2, per_class=2, exclude=exclude)


def test_holm_step_down():
    assert L.holm([0.01, 0.04]) == [True, True]
    assert L.holm([0.03, 0.04]) == [False, False]  # 0.03 > 0.05 / 2
    assert L.holm([0.2, 0.001]) == [False, True]


def test_api_refuses_without_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(L.paths, "ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        L.OpenAIScorer()


def test_committed_summaries_record_demo_indices_not_corpus_text():
    """Result files never hold UIT-VSFC sentences: demonstrations are stored as train indices."""
    import json

    for p in L.OUT.glob("*/*/summary.json"):
        for d in json.loads(p.read_text(encoding="utf-8")).get("demos") or []:
            assert set(d) == {"train_index", "label"}, p


class _FakeTok:
    """Prompt "p<k>" encodes to k + 1 copies of token k, left-padded with 0."""

    pad_token_id = 0

    def __call__(self, prompts, **kw):
        import torch

        ids = [[int(p[1:])] * (int(p[1:]) + 1) for p in prompts]
        if "return_tensors" not in kw:
            return {"input_ids": ids}
        width = max(map(len, ids))
        return _Enc(torch.tensor([[0] * (width - len(x)) + x for x in ids]))


class _Enc(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)

    def to(self, device):
        return self


class _FakeLM:
    """Last-position logits: token id in column 0, zeros elsewhere; records batch sizes."""

    def __init__(self, fill=None):
        self.fill, self.batches = fill, []

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, logits_to_keep=None):
        import torch

        self.batches.append(tuple(input_ids.shape))
        logits = torch.zeros(input_ids.shape[0], 1, 5)
        logits[:, -1, 0] = input_ids[:, -1].float()
        if self.fill is not None:
            logits[:] = self.fill
        return type("Out", (), {"logits": logits})()

    __call__ = forward


def _scorer(model):
    s = L.HFScorer.__new__(L.HFScorer)
    s.device, s.single_token, s.label_ids = "cpu", True, [[0], [1], [2]]
    s.tok, s.model = _FakeTok(), model
    s.prefix_cache, s.prefix_cache_check = True, {}
    return s


def test_scores_come_back_in_input_order_within_the_token_budget():
    pytest.importorskip("torch")
    ks = [7, 1, 12, 3, 9, 0, 5]
    lm = _FakeLM()
    out = _scorer(lm).score([f"p{k}" for k in ks], batch_size=3, max_tokens=20)
    assert list(np.argsort(out[:, 0])) == list(np.argsort(ks))  # order restored
    assert all((n <= 3 and n * width <= 20) or n == 1 for n, width in lm.batches)
    assert sum(n for n, _ in lm.batches) == len(ks)


def test_non_finite_scores_are_refused():
    """An fp16 overflow must stop the run, not turn NaN into a label."""
    pytest.importorskip("torch")
    with pytest.raises(FloatingPointError, match="non-finite"):
        _scorer(_FakeLM(fill=float("nan"))).score(["p1"])


def _arm(diff: float, p: float) -> dict:
    return {
        "model": "m",
        "revision": "r",
        "per_class": {"neutral": {"f1": 0.7 + diff}},
        "vs_encoder": {
            "ce": {
                "neutral_f1": {
                    "observed_diff": diff,
                    "ci_low": diff - 0.1,
                    "ci_high": diff + 0.1,
                    "p_value": p,
                }
            }
        },
    }


def test_holm_adjusted_matches_the_step_down():
    adj = L.holm_adjusted([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])  # 3*0.01; max(2*0.03, 0.03); max(1*0.04, 0.06)
    for p in ([0.01, 0.04], [0.03, 0.04], [0.02, 0.2, 0.001]):
        assert [a <= 0.05 for a in L.holm_adjusted(p)] == L.holm(p)


def test_h7_waits_for_both_arms():
    d = L.h7_decide({"qwen3-4b": None, "gpt-4o-mini": _arm(0.24, 0.0002)})
    assert not d["decided"] and d["missing"] == ["qwen3-4b"] and d["outcome"] is None


def test_h7_needs_holm_and_a_positive_difference():
    d = L.h7_decide({"qwen3-4b": _arm(-0.30, 0.0002), "gpt-4o-mini": _arm(0.24, 0.0002)})
    assert d["decided"] and d["better_on_neutral"] == [
        "gpt-4o-mini"
    ]  # a significant loss is not "better"
    assert d["outcome"] == "llm_better_on_neutral"
    d = L.h7_decide({"qwen3-4b": _arm(0.05, 0.04), "gpt-4o-mini": _arm(0.05, 0.03)})
    assert (
        d["better_on_neutral"] == [] and d["outcome"] == "encoder_better_or_equal"
    )  # 0.03 > 0.05 / 2


def _tiny_qwen3_scorer():
    """A 2-layer Qwen3 with random weights and a word-level tokenizer: real attention, masks,
    rotary positions and KV cache, small enough for CPU in CI."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3ForCausalLM"):
        pytest.skip("transformers without Qwen3")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    words = [
        "[PAD]",
        "[UNK]",
        *L.LABELS,
        "sys",
        "rule",
        "one",
        "two",
        "Feedback:",
        "a",
        "b",
        "c",
        "d",
        "e",
        "end",
    ]
    vocab = {w: i for i, w in enumerate(words)}
    tk = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]"
    )
    tok.padding_side = "left"
    torch.manual_seed(0)
    cfg = transformers.Qwen3Config(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
    )
    s = L.HFScorer.__new__(L.HFScorer)
    s.device, s.tok = "cpu", tok
    s.model = transformers.Qwen3ForCausalLM(cfg).eval()
    s.label_ids = [tok.encode(lab, add_special_tokens=False) for lab in L.LABELS]
    s.single_token = all(len(i) == 1 for i in s.label_ids)
    s.prefix_cache, s.prefix_cache_check = True, {}
    return s


PREFIX = " ".join(["sys", "rule", "one", "two"] * 5) + " Feedback:"
TEXTS = ["a", "b c", "a b c d e", "e", "d d", "c a b", "a b c d e a b c d e end", "b", "e d c b a"]


def test_prefix_cache_scores_equal_full_scores_on_a_real_model():
    s = _tiny_qwen3_scorer()
    prompts = [f"{PREFIX} {t} end" for t in TEXTS]
    cached = s.score(prompts, batch_size=4, max_tokens=200)  # several padded batches
    report = dict(s.prefix_cache_check)
    s.prefix_cache = False
    full = s.score(prompts, batch_size=4, max_tokens=200)
    assert report["used"] and report["passed"]
    assert report["shared_prefix_tokens"] == len(s.tok(PREFIX, add_special_tokens=False).input_ids)
    assert np.abs(cached - full).max() < 1e-4
    one_by_one = np.vstack([s.score([p]) for p in prompts])  # no padding at all
    assert np.abs(cached - one_by_one).max() < 1e-4


def test_no_shared_prefix_means_full_scoring():
    s = _tiny_qwen3_scorer()
    s.score(["a b end", "c d end", "e end"])
    assert (
        s.prefix_cache_check["used"] is False and s.prefix_cache_check["shared_prefix_tokens"] == 0
    )


def test_a_failed_self_check_falls_back_to_full_scoring(monkeypatch):
    s = _tiny_qwen3_scorer()
    prompts = [f"{PREFIX} {t} end" for t in TEXTS]
    s.prefix_cache = False
    full = s.score(prompts)
    s.prefix_cache = True
    wrong = L.HFScorer._forward_cached
    monkeypatch.setattr(L.HFScorer, "_forward_cached", lambda self, *a: wrong(self, *a) + 0.5)
    out = s.score(prompts)
    assert s.prefix_cache_check["passed"] is False and s.prefix_cache_check["used"] is False
    assert np.abs(out - full).max() < 1e-6
