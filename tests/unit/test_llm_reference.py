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
