"""Cycle 2 H7: an instruction-tuned LLM as a reference point on the encoder's hard cases (Study E).

Scoring is **label likelihood**, not generation. The model reads the prompt, and each label's
log-probability as the answer is read off. Every output is a valid label, with probabilities, and
there is no parsing to fail. Two backends share the prompts:

* ``hf``: a local Hugging Face causal LM (Qwen3), thinking disabled, revision pinned.
* ``openai``: the Chat Completions API with ``max_tokens=1`` and the top-20 log-probabilities of the
  answer token. A label absent from them gets the smallest returned log-probability minus 1
  (``cycle2.yaml`` v2). The key is read from ``OPENAI_API_KEY`` or a git-ignored ``.env``; it is
  never logged or written anywhere.

Prompt development uses the pilot model only, on a fixed train subset. The chosen variant is frozen
in ``results/studies/llm_reference/prompt_dev.json`` and reused unchanged for every other run.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vifeedback import paths
from vifeedback.constants import label_names

LABELS = label_names("sentiment")  # ["negative", "neutral", "positive"], the encoder's order
OUT = paths.RESULTS / "studies" / "llm_reference"
PROMPT_DEV_FILE = OUT / "prompt_dev.json"

API_MODEL = "gpt-4o-mini-2024-07-18"
# USD per 1M tokens, OpenAI's published gpt-4o-mini price (input, output). Recorded with each run so
# a later price change does not silently rewrite the cost column.
API_PRICE_PER_M = (0.15, 0.60)

# --- prompts ------------------------------------------------------------------------------------

_TASK = (
    "You classify the sentiment of Vietnamese student feedback about a university course "
    "(teaching, curriculum, facilities, the lecturer)."
)
_DEFS = (
    "Labels:\n"
    "- positive: the student expresses approval, satisfaction or praise.\n"
    "- negative: the student expresses disapproval, dissatisfaction, a complaint or a problem.\n"
    "- neutral: the student expresses no evaluative attitude, or one too balanced or too unclear "
    "to call."
)
# ANNOTATION_GUIDE § 2: the corpus's own convention (91.1% of nên/cần/mong sentences are negative).
_SUGGESTION = (
    "A suggestion or a request for change counts as negative, because it implies the current state "
    "falls short. A question or a request for information is neutral."
)
_INFORMAL = "The text is lowercase and may lack Vietnamese diacritics or use abbreviations."
_CONTRAST = (
    "If a sentence contrasts two evaluations (A nhưng B, tuy A nhưng B), the part after the "
    "contrast word decides the label."
)
_ANSWER = "Answer with exactly one word: positive, negative or neutral."

VARIANTS: dict[str, list[str]] = {
    "v1_definitions": [_TASK, _DEFS, _ANSWER],
    "v2_policy": [_TASK, _DEFS, _SUGGESTION, _ANSWER],
    "v3_policy_informal": [_TASK, _DEFS, _SUGGESTION, _INFORMAL, _ANSWER],
    "v4_policy_informal_contrast": [_TASK, _DEFS, _SUGGESTION, _INFORMAL, _CONTRAST, _ANSWER],
}


def messages(
    text: str, variant: str, demos: Sequence[tuple[str, str]] = ()
) -> list[dict[str, str]]:
    """System prompt, optional demonstrations as earlier turns, then the sentence."""
    msgs = [{"role": "system", "content": "\n\n".join(VARIANTS[variant])}]
    for t, label in demos:
        msgs += [
            {"role": "user", "content": f"Feedback: {t}"},
            {"role": "assistant", "content": label},
        ]
    msgs.append({"role": "user", "content": f"Feedback: {text}"})
    return msgs


# --- data ---------------------------------------------------------------------------------------


def prompt_dev_subset(
    y: np.ndarray, n: int = 200, per_class_min: int = 40, seed: int = 0
) -> np.ndarray:
    """A fixed train subset for prompt development: stratified, with a floor for the minority.

    Proportional allocation would give neutral about 8 of 200 rows, too few to tell prompts apart
    on macro-F1. The floor makes it 40; the rest is split in proportion.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    take = np.maximum(per_class_min, np.floor(n * counts / counts.sum())).astype(int)
    while take.sum() > n:
        take[np.argmax(take)] -= 1
    idx = [
        rng.choice(np.flatnonzero(y == c), size=t, replace=False)
        for c, t in zip(classes, take, strict=True)
    ]
    return np.sort(np.concatenate(idx))


def draw_demo_indices(
    texts: Sequence[str],
    y: np.ndarray,
    seed: int,
    per_class: int = 2,
    exclude: Sequence[int] = (),
    max_words: int = 30,
) -> list[int]:
    """Few-shot demonstrations from train: ``per_class`` per label, uniformly among sentences of at
    most ``max_words`` words, never from the prompt-development subset. Shuffled, so no label
    always comes last.

    Returns train row indices: results record these, never the sentences, because corpus text is
    not committed."""
    rng = np.random.default_rng(seed)
    ok = np.array([len(t.split()) <= max_words for t in texts])
    ok[list(exclude)] = False
    picks = [
        int(i)
        for c in range(len(LABELS))
        for i in rng.choice(
            np.flatnonzero(ok & (np.asarray(y) == c)), size=per_class, replace=False
        )
    ]
    rng.shuffle(picks)
    return picks


def draw_demos(
    texts: Sequence[str], y: np.ndarray, seed: int, per_class: int = 2, **kw: Any
) -> list[tuple[str, str]]:
    """The demonstrations themselves, for building prompts (see `draw_demo_indices`)."""
    idx = draw_demo_indices(texts, y, seed, per_class, **kw)
    return [(texts[i], LABELS[int(y[i])]) for i in idx]


# --- scoring: local HF model --------------------------------------------------------------------

# A shared prefix shorter than this is not worth a cache. The tolerance sits above fp16 rounding
# (measured with Qwen3-0.6B: full scoring batched vs one prompt at a time differs by up to 0.11 in
# log-probability, and flipped 4 of 800 near-tied NEU-ESC labels; cached scoring flipped 1 of the
# same 800 against one-at-a-time) and far below what a wrong mask or position would cause.
MIN_SHARED_PREFIX = 16
PREFIX_CACHE_TOLERANCE = 0.25


def _common_prefix_len(seqs: Sequence[Sequence[int]]) -> int:
    """Length of the longest token prefix shared by every sequence."""
    first = seqs[0]
    n = len(first)
    for s in seqs[1:]:
        m = min(n, len(s))
        i = 0
        while i < m and s[i] == first[i]:
            i += 1
        n = i
        if n == 0:
            break
    return n


def _token_batches(lengths: Sequence[int], batch_size: int, max_tokens: int) -> list[list[int]]:
    """Indices shortest first, cut into batches of <= batch_size items and <= max_tokens padded tokens."""
    batches: list[list[int]] = []
    cur: list[int] = []
    width = 0
    for i in np.argsort(lengths, kind="stable"):
        w = max(width, lengths[i])
        if cur and (len(cur) >= batch_size or w * (len(cur) + 1) > max_tokens):
            batches.append(cur)
            cur, w = [], lengths[i]
        cur.append(int(i))
        width = w
    if cur:
        batches.append(cur)
    return batches


class HFScorer:
    """Label log-likelihoods from a local causal LM, thinking disabled (Qwen3 chat template)."""

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str = "auto",
        dtype: str = "float16",
        prefix_cache: bool = True,
    ):
        import torch
        from huggingface_hub import model_info
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision or model_info(model_id).sha
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=self.revision)
        self.tok.padding_side = "left"
        import transformers
        from packaging.version import Version

        # `dtype` replaced `torch_dtype` in 4.56; older versions (Kaggle images) would ignore it
        # and load fp32, which does not fit a T4 at 4B parameters.
        kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_id, revision=self.revision, **{kw: getattr(torch, dtype)}
        )
        self.model = model.to(self.device).eval()
        self.label_ids = [self.tok.encode(lab, add_special_tokens=False) for lab in LABELS]
        self.single_token = all(len(ids) == 1 for ids in self.label_ids)
        self.prefix_cache = prefix_cache
        self.prefix_cache_check: dict[str, Any] = {}

    def prompt(self, msgs: list[dict[str, str]]) -> str:
        return str(
            self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        )

    def _forward_kwargs(self) -> dict[str, Any]:
        """Scoring reads one position: skip the KV cache, and compute logits for the last token only
        when the model supports it (the vocabulary projection over every position costs memory)."""
        import inspect

        params = inspect.signature(self.model.forward).parameters
        kw: dict[str, Any] = {"use_cache": False}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in params:
                kw[name] = 1
                break
        return kw

    def score(self, prompts: list[str], batch_size: int = 8, max_tokens: int = 8192) -> np.ndarray:
        """(n, 3) summed label log-probabilities, in the order of `prompts`.

        Prompts are scored shortest first, in batches of at most `batch_size` prompts and
        `max_tokens` padded tokens, so padding is small and long few-shot prompts still fit. The
        score of a prompt does not depend on its batch beyond floating-point rounding.

        Every prompt of a configuration starts with the same system prompt and demonstrations,
        82-95% of its tokens. With `prefix_cache` that shared prefix is run once and its key/value
        cache reused, so only each prompt's own tokens are computed. Before relying on it, a
        self-check scores prompts of different lengths both ways; any disagreement, or a
        transformers version without the cache API, falls back to full scoring. What happened is
        kept in `prefix_cache_check` and written to the run's summary.
        """
        out = np.empty((len(prompts), len(LABELS)))
        if not self.single_token:
            for j, p in enumerate(prompts):
                out[j] = self._score_multi(p)
        else:
            ids = [list(x) for x in self.tok(prompts, add_special_tokens=False)["input_ids"]]
            shared = min(_common_prefix_len(ids), min(map(len, ids)) - 1)
            self.prefix_cache_check = {"used": False, "shared_prefix_tokens": int(shared)}
            done = False
            if self.prefix_cache and shared >= MIN_SHARED_PREFIX:
                try:
                    check = self._check_prefix_cache(ids, shared)
                    self.prefix_cache_check.update(check)
                    if check["passed"]:
                        out[:] = self._score_cached(ids, shared, batch_size, max_tokens)
                        self.prefix_cache_check["used"] = True
                        done = True
                except (AttributeError, TypeError, ValueError, NotImplementedError) as e:
                    # An older or newer transformers without the cache calls used here.
                    self.prefix_cache_check["error"] = f"{type(e).__name__}: {e}"[:300]
            if not done:
                out[:] = self._score_full(ids, batch_size, max_tokens)
        if not np.isfinite(out).all():
            # fp16 can overflow in larger models; a NaN score would silently become a label.
            raise FloatingPointError(
                f"{int((~np.isfinite(out)).any(axis=1).sum())} of {len(out)} prompts gave non-finite "
                "label log-probabilities; re-run with --dtype bfloat16 (Ampere or newer) or float32"
            )
        return out

    def _pad_id(self) -> int:
        pad = self.tok.pad_token_id
        return int(pad if pad is not None else self.tok.eos_token_id)

    def _forward_full(self, batch: list[list[int]]) -> np.ndarray:
        """Label log-probabilities of whole prompts, left-padded into one batch."""
        import torch

        width = max(map(len, batch))
        pad = self._pad_id()
        x = torch.tensor([[pad] * (width - len(s)) + s for s in batch], device=self.device)
        mask = torch.tensor(
            [[0] * (width - len(s)) + [1] * len(s) for s in batch], device=self.device
        )
        kw = self._forward_kwargs()
        with torch.inference_mode():
            logits = self.model(input_ids=x, attention_mask=mask, **kw).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
        return lp[:, [ids[0] for ids in self.label_ids]].cpu().numpy()

    def _forward_cached(self, cache: Any, batch: list[list[int]], shared: int) -> np.ndarray:
        """Label log-probabilities from each prompt's own tokens after the cached shared prefix.

        The own tokens are left-padded, so the pads sit between the prefix and the text: the mask
        hides them and the position ids continue from the prefix without gaps, which is exactly
        the unpadded prompt. The last position is then every prompt's last token.
        """
        import copy

        import torch

        width = max(len(s) - shared for s in batch)
        pad = self._pad_id()
        x, mask, pos = [], [], []
        for s in batch:
            own = s[shared:]
            p = width - len(own)
            x.append([pad] * p + own)
            mask.append([1] * shared + [0] * p + [1] * len(own))
            pos.append([shared] * p + list(range(shared, shared + len(own))))
        kv = copy.deepcopy(cache)
        kv.batch_repeat_interleave(len(batch))
        kw = {**self._forward_kwargs(), "use_cache": True}
        with torch.inference_mode():
            logits = (
                self.model(
                    input_ids=torch.tensor(x, device=self.device),
                    attention_mask=torch.tensor(mask, device=self.device),
                    position_ids=torch.tensor(pos, device=self.device),
                    past_key_values=kv,
                    **kw,
                )
                .logits[:, -1, :]
                .float()
            )
            lp = torch.log_softmax(logits, dim=-1)
        return lp[:, [ids[0] for ids in self.label_ids]].cpu().numpy()

    def _prefix_cache(self, ids: list[list[int]], shared: int) -> Any:
        import torch

        with torch.inference_mode():
            prefix = torch.tensor([ids[0][:shared]], device=self.device)
            return self.model(input_ids=prefix, use_cache=True).past_key_values

    def _check_prefix_cache(self, ids: list[list[int]], shared: int, k: int = 8) -> dict[str, Any]:
        """Score k prompts spread over the length range both ways, in one padded batch each."""
        order = np.argsort([len(s) for s in ids], kind="stable")
        pick = sorted({int(order[int(q)]) for q in np.linspace(0, len(ids) - 1, min(k, len(ids)))})
        batch = [ids[i] for i in pick]
        full = self._forward_full(batch)
        cached = self._forward_cached(self._prefix_cache(ids, shared), batch, shared)
        diff = float(np.abs(full - cached).max())
        same = bool((full.argmax(1) == cached.argmax(1)).all())
        return {
            "check_prompts": len(batch),
            "check_max_abs_diff": diff,
            "check_same_labels": same,
            "passed": bool(same and diff <= PREFIX_CACHE_TOLERANCE),
        }

    def _score_cached(
        self, ids: list[list[int]], shared: int, batch_size: int, max_tokens: int
    ) -> np.ndarray:
        out = np.empty((len(ids), len(LABELS)))
        cache = self._prefix_cache(ids, shared)
        # The budget counts the prefix too: attention and the copied cache span the whole prompt.
        for idx in _token_batches([len(s) for s in ids], batch_size, max_tokens):
            out[idx] = self._forward_cached(cache, [ids[i] for i in idx], shared)
        return out

    def _score_full(self, ids: list[list[int]], batch_size: int, max_tokens: int) -> np.ndarray:
        out = np.empty((len(ids), len(LABELS)))
        for idx in _token_batches([len(s) for s in ids], batch_size, max_tokens):
            out[idx] = self._forward_full([ids[i] for i in idx])
        return out

    def _score_multi(self, prompt: str) -> np.ndarray:
        """Labels of several tokens: one sequence per label, summed log-probabilities."""
        import torch

        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        seqs = [p_ids + ids for ids in self.label_ids]
        width = max(map(len, seqs))
        pad = self.tok.pad_token_id or 0
        x = torch.tensor([[pad] * (width - len(q)) + q for q in seqs], device=self.device)
        mask = torch.tensor(
            [[0] * (width - len(q)) + [1] * len(q) for q in seqs], device=self.device
        )
        lp = torch.log_softmax(self.model(input_ids=x, attention_mask=mask).logits.float(), dim=-1)
        scores = []
        for r, ids in enumerate(self.label_ids):
            end = width  # left padding: the label occupies the last len(ids) positions
            pos = range(end - len(ids), end)
            scores.append(
                sum(float(lp[r, t - 1, tok_id]) for t, tok_id in zip(pos, ids, strict=True))
            )
        return np.array(scores)


# --- scoring: OpenAI API ------------------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    env = paths.ROOT / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "OPENAI_API_KEY":
                key = value.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your own terminal or in .env (git-ignored); "
            "never paste it into a chat or a tracked file."
        )
    return key


def label_logprobs_from_top(top: list[dict[str, Any]]) -> np.ndarray:
    """Map the answer token's top log-probabilities onto the three labels.

    A token counts for a label if, lowercased and stripped, it is a non-empty prefix of that label
    and of no other ("neu" -> neutral; "ne" is ambiguous and ignored). Several matching tokens
    ("positive", " positive", "Positive") are pooled with log-sum-exp. A label with no match gets
    the smallest returned log-probability minus 1.
    """
    floor = min(t["logprob"] for t in top) - 1.0
    out = []
    for lab in LABELS:
        hits = []
        for t in top:
            s = t["token"].strip().lower()
            if s and lab.startswith(s) and sum(other.startswith(s) for other in LABELS) == 1:
                hits.append(t["logprob"])
        out.append(float(np.logaddexp.reduce(hits)) if hits else floor)
    return np.array(out)


class OpenAIScorer:
    """Label log-likelihoods from the first answer token of the Chat Completions API."""

    URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model: str = API_MODEL, workers: int = 4, timeout: float = 60.0):
        self.model_id = model
        self.revision = model  # a dated snapshot is the API's only pin
        self.workers, self.timeout = workers, timeout
        self._key = _api_key()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
        self.served_models: set[str] = set()

    def _call(self, msgs: list[dict[str, str]]) -> np.ndarray:
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self.model_id,
                "messages": msgs,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 20,
                "seed": 0,
            }
        ).encode()
        for attempt in range(6):
            req = urllib.request.Request(
                self.URL,
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self._key}",
                    "Content-Type": "application/json",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2**attempt)
                    continue
                # The error body never contains the key; the request headers are not printed.
                raise RuntimeError(f"OpenAI API error {e.code}: {e.read()[:300]!r}") from None
            except (urllib.error.URLError, TimeoutError):
                if attempt < 5:
                    time.sleep(2**attempt)
                    continue
                raise
        self.usage["prompt_tokens"] += resp["usage"]["prompt_tokens"]
        self.usage["completion_tokens"] += resp["usage"]["completion_tokens"]
        self.usage["calls"] += 1
        self.served_models.add(resp.get("model", ""))
        return label_logprobs_from_top(resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"])

    def score_messages(self, batch: list[list[dict[str, str]]]) -> np.ndarray:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            return np.array(list(pool.map(self._call, batch)))

    def cost_usd(self) -> float:
        i, o = API_PRICE_PER_M
        return (self.usage["prompt_tokens"] * i + self.usage["completion_tokens"] * o) / 1e6


# --- running and reporting ----------------------------------------------------------------------


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run(
    scorer: Any,
    texts: Sequence[str],
    variant: str,
    demos: Sequence[tuple[str, str]] = (),
    batch_size: int = 8,
    max_tokens: int = 8192,
) -> dict[str, Any]:
    """Score every text; returns log-likelihoods, probabilities and timing."""
    msgs = [messages(t, variant, demos) for t in texts]
    t0 = time.perf_counter()
    if isinstance(scorer, OpenAIScorer):
        ll = scorer.score_messages(msgs)
    else:
        ll = scorer.score(
            [scorer.prompt(m) for m in msgs], batch_size=batch_size, max_tokens=max_tokens
        )
    seconds = time.perf_counter() - t0
    return {
        "loglik": ll,
        "probs": softmax(ll),
        "seconds": seconds,
        "seconds_per_1k": 1000 * seconds / max(len(texts), 1),
    }


def neutral_f1_from_confusion(cm: np.ndarray) -> float:
    c = LABELS.index("neutral")
    tp, fp, fn = cm[c, c], cm[:, c].sum() - cm[c, c], cm[c, :].sum() - cm[c, c]
    return float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0


def compare_to_encoder(y: np.ndarray, llm_pred: np.ndarray, enc_pred: np.ndarray) -> dict[str, Any]:
    """Paired example bootstrap, LLM minus encoder: neutral F1 (declared) and macro-F1."""
    from vifeedback.evaluation import bootstrap as B

    k = len(LABELS)
    return {
        "neutral_f1": B.paired_bootstrap(
            y, llm_pred, enc_pred, k, n_resamples=10_000, seed=42, metric=neutral_f1_from_confusion
        ),
        "macro_f1": B.paired_bootstrap(y, llm_pred, enc_pred, k, n_resamples=10_000, seed=42),
    }


def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down: which hypotheses are rejected at family-wise level alpha."""
    order = np.argsort(p_values)
    reject = [False] * len(p_values)
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (len(p_values) - rank):
            reject[i] = True
        else:
            break
    return reject


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm-adjusted p-values (step-down, monotone); `holm` rejects exactly where these are <= alpha."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * float(p_values[i])))
        adjusted[i] = running
    return adjusted


# The two declared H7 arms (cycle2.yaml H7_llm_reference; cycle3.yaml v6 h7_decision).
H7_ARMS = {"qwen3-4b": "Qwen/Qwen3-4B", "gpt-4o-mini": API_MODEL}


def h7_decide(arms: dict[str, dict[str, Any] | None], alpha: float = 0.05) -> dict[str, Any]:
    """Apply the H7 rule: Holm over the arms' zero-shot neutral-F1 p-values vs CE on challenge v1.

    `arms` maps each arm to its zero-shot challenge summary.json, or None while that run does not
    exist. With an arm missing nothing is decided (cycle3.yaml v6 h7_decision).
    """
    rows: dict[str, dict[str, Any] | None] = {}
    for name, s in arms.items():
        if s is None:
            rows[name] = None
            continue
        n = s["vs_encoder"]["ce"]["neutral_f1"]
        rows[name] = {
            "model": s["model"],
            "revision": s["revision"],
            "neutral_f1": s["per_class"]["neutral"]["f1"],
            "neutral_f1_minus_ce": n["observed_diff"],
            "ci95": [n["ci_low"], n["ci_high"]],
            "p": n["p_value"],
        }
    missing = [k for k, v in rows.items() if v is None]
    if missing:
        return {"decided": False, "missing": missing, "arms": rows, "outcome": None}
    present = {k: v for k, v in rows.items() if v is not None}
    names = list(present)
    p_values = [present[k]["p"] for k in names]
    for k, rejected, adj in zip(names, holm(p_values, alpha), holm_adjusted(p_values), strict=True):
        present[k]["p_holm"] = adj
        present[k]["better_on_neutral"] = bool(rejected and present[k]["neutral_f1_minus_ce"] > 0)
    better = [k for k in names if present[k]["better_on_neutral"]]
    return {
        "decided": True,
        "missing": [],
        "arms": rows,
        "better_on_neutral": better,
        "outcome": "llm_better_on_neutral" if better else "encoder_better_or_equal",
    }


def frozen_variant() -> str:
    if not PROMPT_DEV_FILE.exists():
        raise FileNotFoundError(
            f"{PROMPT_DEV_FILE} not found: run `vifeedback study llm-prompt-dev` with the pilot first"
        )
    return str(json.loads(PROMPT_DEV_FILE.read_text(encoding="utf-8"))["frozen_variant"])


def slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def write(path: Path, obj: Any) -> None:
    from vifeedback.evaluation.report import yaml_safe

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(yaml_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")
