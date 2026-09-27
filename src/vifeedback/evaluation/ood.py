"""Cycle 3 S3: an out-of-scope score for inputs that are not course feedback (cycle3.yaml).

The encoder has no abstain class, and its max-probability does not separate off-topic text (the
challenge set's 20 off-topic rows get a mean confidence of 0.86-0.90). Three scores are compared,
higher = more in-domain:

* max softmax probability (the baseline);
* negative energy, logsumexp of the logits (Liu et al., 2020);
* negative Mahalanobis distance of the sentence feature to the nearest class mean, with a shared
  covariance fitted on train (Lee et al., 2018).

Method and threshold are chosen on development data (challenge v1 off-topic rows plus U4 generated
sentences); the decision is confirmed on challenge v2's off-topic rows.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

OFF_TOPICS = (
    "thời tiết",
    "mua sắm và giá cả",
    "thể thao",
    "phim ảnh và âm nhạc",
    "ăn uống",
    "giao thông",
    "quảng cáo và tin nhắn rác",
    "chuyện gia đình",
    "sức khỏe",
    "công nghệ và điện thoại",
)


def generate_offtopic(
    n_per_topic: int = 50, seed: int = 0, model: str = "gpt-4o-mini-2024-07-18"
) -> dict[str, Any]:
    """U4 (NEXT_PLAN v3 § 2): off-topic Vietnamese sentences for development only.

    Nothing from the corpus is sent. A third of the sentences are asked for without diacritics or
    with abbreviations, so the set does not reward a detector for spotting informal typing.
    """
    import urllib.request

    from vifeedback.evaluation.llm_reference import _api_key

    key = _api_key()
    rows: list[dict[str, str]] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for i, topic in enumerate(OFF_TOPICS):
        prompt = (
            f"Viết {n_per_topic} câu tiếng Việt ngắn (5-20 từ), khác nhau, như người dùng mạng xã hội "
            f"gõ, về chủ đề: {topic}. Không nhắc tới trường học, môn học, giảng viên hay sinh viên. "
            "Khoảng một phần ba số câu viết không dấu hoặc dùng từ viết tắt. "
            'Trả về JSON dạng {"sentences": ["...", "..."]}.'
        )
        body = json.dumps(
            {
                "model": model,
                "temperature": 1.0,
                "seed": seed + i,
                "response_format": {"type": "json_object"},
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
        usage["prompt_tokens"] += resp["usage"]["prompt_tokens"]
        usage["completion_tokens"] += resp["usage"]["completion_tokens"]
        for s in json.loads(resp["choices"][0]["message"]["content"]).get("sentences", []):
            if isinstance(s, str) and s.strip():
                rows.append({"topic": topic, "text": s.strip()})
    return {"model": model, "seed": seed, "usage": usage, "rows": rows}


def encode(
    checkpoint: str, texts: list[str], batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Sentence features (final <s> hidden state, what the classification head reads) and logits."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(dev).eval()
    feats, logits = [], []
    with torch.inference_mode():
        for s in range(0, len(texts), batch_size):
            enc = tok(
                texts[s : s + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=96,
            ).to(dev)
            out = model(**enc, output_hidden_states=True)
            feats.append(out.hidden_states[-1][:, 0, :].float().cpu().numpy())
            logits.append(out.logits.float().cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def fit_mahalanobis(features: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    means = np.stack([features[y == c].mean(axis=0) for c in np.unique(y)])
    centered = features - means[y]
    cov = centered.T @ centered / len(features)
    precision = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    return {"means": means, "precision": precision}


def scores(
    logits: np.ndarray, features: np.ndarray, maha: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Higher = more in-domain, for every method."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    energy = np.log(np.exp(z).sum(axis=1)) + logits.max(axis=1)
    d = np.stack(
        [
            np.einsum("ij,jk,ik->i", features - m, maha["precision"], features - m)
            for m in maha["means"]
        ],
        axis=1,
    )
    return {
        "max_probability": p.max(axis=1),
        "neg_energy": energy,
        "neg_mahalanobis": -d.min(axis=1),
    }


def evaluate(in_domain: np.ndarray, out_of_scope: np.ndarray, keep: float = 0.95) -> dict[str, Any]:
    """AUROC (in-domain vs out-of-scope), and the threshold keeping `keep` of in-domain rows."""
    from sklearn.metrics import roc_auc_score

    y = np.r_[np.ones(len(in_domain)), np.zeros(len(out_of_scope))]
    s = np.r_[in_domain, out_of_scope]
    thr = float(np.quantile(in_domain, 1 - keep))
    return {
        "auroc": float(roc_auc_score(y, s)),
        "threshold": thr,
        "in_domain_flagged": float((in_domain < thr).mean()),
        "out_of_scope_caught": float((out_of_scope < thr).mean()),
    }
