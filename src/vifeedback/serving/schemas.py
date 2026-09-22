"""Request and response contracts.

These types are the API's public surface: changing one is a breaking change for every caller, so
they live apart from the app and are covered by their own tests.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

MAX_BATCH = 64
MAX_CHARS = 2000


class ClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)
    task: Literal["sentiment", "topic"] = "sentiment"
    return_probabilities: bool = True

    @field_validator("texts")
    @classmethod
    def _non_empty_and_bounded(cls, v: list[str]) -> list[str]:
        # An unbounded input is a denial-of-service surface, and a whitespace-only one produces a
        # confident prediction from no evidence. Both are rejected at the edge rather than in the
        # model.
        for i, t in enumerate(v):
            if not t.strip():
                raise ValueError(f"texts[{i}] is empty or whitespace-only")
            if len(t) > MAX_CHARS:
                raise ValueError(f"texts[{i}] exceeds {MAX_CHARS} characters ({len(t)})")
        return v


class Prediction(BaseModel):
    text: str
    label: str
    label_id: int
    confidence: float
    probabilities: dict[str, float] | None = None


class ClassifyResponse(BaseModel):
    predictions: list[Prediction]
    task: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadyResponse(BaseModel):
    """`ready` is false until a model is actually loaded.

    A readiness probe that turns green before the model exists is worse than no probe: the
    orchestrator routes traffic to a process that can only fail.
    """

    ready: bool
    models_loaded: list[str]
    detail: str | None = None


class VersionResponse(BaseModel):
    service_version: str
    model_version: str
    git_sha: str | None = None
    runtime: str
    max_length: int
