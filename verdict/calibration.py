"""Qdrant-backed calibration store for historical incident rates."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

if TYPE_CHECKING:
    from verdict.classifier import ProposedStep, RiskClassifier, RiskScore

EmbedFn = Callable[[str], Sequence[float]]


@dataclass
class IncidentRecord:
    task_class: str
    description: str
    incident: float
    fan_out: int
    risk_score: float
    repo: str
    notes: str = ""
    vector: list[float] = field(default_factory=list)


class CalibrationStore:
    def __init__(
        self,
        client: QdrantClient,
        embed_fn: EmbedFn,
        collection: str = "verdict_incidents",
        vector_size: int | None = None,
    ) -> None:
        self.client = client
        self.embed_fn = embed_fn
        self.collection = collection
        probe = list(embed_fn("verdict-probe"))
        self.vector_size = vector_size or len(probe)
        if self.vector_size <= 0:
            raise ValueError("embed_fn must return a non-empty vector")
        names = {c.name for c in client.get_collections().collections}
        if collection not in names:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            for field_name, schema in [
                ("task_class", PayloadSchemaType.KEYWORD),
                ("incident", PayloadSchemaType.FLOAT),
                ("fan_out", PayloadSchemaType.INTEGER),
                ("repo", PayloadSchemaType.KEYWORD),
            ]:
                client.create_payload_index(
                    collection_name=collection,
                    field_name=field_name,
                    field_schema=schema,
                )

    def record(
        self,
        *,
        repo: str,
        step: ProposedStep,
        risk: RiskScore,
        label: float,
        notes: str | None = None,
    ) -> None:
        vector = list(self.embed_fn(f"{step.task_class}: {step.description}"))
        if len(vector) != self.vector_size:
            raise ValueError(f"embed_fn returned dim={len(vector)}, expected {self.vector_size}")
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "task_class": step.task_class,
                        "description": step.description,
                        "incident": float(label),
                        "fan_out": risk.fan_out,
                        "risk_score": risk.value,
                        "repo": repo,
                        "notes": notes or "",
                    },
                )
            ],
        )

    def historical_rate(
        self,
        *,
        task_class: str,
        description: str,
        fan_out: int,
        top_k: int = 20,
    ) -> float:
        del fan_out  # reserved for future payload-side boosting
        hits = self.client.query_points(
            collection_name=self.collection,
            query=list(self.embed_fn(f"{task_class}: {description}")),
            query_filter=Filter(
                must=[FieldCondition(key="task_class", match=MatchValue(value=task_class))]
            ),
            limit=top_k,
            with_payload=True,
        ).points
        if not hits:
            return 0.2  # cold-start prior
        num = 0.0
        den = 0.0
        for hit in hits:
            if not hit.payload:
                continue
            score = float(hit.score or 0.0)
            num += score * float(hit.payload["incident"])
            den += score
        if den <= 0.0:
            return 0.2
        return max(0.0, min(1.0, num / den))

    def recalibrate(
        self,
        classifier: RiskClassifier,
        label: float,
        iteration: int,
        max_iterations: int,
    ) -> None:
        """Called after a step resolves. Nudges tau, never the weights, since
        the weights encode a modeling choice about which signals matter and
        tau encodes a cost tradeoff the team is entitled to move."""
        if label >= 1.0 and iteration >= max_iterations:
            classifier.config.tau = max(0.35, classifier.config.tau - 0.02)


class InMemoryCalibrationStore:
    """Drop-in stub for tests and environments without Qdrant.

    Uses cosine similarity over caller-provided embeddings (or a hashed
    bag-of-tokens fallback) so demos stay dependency-light.
    """

    def __init__(
        self,
        embed_fn: EmbedFn | None = None,
        *,
        cold_start_prior: float = 0.2,
        class_priors: dict[str, float] | None = None,
    ) -> None:
        self.embed_fn = embed_fn or hashed_embed
        self.cold_start_prior = cold_start_prior
        self.class_priors = class_priors or {}
        self.records: list[IncidentRecord] = []

    def record(
        self,
        *,
        repo: str,
        step: ProposedStep,
        risk: RiskScore,
        label: float,
        notes: str | None = None,
    ) -> None:
        vector = list(self.embed_fn(f"{step.task_class}: {step.description}"))
        self.records.append(
            IncidentRecord(
                task_class=step.task_class,
                description=step.description,
                incident=float(label),
                fan_out=risk.fan_out,
                risk_score=risk.value,
                repo=repo,
                notes=notes or "",
                vector=vector,
            )
        )

    def historical_rate(
        self,
        *,
        task_class: str,
        description: str,
        fan_out: int,
        top_k: int = 20,
    ) -> float:
        del fan_out
        pool = [r for r in self.records if r.task_class == task_class]
        if not pool:
            return self.class_priors.get(task_class, self.cold_start_prior)
        query = list(self.embed_fn(f"{task_class}: {description}"))
        ranked = sorted(
            ((_cosine(query, r.vector), r) for r in pool),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]
        num = sum(score * r.incident for score, r in ranked)
        den = sum(score for score, _ in ranked) or 1.0
        return max(0.0, min(1.0, num / den))

    def recalibrate(
        self,
        classifier: RiskClassifier,
        label: float,
        iteration: int,
        max_iterations: int,
    ) -> None:
        if label >= 1.0 and iteration >= max_iterations:
            classifier.config.tau = max(0.35, classifier.config.tau - 0.02)


def hashed_embed(text: str, dims: int = 64) -> list[float]:
    """Deterministic bag-of-tokens embedding for demos/tests (no API key)."""
    vec = [0.0] * dims
    for token in text.lower().split():
        idx = hash(token) % dims
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return max(0.0, dot / (na * nb))
