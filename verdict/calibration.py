"""Component Five — CalibrationStore: Qdrant-backed incident memory.

Every escalation outcome and every autonomous failure caught by the executor's
retry cap becomes a labeled example. Mining these back into the risk classifier
keeps the boundary between "loop it" and "escalate it" from being a fixed guess.

Label table (see ``docs/DESIGN.md``, Component Five)
----------------------------------------------------
======= ============================================= =====
Event   Meaning                                       label
======= ============================================= =====
reject / verifier exhausted                           1.0
escalation redirected                                 0.5
escalation approved / bounded pass (no later revert)  0.0
======= ============================================= =====

``historical_rate`` queries *similar* past steps (embeddings + ``task_class``
filter), not only exact class matches. ``recalibrate`` only ever nudges
``tau``, never the signal weights ``w_v``, ``w_f``, ``w_h``.

Teams without Qdrant yet can use :class:`InMemoryCalibrationStore` — same
public methods, hashed bag-of-tokens embeddings by default.

Public surface owned by this module
-----------------------------------
* ``CalibrationStore``, ``InMemoryCalibrationStore``
* ``IncidentRecord``, ``hashed_embed`` (helpers)
"""

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
    """One labeled calibration example (used by the in-memory store)."""

    task_class: str
    description: str
    incident: float
    fan_out: int
    risk_score: float
    repo: str
    notes: str = ""
    vector: list[float] = field(default_factory=list)


def outcome_label(
    *,
    escalation_outcome: str | None,
    bounded_passed: bool | None,
) -> float:
    """Map a resolved step to the design-spec calibration label.

    Parameters
    ----------
    escalation_outcome:
        ``\"approved\"`` | ``\"redirected\"`` | ``\"rejected\"`` | ``None``
        when the step never entered escalation.
    bounded_passed:
        ``True`` / ``False`` when a bounded run completed; ``None`` if none.

    Returns
    -------
    float
        ``1.0`` (incident), ``0.5`` (redirect), or ``0.0`` (clean / approved).
    """
    if escalation_outcome == "rejected":
        return 1.0
    if escalation_outcome == "redirected":
        return 0.5
    if escalation_outcome == "approved":
        return 0.0
    if bounded_passed is False:
        return 1.0
    return 0.0


class CalibrationStore:
    """Qdrant-backed store for step embeddings + incident labels.

    Parameters
    ----------
    client:
        Connected :class:`qdrant_client.QdrantClient` (URL, cloud, or ``\":memory:\"``).
    embed_fn:
        Maps ``\"{task_class}: {description}\"`` text to a dense vector. Vector
        size is probed once at construction (override with ``vector_size``).
    collection:
        Qdrant collection name (created on first use).
    vector_size:
        Optional explicit dimensionality; defaults to ``len(embed_fn(probe))``.

    Setup
    -----
    >>> from qdrant_client import QdrantClient  # doctest: +SKIP
    >>> from verdict import CalibrationStore, RiskClassifier, ClassifierConfig
    >>> client = QdrantClient(url="http://localhost:6333")  # or ":memory:"
    >>> store = CalibrationStore(client, embed_fn=my_embed_fn)  # doctest: +SKIP
    """

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
        """Upsert one labeled incident into Qdrant."""
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
        """Similarity-weighted mean incident label for similar past steps.

        Cold start (no hits for ``task_class``) returns the design-spec prior
        ``0.2``. ``fan_out`` is reserved for future payload-side boosting.
        """
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
        num = sum(h.score * float(h.payload["incident"]) for h in hits if h.payload)
        den = sum(h.score for h in hits) or 1.0
        return max(0.0, min(1.0, num / den))

    def recalibrate(
        self,
        classifier: RiskClassifier,
        label: float,
        iteration: int,
        max_iterations: int,
    ) -> None:
        """Nudge ``classifier.config.tau`` after a step resolves.

        Never touches the signal weights. When a step exhausts its verifier
        retries with a hard incident label (``label >= 1``), lower ``tau``
        slightly (floor ``0.35``) so similar future work escalates earlier.
        """
        if label >= 1.0 and iteration >= max_iterations:
            classifier.config.tau = max(0.35, classifier.config.tau - 0.02)


class InMemoryCalibrationStore:
    """Drop-in stub for tests and environments without Qdrant.

    Uses cosine similarity over caller-provided embeddings (or a hashed
    bag-of-tokens fallback) so demos stay dependency-light. Optional
    ``class_priors`` let the worked example reproduce the design-spec numbers
    (``rename≈0.05``, ``boundary≈0.62``) before any incidents are recorded.
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
        """Append one labeled incident to the in-memory list."""
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
        """In-memory analogue of :meth:`CalibrationStore.historical_rate`."""
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
        """Same tau-nudge policy as :meth:`CalibrationStore.recalibrate`."""
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
