from __future__ import annotations

from qdrant_client import QdrantClient
from verdict.calibration import CalibrationStore, hashed_embed
from verdict.classifier import ProposedStep, RiskScore, Route


def test_qdrant_calibration_store_roundtrip() -> None:
    client = QdrantClient(":memory:")
    store = CalibrationStore(client, embed_fn=hashed_embed, collection="verdict_test")
    step = ProposedStep(
        step_id="1",
        description="extract billing into service",
        planned_files=["a.py"],
        task_class="boundary",
    )
    risk = RiskScore(0.72, False, 3, 0.62, Route.ESCALATE)
    store.record(repo="demo", step=step, risk=risk, label=1.0, notes="rejected")
    rate = store.historical_rate(
        task_class="boundary",
        description="extract billing into service",
        fan_out=3,
        top_k=5,
    )
    assert rate == 1.0

    # Unseen class → cold-start prior
    assert (
        store.historical_rate(task_class="migration", description="move tables", fan_out=2) == 0.2
    )
