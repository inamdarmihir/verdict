"""Risk calibration memory backed by mem0 + Qdrant."""
from __future__ import annotations
from typing import Any


def build_memory(qdrant_url: str = "http://localhost:6333", collection_name: str = "verdict_memory"):
    from mem0 import Memory
    return Memory.from_config({
        "vector_store": {
            "provider": "qdrant",
            "config": {"url": qdrant_url, "collection_name": collection_name},
        }
    })


def record_step_outcome(memory, step_description: str, risk_score: float, route: str, outcome: str, agent_id: str = "verdict") -> None:
    """Persist a step risk decision and its outcome for threshold calibration."""
    memory.add(
        f"Step '{step_description}': risk={risk_score:.2f}, route={route}, outcome={outcome}",
        user_id=agent_id,
        metadata={"risk": risk_score, "route": route, "outcome": outcome},
    )


def query_calibration_history(memory, description: str, agent_id: str = "verdict") -> list[dict[str, Any]]:
    """Retrieve past risk decisions similar to a given step description."""
    results = memory.search(description, user_id=agent_id)
    return results.get("results", [])
