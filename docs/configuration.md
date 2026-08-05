# Configuration

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `OPENAI_API_KEY` | — | Required for LangGraph node and Agno agent |
| `VERDICT_AUTO_APPROVE_THRESHOLD` | `0.3` | Risk below which actions auto-execute |
| `VERDICT_BLOCK_THRESHOLD` | `0.7` | Risk above which human approval required |

## ActionRiskClassifier Parameters

```python
ActionRiskClassifier(
    store: VerdictStore,
    memory: mem0.Memory | None = None,
    auto_approve_threshold: float = 0.3,
    block_threshold: float = 0.7,
)
```

## VerdictStore Parameters

```python
VerdictStore(
    client: QdrantClient,
    collection: str = "verdict_history",
)
```

## Risk Dimension Weights

Weights are configurable. The default sum to 1.0:

```python
from verdict.dimensions import RiskDimension

dimensions = [
    RiskDimension("reversibility", weight=0.25),
    RiskDimension("blast_radius", weight=0.20),
    RiskDimension("production_signal", weight=0.20),
    RiskDimension("auth_requirements", weight=0.15),
    RiskDimension("data_sensitivity", weight=0.10),
    RiskDimension("code_change_scope", weight=0.10),
]
```

## Calibration

verdict improves over time by comparing predicted verdicts against actual outcomes. Run `calibration.record_outcome()` after each human-reviewed step and the scoring weights auto-adjust.
