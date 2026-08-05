# API Reference

## ActionRiskClassifier

```python
classifier.classify(action: str, context: dict) -> ClassificationResult
# ClassificationResult: risk_score, verdict, dimension_scores, human_brief

classifier.score_dimensions(action: str, context: dict) -> list[DimensionScore]
# DimensionScore: name, score, justification
```

## VerdictEnum

```python
class VerdictEnum(str, Enum):
    auto_approved = "auto_approved"
    needs_human_review = "needs_human_review"
    auto_blocked = "auto_blocked"
```

## enforce_verdict

```python
from verdict.enforcement import enforce_verdict

async decision = enforce_verdict(
    result: ClassificationResult,
    timeout_seconds: int = 300,  # human review timeout
) -> EnforcementDecision
# EnforcementDecision: approved, modified, modified_action
```

## Memory

```python
from verdict.memory import build_memory, record_step_outcome, query_outcome_history

memory = build_memory(qdrant_url, collection_name)
record_step_outcome(memory, action, verdict, human_decision, outcome)
query_outcome_history(memory, action_type) -> list[dict]
```

## Calibration

```python
from verdict.calibration import record_outcome, recalibrate_weights

record_outcome(store, memory, action, verdict, human_decision, actual_outcome)
new_weights = recalibrate_weights(store, min_samples=50) -> dict[str, float]
```

## Agno + LangGraph

```python
from verdict.agno_agent import build_agno_verdict_agent, build_langgraph_verdict_pipeline

agent = build_agno_verdict_agent(classifier, memory=None, model="gpt-4o")
pipeline = build_langgraph_verdict_pipeline(classifier, memory=None)
```
