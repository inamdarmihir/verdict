# Quick Start

## 1. Install

```bash
pip install verdict-agent
docker run -p 6333:6333 qdrant/qdrant
```

## 2. Classify an action

```python
from verdict.classifier import ActionRiskClassifier
from verdict.store import VerdictStore
from verdict.memory import build_memory
from qdrant_client import QdrantClient

store = VerdictStore(QdrantClient("http://localhost:6333"))
memory = build_memory()
classifier = ActionRiskClassifier(store, memory=memory)

result = classifier.classify(
    action="DELETE FROM orders WHERE created_at < '2024-01-01'",
    context={"environment": "production", "database": "primary"},
)
print(result.risk_score)   # 0.78
print(result.verdict)      # needs_human_review
for d in result.dimension_scores:
    print(f"  {d.name}: {d.score:.2f}  {d.justification}")
```

## 3. Enforce the verdict

```python
from verdict.enforcement import enforce_verdict

async def execute_step(action: str, context: dict):
    result = classifier.classify(action, context)
    decision = await enforce_verdict(result)  # blocks if needs_human_review
    if decision.approved:
        await run(action)
    elif decision.modified:
        await run(decision.modified_action)
```

## 4. Use the LangGraph pipeline

```python
from verdict.agno_agent import build_langgraph_verdict_pipeline

pipeline = build_langgraph_verdict_pipeline(classifier, memory=memory)
result = pipeline.invoke({
    "action": "git push origin main --force",
    "context": {"branch": "main"},
    "risk_score": 0.0, "verdict": "", "human_decision": "",
})
print(result["verdict"])           # needs_human_review
print(result["human_brief"])       # contextual review brief
```

## 5. Use the Agno agent

```python
from verdict.agno_agent import build_agno_verdict_agent

agent = build_agno_verdict_agent(classifier, memory=memory)
agent.print_response("Classify: rm -rf /var/lib/postgresql/data and explain each risk dimension.")
```

## 6. Record outcome for calibration

```python
from verdict.calibration import record_outcome

record_outcome(
    store, memory,
    action="DELETE FROM orders...",
    verdict="needs_human_review",
    human_decision="approved",
    actual_outcome="success_no_incidents",
)
```
