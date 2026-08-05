<div align="center">

# verdict

**Risk classification for agentic coding loops — route high-risk steps to humans before they execute.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Qdrant](https://img.shields.io/badge/vector--db-Qdrant-red.svg)](https://qdrant.tech)
[![Agno](https://img.shields.io/badge/agent-agno%20v2.8.6-blueviolet.svg)](https://github.com/agno-agi/agno)
[![mem0](https://img.shields.io/badge/memory-mem0%20v3.0.0-green.svg)](https://mem0.ai)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph%20v1.2.10-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-informational.svg)](https://inamdarmihir.github.io/verdict/)

</div>

---

## The Problem

Coding agents take irreversible high-risk actions — deleting files, running migrations, mass refactors across hundreds of files — without human oversight. By the time you notice, the damage is done.

## The Solution

**verdict** scores each proposed step with a **deterministic risk formula** (no LLM self-assessment). Low-risk steps run under a bounded executor with a verifier contract. High-risk steps generate a minimal human review package. All decisions calibrate a Qdrant-backed threshold via outcome feedback.

## Risk Formula

```
R(t) = w_v × (1 - verifier_exists)
     + w_f × normalized_fan_out
     + w_h × historical_incident_rate

w_v = 0.50  |  w_f = 0.30  |  w_h = 0.20  |  threshold τ = 0.55
```

## Routing Logic

| Risk Score | Route | What happens |
|---|---|---|
| R < 0.55 | `BoundedExecutor` | Run with verifier contract + iteration cap |
| R ≥ 0.55 | `EscalationArtifact` | Build review package for human |
| Exhausted | `EscalationArtifact` | Cap hit — human decides |

## How It Works

```
ProposedStep
  │
  ▼
RiskClassifier.classify()
  ├─ verifier_exists? ← VerifierContract
  ├─ fan_out count
  └─ historical_incident_rate ← Qdrant CalibrationStore
  │
  ├── R < τ → BoundedExecutor(verifier, cap)
  │            └── success → CheckpointCommit → CalibrationStore.record(OK)
  │
  └── R ≥ τ → EscalationArtifact → ReviewPackage
                   └── human decision → CalibrationStore.record(label)
                                          ├─ τ nudge ± 0.02
                                          └─ mem0 — outcome history
```

## Quick Start

```bash
pip install verdict-agents
docker run -p 6333:6333 qdrant/qdrant
```

```python
from verdict import RiskClassifier, BoundedExecutor
from verdict.classifier import ProposedStep
from verdict.memory import build_memory

classifier = RiskClassifier()
memory = build_memory()

step = ProposedStep(
    description="rename variable across 3 files",
    has_verifier=True,
    fan_out=3,
)
result = classifier.classify(step)
print(result.route)        # Route.bounded_exec
print(result.risk_score)   # 0.31
```

## LangGraph Supervisor

```python
from verdict.graph import build_verdict_graph

graph = build_verdict_graph()
result = graph.invoke({
    "step": {"description": "run database migration on prod", "has_verifier": False, "fan_out": 1},
    "threshold": 0.55,
})
print(result["route"])  # escalate
```

## Agno Risk Supervisor

```python
from verdict.agno_agent import build_agno_risk_supervisor

agent = build_agno_risk_supervisor(classifier, memory=memory)
agent.print_response("Should I delete all files in /tmp/build? It has no verifier.")
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `OPENAI_API_KEY` | — | Required for Agno agent mode |
| `VERDICT_THRESHOLD` | `0.55` | Risk threshold τ for routing |
| `VERDICT_ITERATION_CAP` | `5` | Max retries in BoundedExecutor |

## Tech Stack

| Component | Purpose |
|---|---|
| [Qdrant](https://qdrant.tech) `>=1.18.0` | Calibration store + incident history |
| [Agno](https://github.com/agno-agi/agno) `>=2.8.6` | Risk supervisor agent |
| [mem0](https://mem0.ai) `>=3.0.0` | Step outcome memory |
| [LangGraph](https://langchain-ai.github.io/langgraph/) `>=1.2.10` | classify→exec→calibrate graph |

## License

MIT — see [LICENSE](LICENSE).
