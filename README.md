<div align="center">

# Verdict

**A working risk classifier for agentic software factories.**

[![License](https://img.shields.io/github/license/inamdarmihir/verdict?style=flat-square&color=5B5BD6)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-3572A5?style=flat-square)
[![CI](https://img.shields.io/github/actions/workflow/status/inamdarmihir/verdict/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/inamdarmihir/verdict/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/inamdarmihir/verdict?style=flat-square&color=FB6A76)](https://github.com/inamdarmihir/verdict/stargazers)

**[Design article](https://aihive.hashnode.dev/verdict-a-working-risk-classifier-for-agentic-software-factories)** · **[License](#license)**

</div>

---

Verdict gives an agentic coding loop a structural way to recognize when its own stop condition cannot measure what actually matters for a step — and to route that step to a human instead of grinding forward on a passing-but-blind signal.

> Design thesis: a loop should refuse to terminate successfully when the verifiability gap \(G(t) = S(t) - M(\sigma_t)\) exceeds a threshold, even if tests are green.

The full design spec lives in [`docs/DESIGN.md`](docs/DESIGN.md).

---

## Why Verdict exists

Loops excel at mechanically checkable work (failing tests, renames, lint). They fail silently on architectural decisions: the suite stays green while maintainability collapses. More review agents raise the floor; they do not create the missing oracle.

Verdict makes that silence structural:

1. **Classify** every step before it runs (deterministic signals, not LLM self-assessment)
2. **Bound** low-risk steps under an explicit verifier contract + iteration cap
3. **Escalate** high-risk / exhausted steps into a minimal human review package
4. **Checkpoint** at every merge point (git blast-radius boundary)
5. **Calibrate** escalation outcomes back into a Qdrant-backed incident memory

## Architecture

```
                 ProposedStep
                      │
                      ▼
               ┌─────────────┐
               │  classifier  │──▶ RiskScore
               └──────┬───────┘
               risk.route.value
         ┌────────────┴────────────┐
         ▼                          ▼
  ┌────────────┐            ┌─────────────┐
  │  executor   │            │ escalation   │
  │ (bounded)   │            │ (HITL)       │
  └──────┬─────┘            └──────┬──────┘
         └─────────────┬────────────┘
                       ▼
                 ┌────────────┐
                 │ checkpoint  │  git SHA / blast radius
                 └─────┬──────┘
                       ▼
                 ┌────────────┐
                 │ calibration │  Qdrant incident memory
                 └────────────┘
```

| Module | Public API | Responsibility |
|--------|------------|----------------|
| `verdict.classifier` | `RiskClassifier`, `ProposedStep`, `RiskScore` | Score risk from verifier existence, fan-out, history |
| `verdict.contracts` | `VerifierContract`, `ExecutionResult` | Environment + instruction + scoring fn |
| `verdict.executor` | `BoundedExecutor` | Cap retries; return exhaustion as data |
| `verdict.escalation` | `EscalationArtifact`, `ReviewPackage` | Minimal human review package |
| `verdict.checkpoint` | `CheckpointCommit` | Repo-state blast-radius boundary |
| `verdict.calibration` | `CalibrationStore`, `InMemoryCalibrationStore` | Historical incident rate + τ nudges |
| `verdict.graph` | `build_verdict_graph` | Optional LangGraph supervisor wiring |

Risk score:

\[
R(t) = w_v \cdot (1 - \mathbb{1}_{\text{verifier}}(t)) + w_f \cdot \tilde{f}(t) + w_h \cdot h(t)
\]

Defaults: \(w_v=0.45\), \(w_f=0.30\), \(w_h=0.25\), \(\tau=0.55\), \(F_{\max}=8\). Escalate when \(R(t) \ge \tau\).

## Requirements

- Python 3.11+
- [`qdrant-client`](https://github.com/qdrant/qdrant-client) (core)
- Optional host loop: [LangGraph](https://langchain-ai.github.io/langgraph/) + [langchain-openai](https://python.langchain.com/docs/integrations/chat/openai/)

### Model verification (LangGraph + OpenAI)

The end-to-end LangGraph demo targets **`gpt-5.6-sol`** (alias **`gpt-5.6`**):

| Source | Finding |
|--------|---------|
| [OpenAI Models](https://developers.openai.com/api/docs/models) | GPT-5.6 Sol is the current frontier model for complex reasoning/coding; `gpt-5.6` routes to `gpt-5.6-sol` |
| [LangChain OpenAI chat integration](https://docs.langchain.com/oss/python/integrations/chat/openai) | `ChatOpenAI(model="gpt-5.6-sol")` is documented and supported |
| Package versions used | `langgraph>=1.2`, `langchain-openai>=1.4`, `qdrant-client>=1.18` |

Override with `VERDICT_MODEL` if needed. The demo runs offline without an API key (deterministic stub agent).

## Install

```bash
# Library only (qdrant-client)
pip install -e .

# With LangGraph + OpenAI integration
pip install -e ".[langgraph]"

# Dev tools (lint, typecheck, tests)
pip install -e ".[dev]"
```

## Quick start

```python
from verdict import (
    RiskClassifier,
    ClassifierConfig,
    ProposedStep,
    InMemoryCalibrationStore,
    BoundedExecutor,
    EscalationArtifact,
    CheckpointCommit,
    Route,
)
from verdict.contracts import always_pass_contract

registry = {"rename": always_pass_contract("rename")}
store = InMemoryCalibrationStore(class_priors={"rename": 0.05, "boundary": 0.62})
classifier = RiskClassifier(ClassifierConfig(), registry, store)

step = ProposedStep(
    step_id="1",
    description="Extract billing into a separate service",
    planned_files=["billing/service.py", "billing/facade.py", "app/main.py"],
    task_class="boundary",
)
risk = classifier.score(step)

if risk.route == Route.BOUNDED:
    result = BoundedExecutor(agent_fn).run(step, registry[step.task_class])
else:
    package = EscalationArtifact.build(step, [], risk)
    # hand package to your HITL / LangGraph interrupt()
```

### LangGraph wiring

```python
from verdict.graph import build_verdict_graph, default_demo_classifier
from verdict.executor import BoundedExecutor
from verdict.checkpoint import CheckpointCommit

classifier, store, registry = default_demo_classifier()
app = build_verdict_graph(
    classifier=classifier,
    executor=BoundedExecutor(agent_fn),
    contract_registry=registry,
    checkpointer=CheckpointCommit(".", use_git=True),
    calibration_store=store,
)
```

See [`examples/langgraph_supervisor.py`](examples/langgraph_supervisor.py) for a complete runnable supervisor.

## Worked example (offline)

The design-spec contrast — mechanical rename vs architectural boundary — runs with no API keys:

```bash
python examples/worked_example.py
# or
python -m verdict
```

Expected routing:

| Step | Task | \(R(t)\) | Route |
|------|------|----------|-------|
| A | Rename `CustomerDTO` → `CustomerRecord` (verifier exists, \(h\approx0.05\)) | **0.3125** | `bounded` |
| B | Extract billing service (no verifier, \(h\approx0.62\)) | **0.7175** | `escalate` |

## End-to-end LangGraph demo

```bash
pip install -e ".[langgraph]"

# Offline (no model calls)
python examples/langgraph_supervisor.py

# Live model (optional)
export OPENAI_API_KEY=sk-...
export VERDICT_MODEL=gpt-5.6-sol   # default
python examples/langgraph_supervisor.py
```

Flow per step: `classify → bounded_exec|escalate → checkpoint → calibrate`.

## Qdrant calibration

```python
from qdrant_client import QdrantClient
from verdict import CalibrationStore, RiskClassifier, ClassifierConfig

client = QdrantClient(url="http://localhost:6333")  # or QdrantClient(":memory:")
store = CalibrationStore(client, embed_fn=my_embed_fn)  # vectors sized from embed_fn
classifier = RiskClassifier(ClassifierConfig(), contract_registry, store)
```

Labels recorded after each step:

| Event | Label \(\ell\) |
|-------|----------------|
| Escalation rejected / verifier exhausted | `1.0` |
| Escalation redirected | `0.5` |
| Escalation approved / bounded pass | `0.0` |

`recalibrate` only ever nudges \(\tau\), never the signal weights.

## Project layout

```
verdict/
  __init__.py          # public exports
  classifier.py        # RiskClassifier
  contracts.py         # VerifierContract, ExecutionResult
  executor.py          # BoundedExecutor
  escalation.py        # EscalationArtifact / ReviewPackage
  checkpoint.py        # CheckpointCommit
  calibration.py       # CalibrationStore (+ in-memory stub)
  graph.py             # optional LangGraph supervisor
examples/
  worked_example.py
  langgraph_supervisor.py
tests/
docs/
  DESIGN.md            # original design article / spec
```

## Development

```bash
pip install -e ".[dev]"

ruff check verdict tests examples
ruff format verdict tests examples
mypy verdict
pytest --cov=verdict
```

CI runs the same checks on Python 3.11 and 3.12 (see [`.github/workflows/ci.yml`](.github/workflows/ci.yml)).

## Design notes

- The classifier is deliberately boring: silent failure in the router reproduces the problem the library exists to prevent.
- Host frameworks (LangGraph, custom supervisors, CI hooks) are callers, not hard dependencies of the core package.
- `InMemoryCalibrationStore` lets you adopt Verdict before standing up Qdrant.
- Checkpoint commits persist **repo** state (blast radius). Pair them with LangGraph’s `MemorySaver` (or similar) for **graph** resumability — different axes.

## References

- Design article: [`docs/DESIGN.md`](docs/DESIGN.md)
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Qdrant Documentation](https://qdrant.tech/documentation)
- [OpenAI GPT-5.6 Sol](https://developers.openai.com/api/docs/models/gpt-5.6-sol)

## License

MIT — see [`LICENSE`](LICENSE).

---

<div align="center"><sub>Part of the <a href="https://aihive.hashnode.dev">AIHive</a> series — <a href="https://aihive.hashnode.dev/verdict-a-working-risk-classifier-for-agentic-software-factories">read the design article</a></sub></div>
