"""Verdict — a working risk classifier for agentic software factories.

Verdict gives an agentic coding loop a structural way to recognize when its
own stop condition cannot measure what actually matters for a step, and to
route that step to a human instead of grinding forward on a passing-but-blind
signal.

Design thesis
-------------
A loop should refuse to terminate successfully when the verifiability gap

    G(t) = S(t) - M(sigma_t)

exceeds a threshold, even if tests are green. Escalation is that refusal.

Five components (see ``docs/DESIGN.md``)
----------------------------------------
1. **classifier** — score every proposed step before it runs
2. **contracts / executor** — bound low-risk steps under a verifier + iteration cap
3. **escalation** — build a minimal human review package for high-risk steps
4. **checkpoint** — git blast-radius boundary at every merge point
5. **calibration** — Qdrant-backed incident memory that nudges the threshold tau

Quick start
-----------
Install the library (Python 3.11+)::

    pip install -e .
    # optional LangGraph host loop:
    pip install -e ".[langgraph]"

Score a step and route it::

    from verdict import (
        RiskClassifier, ClassifierConfig, ProposedStep,
        InMemoryCalibrationStore, BoundedExecutor,
        EscalationArtifact, Route,
    )
    from verdict.contracts import always_pass_contract

    registry = {"rename": always_pass_contract("rename")}
    store = InMemoryCalibrationStore(class_priors={"rename": 0.05, "boundary": 0.62})
    classifier = RiskClassifier(ClassifierConfig(), registry, store)

    step = ProposedStep(
        step_id="1",
        description="Extract billing into a separate service",
        planned_files=["billing/service.py", "app/main.py"],
        task_class="boundary",
    )
    risk = classifier.score(step)
    if risk.route == Route.BOUNDED:
        ...
    else:
        package = EscalationArtifact.build(step, [], risk)

Offline demos (no API keys)::

    python -m verdict                       # worked example
    python examples/langgraph_supervisor.py # LangGraph supervisor
"""

from __future__ import annotations

from verdict.calibration import CalibrationStore, InMemoryCalibrationStore, outcome_label
from verdict.checkpoint import CheckpointCommit
from verdict.classifier import (
    ClassifierConfig,
    ProposedStep,
    RiskClassifier,
    RiskScore,
    Route,
    extract_changed_symbols,
    python_fan_out,
)
from verdict.contracts import ExecutionResult, VerifierContract, pytest_contract
from verdict.escalation import EscalationArtifact, ReviewPackage
from verdict.executor import BoundedExecutor

__all__ = [
    "BoundedExecutor",
    "CalibrationStore",
    "CheckpointCommit",
    "ClassifierConfig",
    "EscalationArtifact",
    "ExecutionResult",
    "InMemoryCalibrationStore",
    "ProposedStep",
    "ReviewPackage",
    "RiskClassifier",
    "RiskScore",
    "Route",
    "VerifierContract",
    "extract_changed_symbols",
    "outcome_label",
    "pytest_contract",
    "python_fan_out",
]

__version__ = "0.1.0"
