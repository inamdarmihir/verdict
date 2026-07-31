"""Optional LangGraph supervisor that routes through Verdict."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, TypedDict

from verdict.calibration import InMemoryCalibrationStore
from verdict.checkpoint import CheckpointCommit
from verdict.classifier import (
    ClassifierConfig,
    ProposedStep,
    RiskClassifier,
    RiskScore,
    Route,
)
from verdict.contracts import ExecutionResult, VerifierContract
from verdict.escalation import EscalationArtifact, ReviewPackage
from verdict.executor import BoundedExecutor


class SupervisorState(TypedDict, total=False):
    steps: list[ProposedStep]
    step_index: int
    risk: RiskScore
    execution_history: list[ExecutionResult]
    review_package: ReviewPackage | None
    escalation_outcome: str | None
    human_notes: str | None
    checkpoint_sha: str | None
    last_result: ExecutionResult | None
    status: str


def build_verdict_graph(
    *,
    classifier: RiskClassifier,
    executor: BoundedExecutor,
    contract_registry: dict[str, VerifierContract],
    checkpointer: CheckpointCommit,
    repo: str = "local",
    calibration_store: InMemoryCalibrationStore | None = None,
    use_interrupt: bool = True,
) -> Any:
    """Build a LangGraph that classify → bounded|escalate → checkpoint → calibrate.

    Requires the optional ``langgraph`` extra:
    ``pip install 'verdict-agents[langgraph]'``.
    """
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.graph import END, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:  # pragma: no cover - exercised when extra missing
        raise ImportError(
            "LangGraph is required for verdict.graph. "
            "Install with: pip install 'verdict-agents[langgraph]'"
        ) from exc

    store = calibration_store

    def classify_node(state: SupervisorState) -> dict[str, Any]:
        step = state["steps"][state["step_index"]]
        risk = classifier.score(step)
        return {"risk": risk, "execution_history": [], "status": "classified"}

    def route_after_classify(state: SupervisorState) -> Literal["bounded_exec", "escalate"]:
        risk = state["risk"]
        return "escalate" if risk.route == Route.ESCALATE else "bounded_exec"

    def bounded_exec_node(state: SupervisorState) -> dict[str, Any]:
        step = state["steps"][state["step_index"]]
        risk = state["risk"]
        contract = contract_registry[step.task_class]
        result = executor.run(step, contract)
        history = list(state.get("execution_history") or [])
        history.append(result)
        if result.passed:
            return {
                "execution_history": history,
                "last_result": result,
                "status": "bounded_pass",
            }
        # Exhaustion → reclassify into escalation with enriched risk reasons.
        exhausted_risk = RiskScore(
            value=max(risk.value, classifier.config.tau),
            verifier_exists=risk.verifier_exists,
            fan_out=risk.fan_out,
            historical_incident_rate=risk.historical_incident_rate,
            route=Route.ESCALATE,
            reasons=[*risk.reasons, "verifier_exhausted"],
        )
        return {
            "execution_history": history,
            "last_result": result,
            "risk": exhausted_risk,
            "status": "verifier_exhausted",
        }

    def route_after_bounded(state: SupervisorState) -> Literal["checkpoint", "escalate"]:
        result = state.get("last_result")
        if result is not None and result.passed:
            return "checkpoint"
        return "escalate"

    def escalate_node(state: SupervisorState) -> dict[str, Any]:
        step = state["steps"][state["step_index"]]
        package = EscalationArtifact.build(
            step,
            list(state.get("execution_history") or []),
            state["risk"],
        )
        if use_interrupt:
            decision = interrupt(
                {
                    "type": "verdict_escalation",
                    "package": _package_payload(package),
                }
            )
            outcome = str(decision.get("outcome", "rejected"))
            notes = decision.get("notes")
        else:
            # Non-interactive fallback used by demos/tests.
            outcome = "redirected"
            notes = "auto-redirect (interrupt disabled)"
        return {
            "review_package": package,
            "escalation_outcome": outcome,
            "human_notes": notes,
            "status": f"escalated:{outcome}",
        }

    def checkpoint_node(state: SupervisorState) -> dict[str, Any]:
        step = state["steps"][state["step_index"]]
        sha = checkpointer.commit(step.step_id, step.description)
        return {"checkpoint_sha": sha, "status": state.get("status", "checkpointed")}

    def calibrate_node(state: SupervisorState) -> dict[str, Any]:
        if store is None:
            return {"status": state.get("status", "done")}
        step = state["steps"][state["step_index"]]
        risk = state["risk"]
        outcome = state.get("escalation_outcome")
        last = state.get("last_result")
        if outcome == "rejected" or (last is not None and not last.passed and outcome is None):
            label = 1.0
        elif outcome == "redirected":
            label = 0.5
        else:
            label = 0.0
        store.record(repo=repo, step=step, risk=risk, label=label, notes=state.get("human_notes"))
        iterations = last.iterations_used if last is not None else 0
        max_iters = 2
        if step.task_class in contract_registry:
            max_iters = contract_registry[step.task_class].max_iterations
        store.recalibrate(classifier, label, iterations, max_iters)
        return {"status": "calibrated"}

    graph = StateGraph(SupervisorState)
    graph.add_node("classify", classify_node)
    graph.add_node("bounded_exec", bounded_exec_node)
    graph.add_node("escalate", escalate_node)
    graph.add_node("checkpoint", checkpoint_node)
    graph.add_node("calibrate", calibrate_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {"bounded_exec": "bounded_exec", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "bounded_exec",
        route_after_bounded,
        {"checkpoint": "checkpoint", "escalate": "escalate"},
    )
    graph.add_edge("escalate", "checkpoint")
    graph.add_edge("checkpoint", "calibrate")
    graph.add_edge("calibrate", END)

    return graph.compile(checkpointer=MemorySaver())


def default_demo_classifier(
    contract_registry: dict[str, Any] | None = None,
    *,
    class_priors: dict[str, float] | None = None,
) -> tuple[RiskClassifier, InMemoryCalibrationStore, dict[str, Any]]:
    """Convenience factory used by examples and integration tests."""
    from verdict.contracts import always_pass_contract

    registry: dict[str, Any] = contract_registry or {
        "rename": always_pass_contract("rename"),
        "bugfix": always_pass_contract("bugfix"),
    }
    store = InMemoryCalibrationStore(
        class_priors=class_priors
        or {
            "rename": 0.05,
            "bugfix": 0.15,
            "boundary": 0.62,
            "migration": 0.45,
            "general": 0.2,
        }
    )
    classifier = RiskClassifier(ClassifierConfig(), registry, store)
    return classifier, store, registry


def _package_payload(package: ReviewPackage) -> dict[str, Any]:
    payload = asdict(package)
    risk = payload.get("risk")
    if isinstance(risk, dict) and isinstance(risk.get("route"), Route):
        risk["route"] = risk["route"].value
    elif hasattr(package.risk.route, "value"):
        payload["risk"]["route"] = package.risk.route.value
    return payload
