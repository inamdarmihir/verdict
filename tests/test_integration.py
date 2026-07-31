from __future__ import annotations

import json
from pathlib import Path

from examples.langgraph_supervisor import run_demo
from examples.worked_example import run as run_worked
from verdict.checkpoint import CheckpointCommit
from verdict.classifier import ProposedStep, Route
from verdict.contracts import ExecutionResult, VerifierContract
from verdict.executor import BoundedExecutor
from verdict.graph import build_verdict_graph, default_demo_classifier


def test_worked_example_routing() -> None:
    report = run_worked()
    assert report["step_a"]["route"] == "bounded"  # type: ignore[index]
    assert report["step_b"]["route"] == "escalate"  # type: ignore[index]
    assert report["step_a"]["risk"] == 0.3125  # type: ignore[index]
    assert abs(float(report["step_b"]["risk"]) - 0.7175) < 1e-9  # type: ignore[index]


def test_langgraph_demo_offline() -> None:
    report = run_demo(use_model=False)
    assert report["model"] is None
    assert report["model_default"] == "gpt-5.6-sol"
    routes = {o["step_id"]: o["pre_route"] for o in report["outcomes"]}
    assert routes["lg-rename"] == "bounded"
    assert routes["lg-boundary"] == "escalate"
    assert report["calibration_records"] == 2


def test_graph_exhaustion_escalates(tmp_path: Path) -> None:
    def failing_score(step: ProposedStep) -> ExecutionResult:
        del step
        return ExecutionResult(passed=False, score=0.0, details="nope")

    def agent(step: ProposedStep, instruction: str) -> None:
        del step, instruction

    contract = VerifierContract(
        name="fail",
        task_class="bugfix",
        environment={},
        instruction_template="Fix: {description}",
        score_fn=failing_score,
        max_iterations=2,
    )
    classifier, store, registry = default_demo_classifier(contract_registry={"bugfix": contract})
    app = build_verdict_graph(
        classifier=classifier,
        executor=BoundedExecutor(agent),
        contract_registry=registry,
        checkpointer=CheckpointCommit(str(tmp_path), use_git=False),
        calibration_store=store,
        use_interrupt=False,
    )
    step = ProposedStep("ex", "impossible fix", ["a.py"], task_class="bugfix")
    # Force bounded path by ensuring verifier exists and low priors.
    risk = classifier.score(step)
    assert risk.route == Route.BOUNDED
    out = app.invoke(
        {"steps": [step], "step_index": 0},
        config={"configurable": {"thread_id": "exhaust"}},
    )
    assert out["status"] == "calibrated"
    assert out["escalation_outcome"] == "redirected"
    assert out["risk"].route == Route.ESCALATE
    assert "verifier_exhausted" in out["risk"].reasons


def test_package_exports_stable() -> None:
    import verdict

    assert verdict.__version__ == "0.1.0"
    assert hasattr(verdict, "RiskClassifier")
    assert hasattr(verdict, "CalibrationStore")


def test_worked_example_cli() -> None:
    report = run_worked()
    payload = json.dumps(report)
    assert "bounded" in payload
