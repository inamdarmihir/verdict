from __future__ import annotations

from verdict.classifier import ProposedStep, RiskScore, Route
from verdict.contracts import ExecutionResult
from verdict.escalation import EscalationArtifact


def test_escalation_package_includes_verifier_question() -> None:
    step = ProposedStep(
        step_id="b",
        description="Extract billing",
        planned_files=["a.py", "b.py"],
        proposed_diff="+def new_api():\n+    pass\n",
        task_class="boundary",
    )
    risk = RiskScore(
        value=0.72,
        verifier_exists=False,
        fan_out=6,
        historical_incident_rate=0.62,
        route=Route.ESCALATE,
        reasons=["no verifier"],
    )
    history = [ExecutionResult(passed=False, score=0.0, details="fail", iterations_used=2)]
    package = EscalationArtifact.build(step, history, risk)
    assert package.step_id == "b"
    assert any("without a stop-condition oracle" in q for q in package.questions_for_reviewer)
    assert any("Verifier exhausted" in q for q in package.questions_for_reviewer)
    assert any("fan-out" in q for q in package.questions_for_reviewer)
    assert package.interface_signatures[0]["symbol"] == "new_api"
