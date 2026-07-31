#!/usr/bin/env python3
"""Worked example from the design spec: mechanical rename vs architectural boundary.

Runs entirely offline (no API key, no Qdrant) using InMemoryCalibrationStore.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verdict.calibration import InMemoryCalibrationStore
from verdict.checkpoint import CheckpointCommit
from verdict.classifier import ClassifierConfig, ProposedStep, RiskClassifier, Route
from verdict.contracts import always_pass_contract
from verdict.escalation import EscalationArtifact
from verdict.executor import BoundedExecutor


def _noop_agent(step: ProposedStep, instruction: str) -> None:
    del step, instruction


def run() -> dict[str, object]:
    registry = {
        "rename": always_pass_contract("rename"),
        # intentionally no "boundary" contract — absence is a hard signal
    }
    store = InMemoryCalibrationStore(
        class_priors={"rename": 0.05, "boundary": 0.62, "general": 0.2}
    )
    classifier = RiskClassifier(ClassifierConfig(), registry, store)
    executor = BoundedExecutor(_noop_agent)
    checkpoints = CheckpointCommit(repo_path=str(ROOT), use_git=False)

    step_a = ProposedStep(
        step_id="A-rename",
        description="Rename CustomerDTO to CustomerRecord across the codebase",
        planned_files=[f"src/module_{i}.py" for i in range(14)],
        proposed_diff=(
            "--- a/src/dto.py\n+++ b/src/dto.py\n"
            "-class CustomerDTO:\n+class CustomerRecord:\n"
            "     pass\n"
        ),
        task_class="rename",
        repo_path=str(ROOT),
    )
    step_b = ProposedStep(
        step_id="B-boundary",
        description=("Extract billing into a separate service and leave a façade in the monolith"),
        planned_files=["billing/service.py", "billing/facade.py", "app/main.py"],
        proposed_diff=None,
        task_class="boundary",
        repo_path=str(ROOT),
    )

    risk_a = classifier.score(step_a)
    risk_b = classifier.score(step_b)

    result_a = None
    package_b = None
    if risk_a.route == Route.BOUNDED:
        result_a = executor.run(step_a, registry["rename"])
        sha_a = checkpoints.commit(step_a.step_id, step_a.description)
        store.record(repo="demo", step=step_a, risk=risk_a, label=0.0, notes="bounded pass")
    else:
        sha_a = None

    if risk_b.route == Route.ESCALATE:
        package_b = EscalationArtifact.build(step_b, [], risk_b)
        sha_b = checkpoints.commit(step_b.step_id, step_b.description)
        store.record(
            repo="demo",
            step=step_b,
            risk=risk_b,
            label=0.5,
            notes="human redirected: keep billing in-process behind explicit interfaces",
        )
        store.recalibrate(classifier, label=0.5, iteration=0, max_iterations=2)
    else:
        sha_b = None

    return {
        "step_a": {
            "risk": risk_a.value,
            "route": risk_a.route.value,
            "fan_out": risk_a.fan_out,
            "historical_incident_rate": risk_a.historical_incident_rate,
            "reasons": risk_a.reasons,
            "execution_passed": None if result_a is None else result_a.passed,
            "checkpoint": sha_a,
        },
        "step_b": {
            "risk": risk_b.value,
            "route": risk_b.route.value,
            "fan_out": risk_b.fan_out,
            "historical_incident_rate": risk_b.historical_incident_rate,
            "reasons": risk_b.reasons,
            "questions": None if package_b is None else package_b.questions_for_reviewer,
            "checkpoint": sha_b,
        },
        "tau_after": classifier.config.tau,
    }


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))
    a_ok = report["step_a"]["route"] == "bounded"  # type: ignore[index]
    b_ok = report["step_b"]["route"] == "escalate"  # type: ignore[index]
    if not (a_ok and b_ok):
        raise SystemExit(
            f"Worked example failed expected routing (A bounded={a_ok}, B escalate={b_ok})"
        )
    print("\nOK: Step A → bounded, Step B → escalate")


if __name__ == "__main__":
    main()
