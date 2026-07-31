"""Verifier contracts: environment + instruction + scoring function."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verdict.classifier import ProposedStep


@dataclass
class ExecutionResult:
    passed: bool
    score: float
    details: str
    iterations_used: int = 0


@dataclass
class VerifierContract:
    name: str
    task_class: str
    environment: dict[str, Any]
    instruction_template: str
    score_fn: Callable[[ProposedStep], ExecutionResult]
    max_iterations: int = 2


def pytest_contract(test_path: str, repo_path: str) -> VerifierContract:
    def score_fn(step: ProposedStep) -> ExecutionResult:
        from subprocess import run

        proc = run(
            ["pytest", test_path, "-q"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        passed = proc.returncode == 0
        return ExecutionResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            details=(proc.stdout[-4000:] + proc.stderr[-2000:]),
        )

    return VerifierContract(
        name=f"pytest:{test_path}",
        task_class="bugfix",
        environment={"cwd": ".", "network": False, "timeout_s": 120},
        instruction_template=(
            "Apply a minimal patch for: {description}. "
            "Do not change public APIs. Stop when the attached verifier passes."
        ),
        score_fn=score_fn,
        max_iterations=2,
    )


def always_pass_contract(task_class: str = "rename", name: str = "always-pass") -> VerifierContract:
    """Deterministic contract useful for demos and unit tests."""

    def score_fn(step: ProposedStep) -> ExecutionResult:
        return ExecutionResult(
            passed=True,
            score=1.0,
            details=f"synthetic pass for step={step.step_id}",
        )

    return VerifierContract(
        name=name,
        task_class=task_class,
        environment={"cwd": ".", "network": False, "timeout_s": 30},
        instruction_template=(
            "Apply a mechanical change for: {description}. "
            "Keep public APIs stable. Stop when the attached verifier passes."
        ),
        score_fn=score_fn,
        max_iterations=2,
    )
