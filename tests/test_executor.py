from __future__ import annotations

from verdict.classifier import ProposedStep
from verdict.contracts import ExecutionResult, VerifierContract
from verdict.executor import BoundedExecutor


def test_bounded_executor_passes_on_first_try() -> None:
    calls: list[int] = []

    def agent(step: ProposedStep, instruction: str) -> None:
        del step, instruction
        calls.append(1)

    def score_fn(step: ProposedStep) -> ExecutionResult:
        del step
        return ExecutionResult(passed=True, score=1.0, details="ok")

    contract = VerifierContract(
        name="ok",
        task_class="bugfix",
        environment={},
        instruction_template="fix: {description}",
        score_fn=score_fn,
        max_iterations=3,
    )
    step = ProposedStep("1", "fix bug", ["a.py"], task_class="bugfix")
    result = BoundedExecutor(agent).run(step, contract)
    assert result.passed is True
    assert result.iterations_used == 1
    assert len(calls) == 1


def test_bounded_executor_exhausts() -> None:
    def agent(step: ProposedStep, instruction: str) -> None:
        del step, instruction

    def score_fn(step: ProposedStep) -> ExecutionResult:
        del step
        return ExecutionResult(passed=False, score=0.0, details="still failing")

    contract = VerifierContract(
        name="fail",
        task_class="bugfix",
        environment={},
        instruction_template="Fix: {description}",
        score_fn=score_fn,
        max_iterations=2,
    )
    step = ProposedStep("1", "fix bug", ["a.py"], task_class="bugfix")
    result = BoundedExecutor(agent).run(step, contract)
    assert result.passed is False
    assert result.iterations_used == 2
