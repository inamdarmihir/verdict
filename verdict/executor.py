"""Bounded execution under a verifier contract and iteration cap."""

from __future__ import annotations

from collections.abc import Callable

from verdict.classifier import ProposedStep
from verdict.contracts import ExecutionResult, VerifierContract


class BoundedExecutor:
    def __init__(self, agent_fn: Callable[[ProposedStep, str], None]) -> None:
        """`agent_fn` mutates the working tree in place, given a step and a
        rendered instruction — the executor doesn't care how it does it."""
        self.agent_fn = agent_fn

    def run(
        self,
        step: ProposedStep,
        contract: VerifierContract,
        max_iters: int | None = None,
    ) -> ExecutionResult:
        iterations = max_iters if max_iters is not None else contract.max_iterations
        if iterations < 1:
            raise ValueError("max_iters must be >= 1")
        instruction = contract.instruction_template.format(description=step.description)
        result = ExecutionResult(passed=False, score=0.0, details="", iterations_used=0)
        for i in range(1, iterations + 1):
            self.agent_fn(step, instruction)
            result = contract.score_fn(step)
            result.iterations_used = i
            if result.passed:
                return result
        return result
