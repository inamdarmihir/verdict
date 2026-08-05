"""Component Two — BoundedExecutor: capped loop under a verifier contract.

Steps that clear the classifier run inside :class:`BoundedExecutor` — not under
a general instruction to "write good code," but under a
:class:`~verdict.contracts.VerifierContract` specific to that step.

Iteration caps matter as much as the verifier. An agent that cannot pass its
own contract in a small, fixed number of tries should not keep grinding — it
should be reclassified and routed to escalation
(``exhaust(t) = 1[i_t >= I_max and M(sigma_t) < 1]``). Calibration later treats
exhaustion as a positive label for "should have been higher risk."

Public surface owned by this module
-----------------------------------
* ``BoundedExecutor``
"""

from __future__ import annotations

from collections.abc import Callable

from verdict.classifier import ProposedStep
from verdict.contracts import ExecutionResult, VerifierContract


class BoundedExecutor:
    """Run a step under a verifier contract with a hard iteration cap.

    Parameters
    ----------
    agent_fn:
        Callable ``(step, instruction) -> None`` that mutates the working tree
        in place. The executor does not care how the agent applies patches —
        LLM tool loops, deterministic transformers, or stubs for tests are all
        valid.

    Example
    -------
    >>> from verdict.executor import BoundedExecutor
    >>> from verdict.contracts import always_pass_contract
    >>> from verdict.classifier import ProposedStep
    >>> def agent(step, instruction): ...
    >>> result = BoundedExecutor(agent).run(
    ...     ProposedStep("1", "rename", ["a.py"], task_class="rename"),
    ...     always_pass_contract("rename"),
    ... )
    >>> result.passed
    True
    """

    def __init__(self, agent_fn: Callable[[ProposedStep, str], None]) -> None:
        self.agent_fn = agent_fn

    def run(
        self,
        step: ProposedStep,
        contract: VerifierContract,
        max_iters: int | None = None,
    ) -> ExecutionResult:
        """Attempt ``step`` until the contract passes or the iteration cap hits.

        Returns an :class:`~verdict.contracts.ExecutionResult` whether or not
        the step passed — exhaustion is expected data for the escalation path,
        not an exception.

        Parameters
        ----------
        step:
            Proposed work unit.
        contract:
            Oracle + instruction template for this ``task_class``.
        max_iters:
            Override ``contract.max_iterations`` when provided. Must be ``>= 1``.
        """
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
