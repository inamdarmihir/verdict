"""Verifier contracts: environment + instruction + scoring function.

Component Two of Verdict (paired with :mod:`verdict.executor`) borrows the
three-part shape popularized by Harbor-style agent environments:

* **environment** — sandbox constraints (cwd, network, timeout, …)
* **instruction** — step description plus hard constraints
* **scoring function** — returns a scalar in ``[0, 1]`` as an :class:`ExecutionResult`

The contract is attached *before* the agent iterates, so the stop condition is
not improvised mid-loop. Every autonomous step must name its oracle up front;
if you cannot name one, :meth:`~verdict.classifier.RiskClassifier.score` should
already have routed you to escalation.

Public surface owned by this module
-----------------------------------
* ``ExecutionResult``, ``VerifierContract``
* ``pytest_contract``, ``always_pass_contract``
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from verdict.classifier import ProposedStep


@dataclass
class ExecutionResult:
    """Outcome of running a step under a :class:`VerifierContract`.

    Exhaustion is a normal return value, not an exception — the escalation
    module needs the full attempt history to build a useful review package.

    Attributes
    ----------
    passed:
        Whether the oracle considers the step done.
    score:
        Scalar in ``[0, 1]`` (typically ``1.0`` / ``0.0`` for pass/fail oracles).
    details:
        Truncated verifier output suitable for logs or review packages.
    iterations_used:
        How many agent attempts produced this result (set by the executor).
    """

    passed: bool
    score: float
    details: str
    iterations_used: int = 0


@dataclass
class VerifierContract:
    """Named oracle attached to a ``task_class`` before bounded execution.

    Parameters
    ----------
    name:
        Human-readable contract id (e.g. ``pytest:tests/test_billing.py``).
    task_class:
        Key used in the classifier's contract registry.
    environment:
        Sandbox metadata the host loop may enforce (cwd, network, timeout, …).
    instruction_template:
        Format string with a ``{description}`` placeholder filled from the step.
    score_fn:
        Callable that inspects the working tree (or other state) and returns an
        :class:`ExecutionResult`.
    max_iterations:
        Default retry cap for :class:`~verdict.executor.BoundedExecutor`.
    """

    name: str
    task_class: str
    environment: dict[str, Any]
    instruction_template: str
    score_fn: Callable[[ProposedStep], ExecutionResult]
    max_iterations: int = 2


def pytest_contract(test_path: str, repo_path: str) -> VerifierContract:
    """Build a contract that scores a step by running ``pytest`` on ``test_path``.

    The important property is not that every contract is a unit test — it is
    that every autonomous step names its oracle up front.

    Parameters
    ----------
    test_path:
        Path (relative to ``repo_path``) passed to ``pytest``.
    repo_path:
        Working directory for the pytest subprocess.

    Returns
    -------
    VerifierContract
        A ``task_class="bugfix"`` contract with ``max_iterations=2``.
    """

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
    """Deterministic contract useful for demos, unit tests, and offline setup.

    Use this while wiring Verdict into a host loop before you have a real
    oracle for a mechanical ``task_class`` (e.g. renames with a grep gate).
    """

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
