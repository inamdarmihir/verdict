"""Component Three — EscalationArtifact: minimal human review packages.

Steps that do not clear the classifier — or that exhaust their retries in
:class:`~verdict.executor.BoundedExecutor` — get routed here instead of
continuing to iterate blind.

The module produces the smallest artifact that lets a human decide quickly,
then hands control back to whatever pause/resume mechanism the host loop uses
(LangGraph ``interrupt()``, a CI approval gate, a Slack form, …).

Prefer three compact views over full design documents:

* **call-stack diff** — control-flow edges gained/lost (approximated via fan-out)
* **file-tree diff** — added/removed/moved paths
* **interface signatures** — before/after for functions at stake

Public surface owned by this module
-----------------------------------
* ``ReviewPackage``, ``EscalationArtifact``
"""

from __future__ import annotations

from dataclasses import dataclass

from verdict.classifier import ProposedStep, RiskScore, extract_changed_symbols
from verdict.contracts import ExecutionResult


@dataclass
class ReviewPackage:
    """Compressed input to human judgment for a high-risk step.

    None of this replaces judgment — it compresses the input to judgment.

    Attributes
    ----------
    step_id:
        Correlates the package with checkpoints and calibration labels.
    risk:
        The :class:`~verdict.classifier.RiskScore` that triggered escalation.
    summary:
        Step description (short intent statement).
    file_tree_diff:
        Compact path-level view of planned changes.
    interface_signatures:
        Before/after sketches for symbols extracted from the proposed diff.
    call_stack_diff:
        Approximate control-flow / fan-out notes for the reviewer.
    proposed_diff_excerpt:
        Truncated unified diff (capped so review UIs stay readable).
    questions_for_reviewer:
        Focused prompts — always includes the mechanical-vs-boundary question,
        plus verifier / fan-out / exhaustion probes when relevant.
    """

    step_id: str
    risk: RiskScore
    summary: str
    file_tree_diff: list[str]
    interface_signatures: list[dict[str, str]]
    call_stack_diff: list[str]
    proposed_diff_excerpt: str
    questions_for_reviewer: list[str]


class EscalationArtifact:
    """Pure transform from step + history + risk → :class:`ReviewPackage`.

    Implemented as a static builder rather than an instance because the module
    has no state of its own to hold.
    """

    @staticmethod
    def build(
        step: ProposedStep,
        execution_history: list[ExecutionResult],
        risk: RiskScore,
    ) -> ReviewPackage:
        """Build the minimal review package for a human-in-the-loop pause.

        Parameters
        ----------
        step:
            The proposed work that needs review.
        execution_history:
            Prior bounded-execution attempts (empty when escalated pre-run).
        risk:
            Classifier (or exhaustion-enriched) risk score.

        Returns
        -------
        ReviewPackage
            Ready to serialize into a LangGraph ``interrupt()`` payload or any
            other HITL channel.

        Example (LangGraph)
        -------------------
        >>> from langgraph.types import interrupt  # doctest: +SKIP
        >>> package = EscalationArtifact.build(step, history, risk)  # doctest: +SKIP
        >>> decision = interrupt({"type": "verdict_escalation", "package": ...})  # doctest: +SKIP
        """
        symbols = extract_changed_symbols(step.proposed_diff or "")
        questions = [
            "Is this change mechanical, or does it set a new boundary/invariant?",
            "What would a correct verifier for this step look like if we had one?",
        ]
        if not risk.verifier_exists:
            questions.append("Approve proceeding without a stop-condition oracle?")
        if risk.fan_out >= 5:
            questions.append("Is the fan-out expected (rename) or a smell (shotgun surgery)?")
        if execution_history and not execution_history[-1].passed:
            questions.append(
                f"Verifier exhausted after {execution_history[-1].iterations_used} attempts "
                "— redirect, or accept without a passing oracle?"
            )
        return ReviewPackage(
            step_id=step.step_id,
            risk=risk,
            summary=step.description,
            file_tree_diff=[f"+/- {f}" for f in step.planned_files],
            interface_signatures=[
                {"symbol": s, "after": f"{s}(...)", "before": "unknown"} for s in symbols
            ],
            call_stack_diff=[f"approx call-site fan-out for {symbols}: {risk.fan_out} files"],
            proposed_diff_excerpt=(step.proposed_diff or "")[:6000],
            questions_for_reviewer=questions,
        )
