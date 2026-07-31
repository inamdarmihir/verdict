"""Escalation artifacts: minimal review packages for human-in-the-loop."""

from __future__ import annotations

from dataclasses import dataclass

from verdict.classifier import ProposedStep, RiskScore, extract_changed_symbols
from verdict.contracts import ExecutionResult


@dataclass
class ReviewPackage:
    step_id: str
    risk: RiskScore
    summary: str
    file_tree_diff: list[str]
    interface_signatures: list[dict[str, str]]
    call_stack_diff: list[str]
    proposed_diff_excerpt: str
    questions_for_reviewer: list[str]


class EscalationArtifact:
    @staticmethod
    def build(
        step: ProposedStep,
        execution_history: list[ExecutionResult],
        risk: RiskScore,
    ) -> ReviewPackage:
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
