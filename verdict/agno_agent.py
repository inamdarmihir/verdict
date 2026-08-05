"""Agno-based risk supervisor agent for coding loops."""
from __future__ import annotations
from typing import Any, Callable


def build_agno_risk_supervisor(classifier, memory=None, model: str = "gpt-4o"):
    """Agno Agent that scores proposed steps and routes them based on risk."""
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    def assess_step(description: str, has_verifier: bool, fan_out: int) -> dict[str, Any]:
        """Score a proposed coding step and decide whether to execute or escalate."""
        from verdict.classifier import ProposedStep, RiskScore
        step = ProposedStep(
            description=description,
            has_verifier=has_verifier,
            fan_out=fan_out,
        )
        result = classifier.classify(step)
        if memory:
            history = query_past_outcomes(description)
            past_escalations = sum(1 for h in history if "escalate" in str(h.get("memory", "")))
            if past_escalations > 0:
                result = result._replace(risk_score=min(1.0, result.risk_score + 0.1 * past_escalations))
        return {
            "description": description,
            "risk_score": result.risk_score,
            "route": result.route.value,
        }

    def query_past_outcomes(description: str) -> list[dict[str, Any]]:
        """Retrieve past risk decisions for similar steps from mem0."""
        if not memory:
            return []
        from verdict.memory import query_calibration_history
        return query_calibration_history(memory, description)

    def record_outcome(step_description: str, risk_score: float, route: str, outcome: str) -> None:
        """Record a step's actual outcome back to mem0 for calibration."""
        if memory:
            from verdict.memory import record_step_outcome
            record_step_outcome(memory, step_description, risk_score, route, outcome)

    agent = Agent(
        model=OpenAIChat(id=model),
        name="VerdictRiskSupervisor",
        description="Risk-classifies proposed coding steps and routes them to bounded execution or human escalation.",
        instructions=[
            "You are a risk supervisor for an agentic coding loop.",
            "Assess each proposed step before execution. Steps with risk > 0.55 must be escalated.",
            "Check past outcomes for similar steps to improve risk calibration.",
            "Always record the final outcome to improve future assessments.",
        ],
        tools=[assess_step, query_past_outcomes, record_outcome],
        show_tool_calls=True,
        markdown=True,
    )
    return agent
