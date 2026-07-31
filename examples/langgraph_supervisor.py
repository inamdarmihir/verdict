#!/usr/bin/env python3
"""End-to-end LangGraph supervisor wired to Verdict.

Model: OpenAI ``gpt-5.6-sol`` (alias ``gpt-5.6``) — verified as the current
frontier model in OpenAI's model docs and exercised via ``langchain-openai``'s
``ChatOpenAI`` integration (see LangChain OpenAI chat docs).

Usage:
  export OPENAI_API_KEY=sk-...
  pip install -e '.[langgraph]'
  python examples/langgraph_supervisor.py

Without an API key the demo still runs the Verdict routing path using a
deterministic stub agent (no model calls).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verdict.checkpoint import CheckpointCommit
from verdict.classifier import ProposedStep, Route
from verdict.contracts import always_pass_contract
from verdict.executor import BoundedExecutor
from verdict.graph import build_verdict_graph, default_demo_classifier

# Verified latest frontier model (OpenAI API + langchain-openai ChatOpenAI).
DEFAULT_MODEL = os.environ.get("VERDICT_MODEL", "gpt-5.6-sol")


def build_agent_fn(model_name: str = DEFAULT_MODEL):
    """Return an agent_fn. Uses ChatOpenAI when OPENAI_API_KEY is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:

        def stub(step: ProposedStep, instruction: str) -> None:
            del step, instruction

        return stub, None

    from langchain_openai import ChatOpenAI

    # use_responses_api aligns with OpenAI guidance for agentic GPT-5.6 workloads
    llm = ChatOpenAI(model=model_name, api_key=api_key, use_responses_api=True)

    def agent_fn(step: ProposedStep, instruction: str) -> None:
        # Demonstration only: ask the model for a one-line plan. Real loops
        # would apply patches to the working tree here.
        prompt = (
            f"You are a coding agent inside a Verdict bounded executor.\n"
            f"Instruction: {instruction}\n"
            f"Step: {step.description}\n"
            f"Task class: {step.task_class}\n"
            "Reply with a single sentence describing the minimal patch you would apply."
        )
        _ = llm.invoke(prompt)

    return agent_fn, model_name


def run_demo(*, use_model: bool = True) -> dict[str, Any]:
    classifier, store, registry = default_demo_classifier(
        contract_registry={
            "rename": always_pass_contract("rename"),
            "bugfix": always_pass_contract("bugfix"),
        },
        class_priors={"rename": 0.05, "boundary": 0.62, "bugfix": 0.15},
    )
    model_name = DEFAULT_MODEL if use_model else None
    agent_fn, active_model = build_agent_fn(model_name or DEFAULT_MODEL)
    if not use_model or os.environ.get("OPENAI_API_KEY") is None:
        active_model = None

    executor = BoundedExecutor(agent_fn)
    checkpointer = CheckpointCommit(repo_path=str(ROOT), use_git=False)
    app = build_verdict_graph(
        classifier=classifier,
        executor=executor,
        contract_registry=registry,
        checkpointer=checkpointer,
        calibration_store=store,
        use_interrupt=False,
        repo="langgraph-demo",
    )

    steps = [
        ProposedStep(
            step_id="lg-rename",
            description="Rename CustomerDTO to CustomerRecord",
            planned_files=[f"src/f{i}.py" for i in range(14)],
            task_class="rename",
            repo_path=str(ROOT),
        ),
        ProposedStep(
            step_id="lg-boundary",
            description="Extract billing into a separate service with a façade",
            planned_files=["billing/service.py", "billing/facade.py", "app/main.py"],
            task_class="boundary",
            repo_path=str(ROOT),
        ),
    ]

    outcomes: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        # Pre-score for the report (graph also scores internally).
        risk = classifier.score(step)
        result = app.invoke(
            {
                "steps": [step],
                "step_index": 0,
            },
            config={"configurable": {"thread_id": f"verdict-demo-{idx}"}},
        )
        outcomes.append(
            {
                "step_id": step.step_id,
                "task_class": step.task_class,
                "pre_risk": risk.value,
                "pre_route": risk.route.value,
                "status": result.get("status"),
                "checkpoint_sha": result.get("checkpoint_sha"),
                "escalation_outcome": result.get("escalation_outcome"),
                "expected_route": (
                    Route.BOUNDED.value if step.task_class == "rename" else Route.ESCALATE.value
                ),
            }
        )

    return {
        "model": active_model,
        "model_default": DEFAULT_MODEL,
        "model_note": (
            "Using gpt-5.6-sol — OpenAI's current frontier model "
            "(alias: gpt-5.6). Supported by langchain-openai ChatOpenAI."
        ),
        "outcomes": outcomes,
        "calibration_records": len(store.records),
        "tau": classifier.config.tau,
    }


def main() -> None:
    report = run_demo(use_model=True)
    print(json.dumps(report, indent=2))
    rename = next(o for o in report["outcomes"] if o["step_id"] == "lg-rename")
    boundary = next(o for o in report["outcomes"] if o["step_id"] == "lg-boundary")
    if rename["pre_route"] != "bounded" or boundary["pre_route"] != "escalate":
        raise SystemExit("LangGraph demo routing mismatch")
    print("\nOK: LangGraph supervisor routed rename→bounded, boundary→escalate")


if __name__ == "__main__":
    main()
