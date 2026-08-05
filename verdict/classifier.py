"""Component One — RiskClassifier: score a proposed step before it runs.

The classifier is a small deterministic program with optional learned inputs
(historical incident rates), not another agent. Self-assessment fails silently;
checkable signals do not.

Risk score (see ``docs/DESIGN.md``, Component One)::

    R(t) = w_v * (1 - I_verifier(t)) + w_f * f_tilde(t) + w_h * h(t)

where ``f_tilde(t) = min(1, f(t) / F_max)`` is normalized fan-out and ``h(t)``
is the historical incident rate for similar steps. Route to escalation when
``R(t) >= tau``.

Defaults: ``w_v=0.45``, ``w_f=0.30``, ``w_h=0.25``, ``tau=0.55``, ``F_max=8``.

Public surface owned by this module
-----------------------------------
* ``Route``, ``ProposedStep``, ``RiskScore``, ``ClassifierConfig``
* ``RiskClassifier``
* ``extract_changed_symbols``, ``python_fan_out``
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class HistoricalRateProvider(Protocol):
    """Minimal calibration interface the classifier depends on.

    Both :class:`~verdict.calibration.CalibrationStore` (Qdrant) and
    :class:`~verdict.calibration.InMemoryCalibrationStore` satisfy this
    protocol, so teams can adopt Verdict before standing up Qdrant.
    """

    def historical_rate(
        self,
        *,
        task_class: str,
        description: str,
        fan_out: int,
        top_k: int = 20,
    ) -> float:
        """Return a class-conditional incident rate in ``[0, 1]``."""
        ...


class Route(StrEnum):
    """Where the supervisor should send a scored step."""

    BOUNDED = "bounded"
    """Clear the classifier — run under a verifier contract + iteration cap."""

    ESCALATE = "escalate"
    """Risk (or later exhaustion) requires a human review package."""


@dataclass
class ProposedStep:
    """A single unit of work an agentic plan wants to perform.

    Parameters
    ----------
    step_id:
        Stable identifier used in checkpoints and calibration records.
    description:
        Human-readable intent of the step (also used as the embedding text).
    planned_files:
        Files the agent expects to touch. Used as a fan-out fallback when no
        proposed diff / symbols are available yet.
    proposed_diff:
        Optional unified diff. When present, symbols are extracted for static
        fan-out analysis.
    task_class:
        Coarse class used for verifier lookup and historical priors
        (``rename`` | ``migration`` | ``boundary`` | ``bugfix`` | …).
    repo_path:
        Working tree root for static fan-out analysis. Defaults to ``"."``.
    """

    step_id: str
    description: str
    planned_files: list[str]
    proposed_diff: str | None = None
    task_class: str = "general"  # rename | migration | boundary | bugfix | …
    repo_path: str = "."


@dataclass
class RiskScore:
    """Result of :meth:`RiskClassifier.score`.

    Attributes
    ----------
    value:
        Scalar risk ``R(t)`` in ``[0, 1]`` (weights may push slightly outside
        only if misconfigured; validated config keeps it in range for normal inputs).
    verifier_exists:
        Whether a :class:`~verdict.contracts.VerifierContract` is registered
        for the step's ``task_class``.
    fan_out:
        Estimated blast radius (static call-site count or planned-file count).
    historical_incident_rate:
        ``h(t)`` from the calibration store.
    route:
        ``bounded`` or ``escalate`` based on ``value >= tau``.
    reasons:
        Human-readable contributors to the score (for review packages / logs).
    """

    value: float
    verifier_exists: bool
    fan_out: int
    historical_incident_rate: float
    route: Route
    reasons: list[str] = field(default_factory=list)


@dataclass
class ClassifierConfig:
    """Weights and thresholds for :class:`RiskClassifier`.

    Weights must sum to ``1.0``. Calibration may nudge ``tau`` over time but
    never the weights — weights encode a modeling choice about which signals
    matter; ``tau`` encodes a cost tradeoff the team is entitled to move.
    """

    w_v: float = 0.45
    w_f: float = 0.30
    w_h: float = 0.25
    tau: float = 0.55
    f_max: float = 8.0

    def __post_init__(self) -> None:
        weight_sum = self.w_v + self.w_f + self.w_h
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {weight_sum}")
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        if self.f_max <= 0:
            raise ValueError(f"f_max must be positive, got {self.f_max}")


def extract_changed_symbols(diff_text: str) -> list[str]:
    """Parse ``def`` / ``class`` names from added lines of a unified diff.

    Agents understate scope; symbol extraction feeds static fan-out analysis
    so fan-out is not "number of files the agent *says* it will touch."

    Parameters
    ----------
    diff_text:
        Unified diff text (``+`` lines are inspected; ``+++`` headers skipped).

    Returns
    -------
    list[str]
        Identifier names suitable for AST name/attr matching.
    """
    symbols: list[str] = []
    for line in diff_text.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        s = line[1:].lstrip()
        if s.startswith("def "):
            symbols.append(s[4:].split("(")[0].strip())
        elif s.startswith("class "):
            symbols.append(s[6:].split("(")[0].split(":")[0].strip())
    return [x for x in symbols if x.isidentifier()]


def python_fan_out(repo_path: str, symbols: list[str]) -> int:
    """Count files referencing any of ``symbols`` via AST name/attr matches.

    This is a static-analysis proxy for the "shotgun surgery" smell: work that
    ripples across many unrelated places is disproportionately likely to be an
    architectural decision wearing a bug-fix costume.

    A production hardening pass should prefer tree-sitter plus a proper name
    resolver; the stdlib AST version here fulfills the same contract.

    Parameters
    ----------
    repo_path:
        Repository root to walk for ``*.py`` files.
    symbols:
        Identifiers extracted from a proposed diff (or known rename targets).

    Returns
    -------
    int
        Number of distinct relative file paths that reference any symbol.
    """
    root, wanted, hits = Path(repo_path), set(symbols), set()
    if not wanted:
        return 0
    skip = {".", "venv", ".venv", "node_modules", "__pycache__"}
    if not root.exists():
        return 0
    for path in root.rglob("*.py"):
        if any(p in skip or p.startswith(".") for p in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        defs = {
            n.name
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        if wanted & (names | attrs | defs):
            hits.add(str(path.relative_to(root)))
    return len(hits)


def step_repo_path(step: ProposedStep) -> str:
    """Return the working-tree root used for static fan-out analysis."""
    return step.repo_path


class RiskClassifier:
    """Score a :class:`ProposedStep` from verifier existence, fan-out, and history.

    The classifier is deliberately boring. A silent failure in the router
    reproduces the exact problem the library exists to prevent.

    Parameters
    ----------
    config:
        Weights and escalation threshold.
    contract_registry:
        Map of ``task_class → VerifierContract`` (or any truthy entry). Presence
        alone is the verifier-existence signal; the executor looks up the
        contract later.
    calibration_store:
        Object exposing ``historical_rate(...)`` (Qdrant store or in-memory stub).

    Example
    -------
    >>> from verdict import RiskClassifier, ClassifierConfig, ProposedStep
    >>> from verdict import InMemoryCalibrationStore
    >>> from verdict.contracts import always_pass_contract
    >>> store = InMemoryCalibrationStore(class_priors={"rename": 0.05})
    >>> clf = RiskClassifier(ClassifierConfig(), {"rename": always_pass_contract()}, store)
    >>> risk = clf.score(ProposedStep("1", "rename DTO", ["a.py"] * 14, task_class="rename"))
    >>> risk.route.value
    'bounded'
    """

    def __init__(
        self,
        config: ClassifierConfig,
        contract_registry: dict[str, Any],
        calibration_store: HistoricalRateProvider,
    ) -> None:
        self.config = config
        self.contract_registry = contract_registry
        self.calibration_store = calibration_store

    def score(self, step: ProposedStep) -> RiskScore:
        """Compute ``R(t)`` and the bounded/escalate route for ``step``.

        Pure with respect to the working tree: never mutates files or graph
        state. Safe to call from CI hooks, plain scripts, or LangGraph nodes.
        """
        verifier_exists = step.task_class in self.contract_registry
        symbols = extract_changed_symbols(step.proposed_diff or "")
        planned = max(1, len(step.planned_files)) if step.planned_files else 1
        # Fall back to planned_files when static analysis finds no call sites yet
        # (common for greenfield symbols in a proposed diff).
        static = python_fan_out(step_repo_path(step), symbols) if symbols else 0
        fan_out = static or planned
        h = self.calibration_store.historical_rate(
            task_class=step.task_class,
            description=step.description,
            fan_out=fan_out,
        )
        cfg = self.config
        f_tilde = min(1.0, fan_out / cfg.f_max)
        value = cfg.w_v * (0.0 if verifier_exists else 1.0) + cfg.w_f * f_tilde + cfg.w_h * h
        reasons: list[str] = []
        if not verifier_exists:
            reasons.append(f"no verifier contract for task_class={step.task_class}")
        if f_tilde >= 0.5:
            reasons.append(f"fan_out={fan_out} (normalized={f_tilde:.2f})")
        if h >= 0.3:
            reasons.append(f"historical_incident_rate={h:.2f}")
        route = Route.ESCALATE if value >= cfg.tau else Route.BOUNDED
        return RiskScore(value, verifier_exists, fan_out, h, route, reasons)
