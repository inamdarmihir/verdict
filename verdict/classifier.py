"""Risk classifier: score a proposed step before it runs."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


class HistoricalRateProvider(Protocol):
    """Minimal calibration interface the classifier depends on."""

    def historical_rate(
        self,
        *,
        task_class: str,
        description: str,
        fan_out: int,
        top_k: int = 20,
    ) -> float: ...


class Route(StrEnum):
    BOUNDED = "bounded"
    ESCALATE = "escalate"


@dataclass
class ProposedStep:
    step_id: str
    description: str
    planned_files: list[str]
    proposed_diff: str | None = None
    task_class: str = "general"  # rename | migration | boundary | bugfix | …
    repo_path: str = "."


@dataclass
class RiskScore:
    value: float
    verifier_exists: bool
    fan_out: int
    historical_incident_rate: float
    route: Route
    reasons: list[str] = field(default_factory=list)


@dataclass
class ClassifierConfig:
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
    """Count files referencing any of `symbols` via AST name/attr matches.

    A real release should prefer tree-sitter plus a proper name resolver;
    the stdlib AST version here shows the contract this function fulfills.
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
    return step.repo_path


class RiskClassifier:
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
