"""Verdict: risk classification and escalation for agentic coding loops."""

from __future__ import annotations

from verdict.calibration import CalibrationStore, InMemoryCalibrationStore
from verdict.checkpoint import CheckpointCommit
from verdict.classifier import (
    ClassifierConfig,
    ProposedStep,
    RiskClassifier,
    RiskScore,
    Route,
    extract_changed_symbols,
    python_fan_out,
)
from verdict.contracts import ExecutionResult, VerifierContract, pytest_contract
from verdict.escalation import EscalationArtifact, ReviewPackage
from verdict.executor import BoundedExecutor

__all__ = [
    "BoundedExecutor",
    "CalibrationStore",
    "CheckpointCommit",
    "ClassifierConfig",
    "EscalationArtifact",
    "ExecutionResult",
    "InMemoryCalibrationStore",
    "ProposedStep",
    "ReviewPackage",
    "RiskClassifier",
    "RiskScore",
    "Route",
    "VerifierContract",
    "extract_changed_symbols",
    "pytest_contract",
    "python_fan_out",
]

__version__ = "0.1.0"
