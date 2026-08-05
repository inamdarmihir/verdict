from __future__ import annotations

from verdict.calibration import InMemoryCalibrationStore, hashed_embed, outcome_label
from verdict.classifier import ClassifierConfig, ProposedStep, RiskClassifier, RiskScore, Route
from verdict.contracts import always_pass_contract


def test_outcome_label_matches_design_table() -> None:
    assert outcome_label(escalation_outcome="rejected", bounded_passed=None) == 1.0
    assert outcome_label(escalation_outcome="redirected", bounded_passed=False) == 0.5
    assert outcome_label(escalation_outcome="approved", bounded_passed=None) == 0.0
    assert outcome_label(escalation_outcome=None, bounded_passed=True) == 0.0
    assert outcome_label(escalation_outcome=None, bounded_passed=False) == 1.0


def test_hashed_embed_is_deterministic() -> None:
    assert hashed_embed("rename: CustomerDTO") == hashed_embed("rename: CustomerDTO")
    assert hashed_embed("a") != hashed_embed("b totally different tokens here")


def test_inmemory_historical_rate_cold_start() -> None:
    store = InMemoryCalibrationStore(class_priors={"boundary": 0.62})
    assert (
        store.historical_rate(task_class="boundary", description="extract billing", fan_out=3)
        == 0.62
    )
    assert store.historical_rate(task_class="unknown", description="x", fan_out=1) == 0.2


def test_record_and_query_similar() -> None:
    store = InMemoryCalibrationStore()
    step = ProposedStep("1", "extract billing service", ["a.py"], task_class="boundary")
    risk = RiskScore(0.7, False, 3, 0.6, Route.ESCALATE)
    store.record(repo="demo", step=step, risk=risk, label=0.5, notes="redirected")
    rate = store.historical_rate(
        task_class="boundary", description="extract billing service", fan_out=3
    )
    assert rate == 0.5


def test_recalibrate_lowers_tau_on_exhaustion() -> None:
    store = InMemoryCalibrationStore()
    clf = RiskClassifier(
        ClassifierConfig(tau=0.55),
        {"bugfix": always_pass_contract("bugfix")},
        store,
    )
    store.recalibrate(clf, label=1.0, iteration=2, max_iterations=2)
    assert clf.config.tau == 0.53
