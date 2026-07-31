from __future__ import annotations

from pathlib import Path

import pytest
from verdict.calibration import InMemoryCalibrationStore
from verdict.classifier import (
    ClassifierConfig,
    ProposedStep,
    RiskClassifier,
    Route,
    extract_changed_symbols,
    python_fan_out,
)
from verdict.contracts import always_pass_contract


def test_extract_changed_symbols() -> None:
    diff = """\
--- a/x.py
+++ b/x.py
@@
+def helper_fn(a):
+    return a
+class Widget:
+    pass
+not a definition
"""
    assert extract_changed_symbols(diff) == ["helper_fn", "Widget"]


def test_python_fan_out(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def helper_fn():\n    return 1\n", encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "from a import helper_fn\nprint(helper_fn())\n", encoding="utf-8"
    )
    (tmp_path / "c.py").write_text("x = 1\n", encoding="utf-8")
    assert python_fan_out(str(tmp_path), ["helper_fn"]) == 2


def test_mechanical_rename_routes_bounded() -> None:
    store = InMemoryCalibrationStore(class_priors={"rename": 0.05})
    registry = {"rename": always_pass_contract("rename")}
    clf = RiskClassifier(ClassifierConfig(), registry, store)
    step = ProposedStep(
        step_id="a",
        description="Rename CustomerDTO to CustomerRecord",
        planned_files=[f"f{i}.py" for i in range(14)],
        task_class="rename",
    )
    risk = clf.score(step)
    # R = 0 + 0.30*min(1,14/8) + 0.25*0.05 = 0.3125
    assert risk.route == Route.BOUNDED
    assert risk.value == pytest.approx(0.3125, abs=1e-6)
    assert risk.verifier_exists is True


def test_architectural_boundary_routes_escalate() -> None:
    store = InMemoryCalibrationStore(class_priors={"boundary": 0.62})
    clf = RiskClassifier(ClassifierConfig(), {}, store)
    step = ProposedStep(
        step_id="b",
        description="Extract billing into a separate service",
        planned_files=["billing/service.py", "billing/facade.py", "app/main.py"],
        task_class="boundary",
    )
    risk = clf.score(step)
    # R = 0.45*1 + 0.30*min(1,3/8) + 0.25*0.62 = 0.7175
    assert risk.route == Route.ESCALATE
    assert risk.value == pytest.approx(0.7175, abs=1e-6)
    assert risk.verifier_exists is False
    assert any("no verifier" in r for r in risk.reasons)


def test_config_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="weights must sum"):
        ClassifierConfig(w_v=0.5, w_f=0.5, w_h=0.5)
