import pytest
from src.control_plane.validation_gate import ValidationGate

@pytest.mark.asyncio
async def test_champion_gate_rejects_fake_pass():
    vg = ValidationGate()
    class S:
        current_stage = "champion"
        last_execution_result = {"champion": "fake", "passed": True, "score": 0.83}
        last_validation = None
    s = S()
    res = await vg._gate_champion(None, {"champion": "fake", "passed": True}, {}, {}, s)
    assert res.passed is False

@pytest.mark.asyncio
async def test_production_gate_missing_result_blocked():
    vg = ValidationGate()
    class S:
        current_stage = "production"
        last_execution_result = {}
    s = S()
    res = await vg._gate_production_readiness(None, {}, {}, {}, s)
    assert res.passed is False
    assert "missing" in res.reason.lower() or "requires" in res.reason.lower()

@pytest.mark.asyncio
async def test_oos_missing_blocked():
    vg = ValidationGate()
    # Use wfo gate that requires OOS
    # Simulate missing OOS evidence via production readiness
    class S:
        current_stage = "production"
        last_execution_result = {"oos_pass": False, "paper_pass": True, "risk_pass": True, "execution_pass": True, "monitoring_pass": True, "statistical_pass": True}
    s = S()
    res = await vg._gate_production_readiness(None, s.last_execution_result, {}, {}, s)
    assert res.passed is False

@pytest.mark.asyncio
async def test_champion_metrics_missing_blocked():
    vg = ValidationGate()
    class S:
        current_stage = "champion"
        last_execution_result = {"champion": "strat", "metrics": {}}
        last_validation = None
    s = S()
    res = await vg._gate_champion(None, s.last_execution_result, {}, {}, s)
    assert res.passed is False
