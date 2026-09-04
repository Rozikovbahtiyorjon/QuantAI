import pytest
import pandas as pd
from src.research.oos_firewall import OOSFirewall
from src.research.nested_research_pipeline import HoldoutLock

def make_df(n=5000):
    return pd.DataFrame({"close": list(range(n)), "timestamp": pd.date_range("2020-01-01", periods=n, freq="1h")})

def test_oos_touched_twice_blocked():
    fw = OOSFirewall()
    df = make_df()
    dev, lock = fw.split(df)
    assert lock.touch_count == 0
    # Must FREEZE before FINAL HOLDOUT (P0.10 lifecycle: DEVELOPMENT->OPTIMIZATION->FREEZE->HOLDOUT)
    fw.freeze()
    # First touch should succeed
    fw.validate_holdout(champion_spec=None, validator_fn=lambda spec, holdout: {"ok": True})
    assert lock.sealed is True
    # Second touch should be blocked (one-shot)
    with pytest.raises(RuntimeError):
        fw.validate_holdout(champion_spec=None, validator_fn=lambda spec, holdout: {"ok": True})

def test_oos_without_freeze_blocked():
    fw = OOSFirewall()
    df = make_df()
    fw.split(df)
    # Without freeze, validate must be blocked (fail-closed)
    with pytest.raises(RuntimeError, match="must FREEZE"):
        fw.validate_holdout(champion_spec=None, validator_fn=lambda spec, holdout: {"ok": True})

def test_budget_exceeded_blocked():
    from src.research.research_budget import ResearchBudget
    b = ResearchBudget(max_experiments=1)
    b.check_experiment()
    with pytest.raises(Exception):
        b.check_experiment()

@pytest.mark.asyncio
async def test_oos_missing_blocked():
    from src.control_plane.validation_gate import ValidationGate
    vg = ValidationGate()
    class S:
        current_stage = "wfo"
        last_execution_result = {}
    s = S()
    res = await vg._gate_production_readiness(None, {}, {}, {}, s)
    assert res.passed is False

@pytest.mark.asyncio
async def test_champion_metrics_missing_blocked():
    from src.control_plane.validation_gate import ValidationGate
    vg = ValidationGate()
    class S:
        current_stage = "champion"
        last_execution_result = {"champion": "strat", "metrics": {}}
        last_validation = None
    s = S()
    res = await vg._gate_champion(None, s.last_execution_result, {}, {}, s)
    assert res.passed is False

@pytest.mark.asyncio
async def test_production_gate_missing_blocked():
    from src.control_plane.validation_gate import ValidationGate
    vg = ValidationGate()
    class S:
        current_stage = "production"
        last_execution_result = {}
    s = S()
    res = await vg._gate_production_readiness(None, {}, {}, {}, s)
    assert res.passed is False
