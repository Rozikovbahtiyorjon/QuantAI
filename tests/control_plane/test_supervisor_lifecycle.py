import pytest
from src.control_plane.supervisor import AISupervisor
from src.control_plane.validation_gate import GateResult, GateStatus

@pytest.mark.asyncio
async def test_supervisor_autonomous_transition():
    sup = AISupervisor()
    async def mock_save(*a, **kw): return
    sup.checkpoint_manager.save = mock_save
    async def mock_store(*a, **kw): return "id"
    sup.evidence_manager.store = mock_store
    class T:
        id = "t1"
        name = "research task"
    sup.state.current_task = T()
    sup.state_manager.current_stage = "research"
    sup.state_manager.stage_data.clear()
    res = GateResult(gate_name="aggregate", status=GateStatus.PASSED, passed=True, reason="All gates passed")
    await sup._handle_success(res)
    assert sup.state_manager.current_stage == "architecture"

@pytest.mark.asyncio
async def test_supervisor_no_transition_without_validation():
    sup = AISupervisor()
    sup.state_manager.current_stage = "research"
    sup.state_manager.stage_data.clear()
    can = await sup.state_manager.can_transition("architecture", sup.state)
    assert can is False

@pytest.mark.asyncio
async def test_supervisor_escalate_on_failure():
    sup = AISupervisor()
    # Mock retry to avoid external deps
    async def mock_diagnose(*a, **kw):
        return {"diagnosis": "test"}
    async def mock_repair(*a, **kw):
        from src.control_plane.retry_engine import RepairResult
        return RepairResult(success=False, action=None)
    sup.retry_engine.diagnose = mock_diagnose  # type: ignore
    sup.retry_engine.repair = mock_repair  # type: ignore
    class T:
        id = "t2"
        name = "fail task"
    sup.state.current_task = T()
    sup.state.last_execution_result = {"success": False}
    sup.state.last_test_result = {"passed": False, "tests_run": 1}
    sup.state.last_validation = None
    res = GateResult(gate_name="aggregate", status=GateStatus.FAILED, passed=False, reason="failed")
    await sup._handle_failure(res)
    assert "t2" in sup.state.failed_tasks or sup.state.escalated is True
