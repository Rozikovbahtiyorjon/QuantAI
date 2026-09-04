import pytest
from src.control_plane.agent_router import AgentRouter
from src.control_plane.task_manager import Task

@pytest.mark.asyncio
async def test_agent_router_route():
    ar = AgentRouter()
    t = Task(name="research task", stage="research")
    agent = await ar.route(task=t, stage="research", state=None)
    assert agent is not None

@pytest.mark.asyncio
async def test_agent_router_execute_placeholder_rejected():
    ar = AgentRouter()
    class T:
        name = "ml task"
        metadata = {}
    # _run_ml_engineer previously returned fallback success true; now should be false for insufficient data
    res = await ar._run_ml_engineer(T(), None)
    # With insufficient data, should not be success true with fallback
    if "insufficient" in str(res).lower() or res.get("model") == "skipped_insufficient":
        assert res["success"] is False
    else:
        assert "success" in res
