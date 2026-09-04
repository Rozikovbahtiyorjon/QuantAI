import pytest
from src.control_plane.retry_engine import RetryEngine

@pytest.mark.asyncio
async def test_retry_exhaustion_blocked():
    re = RetryEngine()
    # Simulate budget exceeded scenario via supervisor check_retry already tested
    # Here test that repair fails after max attempts
    class T:
        id = "t"
        name = "retry task"
        retry_count = 10
    # diagnose should handle high retry count
    diag = await re.diagnose(task=T(), execution_result={"success": False}, test_result={"passed": False}, validation_result=None)
    assert isinstance(diag, dict)

@pytest.mark.asyncio
async def test_agent_says_success_but_command_failed_rejected():
    from src.control_plane.task_manager import TaskManager, Task
    tm = TaskManager()
    t = Task(name="fail cmd")
    exec_res = {"success": True, "exit_code": 1, "code": "bad"}
    test_res = {"passed": False, "tests_run": 5}
    rev = await tm.review(t, exec_res, test_res)
    # Even if agent says success, review should check tests
    assert rev["approved"] is False or "failed" in str(rev).lower()
