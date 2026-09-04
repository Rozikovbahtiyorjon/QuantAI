import pytest
from src.control_plane.task_manager import TaskManager, Task

@pytest.mark.asyncio
async def test_run_tests_real():
    tm = TaskManager()
    t = Task(name="test task", stage="testing")
    res = await tm.run_tests(t, None)
    # Must not be placeholder 0 tests passed true
    assert not (res["passed"] is True and res["tests_run"] == 0)
    # Should have real tests_run >0 or passed False
    assert "tests_run" in res

@pytest.mark.asyncio
async def test_review_requires_verified():
    import tempfile, pathlib
    tm = TaskManager()
    t = Task(name="review task")
    # agent says success but command failed → rejected (P1.15: 0 tests_run)
    exec_res = {"success": True, "code": "", "tests_run": 0}
    test_res = {"passed": True, "tests_run": 0}
    rev = await tm.review(t, exec_res, test_res)
    assert rev["approved"] is False
    # proper verified should approve — needs real artifact file + exit_code 0 + metrics
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tf:
        tf.write("print('valid')")
        tf_path = tf.name
    try:
        exec_ok = {"success": True, "code": "print(1)", "tests_run": 5, "artifact_paths": [tf_path], "exit_code": 0}
        test_ok = {"passed": True, "tests_run": 5}
        rev2 = await tm.review(t, exec_ok, test_ok)
        assert rev2["approved"] is True
    finally:
        try:
            pathlib.Path(tf_path).unlink()
        except Exception:
            pass
    # fake metrics must be rejected (P1.16)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as tf2:
        tf2.write("print('valid')")
        tf2_path = tf2.name
    try:
        exec_fake = {"success": True, "code": "print(1)", "artifact_paths": [tf2_path], "exit_code": 0, "metrics": {"bal_acc": 0.39}}
        test_ok = {"passed": True, "tests_run": 5}
        rev3 = await tm.review(t, exec_fake, test_ok)
        assert rev3["approved"] is False
    finally:
        try:
            pathlib.Path(tf2_path).unlink()
        except Exception:
            pass

@pytest.mark.asyncio
async def test_task_lifecycle():
    tm = TaskManager()
    t = Task(name="lifecycle", stage="research")
    tid = await tm.create_task(t)
    fetched = await tm.get_task(tid)
    assert fetched is not None
    await tm.assign_task(fetched)
    await tm.start_task(tid, "quant_researcher")
    assert fetched.status.value in ("assigned", "running")
    await tm.complete_task(tid, result={"success": True})
    assert fetched.status.value == "completed"
