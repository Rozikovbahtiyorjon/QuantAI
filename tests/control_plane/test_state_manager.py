import pytest
from src.control_plane.state_manager import StateManager

@pytest.mark.asyncio
async def test_state_manager_fail_closed_no_evidence():
    sm = StateManager()
    # research -> architecture requires alpha_hypothesis_defined
    assert await sm.can_transition("architecture", None) is False
    # should not transition
    assert await sm.transition_to("architecture", None) is False
    assert sm.current_stage == "research"

@pytest.mark.asyncio
async def test_state_manager_pass_with_evidence():
    sm = StateManager()
    sm.set_criterion_evidence("alpha_hypothesis_defined", {"passed": True})
    sm.set_criterion_evidence("data_validated", {"passed": True})
    assert await sm.can_transition("architecture", None) is True
    assert await sm.transition_to("architecture", None) is True
    assert sm.current_stage == "architecture"

@pytest.mark.asyncio
async def test_state_manager_no_skip_stages():
    sm = StateManager()
    sm.set_criterion_evidence("alpha_hypothesis_defined", {"passed": True})
    sm.set_criterion_evidence("data_validated", {"passed": True})
    # cannot skip to implementation
    assert await sm.can_transition("implementation", None) is False

@pytest.mark.asyncio
async def test_state_manager_no_backward():
    sm = StateManager()
    sm.set_criterion_evidence("alpha_hypothesis_defined", {"passed": True})
    sm.set_criterion_evidence("data_validated", {"passed": True})
    await sm.transition_to("architecture", None)
    assert await sm.can_transition("research", None) is False
