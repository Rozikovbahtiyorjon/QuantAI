"""
QuantAI State Manager
Manages supervisor state and pipeline stages
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .supervisor import SupervisorState


class PipelineStage(str, Enum):
    """Pipeline stages in order"""
    RESEARCH = "research"
    ARCHITECTURE = "architecture"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    BACKTEST = "backtest"
    WFO = "wfo"
    PAPER = "paper"
    OPTIMIZATION = "optimization"
    CHAMPION = "champion"
    PRODUCTION = "production"


@dataclass
class PipelineStageInfo:
    name: str
    stage: str
    entry_criteria: List[str]
    exit_criteria: List[str]
    required_agents: List[str]
    estimated_duration_hours: float
    required_gates: List[str]


@dataclass
class SupervisorState:
    """Current state of the AI Supervisor"""
    current_stage: str = "research"
    active_tasks: Dict[str, Any] = field(default_factory=dict)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    current_agent: Optional[str] = None
    last_checkpoint: Optional[datetime] = None
    iteration: int = 0
    errors: List[str] = field(default_factory=list)
    safety_violations: List[str] = field(default_factory=list)
    escalated: bool = False
    last_analysis: Optional[Dict[str, Any]] = None
    last_execution_result: Optional[Dict[str, Any]] = None
    last_test_result: Optional[Dict[str, Any]] = None
    last_review: Optional[Dict[str, Any]] = None
    last_validation: Optional[Any] = None
    last_execution_result: Optional[Dict[str, Any]] = None
    last_test_result: Optional[Dict[str, Any]] = None
    current_task: Optional[Any] = None


class StateManager:
    """
    Manages supervisor state and pipeline progression.
    Tracks current stage, transitions, and stage-specific requirements.
    """
    
    def __init__(self):
        self.current_stage: str = "research"
        self.stage_history: List[Dict[str, Any]] = []
        self.stage_start_time: Optional[datetime] = None
        self.stage_data: Dict[str, Any] = {}
        
        # Define pipeline stages
        self.stages: Dict[str, PipelineStageInfo] = {
            "research": PipelineStageInfo(
                name="Research",
                stage="research",
                entry_criteria=[],
                exit_criteria=["alpha_hypothesis_defined", "data_validated"],
                required_agents=["quant_researcher", "data_engineer"],
                estimated_duration_hours=24.0,
                required_gates=["data_validation_gate", "hypothesis_gate"]
            ),
            "architecture": PipelineStageInfo(
                name="Architecture",
                stage="architecture",
                entry_criteria=["alpha_hypothesis_defined"],
                exit_criteria=["architecture_documented", "interfaces_defined"],
                required_agents=["quant_engineer", "quant_researcher"],
                estimated_duration_hours=8.0,
                required_gates=["architecture_review_gate"]
            ),
            "implementation": PipelineStageInfo(
                name="Implementation",
                stage="implementation",
                entry_criteria=["architecture_documented"],
                exit_criteria=["code_complete", "unit_tests_pass", "integration_tests_pass"],
                required_agents=["quant_engineer", "data_engineer", "code_reviewer"],
                estimated_duration_hours=48.0,
                required_gates=["code_review_gate", "unit_test_gate", "integration_test_gate"]
            ),
            "testing": PipelineStageInfo(
                name="Testing",
                stage="testing",
                entry_criteria=["implementation_complete"],
                exit_criteria=["compile_pass", "required_tests_pass", "no_lookahead_pass", "risk_tests_pass"],
                required_agents=["qa_engineer", "ml_engineer"],
                estimated_duration_hours=16.0,
                required_gates=["compile_gate", "unit_test_gate", "no_lookahead_gate", "risk_test_gate"]
            ),
            "backtest": PipelineStageInfo(
                name="Backtest",
                stage="backtest",
                entry_criteria=["all_tests_pass"],
                exit_criteria=["backtest_profitable", "risk_limits_ok"],
                required_agents=["ml_engineer", "quant_researcher", "risk_manager"],
                estimated_duration_hours=8.0,
                required_gates=["backtest_profitability_gate", "risk_limits_gate"]
            ),
            "wfo": PipelineStageInfo(
                name="Walk-Forward Optimization",
                stage="wfo",
                entry_criteria=["backtest_pass"],
                exit_criteria=["oos_profitable", "stable_params"],
                required_agents=["ml_engineer", "quant_researcher"],
                estimated_duration_hours=24.0,
                required_gates=["wfo_stability_gate", "overfitting_gate"]
            ),
            "paper": PipelineStageInfo(
                name="Paper Trading",
                stage="paper",
                entry_criteria=["wfo_pass"],
                exit_criteria=["paper_profitable", "consistent_performance"],
                required_agents=["risk_manager", "execution_engineer"],
                estimated_duration_hours=720.0,  # 30 days
                required_gates=["paper_profitability_gate", "consistency_gate"]
            ),
            "optimization": PipelineStageInfo(
                name="Optimization",
                stage="optimization",
                entry_criteria=["paper_pass"],
                exit_criteria=["optimized_params_stable", "improved_metrics"],
                required_agents=["ml_engineer", "quant_researcher"],
                estimated_duration_hours=24.0,
                required_gates=["optimization_gate", "overfitting_gate"]
            ),
            "champion": PipelineStageInfo(
                name="Champion Selection",
                stage="champion",
                entry_criteria=["optimization_pass"],
                exit_criteria=["champion_selected", "promotion_criteria_met"],
                required_agents=["portfolio_manager", "risk_manager"],
                estimated_duration_hours=4.0,
                required_gates=["champion_gate", "promotion_gate"]
            ),
            "production": PipelineStageInfo(
                name="Production Deployment",
                stage="production",
                entry_criteria=["champion_selected", "all_gates_pass"],
                exit_criteria=["live_profitable", "monitoring_active"],
                required_agents=["execution_engineer", "risk_manager", "portfolio_manager"],
                estimated_duration_hours=0.0,  # Continuous
                required_gates=["production_readiness_gate", "live_monitoring_gate"]
            ),
        }
    
    async def get_current_stage(self) -> str:
        """Get current pipeline stage"""
        return self.current_stage
    
    async def get_stage_info(self, stage: str) -> Optional[PipelineStageInfo]:
        """Get stage information"""
        return self.stages.get(stage)
    
    async def get_current_stage_info(self) -> PipelineStageInfo:
        """Get current stage info"""
        return self.stages.get(self.current_stage, self.stages["research"])
    
    async def can_transition(self, target_stage: str, state: Any) -> bool:
        """Check if transition to target stage is allowed"""
        current_info = self.stages.get(self.current_stage)
        target_info = self.stages.get(target_stage)
        
        if not current_info or not target_info:
            return False
        
        # Check if target is next logical stage
        stages_order = list(self.stages.keys())
        current_idx = stages_order.index(self.current_stage)
        target_idx = stages_order.index(target_stage)
        
        if target_idx <= current_idx:
            return False  # Can't go backwards
        
        if target_idx > current_idx + 1:
            return False  # Can't skip stages
        
        # Check exit criteria of current stage
        current_info = self.stages[self.current_stage]
        for criterion in current_info.exit_criteria:
            if not await self._check_criterion(criterion):
                return False
        
        # Check entry criteria of target stage
        target_info = self.stages[target_stage]
        for criterion in target_info.entry_criteria:
            if not await self._check_criterion(criterion):
                return False
        
        return True
    
    async def _check_criterion(self, criterion: str) -> bool:
        """Check if a criterion is met — fail-closed: unknown criterion → False."""
        # Point 5: placeholder return True caused autonomous stage transition without evidence.
        # Now fail-closed: require explicit evidence in stage_data.
        # Mapping from criterion names to required evidence keys/data.
        # If stage_data lacks required gate result, return False.
        # This forces can_transition to require real gate PASS before stage exit.
        # For backward compat, if criterion explicitly stored as True in stage_data, respect it.
        if criterion in self.stage_data:
            val = self.stage_data[criterion]
            if isinstance(val, bool):
                return val
            if isinstance(val, dict):
                # Gate result dict must have status PASS
                if val.get("status") == "PASS" or val.get("passed") is True or val.get("verdict") == "PASS":
                    return True
                return False
            return bool(val)
        # No evidence for this criterion — fail-closed
        return False

    def set_criterion_evidence(self, criterion: str, evidence: Any) -> None:
        """Record gate evidence for criterion to allow transition."""
        self.stage_data[criterion] = evidence
    
    async def transition_to(self, target_stage: str, state: Any) -> bool:
        """Transition to a new stage"""
        if not await self.can_transition(target_stage, None):
            return False
        
        # Record transition
        self.stage_history.append({
            "from_stage": self.current_stage,
            "to_stage": target_stage,
            "timestamp": datetime.now(timezone.utc),
            "iteration": getattr(state, 'iteration', 0)
        })
        
        self.current_stage = target_stage
        self.stage_start_time = datetime.now(timezone.utc)
        self.stage_data = {}
        
        return True
    
    async def get_stage_progress(self) -> Dict[str, Any]:
        """Get progress in current stage"""
        if not self.stage_start_time:
            return {"progress": 0.0, "elapsed_hours": 0}
        
        elapsed = (datetime.now(timezone.utc) - self.stage_start_time).total_seconds() / 3600
        stage_info = self.stages.get(self.current_stage)
        
        if not stage_info or stage_info.estimated_duration_hours <= 0:
            return {"progress": 1.0, "elapsed_hours": elapsed}
        
        progress = min(elapsed / stage_info.estimated_duration_hours, 1.0)
        return {
            "progress": progress,
            "elapsed_hours": elapsed,
            "estimated_total_hours": stage_info.estimated_duration_hours,
            "remaining_hours": max(0, stage_info.estimated_duration_hours - elapsed)
        }


__all__ = [
    "PipelineStage",
    "PipelineStageInfo",
    "StateManager",
]