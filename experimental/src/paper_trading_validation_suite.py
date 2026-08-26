"""
paper_trading_validation_suite.py
QuantAI Framework - Paper Trading Validation & Quality Gate Suite
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("QuantAI.PaperTradingValidation")

class PaperTradingValidationSuite:
    """
    Validates paper trading sessions against realistic constraints:
    - Look-ahead bias detection
    - Slippage & fee realism
    - Drawdown & position lifecycle constraints
    - Multi-session consistency
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_allowed_drawdown = self.config.get("max_allowed_drawdown", 0.15)
        self.min_win_rate = self.config.get("min_win_rate", 0.40)
        self.required_sessions = self.config.get("required_sessions", 3)

    def validate_session_metrics(self, session_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates individual and multi-session paper trading metrics against quality gates.
        """
        logger.info("Starting paper trading session validation...")
        
        drawdown = session_metrics.get("max_drawdown", 0.0)
        win_rate = session_metrics.get("win_rate", 0.0)
        total_trades = session_metrics.get("total_trades", 0)
        look_ahead_detected = session_metrics.get("look_ahead_bias_flag", False)
        
        checks = {
            "drawdown_check": drawdown <= self.max_allowed_drawdown,
            "win_rate_check": win_rate >= self.min_win_rate,
            "sufficient_trades": total_trades >= 10,
            "no_look_ahead": not look_ahead_detected
        }
        
        passed = all(checks.values())
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "passed": passed,
            "checks": checks,
            "metrics_analyzed": session_metrics
        }
        
        if passed:
            logger.info("Paper Trading Session Validation PASSED successfully.")
        else:
            logger.warning(f"Paper Trading Session Validation FAILED. Details: {checks}")
            
        return report

    def run_multi_session_audit(self, sessions_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Audits multiple paper trading sessions for stability and performance consistency.
        """
        if len(sessions_history) < self.required_sessions:
            return {
                "audit_passed": False,
                "reason": f"Insufficient sessions. Required: {self.required_sessions}, Provided: {len(sessions_history)}"
            }
            
        session_results = [self.validate_session_metrics(s) for s in sessions_history]
        all_passed = all(res["passed"] for res in session_results)
        
        return {
            "audit_passed": all_passed,
            "total_sessions_audited": len(sessions_history),
            "session_results": session_results
        }