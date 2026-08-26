"""
====================================================
QuantAI Professional
Portfolio & Risk Management - Phase 5
====================================================

Phase 5 Components:
1. Kelly Criterion Position Sizing
2. Portfolio Correlation Risk Management
3. Cross-Margin Management
4. Dynamic Risk Budget Manager

====================================================
"""

from __future__ import annotations

from src.risk.kelly_sizer import (
    KellyFraction,
    KellyResult,
    KellySizer,
    VolatilityAdjustedKellySizer,
)

from src.risk.portfolio_correlation import (
    CorrelationPair,
    CorrelationCluster,
    CorrelationRiskResult,
    CorrelationMatrix,
    PortfolioCorrelationManager,
)

from src.risk.cross_margin import (
    MarginMode,
    PositionMargin,
    CrossMarginAccount,
    CrossMarginManager,
)

from src.risk.dynamic_risk_budget import (
    RiskBudgetMode,
    RiskBudget,
    RiskBudgetManager,
)

__all__ = [
    # Kelly Criterion
    "KellyFraction",
    "KellyResult",
    "KellySizer",
    "VolatilityAdjustedKellySizer",
    
    # Portfolio Correlation
    "CorrelationPair",
    "CorrelationCluster",
    "CorrelationRiskResult",
    "CorrelationMatrix",
    "PortfolioCorrelationManager",
    
    # Cross-Margin
    "MarginMode",
    "PositionMargin",
    "CrossMarginAccount",
    "CrossMarginManager",
    
    # Dynamic Risk Budget
    "RiskBudgetMode",
    "RiskBudget",
    "RiskBudgetManager",
]

__version__ = "5.1.0"