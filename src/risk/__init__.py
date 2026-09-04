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

# Task 7: Factor Risk (canonical)
try:
    from src.risk.factor_risk import (
        FactorRiskReport,
        FactorGateResult,
        DEFAULT_CRYPTO_FACTOR_MAP,
        compute_gross_exposure,
        compute_net_exposure,
        gross_exposure,
        net_exposure,
        compute_beta,
        beta_to_market,
        compute_portfolio_beta,
        compute_net_beta_exposure,
        crypto_beta,
        compute_factor_exposure,
        factor_exposure,
        compute_factor_concentration,
        factor_concentration,
        herfindahl_index,
        max_factor_weight,
        compute_correlation_adjusted_exposure,
        correlation_adjusted_exposure as factor_correlation_adjusted_exposure,
        compute_factor_risk,
        assess_factor_risk,
        check_factor_risk_gate,
    )
except Exception:  # pragma: no cover
    pass

from src.risk.correlation import (
    CorrelationExposure,
    correlation_adjusted_exposure,
    compute_gross_exposure as corr_gross_exposure,
    compute_net_exposure as corr_net_exposure,
    compute_beta as corr_compute_beta,
    beta_to_market as corr_beta_to_market,
    compute_portfolio_beta as corr_compute_portfolio_beta,
    compute_factor_exposure as corr_compute_factor_exposure,
    compute_factor_concentration as corr_compute_factor_concentration,
    liquidation_distance,
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

    # Factor Risk (Task 7)
    "FactorRiskReport",
    "FactorGateResult",
    "DEFAULT_CRYPTO_FACTOR_MAP",
    "compute_gross_exposure",
    "compute_net_exposure",
    "gross_exposure",
    "net_exposure",
    "compute_beta",
    "beta_to_market",
    "compute_portfolio_beta",
    "compute_net_beta_exposure",
    "crypto_beta",
    "compute_factor_exposure",
    "factor_exposure",
    "compute_factor_concentration",
    "factor_concentration",
    "herfindahl_index",
    "max_factor_weight",
    "compute_correlation_adjusted_exposure",
    "factor_correlation_adjusted_exposure",
    "compute_factor_risk",
    "assess_factor_risk",
    "check_factor_risk_gate",
    # Correlation (legacy re-export)
    "CorrelationExposure",
    "correlation_adjusted_exposure",
    "liquidation_distance",
]

__version__ = "5.1.0"