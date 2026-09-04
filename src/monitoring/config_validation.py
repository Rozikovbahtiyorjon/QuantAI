"""
====================================================
QuantAI Professional
Configuration Validation
====================================================

Startup validation for all critical configuration parameters.
Prevents runtime failures due to misconfiguration.
====================================================
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from config.settings import Settings
from pydantic import ValidationError


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False
    
    def add_warning(self, msg: str):
        self.warnings.append(msg)
    
    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ConfigValidator:
    """
    Comprehensive configuration validator.
    
    Validates:
    - Required environment variables
    - API credentials
    - Path existence
    - Numeric ranges
    - Logical consistency
    """
    
    def __init__(
        self,
        settings,
    ):
        self.settings = settings
        self.result = ValidationResult(valid=True)
    
    def validate_all(self) -> "ValidationResult":
        """Run all validation checks."""
        self._validate_required_env()
        # Skip Binance validation in PAPER mode
        if self.settings.exchange.mode != "PAPER":
            self._validate_binance_config()
        self._validate_paths()
        self._validate_risk_params()
        self._validate_ml_config()
        self._validate_trading_params()
        self._validate_logging()
        
        return self.result
    
    def _validate_required_env(self):
        """Check required environment variables."""
        # Skip Binance API keys in PAPER mode
        if self.settings.exchange.mode == "PAPER":
            return
            
        required = [
            "BINANCE_TESTNET_API_KEY",
            "BINANCE_TESTNET_API_SECRET",
        ]
        
        for var in required:
            if not os.getenv(var):
                if not os.getenv("BINANCE_API_KEY"):
                    self.result.add_error(f"Required env var not set: {var}")
    
    def _validate_binance_config(self):
        """Validate Binance configuration."""
        b = self.settings.exchange
        
        if not b.api_key:
            self.result.add_error("Binance API key not configured")
        elif len(b.api_key) < 10:
            self.result.add_warning("Binance API key seems too short")
        
        if not b.api_secret:
            self.result.add_error("Binance API secret not configured")
        elif len(b.api_secret) < 10:
            self.result.add_warning("Binance API secret seems too short")
        
        if b.testnet:
            self.result.add_warning("Running in TESTNET mode")
        
        # Validate symbol format
        if not b.symbol or "/" not in b.symbol:
            self.result.add_error(f"Invalid symbol format: {b.symbol}")
        
        # Validate timeframe
        valid_timeframes = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
        if b.timeframe not in valid_timeframes:
            self.result.add_error(f"Invalid timeframe: {b.timeframe}. Valid: {valid_timeframes}")
        
        if b.limit < 100 or b.limit > 5000:
            self.result.add_warning(f"Limit {b.limit} outside recommended range (100-5000)")
    
    def _validate_paths(self):
        """Validate file paths exist or can be created."""
        # Model path
        model_path = Path(self.settings.ml.model_path)
        if not model_path.parent.exists():
            self.result.add_warning(f"Model directory does not exist: {model_path.parent}")
        
        # Log file
        log_file = self.settings.logging.trades_file
        log_path = Path(log_file)
        if not log_path.parent.exists():
            self.result.add_warning(f"Log directory does not exist: {log_path.parent}")
        
        # Data directory
        data_dir = Path("data")
        if not data_dir.exists():
            self.result.add_warning(f"Data directory does not exist: {data_dir}")
    
    def _validate_risk_params(self):
        """Validate risk management parameters."""
        r = self.settings.risk
        a = self.settings.account
        
        # Drawdown limit
        if r.max_drawdown_pct <= 0 or r.max_drawdown_pct > 50:
            self.result.add_error(f"drawdown_limit_pct must be 0-50, got {r.max_drawdown_pct}")
        
        # Correlation limits
        if r.max_correlation < 0 or r.max_correlation > 1:
            self.result.add_error(f"max_correlation must be 0-1, got {r.max_correlation}")
        
        if r.max_correlated_assets < 1:
            self.result.add_error(f"max_correlated_assets must be >= 1, got {r.max_correlated_assets}")
        
        # Position sizing
        if self.settings.risk.risk_per_trade <= 0 or self.settings.risk.risk_per_trade > 1:
            self.result.add_error(f"risk_per_trade must be 0-1, got {self.settings.risk.risk_per_trade}")
        
        # Exposure limits
        if r.max_total_exposure_pct <= 0 or r.max_total_exposure_pct > 100:
            self.result.add_error(f"max_total_exposure_pct must be 0-100, got {r.max_total_exposure_pct}")
        
        if r.max_position_exposure_pct <= 0 or r.max_position_exposure_pct > 100:
            self.result.add_error(f"max_position_exposure_pct must be 0-100, got {r.max_position_exposure_pct}")
    
    def _validate_ml_config(self):
        """Validate ML configuration."""
        ml = self.settings.ml
        
        if ml.ml_enabled:
            # Check model path
            model_path = Path(ml.model_path)
            if not model_path.parent.exists():
                self.result.add_warning(f"Model directory does not exist: {ml.model_path}")
            
            # Validate CV params
            if ml.n_splits < 2 or ml.n_splits > 20:
                self.result.add_warning(f"n_splits={ml.n_splits} outside typical range (2-20)")
            
            if ml.embargo_pct < 0 or ml.embargo_pct > 0.5:
                self.result.add_warning(f"embargo_pct={ml.embargo_pct} outside typical range (0-0.5)")
            
            # Training params
            if ml.n_estimators < 50 or ml.n_estimators > 2000:
                self.result.add_warning(f"n_estimators={ml.n_estimators} outside typical range (50-2000)")
            
            if ml.max_depth < 2 or ml.max_depth > 20:
                self.result.add_warning(f"max_depth={ml.max_depth} outside typical range (2-20)")
            
            if ml.learning_rate <= 0 or ml.learning_rate > 1:
                self.result.add_error(f"learning_rate must be 0-1, got {ml.learning_rate}")
            
            if ml.subsample <= 0 or ml.subsample > 1:
                self.result.add_error(f"subsample must be 0-1, got {ml.subsample}")
            
            if ml.colsample_bytree <= 0 or ml.colsample_bytree > 1:
                self.result.add_error(f"colsample_bytree must be 0-1, got {ml.colsample_bytree}")
    
    def _validate_trading_params(self):
        """Validate trading parameters (Core 4 indicators only)."""
        b = self.settings.backtest
        i = self.settings.indicators
        
        # EMA periods
        if i.ema_fast >= i.ema_slow:
            self.result.add_error("ema_fast must be < ema_slow")
        if i.ema_slow >= i.ema_trend:
            self.result.add_error("ema_slow must be < ema_trend")
        
        # RSI
        if i.rsi_period < 2:
            self.result.add_error(f"rsi_period must be >= 2, got {i.rsi_period}")
        if i.rsi_oversold >= i.rsi_overbought:
            self.result.add_error("rsi_oversold must be < rsi_overbought")
        if not (0 <= i.rsi_buy <= 100):
            self.result.add_error(f"rsi_buy must be 0-100, got {i.rsi_buy}")
        if not (0 <= i.rsi_sell <= 100):
            self.result.add_error(f"rsi_sell must be 0-100, got {i.rsi_sell}")
        
        # ATR
        if i.atr_period < 2:
            self.result.add_error(f"atr_period must be >= 2, got {i.atr_period}")
        if i.atr_stop_multiplier <= 0:
            self.result.add_error(f"atr_stop_multiplier must be > 0, got {i.atr_stop_multiplier}")
        if i.atr_take_multiplier <= 0:
            self.result.add_error(f"atr_take_multiplier must be > 0, got {i.atr_take_multiplier}")
        
        # Volume
        if i.volume_ma < 2:
            self.result.add_error(f"volume_ma must be >= 2, got {i.volume_ma}")
        if i.volume_filter <= 0:
            self.result.add_error(f"volume_filter must be > 0, got {i.volume_filter}")
        
        # Backtest
        if b.risk_reward_ratio <= 0:
            self.result.add_error(f"risk_reward_ratio must be > 0, got {b.risk_reward_ratio}")
        
        # SuperTrend (kept for backward compat)
        if self.settings.indicators.supertrend_period < 2:
            self.result.add_error(f"supertrend_period must be >= 2, got {self.settings.indicators.supertrend_period}")
        if self.settings.indicators.supertrend_multiplier <= 0:
            self.result.add_error(f"supertrend_multiplier must be > 0, got {self.settings.indicators.supertrend_multiplier}")
    
    def _validate_logging(self):
        """Validate logging configuration."""
        l = self.settings.logging
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if l.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            self.result.add_error(f"log_level must be one of {valid_levels}, got {l.log_level}")


def validate_config(settings_obj) -> "ValidationResult":
    """Convenience function to validate configuration."""
    validator = ConfigValidator(settings_obj)
    return validator.validate_all()


def validate_config_or_exit(settings_obj) -> "ValidationResult":
    """Validate config and exit on failure."""
    result = validate_config(settings_obj)
    
    if not result.valid:
        print("\nConfiguration validation failed:")
        for error in result.errors:
            print(f"  ERROR: {error}")
        for warning in result.warnings:
            print(f"  WARNING: {warning}")
        sys.exit(1)
    
    if result.warnings:
        for warning in result.warnings:
            print(f"  WARNING: {warning}")
    
    return result


def run_startup_validation(settings_obj) -> "ValidationResult":
    """Run startup validation and return result."""
    validator = ConfigValidator(settings_obj)
    return validator.validate_all()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ValidationResult",
    "ConfigValidator",
    "validate_config",
    "validate_config_or_exit",
    "run_startup_validation",
]