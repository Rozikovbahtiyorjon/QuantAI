"""
====================================================
QuantAI Professional
Testnet Configuration
====================================================

Testnet-specific settings for Binance Testnet deployment.
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from config.settings import Settings, settings


@dataclass
class TestnetConfig:
    """Testnet-specific configuration overrides."""
    
    # Binance Testnet endpoints
    base_url: str = "https://testnet.binancefuture.com"
    ws_base_url: str = "wss://stream.binancefuture.com"
    
    # Testnet API credentials (set via environment variables)
    api_key: str = ""
    api_secret: str = ""
    
    # Testnet trading parameters
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    limit: int = 500
    
    # Testnet risk limits (more conservative)
    max_drawdown_pct: float = 5.0
    max_total_exposure_pct: float = 30.0
    max_position_exposure_pct: float = 3.0
    risk_per_trade: float = 0.005  # 0.5% per trade
    
    # Testnet trading mode
    testnet_mode: bool = True
    dry_run: bool = True  # Start with dry-run
    
    # Testnet monitoring
    log_level: str = "DEBUG"
    metrics_port: int = 9090
    health_check_interval: int = 30
    
    # Testnet limits (more conservative)
    max_open_positions: int = 2
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 5.0
    
    # Testnet ML settings
    ml_enabled: bool = True
    ml_walk_forward: bool = True
    
    # Testnet paper trading
    paper_trading: bool = True
    initial_balance: float = 10000.0  # Larger testnet balance
    
    def to_settings_dict(self) -> dict:
        """Convert to settings dictionary for Settings override."""
        return {
            "exchange": {
                "exchange": "binance",
                "symbol": self.symbol,
                "timeframe": self.timeframe,
                "limit": self.limit,
            },
            "account": {
                "initial_balance": self.initial_balance,
                "risk_per_trade": self.risk_per_trade,
                "max_open_positions": self.max_open_positions,
                "max_risk_percent": self.max_drawdown_pct,
                "min_position_size": 0.001,
                "max_position_size": 1.0,
            },
            "commission": {
                "commission": 0.0004,
                "slippage": 0.0002,
            },
            "backtest": {
                "enable_trailing_stop": True,
                "enable_break_even": True,
                "enable_partial_close": False,
                "risk_reward_ratio": 2.0,
            },
            "ml": {
                "ml_enabled": self.ml_enabled,
                "model_path": "models/quantai_v5.pkl",
                "cv_type": "purged",
                "n_splits": 5,
                "embargo_pct": 0.01,
                "purge_pct": 0.0,
                "n_test_folds": 2,
            },
            "risk": {
                "drawdown_limit_pct": self.max_drawdown_pct,
                "max_correlation": 0.85,
                "max_correlated_assets": 2,
                "position_sizer_method": "kelly",
                "max_total_exposure_percent": self.max_total_exposure_pct,
                "max_position_exposure_percent": self.max_position_exposure_pct,
            },
            "logging": {
                "log_level": self.log_level,
                "save_trades": True,
                "trades_file": "trades_testnet.csv",
            },
            "ml": {
                "cv_type": "purged",
                "n_splits": 5,
                "embargo_pct": 0.01,
                "purge_pct": 0.0,
                "n_test_folds": 2,
                "test_size": 0.20,
                "random_state": 42,
                "n_estimators": 300,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "use_class_weights": True,
            },
            "binance": {
                "api_key": "",
                "api_secret": "",
                "testnet": True,
                "recv_window": 5000,
            },
        }


# Global testnet config instance
testnet_config = TestnetConfig()

# Override settings for testnet
def apply_testnet_overrides(settings_obj) -> None:
    """Apply testnet overrides to settings object."""
    config = testnet_config.to_settings_dict()
    
    # Apply nested overrides
    def apply_nested(obj, path, value):
        keys = path.split("__")
        for key in keys[:-1]:
            obj = getattr(obj, key)
        setattr(obj, keys[-1], value)
    
    for key, value in config.items():
        if "__" in key:
            # Nested setting
            parts = key.split("__")
            obj = settings
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            # Top-level setting
            setattr(settings, key, value)


# Environment variable loading for testnet
def load_testnet_env() -> dict:
    """Load testnet environment variables."""
    import os
    return {
        "BINANCE_API_KEY": os.getenv("BINANCE_TESTNET_API_KEY", ""),
        "BINANCE_API_SECRET": os.getenv("BINANCE_TESTNET_API_SECRET", ""),
        "BINANCE_TESTNET": "true",
        "LOG_LEVEL": "DEBUG",
        "ML_ENABLED": "true",
        "TELEGRAM_ENABLED": "false",
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TestnetConfig",
    "testnet_config",
    "apply_testnet_overrides",
    "load_testnet_env",
]