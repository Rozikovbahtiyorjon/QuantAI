"""
====================================================
QuantAI Professional Configuration (Pydantic v2)
====================================================
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeSettings(BaseSettings):
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "15m"
    limit: int = 1000
    api_key: str = Field(default="", alias="BINANCE_TESTNET_API_KEY")
    api_secret: str = Field(default="", alias="BINANCE_TESTNET_API_SECRET")
    testnet: bool = Field(default=False, alias="BINANCE_TESTNET")
    recv_window: int = 5000

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        valid = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
        if v not in valid:
            raise ValueError(f"Invalid timeframe: {v}. Must be one of {valid}")
        return v


class AccountSettings(BaseSettings):
    initial_balance: float = 1000.0
    risk_per_trade: float = 0.01
    max_open_positions: int = 1
    max_risk_percent: float = 1.0
    min_position_size: float = 0.001
    max_position_size: float = 1.0


class CommissionSettings(BaseSettings):
    commission: float = 0.0004
    slippage: float = 0.0002


class IndicatorSettings(BaseSettings):
    # EMA
    ema_fast: int = 20
    ema_slow: int = 50
    ema_trend: int = 200

    # RSI
    rsi_period: int = 14
    rsi_buy: float = 55
    rsi_sell: float = 45
    rsi_overbought: float = 70
    rsi_oversold: float = 30

    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # ADX
    adx_period: int = 14
    adx_min: float = 25.0

    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0

    # ATR
    atr_period: int = 14
    atr_stop_multiplier: float = 1.5
    atr_take_multiplier: float = 3.0
    trailing_stop_multiplier: float = 2.0

    # Volume
    volume_ma: int = 20
    volume_filter: float = 1.2

    # SuperTrend (kept for backward compat)
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0


class StrategySettings(BaseSettings):
    # Fusion rules
    min_confidence: float = 60.0
    ai_weight: float = 0.60
    ml_weight: float = 0.40
    conflict_penalty: float = 0.70
    ml_hold_blocks_ai: bool = True
    ai_hold_blocks_all: bool = True
    
    # Order Flow
    orderflow_enabled: bool = True
    orderflow_conflict_threshold: float = 0.15
    orderflow_vpin_toxic: float = 0.8
    orderflow_vpin_warning: float = 0.6
    orderflow_kyle_max: float = 0.01
    orderflow_liq_proximity: float = 0.5
    orderflow_pressure_threshold: float = 0.3
    
    # Confidence Engine weights
    confidence_trend_weight: float = 1.50
    confidence_momentum_weight: float = 1.20
    confidence_volume_weight: float = 1.10
    confidence_volatility_weight: float = 1.00
    
    # SL/TP
    sl_tp_method: Literal["atr_fixed", "atr_adaptive"] = "atr_adaptive"


class BacktestSettings(BaseSettings):
    enable_trailing_stop: bool = True
    enable_break_even: bool = True
    enable_partial_close: bool = False
    risk_reward_ratio: float = 2.0


class MLSettings(BaseSettings):
    ml_enabled: bool = False
    model_path: str = "models/quantai_v5.pkl"

    # Purged K-Fold CV - combinatorial provides 10x more OOS paths (C5,2)
    cv_type: Literal["purged", "combinatorial"] = "combinatorial"
    n_splits: int = 5
    embargo_pct: float = 0.01
    purge_pct: float = 0.0
    regime_aware: bool = False
    regime_min_samples: int = 150
    n_test_folds: int = 2

    # ML Engine config
    test_size: float = 0.20
    random_state: int = 42
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.90
    colsample_bytree: float = 0.90
    use_class_weights: bool = True


class TelegramSettings(BaseSettings):
    enabled: bool = False
    token: str = ""
    chat_id: str = ""


class LoggingSettings(BaseSettings):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_trades: bool = True
    trades_file: str = "trades.csv"


class RiskSettings(BaseSettings):
    # Drawdown limits
    drawdown_limit_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    
    # Exposure limits
    max_total_exposure_pct: float = 60.0
    max_position_exposure_pct: float = 5.0
    max_open_positions: int = 1
    
    # Position sizing
    position_sizer_method: Literal["fixed_fractional", "kelly", "volatility_adjusted"] = "fixed_fractional"
    risk_per_trade: float = 0.01  # 1% per trade
    max_risk_percent: float = 1.0
    min_position_size: float = 0.001
    max_position_size: float = 1.0
    risk_reward_ratio: float = 2.0
    min_risk_reward_ratio: float = 1.5
    
    # Leverage limits
    max_leverage: float = 10.0
    min_leverage: float = 1.0
    
    # Correlation limits
    max_correlation: float = 0.85
    max_correlated_assets: int = 2
    
    # Stop loss / Take profit multipliers (ATR-based)
    atr_stop_multiplier: float = 1.5
    atr_take_multiplier: float = 3.0
    trailing_stop_multiplier: float = 2.0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    project_name: str = "QuantAI"
    version: str = "5.1.0"

    exchange: ExchangeSettings = ExchangeSettings()
    account: AccountSettings = AccountSettings()
    commission: CommissionSettings = CommissionSettings()
    indicators: IndicatorSettings = IndicatorSettings()
    backtest: BacktestSettings = BacktestSettings()
    ml: MLSettings = MLSettings()
    telegram: TelegramSettings = TelegramSettings()
    logging: LoggingSettings = LoggingSettings()
    risk: RiskSettings = RiskSettings()
    strategy: StrategySettings = StrategySettings()

    # Convenience properties for backward compatibility
    @property
    def SYMBOL(self) -> str:
        return self.exchange.symbol

    @property
    def TIMEFRAME(self) -> str:
        return self.exchange.timeframe

    @property
    def LIMIT(self) -> int:
        return self.exchange.limit

    @property
    def INITIAL_BALANCE(self) -> float:
        return self.account.initial_balance

    @property
    def RISK_PER_TRADE(self) -> float:
        return self.account.risk_per_trade

    @property
    def MAX_OPEN_POSITIONS(self) -> int:
        return self.account.max_open_positions

    @property
    def COMMISSION(self) -> float:
        return self.commission.commission

    @property
    def SLIPPAGE(self) -> float:
        return self.commission.slippage

    @property
    def EMA_FAST(self) -> int:
        return self.indicators.ema_fast

    @property
    def EMA_SLOW(self) -> int:
        return self.indicators.ema_slow

    @property
    def EMA_TREND(self) -> int:
        return self.indicators.ema_trend

    @property
    def RSI_PERIOD(self) -> int:
        return self.indicators.rsi_period

    @property
    def RSI_BUY(self) -> float:
        return self.indicators.rsi_buy

    @property
    def RSI_SELL(self) -> float:
        return self.indicators.rsi_sell

    @property
    def RSI_OVERBOUGHT(self) -> float:
        return self.indicators.rsi_overbought

    @property
    def RSI_OVERSOLD(self) -> float:
        return self.indicators.rsi_oversold

    @property
    def MACD_FAST(self) -> int:
        return self.indicators.macd_fast

    @property
    def MACD_SLOW(self) -> int:
        return self.indicators.macd_slow

    @property
    def MACD_SIGNAL(self) -> int:
        return self.indicators.macd_signal

    @property
    def ATR_PERIOD(self) -> int:
        return self.indicators.atr_period

    @property
    def ATR_STOP_MULTIPLIER(self) -> float:
        return self.indicators.atr_stop_multiplier

    @property
    def ATR_TAKE_MULTIPLIER(self) -> float:
        return self.indicators.atr_take_multiplier

    @property
    def TRAILING_STOP_MULTIPLIER(self) -> float:
        return self.indicators.trailing_stop_multiplier

    @property
    def ADX_PERIOD(self) -> int:
        return self.indicators.adx_period

    @property
    def ADX_MIN(self) -> float:
        return self.indicators.adx_min

    @property
    def BB_PERIOD(self) -> int:
        return self.indicators.bb_period

    @property
    def BB_STD(self) -> float:
        return self.indicators.bb_std

    @property
    def VOLUME_MA(self) -> int:
        return self.indicators.volume_ma

    @property
    def VOLUME_FILTER(self) -> float:
        return self.indicators.volume_filter

    @property
    def SUPERTREND_PERIOD(self) -> int:
        return self.indicators.supertrend_period

    @property
    def SUPERTREND_MULTIPLIER(self) -> float:
        return self.indicators.supertrend_multiplier

    @property
    def RISK_PERCENT(self) -> float:
        return self.account.max_risk_percent

    @property
    def MIN_POSITION_SIZE(self) -> float:
        return self.account.min_position_size

    @property
    def MAX_POSITION_SIZE(self) -> float:
        return self.account.max_position_size

    @property
    def RISK_REWARD_RATIO(self) -> float:
        return self.backtest.risk_reward_ratio

    @property
    def ENABLE_TRAILING_STOP(self) -> bool:
        return self.backtest.enable_trailing_stop

    @property
    def ENABLE_BREAK_EVEN(self) -> bool:
        return self.backtest.enable_break_even

    @property
    def ENABLE_PARTIAL_CLOSE(self) -> bool:
        return self.backtest.enable_partial_close

    @property
    def ML_ENABLED(self) -> bool:
        return self.ml.ml_enabled

    @property
    def MODEL_PATH(self) -> str:
        return self.ml.model_path

    @property
    def STRATEGY_MIN_CONFIDENCE(self) -> float:
        return self.strategy.min_confidence

    @property
    def STRATEGY_AI_WEIGHT(self) -> float:
        return self.strategy.ai_weight

    @property
    def STRATEGY_ML_WEIGHT(self) -> float:
        return self.strategy.ml_weight

    @property
    def STRATEGY_CONFLICT_PENALTY(self) -> float:
        return self.strategy.conflict_penalty

    @property
    def ORDERFLOW_ENABLED(self) -> bool:
        return self.strategy.orderflow_enabled

    @property
    def ORDERFLOW_CONFLICT_THRESHOLD(self) -> float:
        return self.strategy.orderflow_conflict_threshold

    @property
    def TELEGRAM_ENABLED(self) -> bool:
        return self.telegram.enabled

    @property
    def TELEGRAM_TOKEN(self) -> str:
        return self.telegram.token

    @property
    def TELEGRAM_CHAT_ID(self) -> str:
        return self.telegram.chat_id

    @property
    def LOG_LEVEL(self) -> str:
        return self.logging.log_level

    @property
    def SAVE_TRADES(self) -> bool:
        return self.logging.save_trades

    @property
    def TRADES_FILE(self) -> str:
        return self.logging.trades_file


# Global settings instance
settings = Settings()


# Backward compatibility exports
PROJECT_NAME = settings.project_name
VERSION = settings.version

EXCHANGE = settings.exchange.exchange
SYMBOL = settings.exchange.symbol
TIMEFRAME = settings.exchange.timeframe
LIMIT = settings.exchange.limit
CANDLE_LIMIT = LIMIT

INITIAL_BALANCE = settings.account.initial_balance
RISK_PER_TRADE = settings.account.risk_per_trade
MAX_OPEN_POSITIONS = settings.account.max_open_positions

COMMISSION = settings.commission.commission
SLIPPAGE = settings.commission.slippage

EMA_FAST = settings.indicators.ema_fast
EMA_SLOW = settings.indicators.ema_slow
EMA_TREND = settings.indicators.ema_trend

RSI_PERIOD = settings.indicators.rsi_period
RSI_BUY = settings.indicators.rsi_buy
RSI_SELL = settings.indicators.rsi_sell
RSI_OVERBOUGHT = settings.indicators.rsi_overbought
RSI_OVERSOLD = settings.indicators.rsi_oversold

ATR_PERIOD = settings.indicators.atr_period
ATR_STOP_MULTIPLIER = settings.indicators.atr_stop_multiplier
ATR_TAKE_MULTIPLIER = settings.indicators.atr_take_multiplier
TRAILING_STOP_MULTIPLIER = settings.indicators.trailing_stop_multiplier

# MACD / ADX / BB (single source of truth: IndicatorSettings)
MACD_FAST = settings.indicators.macd_fast
MACD_SLOW = settings.indicators.macd_slow
MACD_SIGNAL = settings.indicators.macd_signal

ADX_PERIOD = settings.indicators.adx_period
ADX_MIN = settings.indicators.adx_min

BB_PERIOD = settings.indicators.bb_period
BB_STD = settings.indicators.bb_std

VOLUME_MA = settings.indicators.volume_ma
VOLUME_FILTER = settings.indicators.volume_filter

SUPERTREND_PERIOD = settings.indicators.supertrend_period
SUPERTREND_MULTIPLIER = settings.indicators.supertrend_multiplier

ENABLE_TRAILING_STOP = settings.backtest.enable_trailing_stop
ENABLE_BREAK_EVEN = settings.backtest.enable_break_even
ENABLE_PARTIAL_CLOSE = settings.backtest.enable_partial_close

RISK_REWARD_RATIO = settings.backtest.risk_reward_ratio
MAX_RISK_PERCENT = settings.account.max_risk_percent
MIN_POSITION_SIZE = settings.account.min_position_size
MAX_POSITION_SIZE = settings.account.max_position_size

ML_ENABLED = settings.ml.ml_enabled
MODEL_PATH = settings.ml.model_path

STRATEGY_MIN_CONFIDENCE = settings.strategy.min_confidence
STRATEGY_AI_WEIGHT = settings.strategy.ai_weight
STRATEGY_ML_WEIGHT = settings.strategy.ml_weight
STRATEGY_CONFLICT_PENALTY = settings.strategy.conflict_penalty

ORDERFLOW_ENABLED = settings.strategy.orderflow_enabled
ORDERFLOW_CONFLICT_THRESHOLD = settings.strategy.orderflow_conflict_threshold

TELEGRAM_ENABLED = settings.telegram.enabled
TELEGRAM_TOKEN = settings.telegram.token
TELEGRAM_CHAT_ID = settings.telegram.chat_id

LOG_LEVEL = settings.logging.log_level
SAVE_TRADES = settings.logging.save_trades
TRADES_FILE = settings.logging.trades_file

RISK_PERCENT = settings.account.max_risk_percent


__all__ = [
    "Settings",
    "settings",
    "ExchangeSettings",
    "AccountSettings",
    "CommissionSettings",
    "IndicatorSettings",
    "BacktestSettings",
    "MLSettings",
    "TelegramSettings",
    "LoggingSettings",
    "RiskSettings",
    # Backward compat
    "PROJECT_NAME",
    "VERSION",
    "EXCHANGE",
    "SYMBOL",
    "TIMEFRAME",
    "LIMIT",
    "CANDLE_LIMIT",
    "INITIAL_BALANCE",
    "RISK_PER_TRADE",
    "MAX_OPEN_POSITIONS",
    "COMMISSION",
    "SLIPPAGE",
    "EMA_FAST",
    "EMA_SLOW",
    "EMA_TREND",
    "RSI_PERIOD",
    "ATR_PERIOD",
    "VOLUME_MA",
    "ENABLE_TRAILING_STOP",
    "ENABLE_BREAK_EVEN",
    "ENABLE_PARTIAL_CLOSE",
    "RISK_REWARD_RATIO",
    "MAX_RISK_PERCENT",
    "MIN_POSITION_SIZE",
    "MAX_POSITION_SIZE",
    "ML_ENABLED",
    "MODEL_PATH",
    "TELEGRAM_ENABLED",
    "TELEGRAM_TOKEN",
    "TELEGRAM_CHAT_ID",
    "LOG_LEVEL",
    "SAVE_TRADES",
    "TRADES_FILE",
    "RISK_PERCENT",
]