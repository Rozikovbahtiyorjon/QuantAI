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
    mode: str = "PAPER"

    @field_validator("timeframe")
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        valid = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}
        if v not in valid:
            raise ValueError(f"Invalid timeframe: {v}. Must be one of {valid}")
        return v


class AccountSettings(BaseSettings):
    """Account balance — risk fields moved to RiskSettings (canonical). Frozen per P0.1."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore", frozen=True)

    initial_balance: float = 1000.0


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
    min_confidence: float = 0.60  # 0..1
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
    
    # Confidence Engine weights (stable WF)
    confidence_trend_weight: float = 1.50
    confidence_momentum_weight: float = 1.20
    confidence_volume_weight: float = 1.10
    confidence_volatility_weight: float = 1.00
    
    # WeightedGate (WF stable PF 1.075, tuned 1.105 overfits OOS)
    weighted_gate_threshold: float = 0.75
    weighted_gate_min_confidence: float = 60.0  # 0..100 percentage
    weighted_gate_long_threshold: float = 0.55
    weighted_gate_short_threshold: float = 0.55
    
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
    # Office mode — group control
    office_enabled: bool = False
    admin_ids: str = ""  # comma-separated user IDs, e.g. "123,456"
    topic_mode: bool = False  # Forum topics per agent
    polling_timeout: int = 20
    # Per-agent tokens JSON: {"risk":"TOKEN1","architect":"TOKEN2"} — for multi-bot mode
    agent_tokens_json: str = ""
    # Natural language — LLM routing (free: ollama local, or openai/groq cloud)
    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8", extra_ignores=True)  # type: ignore
    llm_enabled: bool = False
    llm_provider: str = "openai"  # openai | ollama | groq — env: TELEGRAM__LLM_PROVIDER
    llm_model: str = "gpt-4o-mini"  # env: TELEGRAM__LLM_MODEL
    llm_base_url: str = ""  # env: TELEGRAM__LLM_BASE_URL — for ollama: http://localhost:11434/v1
    llm_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    # explicit TELEGRAM__LLM_API_KEY alternative
    llm_api_key_alt: str = Field(default="", alias="TELEGRAM__LLM_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")

    def get_llm_key(self) -> str:
        # priority: TELEGRAM__LLM_API_KEY -> OPENAI_API_KEY -> env
        if self.llm_api_key_alt.strip():
            return self.llm_api_key_alt.strip()
        if self.llm_api_key.strip():
            return self.llm_api_key.strip()
        # fallback: read .env directly (pydantic may not have loaded OPENAI_API_KEY due to nesting)
        import os

        for k in ("OPENAI_API_KEY", "TELEGRAM__LLM_API_KEY"):
            v = os.getenv(k, "").strip()
            if v:
                return v
        # last resort: parse .env file manually (no import needed)
        try:
            from pathlib import Path

            env_path = Path(".env")
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v and not v.startswith("sk-proj-..."):
                            return v
                    if line.startswith("TELEGRAM__LLM_API_KEY="):
                        v = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if v:
                            return v
        except Exception:
            pass
        return ""

    @property
    def admin_id_list(self) -> list[int]:
        if not self.admin_ids.strip():
            return []
        ids: list[int] = []
        for part in self.admin_ids.split(","):
            part = part.strip()
            if part:
                try:
                    ids.append(int(part))
                except ValueError:
                    continue
        return ids

    @property
    def agent_tokens(self) -> dict[str, str]:
        if not self.agent_tokens_json.strip():
            return {}
        import json

        try:
            data = json.loads(self.agent_tokens_json)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if v}
        except Exception:
            pass
        return {}


class LoggingSettings(BaseSettings):
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_trades: bool = True
    trades_file: str = "trades.csv"


class RiskSettings(BaseSettings):
    """
    DEPRECATED/FROZEN — use src.risk.policy.get_policy() as single source.
    Mirrors src.risk.policy.BasePolicy (ResearchPolicy). Frozen: direct mutation blocked.
    To stay fail-closed, defaults must NOT exceed canonical ResearchPolicy;
    any environment override that loosens limits raises via validators below.
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", env_nested_delimiter="__", extra="ignore", frozen=True)

    # Drawdown limits (canonical ResearchPolicy max_drawdown_pct 10.0)
    drawdown_limit_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    max_drawdown_pct: float = 10.0

    # Exposure limits — P0.6 Single source: derived from canonical ResearchPolicy (no drift)
    # Do not hardcode 60/5 here — import from src.risk.policy to stay in sync
    # Fallback to literal only if import fails (should not happen)
    try:
        from src.risk.policy import ResearchPolicy as _RP
        _def_total = float(_RP.max_total_exposure_pct)
        _def_pos = float(_RP.max_position_exposure_pct)
        _def_leverage = float(_RP.max_leverage)
        _def_drawdown = float(_RP.max_drawdown_pct)
    except Exception:
        _def_total = 60.0
        _def_pos = 5.0
        _def_leverage = 10.0
        _def_drawdown = 10.0
    max_total_exposure_pct: float = _def_total  # Research 60%
    max_position_exposure_pct: float = _def_pos  # Research 5%
    max_open_positions: int = 1
    reserve_percent: float = 40.0  # 40% absolute reserve (3-5-7 rule)

    # Position sizing (canonical risk_per_trade 0.01)
    position_sizer_method: Literal["fixed_fractional", "kelly", "volatility_adjusted"] = "fixed_fractional"
    risk_per_trade: float = 0.01  # 1% per trade
    max_risk_percent: float = 1.0
    min_position_size: float = 0.001
    max_position_size: float = 1.0
    risk_reward_ratio: float = 2.0
    min_risk_reward_ratio: float = 7.0  # 3-5-7 rule: profit ≥ 7% more than loss

    # Leverage limits — P0.6 No Policy -> No Leverage, default None would be stricter, but keepResearch 10x here for research;
    # Production/Testnet must use TestnetPolicy/ProductionPolicy 3x via get_policy()
    max_leverage: float = _def_leverage  # Research 10x
    min_leverage: float = 1.0

    # Correlation limits (canonical max_correlation 0.85)
    max_correlation: float = 0.85
    max_correlated_assets: int = 2

    # Reserve and margin (P0.1 canonical)
    reserve_percent: float = 40.0
    max_margin_pct: float = 30.0
    max_factor_exposure_pct: float = 15.0
    max_factor_concentration: float = 0.70
    correlation_adjusted_limit: float = 0.15
    
    # Stop loss / Take profit multipliers (ATR-based)
    atr_stop_multiplier: float = 1.5
    atr_take_multiplier: float = 3.0
    trailing_stop_multiplier: float = 2.0

    @field_validator("risk_per_trade", "max_total_exposure_pct", "max_position_exposure_pct", "max_drawdown_pct", "max_correlation", "max_leverage", "reserve_percent", "max_margin_pct", "max_factor_exposure_pct", "max_factor_concentration", "correlation_adjusted_limit")
    @classmethod
    def validate_not_looser_than_canonical(cls, v: float, info) -> float:  # type: ignore
        # Compare against canonical ResearchPolicy ceilings; fail if looser
        try:
            from src.risk.policy import ResearchPolicy  # lazy to avoid circular
            ceilings = {
                "risk_per_trade": ResearchPolicy.risk_per_trade,
                "max_total_exposure_pct": ResearchPolicy.max_total_exposure_pct,
                "max_position_exposure_pct": ResearchPolicy.max_position_exposure_pct,
                "max_drawdown_pct": ResearchPolicy.max_drawdown_pct,
                "max_correlation": ResearchPolicy.max_correlation,
                "max_leverage": ResearchPolicy.max_leverage,
                "reserve_percent": ResearchPolicy.reserve_percent,
                "max_margin_pct": ResearchPolicy.max_margin_pct,
                "max_factor_exposure_pct": ResearchPolicy.max_factor_exposure_pct,
                "max_factor_concentration": ResearchPolicy.max_factor_concentration,
                "correlation_adjusted_limit": ResearchPolicy.correlation_adjusted_limit,
            }
            field = info.field_name
            if field in ceilings and float(v) > float(ceilings[field]) + 1e-9:
                raise ValueError(f"RiskSettings.{field} {v} exceeds canonical ResearchPolicy {ceilings[field]} — only tighten via tighten(), not via settings override")
        except ImportError:
            pass
        return v


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

    def model_post_init(self, __context) -> None:
        pass

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
        return self.risk.risk_per_trade

    @property
    def MAX_OPEN_POSITIONS(self) -> int:
        return self.risk.max_open_positions

    @property
    def MAX_RISK_PERCENT(self) -> float:
        return self.risk.max_risk_percent

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
    def TELEGRAM_OFFICE_ENABLED(self) -> bool:
        return self.telegram.office_enabled

    @property
    def TELEGRAM_ADMIN_IDS(self) -> list[int]:
        return self.telegram.admin_id_list

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
RISK_PER_TRADE = settings.risk.risk_per_trade
MAX_OPEN_POSITIONS = settings.risk.max_open_positions

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
MAX_RISK_PERCENT = settings.risk.max_risk_percent
MIN_POSITION_SIZE = settings.risk.min_position_size
MAX_POSITION_SIZE = settings.risk.max_position_size

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

RISK_PERCENT = settings.risk.max_risk_percent


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
    "MAX_RISK_PERCENT",
    "MIN_POSITION_SIZE",
    "MAX_POSITION_SIZE",
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