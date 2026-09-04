"""
Risk Rules Specification — P0.2 Formalize Risk Rules

Before changing values, fix the meaning. Changing 60%→5% or RR 1.5→7
automatically is forbidden — spec first, then implementation.

Canonical spec v1.0 — immutable definitions. Any value change requires
spec_version bump and explicit review.
"""
from dataclasses import dataclass
from typing import Dict

SPEC_VERSION = "1.0"
SPEC_HASH = "risk_spec_v1_3-5-7"

@dataclass(frozen=True)
class RiskRuleDef:
    name: str
    definition: str
    formula: str
    unit: str
    invariant: str
    example: str

RISK_RULES: Dict[str, RiskRuleDef] = {
    "risk_per_trade": RiskRuleDef(
        name="Risk per Trade",
        definition="Max loss per single trade as % of equity, defined by stop distance",
        formula="risk_amount = equity * risk_per_trade; position_size = risk_amount / stop_distance",
        unit="percent (0.01 = 1%)",
        invariant="0 < risk_per_trade <= 0.03; Production 0.005",
        example="equity 10k, risk 1% => risk_amount 100, stop 2% => position 5k notional"
    ),
    "position_exposure": RiskRuleDef(
        name="Position Exposure",
        definition="Max notional of a single position as % of equity (gross)",
        formula="position_exposure = quantity * entry_price / equity",
        unit="percent",
        invariant="position_exposure <= max_position_exposure_pct; tightens Research 5.0 → Production 3.0",
        example="5.0% of 10k = 500 notional max per position"
    ),
    "gross_exposure": RiskRuleDef(
        name="Gross Exposure",
        definition="Sum of absolute notionals of all open positions / equity",
        formula="gross = sum(|notional_i|) / equity",
        unit="percent",
        invariant="gross <= max_total_exposure_pct; Research 60 → Production 20",
        example="2 positions 3% each => gross 6%"
    ),
    "margin_usage": RiskRuleDef(
        name="Margin Usage",
        definition="Margin required / equity, with leverage; isolated vs cross",
        formula="margin = notional / leverage; margin_usage = sum(margin) / equity",
        unit="percent",
        invariant="margin_usage <= max_margin_pct (30%); leverage capped separately",
        example="notional 5% at 3x => margin 1.66%"
    ),
    "capital_reserve": RiskRuleDef(
        name="Capital Reserve",
        definition="Equity that must stay unencumbered — 40% of TOTAL EQUITY (not free margin, not deployable capital, not gross notional). Formal: reserve = equity * reserve_percent; deployable = equity * (1 - reserve_percent); available_for_margin = deployable - margin_used. If deployable < margin_used → reject.",
        formula="reserve = equity * 0.40; deployable = equity - reserve; margin_available = deployable - margin_used; require margin_available >= 0",
        unit="percent of total equity",
        invariant="reserve_percent == 40% always (P0.1); never loosened; deployable 60% is max_total_exposure base; reserve is of equity, not free margin",
        example="equity 10k, reserve 40% => 4k locked forever, deployable 6k, margin_used 2k => margin_available 4k"
    ),
    "reward_risk": RiskRuleDef(
        name="Reward/Risk (RR)",
        definition="RR = reward_distance / stop_risk where reward = take_price - entry, stop_risk = |entry - stop|. NOT expected profit %, NOT win_rate% — purely distance ratio. Must be distinguished from expected_return. RR 2.0 means take is 2x stop distance.",
        formula="RR = (take - entry) / (entry - stop) for long; RR >= min_risk_reward_ratio",
        unit="ratio (reward/stop_risk)",
        invariant="RR >= min_risk_reward_ratio; Research 2.0 → Production 2.0; 7.0 is NOT universal — requires spec bump and must be accompanied by expectancy gate; comment must match math",
        example="entry 100, stop 98 (risk 2), take 104 (reward 4) => RR 2.0"
    ),
    "expected_net_return": RiskRuleDef(
        name="Expected Net Return",
        definition="Minimum expected net return per trade net of costs (commission+slippage+funding+spread) as % of equity or price. Distinct from RR: RR is distance ratio, expected net is probabilistic edge * avg win/loss net of costs. Trading criterion is expectancy, not RR alone.",
        formula="expected_net = (win_rate * avg_win_net - loss_rate * avg_loss_net) / entry ; require >= min_expected_net_return",
        unit="percent of equity or price (e.g., 0.001 = 0.1% net)",
        invariant="expected_net >= min_expected_net_return (e.g., 0.0005-0.001); RR alone insufficient",
        example="win 55% avg_win 0.8% net, loss 45% avg_loss -0.5% net => expected 0.55*0.8 -0.45*0.5 = 0.215% net"
    ),
    "expectancy": RiskRuleDef(
        name="Expectancy (Trading Criterion)",
        definition="PRIMARY trading criterion: expectancy = E[net] = (gross_profit - gross_loss)/trades / avg_risk or per trade net. Must be >0 net of costs. More important than RR alone — RR 7 with 10% win_rate can have negative expectancy.",
        formula="expectancy = (win_rate * avg_win - loss_rate * avg_loss) ; require > min_expectancy (e.g., >0)",
        unit="price or percent net per trade",
        invariant="expectancy > min_expectancy (0) is PRIMARY gate; min_risk_reward_ratio is secondary; both + min_expected_net_return form triple gate",
        example="RR 7 with win 10% avg_win 7R avg_loss 1R => expectancy 0.1*7 -0.9*1 = -0.2R => FAIL despite RR 7"
    ),
    "daily_loss": RiskRuleDef(
        name="Daily Loss",
        definition="Max realized loss per calendar day as % of day-start equity",
        formula="daily_loss = (start_equity - end_equity) / start_equity if end<start",
        unit="percent",
        invariant="daily_loss <= max_daily_loss_pct (5%); resets at UTC midnight",
        example="start 10k, end 9.6k => 4% daily loss"
    ),
    "max_drawdown": RiskRuleDef(
        name="Max Drawdown",
        definition="Peak-to-trough equity decline, trailing peak",
        formula="drawdown = (equity - peak) / peak",
        unit="percent",
        invariant="drawdown >= -max_drawdown_pct (10%); triggers halt",
        example="peak 12k, equity 10.8k => -10% drawdown"
    ),
    "leverage": RiskRuleDef(
        name="Leverage",
        definition="Notional / margin, capped per position and account",
        formula="leverage = notional / margin",
        unit="ratio",
        invariant="1.0 <= leverage <= max_leverage; Production 3.0 (dangerous 50x removed)",
        example="notional 6k, margin 2k => 3x"
    ),
}

def validate_against_spec(policy) -> None:
    """Ensure policy values match spec invariants; changing values requires spec_version bump."""
    for key, rule in RISK_RULES.items():
        # Map spec key to policy field (handle naming differences)
        field_map = {
            "risk_per_trade": "risk_per_trade",
            "position_exposure": "max_position_exposure_pct",
            "gross_exposure": "max_total_exposure_pct",
            "margin_usage": "max_margin_pct",
            "capital_reserve": "reserve_percent",
            "reward_risk": "min_risk_reward_ratio",
            "daily_loss": "max_daily_loss_pct",
            "max_drawdown": "max_drawdown_pct",
            "leverage": "max_leverage",
        }
        field = field_map.get(key)
        if not field or not hasattr(policy, field):
            continue
        val = float(getattr(policy, field))
        # Invariant checks (example: reserve never <40)
        if key == "capital_reserve" and val < 40.0 - 1e-9:
            raise ValueError(f"Spec violation {key}: {val} < 40% reserve (spec {SPEC_VERSION})")
        if key == "leverage" and val > 10.0 + 1e-9:
            # Research max is 10, production 3; >10 is dangerous
            raise ValueError(f"Spec violation {key}: {val} > 10 (spec {SPEC_VERSION})")

__all__ = ["SPEC_VERSION", "SPEC_HASH", "RISK_RULES", "RiskRuleDef", "validate_against_spec"]
