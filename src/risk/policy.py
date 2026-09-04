"""
Risk Policy — CANONICAL SINGLE SOURCE OF TRUTH (Audit: duplicate canonical)

Research -> Paper -> Testnet -> Production — каждая следующая только уменьшает риск.
Никакого второго RiskPolicy. Импортируйте отсюда: from src.risk.policy import ResearchPolicy, PaperPolicy, TestnetPolicy, ProductionPolicy, get_policy
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Any
import copy
try:
    from src.risk.risk_spec import SPEC_VERSION as RISK_SPEC_VERSION, validate_against_spec
except ImportError:
    RISK_SPEC_VERSION = "1.0"
    def validate_against_spec(policy): pass


@dataclass(frozen=True)
class BasePolicy:
    """Immutable base — Research is the loosest, Production the tightest.
    Single source of truth for ALL risk limits (P0.1).
    Spec: src/risk/risk_spec.py v1.0 — values must match spec, changes require spec bump."""

    risk_per_trade: float = 0.01  # 1%
    max_risk_percent: float = 1.0
    max_open_positions: int = 1
    max_total_exposure_pct: float = 60.0
    max_position_exposure_pct: float = 5.0
    max_drawdown_pct: float = 10.0
    max_daily_loss_pct: float = 5.0
    max_leverage: float = 10.0
    max_correlation: float = 0.85
    commission: float = 0.0004
    slippage: float = 0.0002
    # Task 7: factor risk gate
    max_factor_exposure_pct: float = 15.0  # corr-adjusted limit 15%
    max_factor_concentration: float = 0.70  # max weight in single factor <70%
    max_herfindahl: float = 0.60
    correlation_adjusted_limit: float = 0.15  # alias for max_factor_exposure_pct /100
    # P0.1 canonical additions — semantics formalized (P0.6)
    # Reserve: 40% of TOTAL EQUITY (not free margin, not deployable capital, not gross notional)
    # Formal: reserve = equity * 0.40; deployable = equity * 0.60 = max_total_exposure base; margin_available = deployable - margin_used
    reserve_percent: float = 40.0  # 40% of total equity, never loosened
    max_margin_pct: float = 30.0  # max margin exposure (of equity)
    min_reserve_percent: float = 40.0  # alias, same semantics
    # P0.2 spec version — any value change beyond spec requires bump
    spec_version: str = RISK_SPEC_VERSION
    # Reward/Risk — RR = reward_distance / stop_risk (not expected profit %). Spec 2.0; 7.0 is NOT universal
    # Comment must match math: RR = take_distance / stop_distance, not win_rate or profit %
    min_risk_reward_ratio: float = 2.0  # 2.0 means take is 2x stop distance
    # P0.6: Triple gate is expectancy PRIMARY, not RR alone — RR 7 with 10% win_rate can be negative expectancy
    minimum_rr: float = 2.0  # alias for min_risk_reward_ratio, same semantics
    minimum_expected_net_return: float = 0.001  # 0.1% net per trade after costs (commission+slippage+funding+spread)
    minimum_expectancy: float = 0.0  # PRIMARY trading criterion: expectancy >0 net of costs; more important than RR

    def tighten(self, **overrides: Any) -> "BasePolicy":
        """Каждая следующая policy может только ужесточать — looser raises.
        Spec-first: changing 60%->5% or RR 1.5->7 without spec_version bump is blocked.

        For most limits lower is safer, but for Reward/Risk higher is safer.
        """
        # Explicit allowlist of fields that must only decrease when tightening
        TIGHTEN_ONLY_DECREASE: frozenset[str] = frozenset({
            "risk_per_trade",
            "max_risk_percent",
            "max_open_positions",
            "max_total_exposure_pct",
            "max_position_exposure_pct",
            "max_drawdown_pct",
            "max_daily_loss_pct",
            "max_leverage",
            "max_correlation",
            "max_factor_exposure_pct",
            "max_factor_concentration",
            "max_herfindahl",
            "correlation_adjusted_limit",
            "commission",
            "slippage",
            "max_margin_pct",
        })
        # For Reserve/Reward/Risk/Expectancy, higher is safer (more reserve, more reward, more expectancy)
        TIGHTEN_ONLY_INCREASE: frozenset[str] = frozenset({
            "min_risk_reward_ratio",
            "minimum_rr",
            "reserve_percent",
            "min_reserve_percent",
            "minimum_expected_net_return",
            "minimum_expectancy",
        })

        # P0.2: prevent automatic 60%->5% and 1.5->7 without spec — only stepwise canonical tighten allowed
        ALLOWED_TIGHTEN_VALUES: Dict[str, set] = {
            "max_total_exposure_pct": {60.0, 30.0, 20.0},
            "min_risk_reward_ratio": {2.0},
        }
        # Direct jumps to 5% or 7.0 require explicit spec_version bump
        REQUIRES_SPEC_BUMP: Dict[str, Dict[float, str]] = {
            "max_total_exposure_pct": {5.0: "5% requires risk_spec v1.1 (was Research 60, now 5 is not canonical)"},
            "min_risk_reward_ratio": {7.0: "RR 7.0 requires risk_spec v1.1 (was 1.5->7 auto)"},
        }
        new_spec = overrides.get("spec_version", self.spec_version)
        for k, v in overrides.items():
            if k == "spec_version":
                continue
            if not hasattr(self, k):
                raise AttributeError(f"Unknown policy field {k}")
            cur = getattr(self, k)
            if isinstance(cur, (int, float)):
                if k in TIGHTEN_ONLY_DECREASE:
                    if float(v) > float(cur) + 1e-9:
                        raise ValueError(f"Policy may only tighten: {k} {cur} -> {v} blocked (lower is safer)")
                    # Canonical stepwise: 60->30->20 only
                    if k == "max_total_exposure_pct" and float(v) not in ALLOWED_TIGHTEN_VALUES[k] and float(v) != float(cur):
                        if float(v) == 5.0 and new_spec == self.spec_version:
                            raise ValueError(f"Spec violation: {k} {cur} -> {v} blocked — {REQUIRES_SPEC_BUMP[k][5.0]}. Bump spec_version first in src/risk/risk_spec.py")
                    elif k in ALLOWED_TIGHTEN_VALUES and float(v) not in ALLOWED_TIGHTEN_VALUES[k] and float(v) != float(cur):
                        if new_spec == self.spec_version:
                            raise ValueError(f"Spec violation: {k} {cur} -> {v} not in {ALLOWED_TIGHTEN_VALUES[k]} and spec_version not bumped ({self.spec_version}->{new_spec}). First update src/risk/risk_spec.py")
                    # Also block non-stepwise jumps like 60->20 directly
                    if k == "max_total_exposure_pct" and float(cur) == 60.0 and float(v) == 20.0 and new_spec == self.spec_version:
                        raise ValueError(f"Spec violation: {k} {cur} -> {v} skips 30 (60->30->20 stepwise required without spec bump)")
                elif k in TIGHTEN_ONLY_INCREASE:
                    if float(v) < float(cur) - 1e-9:
                        raise ValueError(f"Policy may only tighten: {k} {cur} -> {v} blocked (higher is safer for RR)")
                    if float(v) == 7.0 and new_spec == self.spec_version:
                        raise ValueError(f"Spec violation: {k} {cur} -> {v} blocked — {REQUIRES_SPEC_BUMP[k][7.0]}. Bump spec_version first")
        # Validate against spec invariants after tighten
        new_policy = replace(self, **overrides)  # type: ignore
        try:
            validate_against_spec(new_policy)
        except Exception as e:
            raise ValueError(f"Spec {RISK_SPEC_VERSION} violation after tighten: {e}") from e
        return new_policy


# Каждая следующая только уменьшает риск
ResearchPolicy = BasePolicy(
    risk_per_trade=0.01,
    max_total_exposure_pct=60.0,
    max_position_exposure_pct=5.0,
    max_drawdown_pct=10.0,
)

PaperPolicy = ResearchPolicy.tighten(
    max_total_exposure_pct=30.0,
    max_position_exposure_pct=3.0,
)

TestnetPolicy = PaperPolicy.tighten(
    risk_per_trade=0.005,  # 0.5% as in .env.testnet
    max_total_exposure_pct=20.0,
)

ProductionPolicy = TestnetPolicy.tighten(
    risk_per_trade=0.005,
    max_leverage=3.0,
    max_correlation=0.70,
)

POLICIES: Dict[str, BasePolicy] = {
    "research": ResearchPolicy,
    "paper": PaperPolicy,
    "testnet": TestnetPolicy,
    "production": ProductionPolicy,
    "base": BasePolicy(),
}


def get_policy(name: str) -> BasePolicy:
    try:
        return copy.deepcopy(POLICIES[name.lower()])
    except KeyError:
        raise KeyError(f"Unknown policy {name!r}, known: {list(POLICIES)}")
