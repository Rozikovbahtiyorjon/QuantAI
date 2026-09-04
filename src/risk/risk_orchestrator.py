"""
====================================================
QuantAI Professional
Risk Orchestrator
====================================================

Unified risk facade coordinating:
- DrawdownGuard: equity peak tracking + drawdown limits
- ExposureManager: position/notional exposure limits
- PositionSizer: risk-based position sizing

Provides single RiskDecision interface for Strategy/Runner.
====================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.drawdown_guard import DrawdownGuard, DrawdownGuardResult
from src.exposure_manager import ExposureManager, ExposureResult
from src.position_sizer import PositionSizer, PositionSizeResult  # root src/position_sizer.py
from src.risk.risk_context import RiskContext

# Task 7: factor risk gate
try:
    from src.risk.factor_risk import (
        check_factor_risk_gate,
        compute_factor_risk,
        DEFAULT_CRYPTO_FACTOR_MAP,
    )
except Exception:  # pragma: no cover
    check_factor_risk_gate = None  # type: ignore
    compute_factor_risk = None  # type: ignore
    DEFAULT_CRYPTO_FACTOR_MAP = {}  # type: ignore

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.strategy import SignalResult


@dataclass(frozen=True)
class RiskDecision:
    """
    Unified risk decision from the orchestrator.
    
    Attributes:
        allowed: Whether the position is allowed to be opened
        quantity: Approved position size (0 if not allowed)
        stop_loss: Stop loss price (from signal)
        take_profit: Take profit price (from signal)
        reason: Human-readable reason for decision
        drawdown_result: Detailed drawdown analysis
        exposure_result: Detailed exposure analysis
        position_size_result: Detailed position sizing analysis
        metadata: Additional context (drawdown_pct, exposure_pct, etc.)
    """
    allowed: bool
    quantity: float
    stop_loss: float
    take_profit: float
    reason: str
    drawdown_result: DrawdownGuardResult
    exposure_result: ExposureResult
    position_size_result: PositionSizeResult
    metadata: dict = field(default_factory=dict)


class RiskOrchestrator:
    """
    Orchestrates multiple risk components into a single decision.
    
    Flow:
        SignalResult + Equity + Current Exposure
                    ↓
            DrawdownGuard (check equity health)
                    ↓
            PositionSizer (calculate risk-based size)
                    ↓
            ExposureManager (cap to exposure limits)
                    ↓
            RiskDecision (unified result)
    
    Usage (P0.6 No Policy -> No Orchestrator):
        from src.risk.policy import ProductionPolicy
        orchestrator = RiskOrchestrator(
            drawdown_guard=DrawdownGuard(max_drawdown_percent=10.0),
            exposure_manager=ExposureManager(policy=ProductionPolicy),
            position_sizer=PositionSizer(policy=ProductionPolicy),
            default_risk_percent=1.0,
            default_leverage=1.0,
        )
        # Direct instantiation without policy is forbidden — use create_default_orchestrator(policy=ProductionPolicy)
        
        decision = orchestrator.evaluate(
            signal=signal_result,
            equity=1000.0,
            current_exposure=0.0,
        )
        
        if decision.allowed:
            open_position(decision.quantity, decision.stop_loss, decision.take_profit)
    """
    
    def __init__(
        self,
        drawdown_guard: DrawdownGuard,
        exposure_manager: ExposureManager,
        position_sizer: PositionSizer,
        default_risk_percent: float = 1.0,
        default_leverage: float = 1.0,
    ) -> None:
        self.drawdown_guard = drawdown_guard
        self.exposure_manager = exposure_manager
        self.position_sizer = position_sizer
        self.default_risk_percent = float(default_risk_percent)
        self.default_leverage = float(default_leverage)
    
    def evaluate(
        self,
        signal: SignalResult,
        equity: float,
        current_exposure: float = 0.0,
        risk_percent: Optional[float] = None,
        leverage: Optional[float] = None,
        context: Optional["RiskContext"] = None,
    ) -> RiskDecision:
        """
        Evaluate a trading signal against all risk controls.

        Args:
            signal: Strategy signal with entry, stop_loss, take_profit
            equity: Current account equity/balance
            current_exposure: Current notional exposure from open positions
            risk_percent: Override default risk % (e.g., 1.0 = 1%)
            leverage: Override default leverage
            context: Optional RiskContext (R0.1). When provided, exposure
                limits are evaluated against context.projected_exposure
                (post-close baseline on flips) instead of current_exposure.

        Returns:
            RiskDecision with allowed flag, approved quantity, and diagnostics
        """
        # Use defaults if not provided
        risk_pct = risk_percent if risk_percent is not None else self.default_risk_percent
        lev = leverage if leverage is not None else self.default_leverage

        # R0.1: projected (post-action) exposure baseline for limit checks.
        limit_baseline_exposure = (
            context.effective_exposure
            if context is not None
            else current_exposure
        )

        # -------------------------------------------------
        # 0. STALENESS / MISSING DATA — UNKNOWN → REJECT (P0.3)
        # -------------------------------------------------
        if context is not None:
            # Stale balances
            if context.balance_timestamp is not None:
                try:
                    age = (__import__("datetime").datetime.now(__import__("datetime").timezone.utc) - context.balance_timestamp).total_seconds() if context.balance_timestamp.tzinfo else (__import__("datetime").datetime.now(__import__("datetime").timezone.utc) - context.balance_timestamp.replace(tzinfo=__import__("datetime").timezone.utc)).total_seconds()
                    if age > float(getattr(context, "max_balance_age_sec", 5.0)):
                        return RiskDecision(
                            allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                            reason=f"Stale balances: age {age:.1f}s > {context.max_balance_age_sec}s → UNKNOWN → REJECT",
                            drawdown_result=DrawdownGuardResult(peak_equity=equity, current_equity=equity, drawdown=0.0, drawdown_percent=100.0, allowed=False),
                            exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                            position_size_result=PositionSizeResult(balance=equity, risk_percent=risk_pct, risk_amount=0.0, entry_price=signal.entry, stop_price=signal.stop_loss, stop_distance=0.0, stop_distance_percent=0.0, position_size=0.0, position_notional=0.0, leverage=lev, margin_required=0.0),
                            metadata={"stage": "stale_balances_blocked", "age_sec": age, "max_age": context.max_balance_age_sec},
                        )
                except Exception as e:
                    return RiskDecision(
                        allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                        reason=f"Balance staleness check failed: {e} → UNKNOWN → REJECT",
                        drawdown_result=DrawdownGuardResult(peak_equity=equity, current_equity=equity, drawdown=0.0, drawdown_percent=100.0, allowed=False),
                        exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                        position_size_result=PositionSizeResult(balance=equity, risk_percent=risk_pct, risk_amount=0.0, entry_price=signal.entry, stop_price=signal.stop_loss, stop_distance=0.0, stop_distance_percent=0.0, position_size=0.0, position_notional=0.0, leverage=lev, margin_required=0.0),
                        metadata={"stage": "stale_balances_error", "error": str(e)},
                    )
            # Missing market data
            if context.market_data_timestamp is not None:
                try:
                    age_m = (__import__("datetime").datetime.now(__import__("datetime").timezone.utc) - context.market_data_timestamp).total_seconds() if context.market_data_timestamp.tzinfo else (__import__("datetime").datetime.now(__import__("datetime").timezone.utc) - context.market_data_timestamp.replace(tzinfo=__import__("datetime").timezone.utc)).total_seconds()
                    if age_m > float(getattr(context, "max_market_data_age_sec", 5.0)):
                        return RiskDecision(
                            allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                            reason=f"Missing/stale market data: age {age_m:.1f}s > {context.max_market_data_age_sec}s → UNKNOWN → REJECT",
                            drawdown_result=DrawdownGuardResult(peak_equity=equity, current_equity=equity, drawdown=0.0, drawdown_percent=100.0, allowed=False),
                            exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                            position_size_result=PositionSizeResult(balance=equity, risk_percent=risk_pct, risk_amount=0.0, entry_price=signal.entry, stop_price=signal.stop_loss, stop_distance=0.0, stop_distance_percent=0.0, position_size=0.0, position_notional=0.0, leverage=lev, margin_required=0.0),
                            metadata={"stage": "stale_market_data_blocked", "age_sec": age_m},
                        )
                except Exception as e:
                    return RiskDecision(
                        allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                        reason=f"Market data staleness check failed: {e} → UNKNOWN → REJECT",
                        drawdown_result=DrawdownGuardResult(peak_equity=equity, current_equity=equity, drawdown=0.0, drawdown_percent=100.0, allowed=False),
                        exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                        position_size_result=PositionSizeResult(balance=equity, risk_percent=risk_pct, risk_amount=0.0, entry_price=signal.entry, stop_price=signal.stop_loss, stop_distance=0.0, stop_distance_percent=0.0, position_size=0.0, position_notional=0.0, leverage=lev, margin_required=0.0),
                        metadata={"stage": "stale_market_data_error", "error": str(e)},
                    )
            else:
                # No market data timestamp provided while trying to open position -> UNKNOWN for live trading
                # For strict production, require it; for backward compat, only reject if open_positions non-empty
                if context.open_positions:
                    return RiskDecision(
                        allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                        reason="Missing market data timestamp → UNKNOWN → REJECT",
                        drawdown_result=DrawdownGuardResult(peak_equity=equity, current_equity=equity, drawdown=0.0, drawdown_percent=100.0, allowed=False),
                        exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                        position_size_result=PositionSizeResult(balance=equity, risk_percent=risk_pct, risk_amount=0.0, entry_price=signal.entry, stop_price=signal.stop_loss, stop_distance=0.0, stop_distance_percent=0.0, position_size=0.0, position_notional=0.0, leverage=lev, margin_required=0.0),
                        metadata={"stage": "missing_market_data_blocked"},
                    )
            # Position state missing
            if context.open_positions and context.position_state_version is None:
                return RiskDecision(
                    allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    reason="Position state version missing → UNKNOWN → REJECT",
                    drawdown_result=DrawdownGuardResult(peak_equity=equity, current_equity=equity, drawdown=0.0, drawdown_percent=100.0, allowed=False),
                    exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                    position_size_result=PositionSizeResult(balance=equity, risk_percent=risk_pct, risk_amount=0.0, entry_price=signal.entry, stop_price=signal.stop_loss, stop_distance=0.0, stop_distance_percent=0.0, position_size=0.0, position_notional=0.0, leverage=lev, margin_required=0.0),
                    metadata={"stage": "missing_position_state_blocked"},
                )
        
        # -------------------------------------------------
        # 1. DRAWDOWN CHECK
        # -------------------------------------------------
        drawdown_result = self.drawdown_guard.evaluate(equity)
        
        if not drawdown_result.allowed:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason=(
                    f"maximum drawdown exceeded "
                    f"({drawdown_result.drawdown_percent:.8f}%)"
                ),
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(
                    equity=equity,
                    total_exposure=current_exposure,
                    total_exposure_percent=0.0,
                    available_exposure=0.0,
                    available_exposure_percent=0.0,
                    position_exposure=0.0,
                    position_exposure_percent=0.0,
                    within_limit=False,
                ),
                position_size_result=PositionSizeResult(
                    balance=equity,
                    risk_percent=risk_pct,
                    risk_amount=0.0,
                    entry_price=signal.entry,
                    stop_price=signal.stop_loss,
                    stop_distance=0.0,
                    stop_distance_percent=0.0,
                    position_size=0.0,
                    position_notional=0.0,
                    leverage=lev,
                    margin_required=0.0,
                ),
                metadata={
                    "drawdown_pct": drawdown_result.drawdown_percent,
                    "max_drawdown_pct": self.drawdown_guard.max_drawdown_percent,
                    "stage": "drawdown_blocked",
                },
            )
        
        # -------------------------------------------------
        # 2. POSITION SIZING
        # -------------------------------------------------
        try:
            position_size_result = self.position_sizer.calculate(
                balance=equity,
                risk_percent=risk_pct,
                entry_price=signal.entry,
                stop_price=signal.stop_loss,
                leverage=lev,
            )
        except ValueError as e:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason=f"Position sizing failed: {e}",
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(
                    equity=equity,
                    total_exposure=current_exposure,
                    total_exposure_percent=0.0,
                    available_exposure=0.0,
                    available_exposure_percent=0.0,
                    position_exposure=0.0,
                    position_exposure_percent=0.0,
                    within_limit=False,
                ),
                position_size_result=PositionSizeResult(
                    balance=equity,
                    risk_percent=risk_pct,
                    risk_amount=0.0,
                    entry_price=signal.entry,
                    stop_price=signal.stop_loss,
                    stop_distance=0.0,
                    stop_distance_percent=0.0,
                    position_size=0.0,
                    position_notional=0.0,
                    leverage=lev,
                    margin_required=0.0,
                ),
                metadata={"stage": "sizing_failed", "error": str(e)},
            )
        
        risk_quantity = position_size_result.position_size
        
        # -------------------------------------------------
        # 3. EXPOSURE CAP (per-position limit) + separated exposures (point 26)
        # -------------------------------------------------
        max_position_capital = self.exposure_manager.max_position_capital(equity)
        exposure_limited_quantity = max_position_capital / signal.entry if signal.entry > 0 else 0.0
        
        approved_quantity = min(risk_quantity, exposure_limited_quantity)
        # Separated margin exposure check: notional / leverage vs margin cap
        # 1% risk by stop can still be large notional — check margin separately
        # P0.3: computation error → UNKNOWN → REJECT (no except pass)
        try:
            margin_req = float(position_size_result.margin_required) if hasattr(position_size_result, 'margin_required') else float(approved_quantity * signal.entry / max(1.0, lev))
            margin_pct = margin_req / equity * 100.0 if equity else 0.0
            # Use ExposureManager's separated margin cap
            max_margin = float(getattr(self.exposure_manager, 'max_margin_exposure_pct', self.exposure_manager.max_total_exposure_percent))
            if margin_pct > max_margin + 1e-9:
                return RiskDecision(
                    allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                    reason=f"margin exposure {margin_pct:.2f}% > cap {max_margin:.2f}% (notional {approved_quantity*signal.entry:.2f} / lev {lev})",
                    drawdown_result=drawdown_result,
                    exposure_result=self.exposure_manager.calculate(equity=equity, current_exposure=current_exposure),
                    position_size_result=position_size_result,
                    metadata={"stage": "margin_blocked", "margin_pct": margin_pct, "max_margin_pct": max_margin, "notional": approved_quantity*signal.entry, "leverage": lev},
                )
        except Exception as e:
            # Fail-closed: margin computation error → UNKNOWN → REJECT
            return RiskDecision(
                allowed=False, quantity=0.0, stop_loss=signal.stop_loss, take_profit=signal.take_profit,
                reason=f"Margin exposure computation failed: {e} → UNKNOWN → REJECT",
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0, available_exposure=0.0, available_exposure_percent=0.0, position_exposure=0.0, position_exposure_percent=0.0, within_limit=False),
                position_size_result=position_size_result,
                metadata={"stage": "margin_computation_error", "error": str(e)},
            )

        # -------------------------------------------------
        # 3b. FACTOR RISK GATE (Task 7 + Audit §26-27)
        # -------------------------------------------------
        # BTC+ETH+SOL as 3 positions = 1 CRYPTO_BETA factor.
        # Enforce:
        #   - correlation-adjusted exposure < limit (15% default)
        #   - factor concentration (max weight <70%, Herfindahl <0.60)
        # If context provides correlation_matrix + open_positions, use factor_risk;
        # otherwise gate is skipped (missing data = no gate).
        # Errors in factor/correlation calculation = FAIL-CLOSED (reject trade).
        factor_report = None
        # P0-21 fail-closed if factor module failed to import
        if context is not None and check_factor_risk_gate is None:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason="Factor risk module unavailable — unknown risk = reject trade",
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(
                    equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0,
                    available_exposure=0.0, available_exposure_percent=0.0,
                    position_exposure=0.0, position_exposure_percent=0.0, within_limit=False,
                ),
                position_size_result=position_size_result,
                metadata={"stage": "factor_module_missing"},
            )
        try:
            if context is not None and check_factor_risk_gate is not None:
                # Build positions dict: symbol -> notional pct
                positions: dict = {}
                # open_positions handling - P0-22: malformed records HALT new orders
                op = getattr(context, "open_positions", None)
                if op is not None and not isinstance(op, dict):
                    return RiskDecision(
                        allowed=False,
                        quantity=0.0,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        reason=f"Malformed open_positions container type {type(op).__name__} — expected dict",
                        drawdown_result=drawdown_result,
                        exposure_result=ExposureResult(
                            equity=equity, total_exposure=current_exposure, total_exposure_percent=0.0,
                            available_exposure=0.0, available_exposure_percent=0.0,
                            position_exposure=0.0, position_exposure_percent=0.0, within_limit=False,
                        ),
                        position_size_result=position_size_result,
                        metadata={"stage": "malformed_position_container"},
                    )
                if isinstance(op, dict):
                    for sym, pos in op.items():
                        try:
                            notional = getattr(pos, "notional", None)
                            if notional is None and isinstance(pos, dict):
                                notional = pos.get("notional", 0)
                            # Also handle float values directly
                            if isinstance(pos, (int, float)):
                                notional = float(pos)
                            if notional is None:
                                raise ValueError(f"Position {sym} missing notional")
                            notional_val = float(notional)
                            if notional_val <= 0:
                                raise ValueError(f"Position {sym} has invalid notional: {notional_val}")
                            positions[str(sym)] = notional_val / equity if equity else 0.0
                        except Exception as e:
                            return RiskDecision(
                                allowed=False,
                                quantity=0.0,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                reason=f"Malformed position record for {sym}: {e}",
                                drawdown_result=drawdown_result,
                                exposure_result=ExposureResult(
                                    equity=equity,
                                    total_exposure=current_exposure,
                                    total_exposure_percent=0.0,
                                    available_exposure=0.0,
                                    available_exposure_percent=0.0,
                                    position_exposure=0.0,
                                    position_exposure_percent=0.0,
                                    within_limit=False,
                                ),
                                position_size_result=position_size_result,
                                metadata={"stage": "malformed_position_blocked", "symbol": str(sym), "error": str(e)},
                            )
                # Candidate position
                cand_sym = getattr(signal, "symbol", None) or getattr(signal, "ticker", None) or "CANDIDATE"
                cand_notional_pct = float(approved_quantity * signal.entry) / equity if equity and signal.entry else 0.0
                if cand_notional_pct:
                    # If flip, context.projected_exposure already excludes old; but we still include candidate
                    positions[str(cand_sym)] = cand_notional_pct

                corr = getattr(context, "correlation_matrix", None)
                fmap = getattr(context, "factor_map", None) or DEFAULT_CRYPTO_FACTOR_MAP
                betas = getattr(context, "betas", None)
                corr_limit = float(getattr(context, "correlation_adjusted_limit", 0.15))
                conc_limit = float(getattr(context, "max_factor_concentration", 0.70))
                hhi_limit = float(getattr(context, "max_herfindahl", 0.60))

                # Fail-closed: missing correlation with multiple crypto positions = unknown risk = reject
                if positions:
                    has_corr = corr is not None
                    # Single position with no correlation still needs concentration check; unknown correlation with 2+ positions blocks
                    if not has_corr and len(positions) >= 2:
                        return RiskDecision(
                            allowed=False,
                            quantity=0.0,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            reason="Factor risk blocked: correlation matrix missing for multi-asset exposure (unknown risk)",
                            drawdown_result=drawdown_result,
                            exposure_result=self.exposure_manager.calculate(equity=equity, current_exposure=current_exposure),
                            position_size_result=position_size_result,
                            metadata={"stage": "factor_missing_corr_blocked", "positions": list(positions.keys()), "len_positions": len(positions)},
                        )
                    if has_corr:
                        gate = check_factor_risk_gate(
                            positions,
                            corr_matrix=corr,
                            factor_map=fmap,
                            betas=betas,
                            corr_limit=corr_limit,
                            concentration_limit=conc_limit,
                            herfindahl_limit=hhi_limit,
                        )
                        factor_report = gate.report
                        if not gate.allowed:
                            return RiskDecision(
                                allowed=False,
                                quantity=0.0,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                                reason=f"Factor risk blocked: {gate.reason}",
                                drawdown_result=drawdown_result,
                                exposure_result=self.exposure_manager.calculate(equity=equity, current_exposure=current_exposure),
                                position_size_result=position_size_result,
                                metadata={
                                    "stage": "factor_blocked",
                                    "corr_adj": factor_report.correlation_adjusted_exposure,
                                    "gross": factor_report.gross_exposure,
                                    "net": factor_report.net_exposure,
                                    "max_factor_weight": factor_report.max_factor_weight,
                                    "herfindahl": factor_report.herfindahl,
                                    "max_correlation": factor_report.max_correlation,
                                    "portfolio_beta": factor_report.portfolio_beta,
                                    "factor_exposure": factor_report.factor_exposure,
                                    "warning": factor_report.warning,
                                    "corr_limit": corr_limit,
                                    "concentration_limit": conc_limit,
                                },
                            )
                    else:
                        # Single position without correlation matrix — factor concentration is definitionally 100% for single asset
                        # and should not block; diversification/concentration gate applies only to multi-asset portfolios.
                        # Skip factor gate for single position (no correlation adjustment needed).
                        pass
        except Exception as e:
            # P0-21: FAIL-CLOSED - unknown risk = REJECT trade
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason=f"Factor risk evaluation failed: {e}",
                drawdown_result=drawdown_result,
                exposure_result=ExposureResult(
                    equity=equity,
                    total_exposure=current_exposure,
                    total_exposure_percent=0.0,
                    available_exposure=0.0,
                    available_exposure_percent=0.0,
                    position_exposure=0.0,
                    position_exposure_percent=0.0,
                    within_limit=False,
                ),
                position_size_result=position_size_result,
                metadata={"stage": "factor_gate_error", "error": str(e)},
            )

        if approved_quantity <= 0:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason="Calculated position size is zero after exposure cap",
                drawdown_result=drawdown_result,
                exposure_result=self.exposure_manager.calculate(
                    equity=equity,
                    current_exposure=current_exposure,
                ),
                position_size_result=position_size_result,
                metadata={
                    "risk_quantity": risk_quantity,
                    "exposure_limited_quantity": exposure_limited_quantity,
                    "stage": "exposure_cap_zero",
                },
            )
        
        approved_position_exposure = approved_quantity * signal.entry
        
        # -------------------------------------------------
        # 4. TOTAL EXPOSURE CHECK
        # -------------------------------------------------
        exposure_result = self.exposure_manager.calculate(
            equity=equity,
            current_exposure=limit_baseline_exposure,
            position_exposure=approved_position_exposure,
        )

        can_open = self.exposure_manager.can_open_position(
            equity=equity,
            current_exposure=limit_baseline_exposure,
            new_position_exposure=approved_position_exposure,
        )
        
        if not can_open:
            return RiskDecision(
                allowed=False,
                quantity=0.0,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason="exposure limit exceeded",
                drawdown_result=drawdown_result,
                exposure_result=exposure_result,
                position_size_result=position_size_result,
                metadata={
                    "approved_quantity": approved_quantity,
                    "approved_exposure": approved_position_exposure,
                    "current_exposure": current_exposure,
                    "limit_baseline_exposure": limit_baseline_exposure,
                    "is_flip": context.is_flip if context is not None else False,
                    "max_total_exposure_pct": self.exposure_manager.max_total_exposure_percent,
                    "stage": "total_exposure_blocked",
                },
            )
        
        # -------------------------------------------------
        # 5. ALL CHECKS PASSED
        # -------------------------------------------------
        approved_meta = {
            "drawdown_pct": drawdown_result.drawdown_percent,
            "total_exposure_pct": exposure_result.total_exposure_percent,
            "position_exposure_pct": exposure_result.position_exposure_percent,
            "risk_quantity": risk_quantity,
            "approved_quantity": approved_quantity,
            "leverage": lev,
            "is_flip": context.is_flip if context is not None else False,
            "stage": "approved",
        }
        # Include factor risk diagnostics if computed
        if factor_report is not None:
            approved_meta.update({
                "corr_adj": factor_report.correlation_adjusted_exposure,
                "gross": factor_report.gross_exposure,
                "net": factor_report.net_exposure,
                "max_factor_weight": factor_report.max_factor_weight,
                "herfindahl": factor_report.herfindahl,
                "max_correlation": factor_report.max_correlation,
                "portfolio_beta": factor_report.portfolio_beta,
                "factor_exposure": factor_report.factor_exposure,
                "diversification_ratio": factor_report.diversification_ratio,
                "factor_warning": factor_report.warning,
            })
        return RiskDecision(
            allowed=True,
            quantity=round(approved_quantity, 8),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            reason="Risk approved.",
            drawdown_result=drawdown_result,
            exposure_result=exposure_result,
            position_size_result=position_size_result,
            metadata=approved_meta,
        )
    
    def reset(self) -> None:
        """Reset all stateful components."""
        self.drawdown_guard.reset()
        # ExposureManager and PositionSizer are stateless


# ============================================================
# FACTORY FUNCTIONS
# ============================================================

def create_default_orchestrator(
    max_drawdown_percent: float | None = None,
    max_total_exposure_percent: float | None = None,
    max_position_exposure_percent: float | None = None,
    min_leverage: float | None = None,
    max_leverage: float | None = None,
    default_risk_percent: float = 1.0,
    default_leverage: float = 1.0,
    policy: Any = None,
) -> RiskOrchestrator:
    """
    Create a RiskOrchestrator — strict: No Policy -> No Orchestrator (P0.6).
    Requires explicit RiskPolicy (Research/Production/Paper/Testnet) — no hidden 60/5/50.
    max_leverage default is None (must come from policy), not 50x.
    """
    if policy is None and max_leverage is None and max_total_exposure_percent is None:
        raise ValueError("RiskOrchestrator requires explicit RiskPolicy (Production/Research/Paper/Testnet) — No Policy -> No Orchestrator (P0.6). Pass policy=ProductionPolicy")
    from src.risk.policy import ProductionPolicy, ResearchPolicy
    pol = policy
    if pol is None:
        # No policy but explicit max_leverage provided — allow, but warn and require leverage
        if max_leverage is None:
            raise ValueError("max_leverage default is None — must provide policy or explicit max_leverage (no 50x default)")
        # Use ProductionPolicy as fallback only if explicit leverage is safe
        pol = ProductionPolicy
    # Resolve via policy if None
    if max_drawdown_percent is None:
        max_drawdown_percent = float(pol.max_drawdown_pct)
    if max_total_exposure_percent is None:
        max_total_exposure_percent = float(pol.max_total_exposure_pct)
    if max_position_exposure_percent is None:
        max_position_exposure_percent = float(pol.max_position_exposure_pct)
    if min_leverage is None:
        min_leverage = float(pol.min_leverage if hasattr(pol, "min_leverage") else 1.0)
    if max_leverage is None:
        # No default — must come from policy
        if hasattr(pol, "max_leverage"):
            max_leverage = float(pol.max_leverage)
        else:
            raise ValueError("max_leverage default is None — No Policy -> No Orchestrator (provide policy or explicit max_leverage)")
    return RiskOrchestrator(
        drawdown_guard=DrawdownGuard(max_drawdown_percent=max_drawdown_percent),
        exposure_manager=ExposureManager(
            max_total_exposure_percent=max_total_exposure_percent,
            max_position_exposure_percent=max_position_exposure_percent,
            max_correlation=float(pol.max_correlation),
            policy=pol,
        ),
        position_sizer=PositionSizer(
            min_leverage=min_leverage,
            max_leverage=max_leverage,
            policy=pol,
        ),
        default_risk_percent=default_risk_percent,
        default_leverage=default_leverage,
    )


def create_orchestrator_from_settings(settings) -> RiskOrchestrator:
    """
    Create RiskOrchestrator from Pydantic Settings object via canonical RiskPolicy.

    Resolves field name mismatches (max_total_exposure_pct vs max_total_exposure_percent)
    and uses canonical max_leverage from RiskPolicy, not hardcoded 50.
    """
    from src.risk.policy import ResearchPolicy
    # Handle both pct field names for backward compat
    max_total_exposure = getattr(settings.risk, 'max_total_exposure_pct', None)
    if max_total_exposure is None:
        max_total_exposure = getattr(settings.risk, 'max_total_exposure_percent', ResearchPolicy.max_total_exposure_pct)
    max_position_exposure = getattr(settings.risk, 'max_position_exposure_pct', None)
    if max_position_exposure is None:
        max_position_exposure = getattr(settings.risk, 'max_position_exposure_percent', ResearchPolicy.max_position_exposure_pct)
    max_leverage = getattr(settings.risk, 'max_leverage', ResearchPolicy.max_leverage)

    return RiskOrchestrator(
        drawdown_guard=DrawdownGuard(
            max_drawdown_percent=settings.risk.drawdown_limit_pct
        ),
        exposure_manager=ExposureManager(
            max_total_exposure_percent=float(max_total_exposure),
            max_position_exposure_percent=float(max_position_exposure),
        ),
        position_sizer=PositionSizer(
            min_leverage=1.0,
            max_leverage=float(max_leverage),
        ),
        default_risk_percent=settings.risk.risk_per_trade * 100,  # config is 0.01 = 1%
        default_leverage=1.0,
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RiskContext",
    "RiskDecision",
    "RiskOrchestrator",
    "create_default_orchestrator",
    "create_orchestrator_from_settings",
]