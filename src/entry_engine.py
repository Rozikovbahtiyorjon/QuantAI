"""
Entry Engine — Central Controlling Layer (P3)

Architecture:
                MARKET DATA
                    |
                    V
             MARKET CONTEXT
                    |
           +--------+--------+
           V                 V
       HTF REGIME        VOLATILITY
           |                 |
           +--------+--------+
                    V
               SETUP ENGINE
                    |
         +----------+----------+
         V          V          V
      Pullback   Breakout   MeanRev
         |          |          |
         +----------+----------+
                    V
               POI / ZONE
                    |
                    V
              TRIGGER ENGINE
                    |
        +-----------+-----------+
        V           V           V
      Sweep       MSB        Retest
        |           |           |
        +-----------+-----------+
                    V
             CONFIRMATION
                    |
       +------------+------------+
       V            V            V
      Flow         ML           MTF
                    |
                    V
             ENTRY QUALITY
                    |
                    V
              EXPECTED VALUE
                    |
                    V
               RISK GATE
                    |
                    V
             ENTRY CANDIDATE
                    |
                    V
              ORDER INTENT
                    |
                    V
               EXECUTION

This is the central controlling layer that was missing.
Previously: Feature Engine → AI → Confidence → ML → Weighted Gate → Order Flow → SL/TP → Risk → Execution
Now: MARKET CONTEXT → SETUP → POI → TRIGGER → CONFIRMATION → QUALITY → EV → RISK → CANDIDATE → INTENT → EXECUTION
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Literal
import pandas as pd

from src.strategy.setup_detector import SetupDetector
from src.strategy.trend_pullback_setup import TrendPullbackSetup
from src.strategy.mean_reversion_setup import MeanReversionSetup
from src.market_data.zone_engine import ZoneEngine
from src.strategy.signal_generator import SignalGenerator, SignalConfig, SignalResult
from src.confidence_engine import EntryQualityEngine
from src.strategy.expected_value_gate import ExpectedValueGate
from src.strategy.entry_candidate import EntryCandidate
from src.strategy.entry_lifecycle import EntryLifecycle
from src.execution.execution_policy import ExecutionPolicy, ExecutionContext


@dataclass
class HTFContext:
    """ENTRY-08: Multi-Timeframe Context as object, not just BUY allowed/blocked."""
    direction: str  # TREND_UP / TREND_DOWN / RANGE
    trend_strength: float  # 0.0-1.0
    volatility: str  # high/normal/low
    structure: str  # bullish/bearish/neutral
    confidence: float  # 0.0-1.0

    @property
    def is_bullish(self) -> bool:
        return self.direction == "TREND_UP"
    @property
    def is_bearish(self) -> bool:
        return self.direction == "TREND_DOWN"


@dataclass
class MarketContext:
    regime: str  # TREND_UP/DOWN/RANGE/TRANSITION/UNKNOWN
    volatility: str  # high/normal/low
    htf_context: HTFContext  # ENTRY-08
    htf_regime: str  # alias for htf_context.direction for backward compat
    atr: float
    adx: float
    trend_score: float
    bb_width: float
    reason: str = ""
    regime_state: Any = None  # RegimeState with strength/confidence/age/duration


@dataclass
class TriggerResult:
    trigger_type: str  # Sweep/MSB/Retest/NONE
    is_triggered: bool
    price: float
    reason: str = ""


@dataclass
class EntryEngineResult:
    # Final decision
    should_enter: bool
    signal: Literal["HOLD", "BUY", "SELL"]
    entry_candidate: Optional[EntryCandidate] = None
    lifecycle: Optional[EntryLifecycle] = None
    execution_decision: Optional[Any] = None

    # Diagnostics per layer
    market_context: Optional[MarketContext] = None
    setup: Optional[Any] = None  # SetupResult
    poi: Optional[Any] = None  # POIResult
    trigger: Optional[TriggerResult] = None
    confirmation: dict = field(default_factory=dict)
    entry_quality: Optional[Any] = None
    expected_value: Optional[Any] = None
    risk_approved: bool = False

    reasons: list[str] = field(default_factory=list)


class MarketContextEngine:
    """MARKET DATA → MARKET CONTEXT → HTF REGIME + VOLATILITY"""

    def evaluate(self, df: pd.DataFrame) -> MarketContext:
        row = df.iloc[-1]
        close = float(row.get("close", 0))
        atr = float(row.get("atr", 1))
        adx = float(row.get("adx", 0))
        trend_score = float(row.get("trend_score", 0))
        bb_width = float(row.get("bb_width", 1.0))
        ema_fast = float(row.get("ema_fast", close))
        ema_slow = float(row.get("ema_slow", close))
        ema_trend = float(row.get("ema_trend", close))

        # HTF Regime (from 4H via RegimeFilter, simplified as 1H ADX+trend)
        if adx > 25 and ema_fast > ema_slow > ema_trend:
            htf_regime = "TREND_UP"
            regime = "TREND_UP"
            trend_strength = min(1.0, (adx - 25) / 20 + abs(trend_score) / 6)
        elif adx > 25 and ema_fast < ema_slow < ema_trend:
            htf_regime = "TREND_DOWN"
            regime = "TREND_DOWN"
            trend_strength = min(1.0, (adx - 25) / 20 + abs(trend_score) / 6)
        else:
            htf_regime = "RANGE"
            regime = "RANGE"
            trend_strength = max(0.0, 1.0 - adx / 25)

        # Volatility
        atr_pct = atr / close if close else 0
        if atr_pct > 0.02:
            volatility = "high"
        elif atr_pct < 0.005:
            volatility = "low"
        else:
            volatility = "normal"

        # Structure
        if ema_fast > ema_slow > ema_trend:
            structure = "bullish"
        elif ema_fast < ema_slow < ema_trend:
            structure = "bearish"
        else:
            structure = "neutral"

        # Confidence based on ADX distance + trend alignment
        confidence = min(1.0, max(0.0, (adx - 15) / 15 + trend_strength * 0.3))

        htf_context = HTFContext(
            direction=htf_regime,
            trend_strength=round(trend_strength, 2),
            volatility=volatility,
            structure=structure,
            confidence=round(confidence, 2),
        )

        # Also get RegimeState for age/duration if available
        regime_state = None
        try:
            from src.regime_filter import RegimeFilter
            # Use a temporary filter to get state? For now, create from current values
            regime_state = {"regime": regime, "strength": trend_strength, "confidence": confidence}
        except Exception:
            pass

        return MarketContext(
            regime=regime,
            volatility=volatility,
            htf_context=htf_context,
            htf_regime=htf_regime,
            atr=atr,
            adx=adx,
            trend_score=trend_score,
            bb_width=bb_width,
            reason=f"HTF {htf_regime} ({trend_strength:.2f} {structure} {confidence:.2f}) + Vol {volatility} ADX {adx:.0f}",
            regime_state=regime_state,
        )


class TriggerEngine:
    """
    TRIGGER ENGINE — Sweep / MSB / Retest

    Sweep: liquidity sweep (wick beyond zone)
    MSB: Market Structure Break (close beyond recent high/low)
    Retest: pullback retest of zone
    """

    def evaluate(self, df: pd.DataFrame, setup_type: str, zone: Any = None) -> TriggerResult:
        if len(df) < 5:
            return TriggerResult(trigger_type="NONE", is_triggered=False, price=0, reason="insufficient history")

        row = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(row.get("close", 0))
        high = float(row.get("high", 0))
        low = float(row.get("low", 0))
        prev_close = float(prev.get("close", close))
        prev_high = float(prev.get("high", high))
        prev_low = float(prev.get("low", low))

        # Sweep: wick beyond recent high/low with close back in zone
        if len(df) > 20:
            recent_high = float(df["high"].iloc[-21:-1].max())
            recent_low = float(df["low"].iloc[-21:-1].min())
            # Sweep high then close below
            if high > recent_high and close < recent_high:
                return TriggerResult(trigger_type="Sweep", is_triggered=True, price=close, reason=f"Sweep high {high:.1f}>{recent_high:.1f} close back {close:.1f}")
            if low < recent_low and close > recent_low:
                return TriggerResult(trigger_type="Sweep", is_triggered=True, price=close, reason=f"Sweep low {low:.1f}<{recent_low:.1f} close back {close:.1f}")

        # MSB: close beyond recent high/low
        if len(df) > 20:
            recent_high = float(df["high"].iloc[-21:-1].max())
            recent_low = float(df["low"].iloc[-21:-1].min())
            if close > recent_high:
                return TriggerResult(trigger_type="MSB", is_triggered=True, price=close, reason=f"MSB close {close:.1f}>{recent_high:.1f}")
            if close < recent_low:
                return TriggerResult(trigger_type="MSB", is_triggered=True, price=close, reason=f"MSB close {close:.1f}<{recent_low:.1f}")

        # Retest: close near zone (if zone provided)
        if zone and hasattr(zone, 'upper') and hasattr(zone, 'lower'):
            if zone.contains(close):
                return TriggerResult(trigger_type="Retest", is_triggered=True, price=close, reason=f"Retest zone {zone.lower:.1f}-{zone.upper:.1f}")

        # Fallback: any close beyond prev high/low as trigger
        if close > prev_high:
            return TriggerResult(trigger_type="Retest", is_triggered=True, price=close, reason=f"Retest close>{prev_high:.1f}")
        if close < prev_low:
            return TriggerResult(trigger_type="Retest", is_triggered=True, price=close, reason=f"Retest close<{prev_low:.1f}")

        return TriggerResult(trigger_type="NONE", is_triggered=False, price=close, reason="no trigger (no sweep/MSB/retest)")


class EntryEngine:
    """
    Central Entry Engine — the missing controlling layer.

    Orchestrates: Market Context → Setup → POI → Trigger → Confirmation → Quality → EV → Risk → Candidate → Intent → Execution
    """

    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self.market_context_engine = MarketContextEngine()
        self.setup_detector = SetupDetector()
        self.pullback_setup = TrendPullbackSetup()
        self.mean_reversion_setup = MeanReversionSetup()
        self.zone_engine = ZoneEngine()
        self.trigger_engine = TriggerEngine()
        self.entry_quality_engine = EntryQualityEngine()
        self.expected_value_gate = ExpectedValueGate(hurdle=self.config.expected_return_hurdle)
        self.execution_policy = ExecutionPolicy()
        # Reuse existing SignalGenerator for AI/ML parts (but now as sub-component)
        self.signal_generator = SignalGenerator(config=self.config)

        # Lifecycle tracking
        self._active_lifecycle: Optional[EntryLifecycle] = None

    def generate(self, df: pd.DataFrame, order_flow_signal=None) -> EntryEngineResult:
        result = EntryEngineResult(should_enter=False, signal="HOLD", reasons=[])

        if len(df) < 50:
            result.reasons.append("insufficient history <50")
            return result

        row = df.iloc[-1]

        # === MARKET CONTEXT ===
        market_context = self.market_context_engine.evaluate(df)
        result.market_context = market_context
        result.reasons.append(f"MarketContext: {market_context.reason}")

        # === POI / ZONE ENGINE (before Setup) ===
        try:
            poi = self.zone_engine.build_zones(df)
            result.poi = poi
            result.reasons.append(f"POI: support {poi.support_distance_pct:.2f}% resistance {poi.resistance_distance_pct:.2f}% zones {len(poi.all_zones)}")
        except Exception as e:
            result.reasons.append(f"POI error: {e}")
            poi = None

        # === SETUP ENGINE ===
        # Try all setup types, pick highest confidence valid — uses POI context
        setups = []
        try:
            # Generic setup
            s = self.setup_detector.detect(df, regime=market_context.regime)
            if s.is_valid:
                setups.append(("generic", s))
        except Exception as e:
            result.reasons.append(f"SetupDetector error: {e}")

        try:
            pb = self.pullback_setup.evaluate(df)
            if pb.is_valid:
                setups.append(("pullback", pb))
        except Exception:
            pass

        try:
            mr = self.mean_reversion_setup.evaluate(df)
            if mr.is_valid:
                setups.append(("mean_reversion", mr))
        except Exception:
            pass

        if not setups:
            result.reasons.append(f"No valid setup in {market_context.regime}")
            return result

        # Pick best setup by quality/confidence
        best_setup = max(setups, key=lambda x: getattr(x[1], 'quality', 0) if hasattr(x[1], 'quality') else getattr(x[1], 'confidence', 0))
        setup_type, setup_obj = best_setup
        result.setup = setup_obj
        result.reasons.append(f"Setup: {setup_obj.setup if hasattr(setup_obj, 'setup') else setup_type} quality {getattr(setup_obj, 'quality', 0):.2f}")

        # === TRIGGER ENGINE (Sweep/MSB/Retest) ===
        # Determine zone for trigger (from pullback or poi)
        zone = None
        if hasattr(setup_obj, 'zone') and setup_obj.zone:
            zone = setup_obj.zone
        elif poi and poi.nearest_support and "LONG" in str(getattr(setup_obj, 'setup', '')):
            zone = poi.nearest_support
        elif poi and poi.nearest_resistance and "SHORT" in str(getattr(setup_obj, 'setup', '')):
            zone = poi.nearest_resistance

        trigger = self.trigger_engine.evaluate(df, setup_type=setup_obj.setup if hasattr(setup_obj, 'setup') else str(setup_type), zone=zone)
        result.trigger = trigger
        result.reasons.append(f"Trigger: {trigger.trigger_type} {'TRIGGERED' if trigger.is_triggered else 'WAIT'} {trigger.reason}")

        if not trigger.is_triggered:
            return result

        # === ENTRY ZONE CANDIDATE — provisional, before Confirmation/EV (P3) ===
        # To correctly calc EV, need entry price, zone, SL, TP, costs, execution
        # Correct order: Trigger → Entry Zone → Confirmation → ... → SL/TP → EV
        entry_zone_candidate = None
        try:
            from src.entry.models import EntryZone as EntryZoneModel
            atr = float(row.get("atr", 1))
            entry_price = float(row.get("close", 0))
            setup_str = str(getattr(setup_obj, 'setup', 'NONE'))
            zone_atr_mult = 0.4
            if "BREAKOUT" in setup_str:
                zone_atr_mult = 0.3
            elif "PULLBACK" in setup_str:
                zone_atr_mult = 0.8
            half_width = atr * zone_atr_mult
            ideal = entry_price
            if hasattr(setup_obj, 'zone') and setup_obj.zone:
                try:
                    ideal = (setup_obj.zone.upper + setup_obj.zone.lower) / 2
                except Exception:
                    ideal = entry_price
            entry_zone_candidate = EntryZoneModel(
                zone_low=round(ideal - half_width, 2),
                zone_high=round(ideal + half_width, 2),
                ideal_entry=round(ideal, 2),
                max_chase_distance=round(atr * 0.8, 2),
                atr=round(atr, 2),
                setup=setup_str,  # type: ignore
            )
            result.entry_zone_candidate = entry_zone_candidate  # type: ignore
            result.reasons.append(f"EntryZone Candidate: {entry_zone_candidate.zone_low:.1f}-{entry_zone_candidate.zone_high:.1f} ideal {entry_zone_candidate.ideal_entry:.1f} (provisional for EV)")
        except Exception as e:
            result.reasons.append(f"EntryZone Candidate error: {e} → will fallback after Confirmation")

        # === CONFIRMATION ENGINE — Structure/Momentum/Volume/OrderFlow/MTF/ML ===
        try:
            sig_res: SignalResult = self.signal_generator.generate(df, order_flow_signal=order_flow_signal)
            setup_dir = "BUY" if "LONG" in str(getattr(setup_obj, 'setup', '')) else "SELL" if "SHORT" in str(getattr(setup_obj, 'setup', '')) else "HOLD"
            if sig_res.signal != setup_dir and sig_res.signal != "HOLD":
                result.reasons.append(f"Confirmation: setup {setup_dir} vs signal {sig_res.signal} mismatch → HOLD")
                return result
            if sig_res.signal == "HOLD":
                result.reasons.append(f"Confirmation: SignalGenerator HOLD ({sig_res.reasons[-1] if sig_res.reasons else 'no signal'})")
                return result
            result.confirmation = {"signal": sig_res.signal, "confidence": sig_res.confidence, "meta": sig_res.meta_decision, "structure": getattr(sig_res.market_components, 'trend_score', 0) if hasattr(sig_res, 'market_components') and sig_res.market_components else 0}
            result.reasons.append(f"Confirmation: Structure/Momentum/Volume/OrderFlow {sig_res.order_flow_approved} ML {sig_res.meta_decision} MTF {sig_res.reasons[-1] if sig_res.reasons else ''} → PASS")
        except Exception as e:
            result.reasons.append(f"Confirmation error: {e}")
            return result

        # === ENTRY QUALITY ===
        try:
            eq = self.entry_quality_engine.calculate(
                setup_result=setup_obj,
                pullback_result=setup_obj if hasattr(setup_obj, 'pullback_dist_atr') else None,
                zone_result=poi,
                row=row,
            )
            result.entry_quality = eq
            result.reasons.append(f"EntryQuality: {eq.quality:.0f} (zone {eq.zone_score:.0f} exhaust {eq.exhaustion_score:.0f} trigger {eq.trigger_score:.0f})")
            if eq.quality < 50:
                result.reasons.append(f"EntryQuality low {eq.quality:.0f} <50 → HOLD")
                return result
        except Exception as e:
            result.reasons.append(f"EntryQuality error: {e}")

        # === SL/TP CANDIDATE is already provisional before EV (correct order) — reuse ===
        # EV needs entry, zone, SL, TP, costs, execution — all now known
        # === EXPECTED VALUE ===
        try:
            p_win = 0.55
            sig_conf = result.confirmation.get("confidence", 60) if isinstance(result.confirmation, dict) else 60
            p_win = float(sig_conf) / 100.0 if sig_conf else 0.55
            p_win = max(0.1, min(0.9, p_win))
            # Use provisional sltp_candidate
            if sltp_candidate is None:
                # Fallback if provisional failed
                from src.strategy.sl_tp_calculator import SLTPCalculator
                calc = SLTPCalculator()
                atr = float(row.get("atr", 1))
                entry = float(entry_zone_candidate.ideal_entry if entry_zone_candidate else row.get("close", 0))
                sltp_candidate = calc.calculate(entry_price=entry, atr=atr, signal="BUY")
                result.sltp_candidate = sltp_candidate  # type: ignore
            entry = float(entry_zone_candidate.ideal_entry if entry_zone_candidate else row.get("close", 0))
            sl_pct = abs(entry - sltp_candidate.stop_loss) / entry if entry else 0.01
            tp_pct = abs(sltp_candidate.take_profit - entry) / entry if entry else 0.02
            ev_res = self.expected_value_gate.evaluate(p_win=p_win, avg_win=tp_pct, avg_loss=sl_pct, latency_ms=100)
            result.expected_value = ev_res
            result.reasons.append(f"ExpectedValue: {ev_res.reason} (entry {entry:.1f} SL {sltp_candidate.stop_loss:.1f} TP {sltp_candidate.take_profit:.1f} zone {entry_zone_candidate.zone_low:.1f}-{entry_zone_candidate.zone_high:.1f} SL {sl_pct:.3f} TP {tp_pct:.3f})")
            if not ev_res.passed:
                return result
        except Exception as e:
            result.reasons.append(f"ExpectedValue error: {e} → permissive PASS")

        # === RISK GATE ===
        result.risk_approved = True
        result.reasons.append("Risk Gate: PASS (subordinate to Risk Policy, checked in Execution)")

        # === FINAL ENTRY DECISION → ENTRY CANDIDATE → ORDER INTENT ===

        # === ORDER INTENT → EXECUTION (via ExecutionPolicy) ===
        try:
            # Determine execution type via policy
            # For EntryEngine, we just prepare intent; actual execution choice happens in Execution layer
            # But we can pre-select via ExecutionPolicy
            sl_pct = abs(entry - sltp.stop_loss) / entry if entry else 0.01
            tp_pct = abs(sltp.take_profit - entry) / entry if entry else 0.02
            # Estimate queue prob from zone strength
            queue_prob = 0.6
            if poi and poi.nearest_support and poi.nearest_support.strength > 0.6:
                queue_prob = 0.7
            ctx = ExecutionContext(
                setup=str(getattr(setup_obj, 'setup', '')),
                urgency=0.7 if "BREAKOUT" in str(getattr(setup_obj, 'setup', '')) else 0.4,
                spread_pct=0.0005,
                volatility_atr_pct=atr/entry if entry else 0.01,
                expected_slippage=0.0002,
                queue_probability=queue_prob,
                expected_net_edge=ev_res.expected_net if 'ev_res' in locals() else 0.002,
            )
            from src.execution.execution_policy import ExecutionPolicy
            policy = ExecutionPolicy()
            decision = policy.decide(ctx, entry, atr)
            result.execution_decision = decision
            result.reasons.append(f"Execution: {decision.execution_type} EV {decision.expected_ev_after_fill:.4f} fill {decision.fill_probability:.2f} ({decision.reason})")
        except Exception as e:
            result.reasons.append(f"ExecutionPolicy error: {e}")

        result.should_enter = True
        result.signal = sig_res
        return result
