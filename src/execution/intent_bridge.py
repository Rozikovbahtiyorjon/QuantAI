"""
QuantAI Execution Intent Bridge (R1)

Unifies the three execution paths behind ONE contract:

    SignalResult + RiskDecision
        ↓
    OrderIntentData  (canonical, src/execution/orders.py)
        ↓
    ExecutionBridge.submit()
        ├── PAPER   -> PaperRunnerExecutor (PaperTradingRunner engine)
        ├── DRY_RUN -> ExecutionEngine.submit_intent
        └── LIVE    -> ExecutionEngine.submit_intent

Guards applied before routing:
    - IdempotencyGuard: duplicate signal submissions are dropped
      (same strategy_signal_id + intent kind within TTL).
    - Rate limiter hook (optional; non-paper modes only).

Flip handling: produces an atomic pair [EXIT(reduce_only), ENTRY].
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.execution.orders import (
    OrderIntent,
    OrderIntentData,
    OrderSide,
    OrderType,
)


# =====================================================
# INTENT BUILDERS
# =====================================================

def build_entry_intent(
    signal: Any,
    decision: Any,
    symbol: str = "BTCUSDT",
    strategy_signal_id: str | None = None,
) -> OrderIntentData:
    """
    Map an approved Signal+RiskDecision pair to a canonical ENTRY intent.
    """
    if not getattr(decision, "allowed", False):
        raise ValueError("cannot build entry intent: risk decision not allowed")

    qty = float(decision.quantity)
    if qty <= 0:
        raise ValueError("cannot build entry intent: non-positive quantity")

    side = OrderSide.BUY if signal.signal == "BUY" else OrderSide.SELL

    return OrderIntentData(
        intent=OrderIntent.ENTRY,
        symbol=symbol,
        side=side,
        quantity=qty,
        order_type=OrderType.MARKET,
        risk_decision_id=getattr(decision, "risk_decision_id", None),
        strategy_signal_id=strategy_signal_id,
        metadata={
            "entry": float(signal.entry),
            "stop_loss": float(signal.stop_loss),
            "take_profit": float(signal.take_profit or 0.0),
            "confidence": float(signal.confidence),
        },
    )


def build_exit_intent(
    position_side: str,
    quantity: float,
    symbol: str = "BTCUSDT",
    price: float | None = None,
    strategy_signal_id: str | None = None,
) -> OrderIntentData:
    """
    Close an existing position: opposite side, reduce-only.
    """
    if quantity <= 0:
        raise ValueError("exit intent requires positive quantity")

    side = OrderSide.SELL if position_side == "LONG" else OrderSide.BUY

    return OrderIntentData(
        intent=OrderIntent.EXIT,
        symbol=symbol,
        side=side,
        quantity=float(quantity),
        order_type=OrderType.MARKET,
        reduce_only=True,
        price=price,
        strategy_signal_id=strategy_signal_id,
    )


# =====================================================
# GUARDS
# =====================================================

class IdempotencyGuard:
    """
    Lightweight TTL duplicate filter for signal submissions.

    Key = (strategy_signal_id or derived from intent pair) + kind.
    Mirrors the semantics of production.OrderDeduplicator without
    requiring its background cleanup task.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        self.ttl = float(ttl_seconds)
        self._seen: dict[str, float] = {}

    def _prune(self, now: float) -> None:
        expired = [k for k, ts in self._seen.items() if now - ts > self.ttl]
        for k in expired:
            del self._seen[k]

    def check_and_register(self, key: str) -> bool:
        """
        Returns True when the key is NEW (action allowed).
        Returns False for duplicates within TTL.
        """
        now = time.monotonic()
        self._prune(now)

        if key in self._seen:
            return False

        self._seen[key] = now
        return True


# =====================================================
# HANDLERS
# =====================================================

@dataclass
class BridgeResult:
    accepted: bool
    reason: str
    intents: list[OrderIntentData] = field(default_factory=list)
    orders: list[Any] = field(default_factory=list)
    paper_trade: Any = None
    paper_step: Any = None


class PaperRunnerExecutor:
    """
    PAPER-mode handler backed by the existing PaperTradingRunner.
    Maps intents to runner actions so paper fills stay identical
    to the validated paper pipeline.
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    async def execute(self, intents: list[OrderIntentData], signal: Any) -> BridgeResult:
        trade = None
        step = None

        for it in intents:
            if it.intent == OrderIntent.EXIT:
                trade = self.runner.engine.close_position(price=signal.entry)
            elif it.intent == OrderIntent.ENTRY:
                self.runner.engine.open_position(
                    side="LONG" if it.side == OrderSide.BUY else "SHORT",
                    price=signal.entry,
                    quantity=it.quantity,
                )

        return BridgeResult(
            accepted=True,
            reason="paper executed",
            intents=intents,
            paper_trade=trade,
            paper_step=step,
        )


class EngineExecutor:
    """
    DRY_RUN / LIVE handler delegating to ExecutionEngine.submit_intent.
    The engine applies its own safety checks and reconciliation.
    """

    def __init__(self, engine: Any, rate_limiter: Any | None = None) -> None:
        self.engine = engine
        self.rate_limiter = rate_limiter

    async def execute(self, intents: list[OrderIntentData], signal: Any) -> BridgeResult:
        orders = []

        for it in intents:
            if self.rate_limiter is not None and hasattr(
                self.rate_limiter, "try_acquire_for_endpoint"
            ):
                if not self.rate_limiter.try_acquire_for_endpoint("order", weight=1):
                    return BridgeResult(
                        accepted=False,
                        reason="rate limit exceeded",
                        intents=intents,
                        orders=orders,
                    )

            order = await self.engine.submit_intent(it)
            orders.append(order)

        rejected = [
            o for o in orders
            if getattr(o, "status", None) is not None
            and getattr(o.status, "value", o.status) == "REJECTED"
        ]
        if rejected:
            return BridgeResult(
                accepted=False,
                reason=f"{len(rejected)} order(s) rejected by exchange",
                intents=intents,
                orders=orders,
            )

        return BridgeResult(accepted=True, reason="routed", intents=intents, orders=orders)


# =====================================================
# BRIDGE
# =====================================================

class ExecutionBridge:
    """
    Single submission point for approved signals.

    Usage:
        bridge = ExecutionBridge.paper(runner)
        result = await bridge.submit(signal, decision)

        bridge = ExecutionBroker.dry_run(engine, rate_limiter)
        ...
    """

    MODE_PAPER = "PAPER"
    MODE_DRY_RUN = "DRY_RUN"
    MODE_LIVE = "LIVE"

    def __init__(
        self,
        mode: str,
        executor: Any,
        dedup: IdempotencyGuard | None = None,
    ) -> None:
        self.mode = mode
        self.executor = executor
        self.dedup = dedup or IdempotencyGuard()

    # -------------------------------------------------- factories

    @classmethod
    def paper(cls, runner: Any) -> "ExecutionBridge":
        return cls(cls.MODE_PAPER, PaperRunnerExecutor(runner))

    @classmethod
    def routed(cls, engine: Any, rate_limiter: Any | None = None) -> "ExecutionBridge":
        mode = cls.MODE_LIVE
        cfg_mode = getattr(engine, "config", None)
        m = getattr(cfg_mode, "mode", None)
        if m is not None:
            mode = getattr(m, "value", str(m))
        return cls(mode, EngineExecutor(engine, rate_limiter))

    # -------------------------------------------------- core

    def _build_intents(
        self,
        signal: Any,
        decision: Any,
        symbol: str,
        strategy_signal_id: str | None,
    ) -> list[OrderIntentData]:
        intents: list[OrderIntentData] = []

        # Position state lives on the paper ENGINE; Runner may expose
        # its own attributes, so always resolve through .engine when present.
        runner = getattr(self.executor, "runner", None)
        target = getattr(runner, "engine", None) if runner is not None else None
        if target is None:
            target = runner

        pos = getattr(target, "position", None)
        has_pos = bool(getattr(target, "has_position", False))

        if has_pos and pos is not None:
            current_side = getattr(pos.side, "value", pos.side)
            requested = "LONG" if signal.signal == "BUY" else "SHORT"

            if current_side != requested:
                intents.append(
                    build_exit_intent(
                        position_side=current_side,
                        quantity=float(pos.quantity),
                        symbol=symbol,
                        strategy_signal_id=strategy_signal_id,
                    )
                )
            else:
                # same-side duplicate: nothing to do
                return []

        intents.append(
            build_entry_intent(signal, decision, symbol, strategy_signal_id)
        )
        return intents

    async def submit(
        self,
        signal: Any,
        decision: Any,
        symbol: str = "BTCUSDT",
        strategy_signal_id: str | None = None,
    ) -> BridgeResult:
        if not getattr(decision, "allowed", False):
            return BridgeResult(
                accepted=False,
                reason=f"risk rejected: {getattr(decision, 'reason', '')}",
            )

        intents = self._build_intents(signal, decision, symbol, strategy_signal_id)

        if not intents:
            return BridgeResult(
                accepted=False,
                reason="duplicate same-side signal",
            )

        key_source = strategy_signal_id or (
            f"{symbol}:{signal.signal}:{signal.entry}:{intents[-1].quantity}"
        )
        if not self.dedup.check_and_register(key_source):
            return BridgeResult(
                accepted=False,
                reason="duplicate submission (idempotency)",
                intents=intents,
            )

        return await self.executor.execute(intents, signal)


__all__ = [
    "build_entry_intent",
    "build_exit_intent",
    "IdempotencyGuard",
    "PaperRunnerExecutor",
    "EngineExecutor",
    "ExecutionBridge",
    "BridgeResult",
]
