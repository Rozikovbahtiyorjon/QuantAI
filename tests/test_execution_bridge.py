"""
R1 Execution Bridge tests.

Covers:
    - SignalResult+RiskDecision -> OrderIntentData mapping
    - flip produces atomic [EXIT(reduce_only), ENTRY] pair
    - idempotency guard drops duplicates
    - PAPER mode end-to-end via PaperTradingRunner (fills identical)
    - DRY/LIVE routing through a stub engine + rate limiter hook
"""

from __future__ import annotations

import pytest

from src.execution.intent_bridge import (
    BridgeResult,
    ExecutionBridge,
    IdempotencyGuard,
    PaperRunnerExecutor,
    build_entry_intent,
    build_exit_intent,
)
from src.execution.orders import (
    OrderIntent,
    OrderIntentData,
    OrderSide,
    OrderStatus,
)
from src.paper_trading_runner import PaperTradingRunner
from src.strategy.signal_generator import SignalResult


def make_signal(signal="BUY", entry=100.0, sl=98.0, tp=106.0):
    return SignalResult(
        signal=signal, entry=entry, stop_loss=sl, take_profit=tp, confidence=70.0
    )


class Decision:
    def __init__(self, qty=0.5, allowed=True, reason="ok"):
        self.allowed = allowed
        self.quantity = qty
        self.reason = reason


class TestIntentBuilders:
    def test_entry_intent_mapping(self) -> None:
        sig = make_signal("BUY")
        it = build_entry_intent(sig, Decision(0.5), "BTCUSDT", sid123 := "s1")

        assert isinstance(it, OrderIntentData)
        assert it.intent == OrderIntent.ENTRY
        assert it.side == OrderSide.BUY
        assert it.quantity == 0.5
        assert it.order_type.value == "MARKET"
        assert it.metadata["stop_loss"] == 98.0
        assert it.strategy_signal_id == "s1"

    def test_entry_rejects_disallowed_decision(self) -> None:
        with pytest.raises(ValueError):
            build_entry_intent(make_signal(), Decision(allowed=False))

    def test_exit_is_reduce_only_opposite_side(self) -> None:
        it = build_exit_intent("LONG", 0.5)
        assert it.intent == OrderIntent.EXIT
        assert it.reduce_only is True
        assert it.side == OrderSide.SELL

        it2 = build_exit_intent("SHORT", 0.25)
        assert it2.side == OrderSide.BUY

    def test_flip_pair_atomic(self) -> None:
        runner = PaperTradingRunner(
            initial_balance=1000.0, commission=0.0,
            enable_risk_controls=False, quantity=0.5,
        )
        runner.process_signal(make_signal("BUY"))
        bridge = ExecutionBridge.paper(runner)

        intents = bridge._build_intents(make_signal("SELL"), Decision(0.5), "BTCUSDT", None)

        assert [i.intent for i in intents] == [OrderIntent.EXIT, OrderIntent.ENTRY]
        assert intents[0].reduce_only is True


class TestIdempotency:
    def test_duplicates_dropped(self) -> None:
        g = IdempotencyGuard(ttl_seconds=60)
        assert g.check_and_register("k1") is True
        assert g.check_and_register("k1") is False
        assert g.check_and_register("k2") is True

    def test_ttl_expiry(self) -> None:
        import time
        g = IdempotencyGuard(ttl_seconds=0.05)
        assert g.check_and_register("k") is True
        time.sleep(0.06)
        assert g.check_and_register("k") is True


class TestPaperMode:
    @pytest.mark.asyncio
    async def test_paper_open_matches_runner(self) -> None:
        runner = PaperTradingRunner(initial_balance=1000.0, commission=0.0004)

        # Pre-size via runner risk path to obtain an approved decision-like qty.
        runner.enable_risk_controls = False
        step = runner.process_signal(make_signal("BUY"))
        assert step.position_opened
        expected_qty = runner.engine.position.quantity
        runner.engine.close_position(price=100.0)  # reset book

        runner_ref = PaperTradingRunner(initial_balance=1000.0, commission=0.0004)
        bridge = ExecutionBridge.paper(runner_ref)

        res = await bridge.submit(make_signal("BUY"), Decision(expected_qty), strategy_signal_id="sig#1")

        assert isinstance(res, BridgeResult)
        assert res.accepted is True
        assert runner_ref.has_position
        assert runner_ref.engine.position.quantity == pytest.approx(expected_qty)

    @pytest.mark.asyncio
    async def test_paper_flip_closes_then_opens(self) -> None:
        runner = PaperTradingRunner(initial_balance=1000.0, commission=0.0,
                                    enable_risk_controls=False, quantity=0.5)
        bridge = ExecutionBridge.paper(runner)

        await bridge.submit(make_signal("BUY"), Decision(0.5))
        res = await bridge.submit(make_signal("SELL"), Decision(0.4),
                                  strategy_signal_id="flip1")

        assert res.accepted
        assert runner.engine.position.side == "SHORT"
        assert runner.engine.position.quantity == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_same_side_duplicate_noop(self) -> None:
        runner = PaperTradingRunner(commission=0.0, enable_risk_controls=False,
                                    quantity=0.3)
        bridge = ExecutionBridge.paper(runner)
        await bridge.submit(make_signal("BUY"), Decision(0.3))
        res = await bridge.submit(make_signal("SELL"), Decision(0.3))  # flip first
        assert res.accepted

        res2 = await bridge.submit(
            make_signal("SELL", entry=101.0), Decision(0.3),
            strategy_signal_id="dup-test",
        )
        # same side as current SHORT position -> no intents
        assert res2.accepted is False
        assert "duplicate same-side" in res2.reason


class StubEngine:
    def __init__(self, reject=False):
        self.submitted = []
        self.reject = reject
        class Cfg: mode = "DRY_RUN"
        self.config = Cfg()

    async def submit_intent(self, intent):
        from src.execution.orders import Order
        o = Order(intent=intent)
        if self.reject:
            o.reject("stub reject")
        else:
            o.update_fill(intent.quantity, intent.price or 100.0)
        self.submitted.append(o)
        return o


class CountingLimiter:
    def __init__(self, max_calls):
        self.calls = 0
        self.max = max_calls

    def try_acquire_for_endpoint(self, endpoint, weight=None):
        if self.calls >= self.max:
            return False
        self.calls += 1
        return True


class TestRoutedModes:
    @pytest.mark.asyncio
    async def test_dry_run_routes_intents_and_applies_rate_limit(self) -> None:
        engine = StubEngine()
        limiter = CountingLimiter(max_calls=1)
        bridge = ExecutionBridge.routed(engine, limiter)

        assert bridge.mode == "DRY_RUN"

        res = await bridge.submit(make_signal("BUY"), Decision(0.5),
                                  strategy_signal_id="r1")
        assert res.accepted
        assert len(engine.submitted) == 1

        # second submission hits rate limit
        res2 = await bridge.submit(make_signal("SELL"), Decision(0.5),
                                   strategy_signal_id="r2",
                                   symbol="BTCUSDT")
        # note: no position tracked in stub -> single ENTRY intent
        assert res2.accepted is False
        assert "rate limit" in res2.reason
        assert len(engine.submitted) == 1

    @pytest.mark.asyncio
    async def test_exchange_reject_surfaces(self) -> None:
        engine = StubEngine(reject=True)
        bridge = ExecutionBridge.routed(engine)

        res = await bridge.submit(make_signal("BUY"), Decision(0.5),
                                  strategy_signal_id="x1")
        assert res.accepted is False
        assert "rejected" in res.reason.lower()

    @pytest.mark.asyncio
    async def test_risk_rejected_short_circuits(self) -> None:
        engine = StubEngine()
        bridge = ExecutionBridge.routed(engine)

        res = await bridge.submit(make_signal("BUY"), Decision(allowed=False))
        assert res.accepted is False
        assert len(engine.submitted) == 0