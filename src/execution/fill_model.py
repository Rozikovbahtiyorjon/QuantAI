"""
Fill Model — Audit #41: Limit order is NOT filled when price touched.

Requires queue + volume + time + market movement → fill probability.
Otherwise backtest systematically too optimistic.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass
class FillResult:
    filled: bool
    fill_price: float | None
    fill_prob: float
    reason: str


class LimitFillModel:
    """
    Queue-aware fill model — production-like simulation (Audit §41, Research §56, P2.3).

    Fill probability for limit orders requires 5 factors:
      1. price touched (bar_low <= limit <= bar_high)
      2. volume (bar_volume / avg_volume)
      3. queue position (queue_ahead_pct * (1 - depth_into_bar))
      4. latency (signal->exchange delay reduces fill, 50ms-3s)
      5. order book depth (spread + volume + time priority)

    P2.3: latency and order book depth are now explicit parameters, not heuristic proxy alone.
    Realistic Binance execution: order-book depth + queue + latency + trade flow.
    DETERMINISTIC (seeded via experiment_seed) for reproducibility per P1.24.
    For full production, L2 order-book replay would replace heuristic.
    """

    def __init__(
        self,
        queue_ahead_pct: float = 0.3,
        min_volume_ratio: float = 1.5,
        experiment_seed: int = 42,
        seed: int | None = None,
        # P2.3 explicit
        base_latency_ms: float = 100.0,
        order_book_depth: float = 1.0,  # 1.0 = normal depth, <1 thin, >1 deep
    ):
        self.queue_ahead_pct = queue_ahead_pct
        self.min_volume_ratio = min_volume_ratio
        self.experiment_seed = experiment_seed
        self.seed = seed if seed is not None else experiment_seed
        self.base_latency_ms = float(base_latency_ms)
        self.order_book_depth = float(order_book_depth)

    def _deterministic_random(self, *keys: str) -> float:
        """Deterministic 0..1 from hash of keys — reproducible, no global random."""
        h = hashlib.sha256("|".join(str(k) for k in keys).encode()).hexdigest()
        # Use first 8 hex chars as int → 0..1
        return int(h[:8], 16) / 0xFFFFFFFF

    def attempt_fill(
        self,
        limit_price: float,
        side: str,  # BUY / SELL
        bar_high: float,
        bar_low: float,
        bar_volume: float,
        avg_volume: float,
        spread: float = 0.0002,
        symbol: str = "BTCUSDT",
        bar_timestamp: str = "",
        order_id: str = "",
        # P2.3 explicit overrides per call (if not provided, uses model defaults)
        latency_ms: float | None = None,
        order_book_spread: float | None = None,
        order_book_depth: float | None = None,
    ) -> FillResult:
        """Deterministic fill decision — P2.3 includes latency + order book.

        Args:
            latency_ms: signal→exchange delay (50ms-3s). Higher → lower fill (price moved).
            order_book_spread: current spread (if None, uses spread param).
            order_book_depth: depth multiplier (1.0 normal, <1 thin → lower fill)
        """
        touched = (bar_low <= limit_price <= bar_high)
        if not touched:
            return FillResult(False, None, 0.0, "price not touched")

        # Resolve P2.3 params
        eff_latency = float(latency_ms) if latency_ms is not None else float(self.base_latency_ms)
        eff_spread = float(order_book_spread) if order_book_spread is not None else float(spread)
        eff_depth = float(order_book_depth) if order_book_depth is not None else float(self.order_book_depth)

        # Heuristic: volume must clear queue ahead + price location in bar
        vol_ratio = bar_volume / (avg_volume + 1e-9)
        # Adjust volume ratio by order book depth (thin book → less effective volume)
        vol_ratio_adj = vol_ratio * eff_depth
        if vol_ratio_adj < self.min_volume_ratio:
            fill_prob = 0.2 * vol_ratio_adj / self.min_volume_ratio
        else:
            bar_range = bar_high - bar_low + 1e-9
            if side == "BUY":
                depth_into_bar = (bar_high - limit_price) / bar_range
            else:
                depth_into_bar = (limit_price - bar_low) / bar_range
            # Queue position: deeper in bar → more volume before order, higher prob
            queue_factor = 1.0 - self.queue_ahead_pct * (1 - depth_into_bar)
            fill_prob = 0.3 + 0.6 * depth_into_bar * min(1.0, vol_ratio_adj / 2.0) * queue_factor
            fill_prob = max(0.0, min(1.0, fill_prob))

        # P2.3: latency penalty — higher latency reduces fill (market moved away)
        # 50ms → 0.99x, 100ms → 0.97x, 250ms → 0.92x, 500ms → 0.85x, 1s → 0.75x, 3s → 0.55x
        # Formula: latency_factor = exp(-latency_ms / 1500)
        import math as _math
        latency_factor = _math.exp(-eff_latency / 1500.0)
        # Clamp to 0.5-1.0 for realistic (never 0)
        latency_factor = max(0.5, min(1.0, 0.4 + 0.6 * latency_factor))
        fill_prob *= latency_factor

        # P2.3: order book spread penalty — wider spread → less likely limit at mid fills
        # eff_spread 0.0001 (tight) → 1.0x, 0.001 (wide) → 0.7x
        spread_factor = 1.0 - min(0.3, (eff_spread - 0.0001) * 100)  # heuristic
        spread_factor = max(0.7, min(1.0, spread_factor))
        fill_prob *= spread_factor

        fill_prob = max(0.0, min(1.0, fill_prob))

        # Deterministic: hash(symbol, timestamp, limit_price, side) → 0..1, no random.random()
        det_rand = self._deterministic_random(symbol, bar_timestamp, f"{limit_price:.2f}", side, order_id, str(self.experiment_seed))
        filled = det_rand < fill_prob
        fill_price = limit_price if filled else None
        if filled:
            # Spread as half-spread cost, deterministic
            if side == "BUY":
                fill_price = limit_price + spread * limit_price * 0.5
            else:
                fill_price = limit_price - spread * limit_price * 0.5
            reason = f"filled (queue-aware heuristic, prob {fill_prob:.2f}, det_rand {det_rand:.2f})"
        else:
            reason = f"queue not cleared (prob {fill_prob:.2f}, det_rand {det_rand:.2f})"

        return FillResult(filled, fill_price, fill_prob, reason)
