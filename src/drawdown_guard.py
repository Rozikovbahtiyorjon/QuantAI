from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DrawdownGuardResult:
    peak_equity: float
    current_equity: float
    drawdown: float
    drawdown_percent: float
    allowed: bool


class DrawdownGuard:
    def __init__(
        self,
        max_drawdown_percent: float = 10.0,
    ) -> None:
        if max_drawdown_percent < 0:
            raise ValueError(
                "max_drawdown_percent cannot be negative."
            )

        self.max_drawdown_percent = float(
            max_drawdown_percent
        )
        self._peak_equity: float | None = None

    def evaluate(
        self,
        current_equity: float,
    ) -> DrawdownGuardResult:
        if current_equity <= 0:
            raise ValueError(
                "current_equity must be greater than zero."
            )

        equity = float(current_equity)

        if self._peak_equity is None:
            self._peak_equity = equity

        if equity > self._peak_equity:
            self._peak_equity = equity

        peak = self._peak_equity

        drawdown = max(
            peak - equity,
            0.0,
        )

        if peak > 0:
            drawdown_percent = (
                drawdown
                / peak
                * 100.0
            )
        else:
            drawdown_percent = 0.0

        drawdown = round(
            drawdown,
            8,
        )

        drawdown_percent = round(
            drawdown_percent,
            8,
        )

        allowed = (
            drawdown_percent
            <= self.max_drawdown_percent
        )

        return DrawdownGuardResult(
            peak_equity=round(
                peak,
                8,
            ),
            current_equity=round(
                equity,
                8,
            ),
            drawdown=drawdown,
            drawdown_percent=drawdown_percent,
            allowed=allowed,
        )

    def is_allowed(
        self,
        current_equity: float,
    ) -> bool:
        return self.evaluate(
            current_equity
        ).allowed

    @property
    def peak_equity(self) -> float | None:
        return self._peak_equity

    def reset(self) -> None:
        self._peak_equity = None