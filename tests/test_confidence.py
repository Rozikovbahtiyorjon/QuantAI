"""
=========================================
QuantAI Confidence Engine Test
=========================================
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.confidence_engine import ConfidenceEngine


def main():

    engine = ConfidenceEngine()

    engine.add_component(
        "trend",
        3,
        "EMA Trend",
    )

    engine.add_component(
        "momentum",
        2,
        "RSI",
    )

    engine.add_component(
        "volume",
        1,
        "Volume",
    )

    engine.add_component(
        "volatility",
        0,
        "ATR",
    )

    engine.add_component(
        "liquidity",
        2,
        "Liquidity",
    )

    engine.add_component(
        "structure",
        1,
        "Structure",
    )

    engine.add_component(
        "regime",
        2,
        "Regime",
    )

    engine.print_report()


if __name__ == "__main__":
    main()