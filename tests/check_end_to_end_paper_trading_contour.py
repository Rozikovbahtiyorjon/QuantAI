from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

SOURCE_FILES = [
    ROOT / "src" / "feature_engine.py",
    ROOT / "src" / "confidence_engine.py",
    ROOT / "src" / "risk_manager.py",
    ROOT / "src" / "order_book_market_data.py",
    ROOT / "src" / "order_flow_intelligence.py",
    ROOT / "src" / "order_flow_strategy_integration.py",
    ROOT / "src" / "strategy.py",
    ROOT / "src" / "paper_trading_engine.py",
    ROOT / "src" / "paper_trading_runner.py",
    ROOT / "src" / "paper_trading_session.py",
    ROOT / "src" / "paper_trading_pipeline.py",
]

TEST_FILE = (
    ROOT
    / "tests"
    / "test_end_to_end_paper_trading_contour.py"
)


def run(
    command: list[str],
) -> None:
    print(
        ">",
        " ".join(command),
    )

    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    if completed.stdout:
        print(
            completed.stdout
        )

    if completed.stderr:
        print(
            completed.stderr
        )

    if completed.returncode != 0:
        raise SystemExit(
            completed.returncode
        )


def main() -> None:
    print(
        "=" * 72
    )

    print(
        "QuantAI End-to-End Paper Trading Contour Validation"
    )

    print(
        "=" * 72
    )

    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            *[
                str(path)
                for path in SOURCE_FILES
            ],
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "py_compile",
            str(TEST_FILE),
        ]
    )

    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(TEST_FILE),
        ]
    )

    print(
        "=" * 72
    )

    print(
        "END-TO-END PAPER TRADING CONTOUR: PASS"
    )

    print(
        "Data -> Features -> ML -> Confidence -> "
        "Strategy -> Order Flow -> Risk -> Trade"
    )

    print(
        "Position / Balance / Realized PnL: VERIFIED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()