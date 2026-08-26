from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[1]

TEST_FILE = (
    ROOT
    / "tests"
    / "test_paper_trading_final_validation.py"
)

SOURCE_FILES = (
    ROOT
    / "src"
    / "paper_trading_engine.py",
    ROOT
    / "src"
    / "paper_trading_runner.py",
    ROOT
    / "src"
    / "paper_trading_session.py",
    ROOT
    / "src"
    / "paper_trading_pipeline.py",
    ROOT
    / "src"
    / "strategy.py",
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
        "QuantAI Paper Trading Final Validation"
    )

    print(
        "=" * 72
    )

    compile_command = [
        sys.executable,
        "-m",
        "py_compile",
        *[
            str(path)
            for path in SOURCE_FILES
        ],
    ]

    run(
        compile_command
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
        "PAPER TRADING FINAL VALIDATION: PASS"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()