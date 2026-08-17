from __future__ import annotations

import subprocess
import sys


def run(
    command: list[str],
) -> int:
    print(
        f"[CHECK] {' '.join(command)}"
    )

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"[FAIL] Command exited with "
            f"code {result.returncode}"
        )

    else:
        print(
            "[PASS] Command completed successfully"
        )

    return result.returncode


def main() -> int:
    print("=" * 72)
    print("QuantAI Strategy + OrderFlow Stage Check")
    print("=" * 72)

    checks = [
        [
            sys.executable,
            "-m",
            "py_compile",
            "src/strategy.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_strategy.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_order_flow_strategy_integration.py",
        ],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_order_flow_intelligence.py",
            "tests/test_order_flow_unified_integration.py",
            "tests/test_order_flow_strategy_integration.py",
            "tests/test_strategy.py",
        ],
    ]

    for command in checks:
        if run(command) != 0:
            print("=" * 72)
            print("STRATEGY + ORDER FLOW STAGE CHECK: FAILED")
            print("=" * 72)
            return 1

    print("=" * 72)
    print("STRATEGY + ORDER FLOW STAGE CHECK: SUCCESS")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )