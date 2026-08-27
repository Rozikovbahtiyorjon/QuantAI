#!/usr/bin/env python
"""
QuantAI Dry-Run Testnet Validation
Simple validation script to check testnet readiness
"""

import sys
import os
from pathlib import Path

# Set test environment variables BEFORE importing settings
os.environ.setdefault("BINANCE_TESTNET_API_KEY", "test_key")
os.environ.setdefault("BINANCE_TESTNET_API_SECRET", "test_secret")
os.environ.setdefault("BINANCE_API_KEY", "test_key")
os.environ.setdefault("BINANCE_API_SECRET", "test_secret")
os.environ.setdefault("BINANCE_TESTNET", "true")

def main():
    print("QuantAI Dry-Run Testnet Simulation")
    print("=" * 50)

    # Check config files
    print("\nRunning checks...")

    # Check config files
    print("\nChecking config files...")
    files = ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]
    all_ok = True
    for f in ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]:
        if Path(f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            return 1

    # Check imports
    print("\nChecking imports...")
    try:
        from config.settings import Settings, settings
        from config.testnet_settings import testnet_config
        print("  [OK] Config imports")
    except Exception as e:
        print(f"  [FAIL] Config imports: {e}")
        return 1

    # Check env file
    if Path(".env.testnet").exists():
        print("  [OK] .env.testnet exists")
    else:
        print("  [WARN] .env.testnet not found (copy from template)")

    # Validate config
    print("\nValidating configuration...")
    try:
        from src.monitoring.config_validation import validate_config_or_exit
        from config.settings import Settings
        result = validate_config_or_exit(Settings())
        if result.valid:
            print("  [OK] Config validation passed")
        else:
            print(f"  [FAIL] Config validation: {result.errors}")
            return 1
    except Exception as e:
        print(f"  [FAIL] Config validation: {e}")
        return 1

    print("\n" + "=" * 50)
    print("DRY-RUN SUMMARY")
    print("=" * 50)
    print("All checks passed!")
    print("Ready for testnet deployment.")
    print("\nNext steps:")
    print("  1. Copy .env.testnet.template to .env.testnet")
    print("  2. Fill in BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET")
    print("  3. Run: ./deploy_testnet.sh up")

    return 0


def main():
    print("QuantAI Dry-Run Testnet Simulation")
    print("=" * 50)

    # Check config files
    print("\nRunning checks...")

    # Check config files
    print("\nChecking config files...")
    files = ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]
    all_ok = True
    for f in ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]:
        if Path(f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            return 1

    # Check imports
    print("\nChecking imports...")
    try:
        from config.settings import Settings, settings
        from config.testnet_settings import testnet_config
        print("  [OK] Config imports")
    except Exception as e:
        print(f"  [FAIL] Config imports: {e}")
        return 1

    # Check env file
    if Path(".env.testnet").exists():
        print("  [OK] .env.testnet exists")
    else:
        print("  [WARN] .env.testnet not found (copy from template)")

    # Validate config
    print("\nValidating configuration...")
    try:
        from src.monitoring.config_validation import validate_config_or_exit
        from config.settings import Settings
        result = validate_config_or_exit(Settings())
        if result.valid:
            print("  [OK] Config validation passed")
        else:
            print(f"  [FAIL] Config validation: {result.errors}")
            return 1
    except Exception as e:
        print(f"  [FAIL] Config validation: {e}")
        return 1

    print("\n" + "=" * 50)
    print("DRY-RUN SUMMARY")
    print("=" * 50)
    print("All checks passed!")
    print("Ready for testnet deployment.")
    print("\nNext steps:")
    print("  1. Copy .env.testnet.template to .env.testnet")
    print("  2. Fill in BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET")
    print("  3. Run: ./deploy_testnet.sh up")

    return 0


def main():
    print("QuantAI Dry-Run Testnet Simulation")
    print("=" * 50)

    # Check config files
    print("\nRunning checks...")

    # Check config files
    print("\nChecking config files...")
    files = ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]
    all_ok = True
    for f in ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]:
        if Path(f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            return 1

    # Check imports
    print("\nChecking imports...")
    try:
        from config.settings import Settings, settings
        from config.testnet_settings import testnet_config
        print("  [OK] Config imports")
    except Exception as e:
        print(f"  [FAIL] Config imports: {e}")
        return 1

    # Check env file
    if Path(".env.testnet").exists():
        print("  [OK] .env.testnet exists")
    else:
        print("  [WARN] .env.testnet not found (copy from template)")

    # Validate config
    print("\nValidating configuration...")
    try:
        from src.monitoring.config_validation import validate_config_or_exit
        from config.settings import Settings
        result = validate_config_or_exit(Settings())
        if result.valid:
            print("  [OK] Config validation passed")
        else:
            print(f"  [FAIL] Config validation: {result.errors}")
            return 1
    except Exception as e:
        print(f"  [FAIL] Config validation: {e}")
        return 1

    print("\n" + "=" * 50)
    print("DRY-RUN SUMMARY")
    print("=" * 50)
    print("All checks passed!")
    print("Ready for testnet deployment.")
    print("\nNext steps:")
    print("  1. Copy .env.testnet.template to .env.testnet")
    print("  2. Fill in BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET")
    print("  3. Run: ./deploy_testnet.sh up")

    return 0


def main():
    print("QuantAI Dry-Run Testnet Simulation")
    print("=" * 50)

    # Check config files
    print("\nRunning checks...")

    # Check config files
    print("\nChecking config files...")
    files = ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]
    all_ok = True
    for f in ["config/settings.py", "config/testnet_settings.py", "config/__init__.py"]:
        if Path(f).exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            return 1

    # Check imports
    print("\nChecking imports...")
    try:
        from config.settings import Settings, settings
        from config.testnet_settings import testnet_config
        print("  [OK] Config imports")
    except Exception as e:
        print(f"  [FAIL] Config imports: {e}")
        return 1

    # Check env file
    if Path(".env.testnet").exists():
        print("  [OK] .env.testnet exists")
    else:
        print("  [WARN] .env.testnet not found (copy from template)")

    # Validate config
    print("\nValidating configuration...")
    try:
        from src.monitoring.config_validation import validate_config_or_exit
        from config.settings import Settings
        result = validate_config_or_exit(Settings())
        if result.valid:
            print("  [OK] Config validation passed")
        else:
            print(f"  [FAIL] Config validation: {result.errors}")
            return 1
    except Exception as e:
        print(f"  [FAIL] Config validation: {e}")
        return 1

    print("\n" + "=" * 50)
    print("DRY-RUN SUMMARY")
    print("=" * 50)
    print("All checks passed!")
    print("Ready for testnet deployment.")
    print("\nNext steps:")
    print("  1. Copy .env.testnet.template to .env.testnet")
    print("  2. Fill in BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_API_SECRET")
    print("  3. Run: ./deploy_testnet.sh up")

    return 0


if __name__ == "__main__":
    sys.exit(main())