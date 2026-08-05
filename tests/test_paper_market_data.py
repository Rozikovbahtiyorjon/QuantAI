"""
=========================================================
QuantAI Professional v5
Paper Market Data Tests
=========================================================
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.paper_market_data import PaperMarketData


# =========================================================
# HELPERS
# =========================================================

def make_dataframe(rows: int = 5) -> pd.DataFrame:

    return pd.DataFrame(
        {
            "timestamp": [
                f"2026-01-{i + 1:02d}"
                for i in range(rows)
            ],
            "open": [
                100.0 + i
                for i in range(rows)
            ],
            "high": [
                101.0 + i
                for i in range(rows)
            ],
            "low": [
                99.0 + i
                for i in range(rows)
            ],
            "close": [
                100.0 + i
                for i in range(rows)
            ],
            "volume": [1000.0] * rows,
        }
    )


# =========================================================
# 1. INITIALIZATION
# =========================================================

def test_initialization():

    provider = PaperMarketData(
        make_dataframe()
    )

    assert provider.position == 0
    assert provider.total_rows == 5
    assert provider.finished is False


# =========================================================
# 2. INVALID DATA TYPE
# =========================================================

def test_invalid_data_type():

    with pytest.raises(TypeError):

        PaperMarketData(
            [1, 2, 3]
        )


# =========================================================
# 3. EMPTY DATA
# =========================================================

def test_empty_dataframe():

    with pytest.raises(ValueError):

        PaperMarketData(
            pd.DataFrame()
        )


# =========================================================
# 4. NEXT ROW
# =========================================================

def test_next_returns_first_row():

    provider = PaperMarketData(
        make_dataframe()
    )

    row = provider.next()

    assert row["close"] == 100.0
    assert provider.position == 1


# =========================================================
# 5. SEQUENTIAL ROWS
# =========================================================

def test_next_returns_rows_sequentially():

    provider = PaperMarketData(
        make_dataframe()
    )

    first = provider.next()
    second = provider.next()
    third = provider.next()

    assert first["close"] == 100.0
    assert second["close"] == 101.0
    assert third["close"] == 102.0

    assert provider.position == 3


# =========================================================
# 6. FINISHED STATE
# =========================================================

def test_finished_after_all_rows():

    provider = PaperMarketData(
        make_dataframe(3)
    )

    provider.next()
    provider.next()
    provider.next()

    assert provider.finished is True
    assert provider.position == 3


# =========================================================
# 7. STOP ITERATION
# =========================================================

def test_next_after_end_raises_stop_iteration():

    provider = PaperMarketData(
        make_dataframe(2)
    )

    provider.next()
    provider.next()

    with pytest.raises(StopIteration):

        provider.next()


# =========================================================
# 8. ITERATOR
# =========================================================

def test_iterator_returns_all_rows():

    provider = PaperMarketData(
        make_dataframe(5)
    )

    rows = list(provider)

    assert len(rows) == 5

    assert rows[0]["close"] == 100.0
    assert rows[1]["close"] == 101.0
    assert rows[2]["close"] == 102.0
    assert rows[3]["close"] == 103.0
    assert rows[4]["close"] == 104.0


# =========================================================
# 9. ITERATOR FINISHES
# =========================================================

def test_iterator_finishes():

    provider = PaperMarketData(
        make_dataframe(3)
    )

    list(provider)

    assert provider.finished is True


# =========================================================
# 10. RESET
# =========================================================

def test_reset():

    provider = PaperMarketData(
        make_dataframe(5)
    )

    provider.next()
    provider.next()

    assert provider.position == 2

    provider.reset()

    assert provider.position == 0
    assert provider.finished is False

    row = provider.next()

    assert row["close"] == 100.0


# =========================================================
# 11. DATAFRAME IS COPIED
# =========================================================

def test_input_dataframe_is_not_used_directly():

    data = make_dataframe()

    provider = PaperMarketData(data)

    data.loc[0, "close"] = 9999.0

    row = provider.next()

    assert row["close"] == 100.0


# =========================================================
# 12. INDEX IS RESET
# =========================================================

def test_index_is_reset():

    data = make_dataframe()

    data.index = [
        10,
        20,
        30,
        40,
        50,
    ]

    provider = PaperMarketData(data)

    row = provider.next()

    assert row.name == 0


# =========================================================
# 13. TOTAL ROWS
# =========================================================

def test_total_rows():

    provider = PaperMarketData(
        make_dataframe(17)
    )

    assert provider.total_rows == 17


# =========================================================
# 14. MULTIPLE ITERATIONS AFTER RESET
# =========================================================

def test_multiple_iterations_after_reset():

    provider = PaperMarketData(
        make_dataframe(3)
    )

    first_run = list(provider)

    provider.reset()

    second_run = list(provider)

    assert len(first_run) == 3
    assert len(second_run) == 3

    assert (
        first_run[0]["close"]
        == second_run[0]["close"]
    )

    assert (
        first_run[-1]["close"]
        == second_run[-1]["close"]
    )