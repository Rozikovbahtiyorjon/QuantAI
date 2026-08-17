import numpy as np
import pandas as pd
import pytest

from src.open_interest_divergence_engine import (
    OpenInterestDivergenceEngine,
    OpenInterestDivergenceResult,
)


def make_data(
    close=None,
    open_interest=None,
    size=20,
):
    if close is None:
        close = np.linspace(100.0, 110.0, size)

    if open_interest is None:
        open_interest = np.linspace(1000.0, 1100.0, size)

    return pd.DataFrame(
        {
            "close": close,
            "open_interest": open_interest,
        }
    )


def test_default_initialization():
    engine = OpenInterestDivergenceEngine()

    assert engine.lookback == 5
    assert engine.zscore_window == 50
    assert engine.min_divergence == pytest.approx(0.10)
    assert engine.min_confidence == pytest.approx(0.25)


def test_invalid_lookback_is_rejected():
    with pytest.raises(ValueError):
        OpenInterestDivergenceEngine(lookback=0)


def test_invalid_zscore_window_is_rejected():
    with pytest.raises(ValueError):
        OpenInterestDivergenceEngine(zscore_window=1)


def test_negative_divergence_threshold_is_rejected():
    with pytest.raises(ValueError):
        OpenInterestDivergenceEngine(min_divergence=-0.1)


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        OpenInterestDivergenceEngine(min_confidence=1.1)


def test_transform_returns_expected_columns():
    engine = OpenInterestDivergenceEngine()

    result = engine.transform(make_data())

    expected = {
        "price_change",
        "open_interest_change",
        "price_zscore",
        "open_interest_zscore",
        "divergence_score",
        "divergence_strength",
        "divergence_confidence",
        "divergence_signal",
    }

    assert expected.issubset(result.columns)


def test_transform_preserves_row_count():
    engine = OpenInterestDivergenceEngine()

    data = make_data(size=31)
    result = engine.transform(data)

    assert len(result) == len(data)


def test_equal_length_is_required_for_series_calculation():
    engine = OpenInterestDivergenceEngine()

    price = pd.Series([100.0, 101.0, 102.0])
    oi = pd.Series([1000.0, 1010.0])

    with pytest.raises(ValueError):
        engine.calculate_divergence(price, oi)


def test_non_series_price_is_rejected():
    engine = OpenInterestDivergenceEngine()

    with pytest.raises(TypeError):
        engine.calculate_divergence(
            [100.0, 101.0],
            pd.Series([1000.0, 1010.0]),
        )


def test_missing_column_is_rejected():
    engine = OpenInterestDivergenceEngine()

    data = pd.DataFrame({"close": [100.0, 101.0]})

    with pytest.raises(ValueError):
        engine.transform(data)


def test_non_numeric_column_is_rejected():
    engine = OpenInterestDivergenceEngine()

    data = pd.DataFrame(
        {
            "close": [100.0, 101.0],
            "open_interest": ["1000", "1010"],
        }
    )

    with pytest.raises(TypeError):
        engine.transform(data)


def test_nan_values_are_rejected():
    engine = OpenInterestDivergenceEngine()

    data = make_data()
    data.loc[5, "open_interest"] = np.nan

    with pytest.raises(ValueError):
        engine.transform(data)


def test_non_positive_price_is_rejected():
    engine = OpenInterestDivergenceEngine()

    data = make_data()
    data.loc[3, "close"] = 0.0

    with pytest.raises(ValueError):
        engine.transform(data)


def test_negative_open_interest_is_rejected():
    engine = OpenInterestDivergenceEngine()

    data = make_data()
    data.loc[3, "open_interest"] = -1.0

    with pytest.raises(ValueError):
        engine.transform(data)


def test_result_type_is_correct():
    engine = OpenInterestDivergenceEngine()

    result = engine.evaluate(make_data())

    assert isinstance(result, OpenInterestDivergenceResult)


def test_signal_is_one_of_supported_values():
    engine = OpenInterestDivergenceEngine()

    result = engine.evaluate(make_data())

    assert result.signal in {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }


def test_bullish_divergence_is_detected():
    close = [
        100.0,
        102.0,
        101.0,
        99.0,
        98.0,
        97.0,
        96.0,
        95.0,
        94.0,
        93.0,
        92.0,
        91.0,
        90.0,
        89.0,
        88.0,
        87.0,
        86.0,
        85.0,
        84.0,
        80.0,
    ]

    open_interest = [
        1000.0,
        1005.0,
        1008.0,
        1010.0,
        1015.0,
        1020.0,
        1025.0,
        1030.0,
        1035.0,
        1040.0,
        1045.0,
        1050.0,
        1055.0,
        1060.0,
        1065.0,
        1070.0,
        1075.0,
        1080.0,
        1090.0,
        1200.0,
    ]

    engine = OpenInterestDivergenceEngine(
        lookback=1,
        zscore_window=5,
        min_divergence=0.0,
        min_confidence=0.0,
    )

    result = engine.evaluate(make_data(close, open_interest))

    assert result.signal == "BULLISH"
    assert result.open_interest_change > 0
    assert result.price_change < 0


def test_bearish_divergence_is_detected():
    close = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
        125.0,
    ]

    open_interest = [
        1200.0,
        1195.0,
        1190.0,
        1185.0,
        1180.0,
        1175.0,
        1170.0,
        1165.0,
        1160.0,
        1155.0,
        1150.0,
        1145.0,
        1140.0,
        1135.0,
        1130.0,
        1125.0,
        1120.0,
        1115.0,
        1110.0,
        900.0,
    ]

    engine = OpenInterestDivergenceEngine(
        lookback=1,
        zscore_window=5,
        min_divergence=0.0,
        min_confidence=0.0,
    )

    result = engine.evaluate(make_data(close, open_interest))

    assert result.signal == "BEARISH"
    assert result.price_change > 0
    assert result.open_interest_change < 0


def test_neutral_signal_when_price_and_oi_move_together():
    close = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
        105.0,
        106.0,
        107.0,
        108.0,
        109.0,
    ]

    open_interest = [
        1000.0,
        1010.0,
        1020.0,
        1030.0,
        1040.0,
        1050.0,
        1060.0,
        1070.0,
        1080.0,
        1090.0,
    ]

    engine = OpenInterestDivergenceEngine(
        lookback=1,
        zscore_window=5,
        min_divergence=0.10,
        min_confidence=0.25,
    )

    result = engine.evaluate(make_data(close, open_interest))

    assert result.signal == "NEUTRAL"


def test_compare_returns_boolean():
    engine = OpenInterestDivergenceEngine()

    result = engine.compare(make_data(), "NEUTRAL")

    assert isinstance(result, bool)


def test_compare_is_case_insensitive():
    engine = OpenInterestDivergenceEngine()

    data = make_data()

    signal = engine.signal(data)

    assert engine.compare(data, signal.lower()) is True


def test_summarize_returns_latest_signal():
    engine = OpenInterestDivergenceEngine()

    data = engine.transform(make_data())
    summary = engine.summarize(data)

    assert set(summary) == {
        "signal",
        "strength",
        "confidence",
    }

    assert summary["signal"] in {
        "BULLISH",
        "BEARISH",
        "NEUTRAL",
    }


def test_summarize_requires_divergence_columns():
    with pytest.raises(ValueError):
        OpenInterestDivergenceEngine.summarize(
            pd.DataFrame({"close": [100.0]})
        )


def test_signal_returns_string():
    engine = OpenInterestDivergenceEngine()

    signal = engine.signal(make_data())

    assert isinstance(signal, str)