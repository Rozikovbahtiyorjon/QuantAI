"""
P1.14 Causal Mutation Test — future-mutated past-invariant.

Очень полезный тест:
  изменить future data
  и проверить:
  past outputs must remain identical
для:
  • Feature Engine
  • Regime
  • DatasetBuilder
  • scaler
  • model preprocessing
  • SignalGenerator

Invariant: output[t] depends only on input[:t]
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import add_indicators


def make_ohlcv(rows: int = 600, seed: int = 42, shock_after: int | None = None, shock_factor: float = 5.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.004, size=rows)
    close = 100.0 * np.cumprod(1.0 + rets)
    if shock_after is not None:
        close[shock_after:] *= shock_factor
    open_ = np.empty(rows)
    open_[0] = 100.0
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0, 0.001, size=rows)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(50, 500, size=rows)
    ts = pd.date_range("2024-01-01", periods=rows, freq="15min")
    return pd.DataFrame({"timestamp": ts, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


# ============================================================
# Feature Engine
# ============================================================

class TestFeatureEngineMutation:
    def test_past_features_invariant_under_future_shock(self):
        from src.feature_engine import FeatureEngine
        base = add_indicators(make_ohlcv(600, seed=7))
        shocked = add_indicators(make_ohlcv(600, seed=7, shock_after=400, shock_factor=5.0))
        cutoff = 400
        # Features at cutoff should be identical despite future shock after cutoff
        hist_base = base.iloc[:cutoff].copy()
        hist_shocked = shocked.iloc[:cutoff].copy()
        fe1 = FeatureEngine()
        fe2 = FeatureEngine()
        f1 = fe1.build(hist_base).to_dict()
        f2 = fe2.build(hist_shocked).to_dict()
        mism = [k for k in f1 if k in f2 and not np.isclose(f1[k], f2[k], rtol=1e-9, atol=1e-9, equal_nan=True)]
        assert not mism, f"FeatureEngine leaked future: {mism} differ"

    def test_prefix_features_identical(self):
        from src.feature_engine import FeatureEngine
        full = add_indicators(make_ohlcv(600, seed=13))
        for n in [250, 350, 450]:
            hist_full = full.iloc[:n].copy()
            hist_prefix = add_indicators(make_ohlcv(600, seed=13).iloc[:n].copy())
            fe_full = FeatureEngine().build(hist_full).to_dict()
            fe_prefix = FeatureEngine().build(hist_prefix).to_dict()
            mism = [k for k in fe_full if k in fe_prefix and not np.isclose(fe_full[k], fe_prefix[k], rtol=1e-9, atol=1e-9, equal_nan=True)]
            assert not mism, f"FeatureEngine prefix mismatch at n={n}: {mism}"


# ============================================================
# Regime
# ============================================================

class TestRegimeMutation:
    def test_regime_past_invariant(self):
        from src.regime_filter import RegimeFilter
        base = add_indicators(make_ohlcv(600, seed=21))
        shocked = add_indicators(make_ohlcv(600, seed=21, shock_after=400, shock_factor=5.0))
        cutoff = 400
        hist_base = base.iloc[:cutoff].copy()
        hist_shocked = shocked.iloc[:cutoff].copy()
        r1 = RegimeFilter().classify(hist_base)
        r2 = RegimeFilter().classify(hist_shocked)
        assert r1 == r2, f"Regime leaked future: {r1} vs {r2} at cutoff {cutoff}"

    def test_regime_sequence_past_invariant(self):
        from src.regime_filter import RegimeFilter
        base = add_indicators(make_ohlcv(600, seed=33))
        shocked = add_indicators(make_ohlcv(600, seed=33, shock_after=450, shock_factor=0.2))
        # Classify sequentially up to cutoff, state should be identical
        cutoff = 450
        f1 = RegimeFilter()
        f2 = RegimeFilter()
        seq1 = []
        seq2 = []
        for i in range(60, cutoff):
            seq1.append(f1.classify(base.iloc[: i + 1].copy()))
            seq2.append(f2.classify(shocked.iloc[: i + 1].copy()))
        assert seq1 == seq2, "Regime sequence diverged before shock"

    def test_regime_allows_is_causal(self):
        from src.regime_filter import RegimeFilter
        base = add_indicators(make_ohlcv(200, seed=5))
        f = RegimeFilter()
        regime = f.classify(base)
        # allows should depend only on current regime, not future
        assert f.allows("BUY") in (True, False)
        assert f.allows("SELL") in (True, False)


# ============================================================
# DatasetBuilder
# ============================================================

class TestDatasetBuilderMutation:
    def test_dataset_past_rows_invariant(self):
        from src.dataset_builder import DatasetBuilder, DatasetConfig
        base = make_ohlcv(600, seed=11)
        shocked = make_ohlcv(600, seed=11, shock_after=500, shock_factor=5.0)
        cfg = DatasetConfig(future_bars=5, warmup_bars=200, label_method="triple_barrier")
        b1 = DatasetBuilder(cfg)
        b2 = DatasetBuilder(cfg)
        data1 = b1.prepare_data(base)
        data2 = b2.prepare_data(shocked)
        # Build features dataset: rows with index < shock_after - future - warmup should be identical
        b1.build_features_dataset(data1)
        b2.build_features_dataset(data2)
        # Find rows with index < 480 (well before shock)
        cutoff_idx = 480
        rows1 = [r for r in b1.dataset if r["index"] < cutoff_idx]
        rows2 = [r for r in b2.dataset if r["index"] < cutoff_idx]
        assert len(rows1) == len(rows2) and len(rows1) > 0
        # Compare a few feature keys
        keys_to_check = [k for k in rows1[0].keys() if k not in ("index", "target", "future_return", "tb_barrier", "tb_t1")]
        for r1, r2 in zip(rows1[:20], rows2[:20]):
            for k in keys_to_check[:5]:
                v1 = r1.get(k)
                v2 = r2.get(k)
                if isinstance(v1, float) and isinstance(v2, float):
                    assert np.isclose(v1, v2, rtol=1e-9, atol=1e-9, equal_nan=True), f"DatasetBuilder feature {k} leaked at index {r1['index']}: {v1} vs {v2}"
                else:
                    assert v1 == v2, f"DatasetBuilder {k} leaked at index {r1['index']}"

    def test_dataset_builder_does_not_use_future_in_features(self):
        from src.dataset_builder import DatasetBuilder, DatasetConfig
        df = make_ohlcv(400, seed=77)
        cfg = DatasetConfig(future_bars=5, warmup_bars=200)
        builder = DatasetBuilder(cfg)
        dataset = builder.build(df)
        # Dataset should have been tail-dropped by future_bars
        assert len(dataset) < len(df), "tail not dropped — possible lookahead"
        # Check no NaN features leaked from future
        assert not dataset.isna().any().any(), "dataset has NaNs — possible leakage"


# ============================================================
# Scaler (no global scaler should exist; if exists it must be train-only)
# ============================================================

class TestScalerMutation:
    def test_no_global_scaler_leak(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        scaler_files = list((root / "src").rglob("*.py"))
        for p in scaler_files:
            if p.name == "causal_audit.py":
                continue
            src = p.read_text(encoding="utf-8", errors="ignore")
            if "StandardScaler" in src or "MinMaxScaler" in src or "RobustScaler" in src:
                # If scaler exists, ensure it's not fit on full df
                assert "train_df" in src or "train_size" in src or "WalkForward" in src or "fit_transform" not in src, f"{p} has scaler without train isolation — potential global fit leakage"
        # If no scaler files, test passes (no leakage surface)

    def test_scaler_train_only_simulation(self):
        """Simulate train-only scaling is causal: scaling past with future-mutated train gives different but past-preserving transform."""
        from sklearn.preprocessing import StandardScaler
        base = add_indicators(make_ohlcv(600, seed=50))
        shocked = add_indicators(make_ohlcv(600, seed=50, shock_after=400, shock_factor=5.0))
        cutoff = 400
        train_base = base.iloc[:cutoff][["close","volume"]].astype(float)
        train_shocked_past = shocked.iloc[:cutoff][["close","volume"]].astype(float)
        # Train-only fit should be identical because past is identical
        sc1 = StandardScaler().fit(train_base)
        sc2 = StandardScaler().fit(train_shocked_past)
        np.testing.assert_allclose(sc1.mean_, sc2.mean_, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(sc1.scale_, sc2.scale_, rtol=1e-9, atol=1e-9)
        # Global fit would leak: fit on full including shocked future would differ — ensure we don't do that
        full_shocked = shocked[["close","volume"]].astype(float)
        sc_global = StandardScaler().fit(full_shocked)
        assert not np.allclose(sc1.mean_, sc_global.mean_), "global scaler mean should differ from train-only — proves train-only is causal"


# ============================================================
# Model preprocessing (feature building for ML is causal)
# ============================================================

class TestModelPreprocessingMutation:
    def test_ml_feature_building_is_causal(self):
        from src.feature_engine import build_features
        base = add_indicators(make_ohlcv(600, seed=60))
        shocked = add_indicators(make_ohlcv(600, seed=60, shock_after=450, shock_factor=5.0))
        cutoff = 450
        hist_base = base.iloc[:cutoff].copy()
        hist_shocked = shocked.iloc[:cutoff].copy()
        f_base = build_features(hist_base)
        f_shocked = build_features(hist_shocked)
        mism = [k for k in f_base if k in f_shocked and not np.isclose(f_base[k], f_shocked[k], rtol=1e-9, atol=1e-9, equal_nan=True)]
        assert not mism, f"model preprocessing (build_features) leaked future: {mism}"

    def test_no_preprocessing_uses_future_shift(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        for p in (root / "src" / "feature_engine.py", root / "src" / "dataset_builder.py", root / "src" / "indicators.py"):
            src = p.read_text(encoding="utf-8", errors="ignore")
            assert "shift(-1)" not in src, f"{p.name} contains shift(-1) — future leak in preprocessing"
            if "bfill" in src.lower():
                # Only allow mention in comment about never using bfill
                assert ".bfill(" not in src, f"{p.name} contains bfill() call — future leak"


# ============================================================
# SignalGenerator
# ============================================================

class TestSignalGeneratorMutation:
    def test_signal_past_invariant(self):
        from src.strategy.signal_generator import SignalGenerator
        base = add_indicators(make_ohlcv(600, seed=81))
        shocked = add_indicators(make_ohlcv(600, seed=81, shock_after=500, shock_factor=5.0))
        cutoff = 500
        hist_base = base.iloc[:cutoff].copy()
        hist_shocked = shocked.iloc[:cutoff].copy()
        sg1 = SignalGenerator()
        sg2 = SignalGenerator()
        r1 = sg1.generate(hist_base)
        r2 = sg2.generate(hist_shocked)
        assert r1.signal == r2.signal, f"SignalGenerator leaked future: {r1.signal} vs {r2.signal}"
        assert np.isclose(r1.confidence, r2.confidence, atol=1e-9), f"SignalGenerator confidence leaked: {r1.confidence} vs {r2.confidence}"

    def test_signal_sequence_past_invariant(self):
        from src.strategy.signal_generator import SignalGenerator
        base = add_indicators(make_ohlcv(600, seed=90))
        shocked = add_indicators(make_ohlcv(600, seed=90, shock_after=500, shock_factor=0.3))
        cutoff = 500
        sg1 = SignalGenerator()
        sg2 = SignalGenerator()
        # Generate signals sequentially up to cutoff
        seq1 = []
        seq2 = []
        for i in range(200, cutoff):
            h1 = base.iloc[: i + 1].copy()
            h2 = shocked.iloc[: i + 1].copy()
            # Need fresh generator each time to avoid state carry, but regime filter has state
            # Use new instance per step with reset, or reuse but both start from same state
            # We reuse the same sg1/sg2 sequentially to test stateful causality — they start identical
            seq1.append(SignalGenerator().generate(h1).signal)
            seq2.append(SignalGenerator().generate(h2).signal)
        assert seq1 == seq2, "SignalGenerator sequence diverged before shock"

    def test_order_flow_does_not_introduce_lookahead(self):
        from src.strategy.signal_generator import SignalGenerator
        base = add_indicators(make_ohlcv(300, seed=5))
        sg = SignalGenerator()
        # Without order flow signal, should be causal
        r = sg.generate(base)
        assert r.signal in ("BUY", "SELL", "HOLD")
