from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

# pytest>=9 默认 importlib 导入模式下，某些环境不会自动把 repo root 加入 sys.path。
# 这里显式注入，保证 `skyeye` 可被稳定导入。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from skyeye.market_regime_layer import MarketRegime
from skyeye.factor_layer.core import (
    _empty_result,
    _select_factor_categories,
    compute_factors,
    compute_factors_from_data_facade,
)
from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.indicators.momentum import compute_momentum_factors
from skyeye.factor_layer.indicators.volatility import compute_volatility_factors
from skyeye.factor_layer.indicators.volume import compute_volume_factors
from skyeye.factor_layer.utils import build_factor_value


def test_select_factor_categories_matches_design_doc():
    assert _select_factor_categories("bull_co_move") == ("trend", "volume")
    assert _select_factor_categories("bull_rotation") == ("momentum", "trend")
    assert _select_factor_categories("range_co_move") == ("oscillator", "volatility")
    assert _select_factor_categories("range_rotation") == ("momentum", "oscillator")
    assert _select_factor_categories("bear_co_move") == ("volatility", "volume")
    assert _select_factor_categories("bear_rotation") == ("momentum", "volatility")


def test_empty_result_preserves_regime_and_empty_buckets():
    regime = MarketRegime(regime="range_co_move", strength=0.4)
    result = _empty_result(regime)

    assert result.regime is regime
    assert result.trend_factors == {}
    assert result.momentum_factors == {}
    assert result.oscillator_factors == {}
    assert result.volatility_factors == {}
    assert result.volume_factors == {}


def test_build_factor_value_returns_latest_value_and_percentile():
    history = pd.Series(range(1, 21), index=pd.date_range("2024-01-01", periods=20, freq="B"), dtype=float)
    payload = build_factor_value(history, window=20)

    assert payload["value"] == 20.0
    assert 0.0 <= payload["percentile"] <= 1.0


def test_build_factor_value_returns_nan_percentile_when_history_short():
    history = pd.Series([1.0, 2.0, 3.0, 4.0], index=pd.date_range("2024-01-01", periods=4, freq="B"))
    payload = build_factor_value(history, window=20)

    assert payload["value"] == 4.0
    assert math.isnan(payload["percentile"])


def test_compute_momentum_factors_returns_mom_roc_bias():
    bars = pd.DataFrame(
        {"close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111]},
        index=pd.date_range("2024-01-01", periods=12, freq="B"),
    )
    factors = compute_momentum_factors(
        bars,
        FactorLayerConfig(mom_period=3, roc_period=3, bias_period=3, percentile_window=8),
    )

    assert set(factors) == {"MOM", "ROC", "BIAS"}
    assert factors["MOM"]["value"] > 0
    assert factors["ROC"]["value"] > 0
    assert not math.isnan(factors["BIAS"]["value"])


def test_volume_factors_return_empty_when_volume_missing():
    bars = pd.DataFrame({"close": [1, 2, 3, 4]}, index=pd.date_range("2024-01-01", periods=4, freq="B"))
    assert compute_volume_factors(bars, FactorLayerConfig()) == {}


def test_volatility_factors_still_return_bbands_without_high_low():
    bars = pd.DataFrame({"close": range(1, 40)}, index=pd.date_range("2024-01-01", periods=39, freq="B"))
    factors = compute_volatility_factors(bars, FactorLayerConfig(percentile_window=20))
    assert "BBANDS_WIDTH" in factors
    assert "ATR" not in factors


def test_compute_factors_only_populates_categories_for_regime():
    bars = pd.DataFrame(
        {
            "open": range(1, 80),
            "high": [x + 1 for x in range(1, 80)],
            "low": [x - 1 for x in range(1, 80)],
            "close": range(1, 80),
            "volume": [1000] * 79,
        },
        index=pd.date_range("2024-01-01", periods=79, freq="B"),
    )
    regime = MarketRegime(regime="bull_co_move", strength=0.8)

    result = compute_factors(bars, regime)

    assert result.trend_factors
    assert result.volume_factors
    assert result.momentum_factors == {}
    assert result.oscillator_factors == {}
    assert result.volatility_factors == {}


def test_compute_factors_from_data_facade_reuses_market_regime_when_missing(monkeypatch):
    calls = {"market": 0, "data": 0}

    class DummyFacade:
        def get_daily_bars(self, *args, **kwargs):
            calls["data"] += 1
            return pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=80, freq="B"),
                    "order_book_id": ["000300.XSHG"] * 80,
                    "open": range(1, 81),
                    "high": [x + 1 for x in range(1, 81)],
                    "low": [x - 1 for x in range(1, 81)],
                    "close": range(1, 81),
                    "volume": [1000] * 80,
                }
            ).set_index("date")

    def _fake_market_regime_from_data_facade(**kwargs):
        calls["market"] += 1
        return MarketRegime(regime="bull_co_move", strength=0.7)

    monkeypatch.setattr("skyeye.factor_layer.core.DataFacade", DummyFacade)
    monkeypatch.setattr("skyeye.factor_layer.core.compute_market_regime_from_data_facade", _fake_market_regime_from_data_facade)

    result = compute_factors_from_data_facade(end_date="2024-05-31")

    assert calls["data"] == 1
    assert calls["market"] == 1
    assert result.regime.regime == "bull_co_move"


def test_compute_factors_percentiles_are_nan_or_between_zero_and_one():
    bars = pd.DataFrame(
        {
            "open": range(1, 320),
            "high": [x + 1 for x in range(1, 320)],
            "low": [x - 1 for x in range(1, 320)],
            "close": range(1, 320),
            "volume": [1000] * 319,
        },
        index=pd.date_range("2023-01-02", periods=319, freq="B"),
    )
    result = compute_factors(bars, MarketRegime(regime="bear_rotation", strength=0.9))

    for bucket in (result.momentum_factors, result.volatility_factors):
        for payload in bucket.values():
            percentile = payload["percentile"]
            assert math.isnan(percentile) or 0.0 <= percentile <= 1.0
