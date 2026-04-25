from __future__ import annotations

import pandas as pd

from skyeye.data.facade import DataFacade
from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.indicators.momentum import compute_momentum_factors
from skyeye.factor_layer.indicators.oscillator import compute_oscillator_factors
from skyeye.factor_layer.indicators.trend import compute_trend_factors
from skyeye.factor_layer.indicators.volatility import compute_volatility_factors
from skyeye.factor_layer.indicators.volume import compute_volume_factors
from skyeye.factor_layer.result import FactorLayerResult
from skyeye.factor_layer.utils import normalize_benchmark_bars
from skyeye.market_regime_layer import (
    MarketRegime,
    compute_market_regime_from_data_facade,
    normalize_single_instrument_bars,
)

REGIME_FACTOR_CATEGORIES = {
    "bull_co_move": ("trend", "volume"),
    "bull_rotation": ("momentum", "trend"),
    "range_co_move": ("oscillator", "volatility"),
    "range_rotation": ("momentum", "oscillator"),
    "bear_co_move": ("volatility", "volume"),
    "bear_rotation": ("momentum", "volatility"),
}

CATEGORY_COMPUTERS = {
    "trend": ("trend_factors", compute_trend_factors),
    "momentum": ("momentum_factors", compute_momentum_factors),
    "oscillator": ("oscillator_factors", compute_oscillator_factors),
    "volatility": ("volatility_factors", compute_volatility_factors),
    "volume": ("volume_factors", compute_volume_factors),
}


def _select_factor_categories(regime_label: str) -> tuple[str, str]:
    return REGIME_FACTOR_CATEGORIES[str(regime_label)]


def _empty_result(regime: MarketRegime) -> FactorLayerResult:
    return FactorLayerResult(regime=regime)


def compute_factors(benchmark_bars, regime: MarketRegime, cfg: FactorLayerConfig | None = None) -> FactorLayerResult:
    cfg = cfg or FactorLayerConfig()
    bars = normalize_benchmark_bars(benchmark_bars)
    if bars.empty or "close" not in bars.columns:
        return _empty_result(regime)

    payload = {
        "trend_factors": {},
        "momentum_factors": {},
        "oscillator_factors": {},
        "volatility_factors": {},
        "volume_factors": {},
    }
    for category in _select_factor_categories(regime.regime):
        field_name, computer = CATEGORY_COMPUTERS[category]
        payload[field_name] = computer(bars, cfg, regime=regime)  # 传递 regime 参数
    return FactorLayerResult(regime=regime, **payload)


def compute_factors_from_data_facade(
    end_date: str | pd.Timestamp,
    benchmark_id: str = "000300.XSHG",
    regime: MarketRegime | None = None,
    cfg: FactorLayerConfig | None = None,
) -> FactorLayerResult:
    end_ts = pd.to_datetime(end_date)
    start_ts = end_ts - pd.Timedelta(days=1200)
    data = DataFacade()
    raw = data.get_daily_bars(
        benchmark_id,
        start_ts.strftime("%Y-%m-%d"),
        end_ts.strftime("%Y-%m-%d"),
        fields=["open", "high", "low", "close", "volume"],
        adjust_type="none",
    )
    if raw is None or getattr(raw, "empty", True):
        resolved_regime = regime or MarketRegime(
            regime="range_co_move",
            strength=0.0,
            diagnostics={"reason": "benchmark_bars_empty"},
        )
        return _empty_result(resolved_regime)

    bars = normalize_single_instrument_bars(raw, benchmark_id)
    resolved_regime = regime or compute_market_regime_from_data_facade(
        end_date=end_date,
        benchmark_id=benchmark_id,
    )
    return compute_factors(bars, resolved_regime, cfg=cfg)
