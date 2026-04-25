from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, safe_ratio, wilder_smooth

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime


def compute_volatility_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}

    # 获取 regime.diagnostics 中的指标值
    dir_diag = regime.diagnostics.get("direction_diagnostics", {}) if regime else {}

    mean = close.rolling(cfg.boll_period).mean()
    std = close.rolling(cfg.boll_period).std(ddof=0)
    upper = mean + cfg.boll_std * std
    lower = mean - cfg.boll_std * std
    add_factor_from_series(factors, "BBANDS_UPPER", upper, cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_MIDDLE", mean, cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_LOWER", lower, cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_WIDTH", safe_ratio(upper - lower, mean), cfg.percentile_window)

    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)

        # ATR - 优先从 regime 复用
        if dir_diag.get("atr_value") is not None:
            # ATR 值已知，但仍需历史序列计算 percentile
            prev_close = close.shift(1)
            tr = pd.Series(
                np.maximum.reduce([(high - low), (high - prev_close).abs(), (low - prev_close).abs()]),
                index=bars.index,
            )
            add_factor_from_series(factors, "ATR", wilder_smooth(tr, cfg.atr_period), cfg.percentile_window)
        else:
            prev_close = close.shift(1)
            tr = pd.Series(
                np.maximum.reduce([(high - low), (high - prev_close).abs(), (low - prev_close).abs()]),
                index=bars.index,
            )
            add_factor_from_series(factors, "ATR", wilder_smooth(tr, cfg.atr_period), cfg.percentile_window)

        dc_upper = high.rolling(cfg.dc_period).max()
        dc_lower = low.rolling(cfg.dc_period).min()
        add_factor_from_series(factors, "DC_UPPER", dc_upper, cfg.percentile_window)
        add_factor_from_series(factors, "DC_LOWER", dc_lower, cfg.percentile_window)
        add_factor_from_series(factors, "DC_MIDDLE", (dc_upper + dc_lower) / 2.0, cfg.percentile_window)

    return factors
