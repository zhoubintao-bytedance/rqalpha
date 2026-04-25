from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, wilder_smooth

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime


def compute_oscillator_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = wilder_smooth(gain, cfg.rsi_period) / wilder_smooth(loss, cfg.rsi_period)
    add_factor_from_series(factors, "RSI", 100.0 - 100.0 / (1.0 + rs), cfg.percentile_window)

    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)
        highest = high.rolling(cfg.kdj_n).max()
        lowest = low.rolling(cfg.kdj_n).min()
        rsv = (close - lowest) / (highest - lowest) * 100.0
        k = rsv.ewm(alpha=1.0 / cfg.kdj_m1, adjust=False).mean()
        d = k.ewm(alpha=1.0 / cfg.kdj_m2, adjust=False).mean()
        add_factor_from_series(factors, "KDJ_K", k, cfg.percentile_window)
        add_factor_from_series(factors, "KDJ_D", d, cfg.percentile_window)
        add_factor_from_series(factors, "KDJ_J", 3.0 * k - 2.0 * d, cfg.percentile_window)

        typical_price = (high + low + close) / 3.0
        ma = typical_price.rolling(cfg.cci_period).mean()
        md = (typical_price - ma).abs().rolling(cfg.cci_period).mean()
        add_factor_from_series(factors, "CCI", (typical_price - ma) / (0.015 * md), cfg.percentile_window)

    return factors
