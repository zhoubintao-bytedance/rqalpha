from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, safe_ratio, wilder_smooth

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime

_FL_BBANDS_UPPER = "fl_bbands_upper"
_FL_BBANDS_MIDDLE = "fl_bbands_middle"
_FL_BBANDS_LOWER = "fl_bbands_lower"
_FL_BBANDS_WIDTH = "fl_bbands_width"
_FL_ATR = "fl_atr"
_FL_DC_UPPER = "fl_dc_upper"
_FL_DC_LOWER = "fl_dc_lower"
_FL_DC_MIDDLE = "fl_dc_middle"


def _compute_volatility_raw(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    cfg: FactorLayerConfig,
) -> dict[str, pd.Series]:
    close = pd.Series(close, dtype=float)
    result: dict[str, pd.Series] = {}

    mean = close.rolling(cfg.boll_period).mean()
    std = close.rolling(cfg.boll_period).std(ddof=0)
    upper = mean + cfg.boll_std * std
    lower = mean - cfg.boll_std * std
    result[_FL_BBANDS_UPPER] = upper
    result[_FL_BBANDS_MIDDLE] = mean
    result[_FL_BBANDS_LOWER] = lower
    result[_FL_BBANDS_WIDTH] = safe_ratio(upper - lower, mean)

    if high is not None and low is not None:
        high_s = pd.Series(high, dtype=float)
        low_s = pd.Series(low, dtype=float)
        prev_close = close.shift(1)
        tr = pd.Series(
            np.maximum.reduce([
                (high_s - low_s),
                (high_s - prev_close).abs(),
                (low_s - prev_close).abs(),
            ]),
            index=close.index,
        )
        result[_FL_ATR] = wilder_smooth(tr, cfg.atr_period)

        dc_upper = high_s.rolling(cfg.dc_period).max()
        dc_lower = low_s.rolling(cfg.dc_period).min()
        result[_FL_DC_UPPER] = dc_upper
        result[_FL_DC_LOWER] = dc_lower
        result[_FL_DC_MIDDLE] = (dc_upper + dc_lower) / 2.0

    return result


def compute_volatility_series(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    cfg: FactorLayerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or FactorLayerConfig()
    raw = _compute_volatility_raw(close, high, low, cfg)
    return pd.DataFrame(raw, index=close.index)


def compute_volatility_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    high = pd.Series(bars["high"], dtype=float) if "high" in bars.columns else None
    low = pd.Series(bars["low"], dtype=float) if "low" in bars.columns else None
    raw = _compute_volatility_raw(close, high, low, cfg)
    factors: dict[str, dict[str, float]] = {}

    add_factor_from_series(factors, "BBANDS_UPPER", raw[_FL_BBANDS_UPPER], cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_MIDDLE", raw[_FL_BBANDS_MIDDLE], cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_LOWER", raw[_FL_BBANDS_LOWER], cfg.percentile_window)
    add_factor_from_series(factors, "BBANDS_WIDTH", raw[_FL_BBANDS_WIDTH], cfg.percentile_window)
    if _FL_ATR in raw:
        add_factor_from_series(factors, "ATR", raw[_FL_ATR], cfg.percentile_window)
        add_factor_from_series(factors, "DC_UPPER", raw[_FL_DC_UPPER], cfg.percentile_window)
        add_factor_from_series(factors, "DC_LOWER", raw[_FL_DC_LOWER], cfg.percentile_window)
        add_factor_from_series(factors, "DC_MIDDLE", raw[_FL_DC_MIDDLE], cfg.percentile_window)
    return factors
