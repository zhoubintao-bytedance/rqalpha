from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, wilder_smooth

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime

_FL_RSI = "fl_rsi"
_FL_KDJ_K = "fl_kdj_k"
_FL_KDJ_D = "fl_kdj_d"
_FL_KDJ_J = "fl_kdj_j"
_FL_CCI = "fl_cci"


def _compute_oscillator_raw(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    cfg: FactorLayerConfig,
) -> dict[str, pd.Series]:
    close = pd.Series(close, dtype=float)
    result: dict[str, pd.Series] = {}

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = wilder_smooth(gain, cfg.rsi_period) / wilder_smooth(loss, cfg.rsi_period)
    result[_FL_RSI] = 100.0 - 100.0 / (1.0 + rs)

    if high is not None and low is not None:
        high_s = pd.Series(high, dtype=float)
        low_s = pd.Series(low, dtype=float)
        highest = high_s.rolling(cfg.kdj_n).max()
        lowest = low_s.rolling(cfg.kdj_n).min()
        rsv = (close - lowest) / (highest - lowest) * 100.0
        k = rsv.ewm(alpha=1.0 / cfg.kdj_m1, adjust=False).mean()
        d = k.ewm(alpha=1.0 / cfg.kdj_m2, adjust=False).mean()
        result[_FL_KDJ_K] = k
        result[_FL_KDJ_D] = d
        result[_FL_KDJ_J] = 3.0 * k - 2.0 * d

        typical_price = (high_s + low_s + close) / 3.0
        ma = typical_price.rolling(cfg.cci_period).mean()
        md = (typical_price - ma).abs().rolling(cfg.cci_period).mean()
        result[_FL_CCI] = (typical_price - ma) / (0.015 * md)

    return result


def compute_oscillator_series(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    cfg: FactorLayerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or FactorLayerConfig()
    raw = _compute_oscillator_raw(close, high, low, cfg)
    return pd.DataFrame(raw, index=close.index)


def compute_oscillator_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    high = pd.Series(bars["high"], dtype=float) if "high" in bars.columns else None
    low = pd.Series(bars["low"], dtype=float) if "low" in bars.columns else None
    raw = _compute_oscillator_raw(close, high, low, cfg)
    factors: dict[str, dict[str, float]] = {}
    add_factor_from_series(factors, "RSI", raw[_FL_RSI], cfg.percentile_window)
    if _FL_KDJ_K in raw:
        add_factor_from_series(factors, "KDJ_K", raw[_FL_KDJ_K], cfg.percentile_window)
        add_factor_from_series(factors, "KDJ_D", raw[_FL_KDJ_D], cfg.percentile_window)
        add_factor_from_series(factors, "KDJ_J", raw[_FL_KDJ_J], cfg.percentile_window)
        add_factor_from_series(factors, "CCI", raw[_FL_CCI], cfg.percentile_window)
    return factors
