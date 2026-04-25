from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, ema, safe_ratio, wilder_smooth

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime

_FL_MACD = "fl_macd"
_FL_ADX = "fl_adx"
_FL_RSRS = "fl_rsrs"


def _ma_name(period: int) -> str:
    return f"fl_ma_{period}"


def _ema_name(period: int) -> str:
    return f"fl_ema_{period}"


def _compute_trend_raw(
    close: pd.Series,
    high: pd.Series | None,
    low: pd.Series | None,
    cfg: FactorLayerConfig,
) -> dict[str, pd.Series]:
    close = pd.Series(close, dtype=float)
    result: dict[str, pd.Series] = {}

    macd = ema(close, cfg.macd_fast) - ema(close, cfg.macd_slow)
    signal = ema(macd, cfg.macd_signal)
    result[_FL_MACD] = macd - signal

    for period in cfg.ma_periods:
        result[_ma_name(period)] = close.rolling(period).mean()
    for period in cfg.ema_periods:
        result[_ema_name(period)] = ema(close, period)

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
        plus_dm = (high_s.diff()).where(lambda x: x > 0, 0.0)
        minus_dm = (-low_s.diff()).where(lambda x: x > 0, 0.0)
        atr = wilder_smooth(tr, cfg.atr_period)
        plus_di = 100.0 * safe_ratio(wilder_smooth(plus_dm, cfg.atr_period), atr)
        minus_di = 100.0 * safe_ratio(wilder_smooth(minus_dm, cfg.atr_period), atr)
        dx = 100.0 * safe_ratio((plus_di - minus_di).abs(), plus_di + minus_di)
        result[_FL_ADX] = wilder_smooth(dx, cfg.atr_period)

        rsrs = safe_ratio(
            high_s.rolling(cfg.rsrs_n).mean(),
            low_s.rolling(cfg.rsrs_n).mean(),
        ).replace([np.inf, -np.inf], np.nan)
        zscore = (rsrs - rsrs.rolling(cfg.rsrs_m).mean()) / rsrs.rolling(cfg.rsrs_m).std(ddof=0)
        result[_FL_RSRS] = zscore

    return result


def compute_trend_series(
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    cfg: FactorLayerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or FactorLayerConfig()
    raw = _compute_trend_raw(close, high, low, cfg)
    return pd.DataFrame(raw, index=close.index)


def compute_trend_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    high = pd.Series(bars["high"], dtype=float) if "high" in bars.columns else None
    low = pd.Series(bars["low"], dtype=float) if "low" in bars.columns else None
    raw = _compute_trend_raw(close, high, low, cfg)
    factors: dict[str, dict[str, float]] = {}

    add_factor_from_series(factors, "MACD", raw[_FL_MACD], cfg.percentile_window)
    for period in cfg.ma_periods:
        add_factor_from_series(factors, f"MA_{period}", raw[_ma_name(period)], cfg.percentile_window)
    for period in cfg.ema_periods:
        add_factor_from_series(factors, f"EMA_{period}", raw[_ema_name(period)], cfg.percentile_window)
    if _FL_ADX in raw:
        add_factor_from_series(factors, "ADX", raw[_FL_ADX], cfg.percentile_window)
        add_factor_from_series(factors, "RSRS", raw[_FL_RSRS], cfg.percentile_window)
    return factors
