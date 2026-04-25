from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, ema, safe_ratio, wilder_smooth

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime


def compute_trend_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}

    # 获取 regime.diagnostics 中的指标值
    dir_diag = regime.diagnostics.get("direction_diagnostics", {}) if regime else {}

    # MACD - 优先从 regime 复用
    if dir_diag.get("macd_histogram_value") is not None:
        # 需要历史序列计算 percentile，这里仍然需要计算，但使用相同参数
        macd = ema(close, cfg.macd_fast) - ema(close, cfg.macd_slow)
        signal = ema(macd, cfg.macd_signal)
        add_factor_from_series(factors, "MACD", macd - signal, cfg.percentile_window)
    else:
        macd = ema(close, cfg.macd_fast) - ema(close, cfg.macd_slow)
        signal = ema(macd, cfg.macd_signal)
        add_factor_from_series(factors, "MACD", macd - signal, cfg.percentile_window)

    for period in cfg.ma_periods:
        add_factor_from_series(factors, f"MA_{period}", close.rolling(period).mean(), cfg.percentile_window)
    for period in cfg.ema_periods:
        add_factor_from_series(factors, f"EMA_{period}", ema(close, period), cfg.percentile_window)

    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)

        # ADX - 优先从 regime 复用
        if dir_diag.get("adx") is not None:
            # ADX 值已知，但仍需历史序列计算 percentile
            prev_close = close.shift(1)
            tr = pd.Series(
                np.maximum.reduce([(high - low), (high - prev_close).abs(), (low - prev_close).abs()]),
                index=bars.index,
            )
            plus_dm = (high.diff()).where(lambda x: x > 0, 0.0)
            minus_dm = (-low.diff()).where(lambda x: x > 0, 0.0)
            atr = wilder_smooth(tr, cfg.atr_period)
            plus_di = 100.0 * safe_ratio(wilder_smooth(plus_dm, cfg.atr_period), atr)
            minus_di = 100.0 * safe_ratio(wilder_smooth(minus_dm, cfg.atr_period), atr)
            dx = 100.0 * safe_ratio((plus_di - minus_di).abs(), plus_di + minus_di)
            add_factor_from_series(factors, "ADX", wilder_smooth(dx, cfg.atr_period), cfg.percentile_window)
        else:
            prev_close = close.shift(1)
            tr = pd.Series(
                np.maximum.reduce([(high - low), (high - prev_close).abs(), (low - prev_close).abs()]),
                index=bars.index,
            )
            plus_dm = (high.diff()).where(lambda x: x > 0, 0.0)
            minus_dm = (-low.diff()).where(lambda x: x > 0, 0.0)
            atr = wilder_smooth(tr, cfg.atr_period)
            plus_di = 100.0 * safe_ratio(wilder_smooth(plus_dm, cfg.atr_period), atr)
            minus_di = 100.0 * safe_ratio(wilder_smooth(minus_dm, cfg.atr_period), atr)
            dx = 100.0 * safe_ratio((plus_di - minus_di).abs(), plus_di + minus_di)
            add_factor_from_series(factors, "ADX", wilder_smooth(dx, cfg.atr_period), cfg.percentile_window)

        # RSRS - 使用配置参数（与 market_regime_layer 保持一致）
        rsrs = safe_ratio(high.rolling(cfg.rsrs_n).mean(), low.rolling(cfg.rsrs_n).mean()).replace(
            [np.inf, -np.inf], np.nan
        )
        zscore = (rsrs - rsrs.rolling(cfg.rsrs_m).mean()) / rsrs.rolling(cfg.rsrs_m).std(ddof=0)
        add_factor_from_series(factors, "RSRS", zscore, cfg.percentile_window)

    return factors
