from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, safe_ratio

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime


def compute_volume_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    if "volume" not in bars.columns:
        return {}

    close = pd.Series(bars["close"], dtype=float)
    volume = pd.Series(bars["volume"], dtype=float)
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    factors: dict[str, dict[str, float]] = {}

    add_factor_from_series(factors, "OBV", obv, cfg.percentile_window)
    add_factor_from_series(factors, "OBV_MA", obv.rolling(cfg.obv_ma_period).mean(), cfg.percentile_window)

    if {"high", "low"}.issubset(bars.columns):
        high = pd.Series(bars["high"], dtype=float)
        low = pd.Series(bars["low"], dtype=float)
        typical_price = (high + low + close) / 3.0
        raw_flow = typical_price * volume
        pos_flow = raw_flow.where(typical_price.diff() > 0, 0.0).rolling(cfg.mfi_period).sum()
        neg_flow = raw_flow.where(typical_price.diff() < 0, 0.0).rolling(cfg.mfi_period).sum().abs()
        mfi = 100.0 - 100.0 / (1.0 + safe_ratio(pos_flow, neg_flow))
        add_factor_from_series(factors, "MFI", mfi, cfg.percentile_window)

    return factors
