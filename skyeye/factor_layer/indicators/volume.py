from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series, safe_ratio

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime

_FL_OBV = "fl_obv"
_FL_OBV_MA = "fl_obv_ma"
_FL_MFI = "fl_mfi"


def _compute_volume_raw(
    close: pd.Series,
    volume: pd.Series | None,
    high: pd.Series | None,
    low: pd.Series | None,
    cfg: FactorLayerConfig,
) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}

    if volume is None:
        return result

    close_s = pd.Series(close, dtype=float)
    vol_s = pd.Series(volume, dtype=float)
    direction = np.sign(close_s.diff()).fillna(0.0)
    obv = (direction * vol_s).cumsum()
    result[_FL_OBV] = obv
    result[_FL_OBV_MA] = obv.rolling(cfg.obv_ma_period).mean()

    if high is not None and low is not None:
        high_s = pd.Series(high, dtype=float)
        low_s = pd.Series(low, dtype=float)
        typical_price = (high_s + low_s + close_s) / 3.0
        raw_flow = typical_price * vol_s
        pos_flow = raw_flow.where(typical_price.diff() > 0, 0.0).rolling(cfg.mfi_period).sum()
        neg_flow = raw_flow.where(typical_price.diff() < 0, 0.0).rolling(cfg.mfi_period).sum().abs()
        mfi = 100.0 - 100.0 / (1.0 + safe_ratio(pos_flow, neg_flow))
        result[_FL_MFI] = mfi

    return result


def compute_volume_series(
    close: pd.Series,
    volume: pd.Series | None = None,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    cfg: FactorLayerConfig | None = None,
) -> pd.DataFrame:
    cfg = cfg or FactorLayerConfig()
    raw = _compute_volume_raw(close, volume, high, low, cfg)
    if not raw:
        return pd.DataFrame(index=close.index)
    return pd.DataFrame(raw, index=close.index)


def compute_volume_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    if "volume" not in bars.columns:
        return {}

    close = pd.Series(bars["close"], dtype=float)
    volume = pd.Series(bars["volume"], dtype=float)
    high = pd.Series(bars["high"], dtype=float) if "high" in bars.columns else None
    low = pd.Series(bars["low"], dtype=float) if "low" in bars.columns else None
    raw = _compute_volume_raw(close, volume, high, low, cfg)
    factors: dict[str, dict[str, float]] = {}

    add_factor_from_series(factors, "OBV", raw[_FL_OBV], cfg.percentile_window)
    add_factor_from_series(factors, "OBV_MA", raw[_FL_OBV_MA], cfg.percentile_window)
    if _FL_MFI in raw:
        add_factor_from_series(factors, "MFI", raw[_FL_MFI], cfg.percentile_window)
    return factors
