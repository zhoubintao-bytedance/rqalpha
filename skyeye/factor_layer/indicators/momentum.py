from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime

# Column naming: lowercase with fl_ prefix for DataFrame output, uppercase short
# names for the benchmark-facing add_factor_from_series bucket interface.
_FL_MOM = "fl_mom"
_FL_ROC = "fl_roc"
_FL_BIAS = "fl_bias"


def _compute_momentum_raw(close: pd.Series, cfg: FactorLayerConfig) -> dict[str, pd.Series]:
    """Compute raw momentum time series (no percentile applied)."""
    close = pd.Series(close, dtype=float)
    return {
        _FL_MOM: close - close.shift(cfg.mom_period),
        _FL_ROC: close / close.shift(cfg.roc_period) - 1.0,
        _FL_BIAS: close / close.rolling(cfg.bias_period).mean() - 1.0,
    }


def compute_momentum_series(close: pd.Series, cfg: FactorLayerConfig) -> pd.DataFrame:
    """Return a DataFrame with full momentum factor time series (fl_* columns)."""
    raw = _compute_momentum_raw(close, cfg)
    return pd.DataFrame(raw, index=close.index)


def compute_momentum_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}
    raw = _compute_momentum_raw(close, cfg)
    add_factor_from_series(factors, "MOM", raw[_FL_MOM], cfg.percentile_window)
    add_factor_from_series(factors, "ROC", raw[_FL_ROC], cfg.percentile_window)
    add_factor_from_series(factors, "BIAS", raw[_FL_BIAS], cfg.percentile_window)
    return factors
