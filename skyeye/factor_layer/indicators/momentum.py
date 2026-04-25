from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from skyeye.factor_layer.config import FactorLayerConfig
from skyeye.factor_layer.utils import add_factor_from_series

if TYPE_CHECKING:
    from skyeye.market_regime_layer import MarketRegime


def compute_momentum_factors(
    bars: pd.DataFrame, cfg: FactorLayerConfig, regime: "MarketRegime | None" = None
) -> dict[str, dict[str, float]]:
    close = pd.Series(bars["close"], dtype=float)
    factors: dict[str, dict[str, float]] = {}
    add_factor_from_series(factors, "MOM", close - close.shift(cfg.mom_period), cfg.percentile_window)
    add_factor_from_series(factors, "ROC", close / close.shift(cfg.roc_period) - 1.0, cfg.percentile_window)
    add_factor_from_series(
        factors,
        "BIAS",
        close / close.rolling(cfg.bias_period).mean() - 1.0,
        cfg.percentile_window,
    )
    return factors
