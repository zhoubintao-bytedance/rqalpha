from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from skyeye.market_regime_layer import MarketRegime


FactorValue: TypeAlias = dict[Literal["value", "percentile"], float]


@dataclass(frozen=True)
class FactorLayerResult:
    regime: MarketRegime
    trend_factors: dict[str, FactorValue] = field(default_factory=dict)
    momentum_factors: dict[str, FactorValue] = field(default_factory=dict)
    oscillator_factors: dict[str, FactorValue] = field(default_factory=dict)
    volatility_factors: dict[str, FactorValue] = field(default_factory=dict)
    volume_factors: dict[str, FactorValue] = field(default_factory=dict)
