"""AX1 feature catalog and metadata contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from skyeye.market_regime_layer import MarketRegimeConfig, required_market_regime_history_days


REGIME_FEATURE_LOOKBACK_DAYS = required_market_regime_history_days(MarketRegimeConfig())


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    scope: str
    asset_type: str
    source_family: str
    required_columns: tuple[str, ...] = ()
    lookback_days: int = 0
    observable_lag_days: int | None = 0
    depends_on: tuple[str, ...] = ()
    status: str = "implemented"
    description: str = ""
    requires_as_of_date: bool = False
    uses_latest_snapshot: bool = False
    decision_time: str = "after_close"
    tradable_lag_days: int | None = 1
    data_source_status: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "scope", str(self.scope))
        object.__setattr__(self, "asset_type", str(self.asset_type).lower())
        object.__setattr__(self, "source_family", str(self.source_family))
        object.__setattr__(self, "required_columns", tuple(str(item) for item in self.required_columns))
        object.__setattr__(self, "depends_on", tuple(str(item) for item in self.depends_on))
        object.__setattr__(self, "status", str(self.status))
        object.__setattr__(self, "lookback_days", int(self.lookback_days or 0))
        if self.observable_lag_days is not None:
            object.__setattr__(self, "observable_lag_days", int(self.observable_lag_days))
        object.__setattr__(self, "decision_time", str(self.decision_time))
        if self.tradable_lag_days is not None:
            object.__setattr__(self, "tradable_lag_days", int(self.tradable_lag_days))
        object.__setattr__(self, "data_source_status", str(self.data_source_status or self.status))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "scope": self.scope,
            "asset_type": self.asset_type,
            "source_family": self.source_family,
            "required_columns": list(self.required_columns),
            "lookback_days": self.lookback_days,
            "observable_lag_days": self.observable_lag_days,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "description": self.description,
            "requires_as_of_date": bool(self.requires_as_of_date),
            "uses_latest_snapshot": bool(self.uses_latest_snapshot),
            "decision_time": self.decision_time,
            "tradable_lag_days": self.tradable_lag_days,
            "data_source_status": self.data_source_status,
        }


class FeatureCatalog:
    def __init__(self, definitions: Iterable[FeatureDefinition]):
        self._definitions = list(definitions)
        self._by_name = {definition.name: definition for definition in self._definitions}
        if len(self._by_name) != len(self._definitions):
            raise ValueError("AX1 feature catalog contains duplicate feature names")

    def __contains__(self, name: str) -> bool:
        return str(name) in self._by_name

    def get(self, name: str) -> FeatureDefinition:
        try:
            return self._by_name[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown AX1 feature: {name}") from exc

    def definitions(self) -> list[FeatureDefinition]:
        return list(self._definitions)

    def names_for_scopes(self, scopes: Sequence[str]) -> list[str]:
        scope_set = {str(scope) for scope in scopes}
        return [
            definition.name
            for definition in self._definitions
            if definition.scope in scope_set and definition.status == "implemented"
        ]

    def require_active(self, feature_names: Sequence[str]) -> list[FeatureDefinition]:
        resolved: list[FeatureDefinition] = []
        for name in feature_names:
            if str(name) not in self._by_name:
                raise ValueError(f"unknown AX1 feature: {name}")
            definition = self._by_name[str(name)]
            if definition.status != "implemented":
                raise ValueError(f"AX1 feature is not implemented: {definition.name}")
            resolved.append(definition)
        return resolved

    def to_dict(self) -> dict:
        return {"features": [definition.to_dict() for definition in self._definitions]}


def build_default_feature_catalog(config: dict | None = None) -> FeatureCatalog:
    config = dict(config or {})
    definitions: list[FeatureDefinition] = []

    def add(**kwargs) -> None:
        definitions.append(FeatureDefinition(**kwargs))

    for name, lookback in (
        ("feature_momentum_5d", 5),
        ("feature_volatility_5d", 5),
        ("feature_dollar_volume", 0),
        ("feature_liquidity_score", 20),
        ("feature_risk_forecast", 20),
        ("feature_cost_forecast", 20),
    ):
        add(
            name=name,
            scope="common",
            asset_type="both",
            source_family="price_volume",
            required_columns=("close", "volume"),
            lookback_days=lookback,
            observable_lag_days=0,
            status="implemented",
            data_source_status="implemented",
        )

    # Amihud illiquidity and realized skewness (inline from OHLCV)
    add(
        name="feature_amihud_illiquidity",
        scope="common",
        asset_type="both",
        source_family="price_volume",
        required_columns=("close", "volume"),
        lookback_days=20,
        observable_lag_days=0,
        status="implemented",
        data_source_status="implemented",
        description="Amihud illiquidity: 20d rolling mean of |return|/dollar_volume",
    )
    add(
        name="feature_realized_skew_20d",
        scope="common",
        asset_type="both",
        source_family="price_volume",
        required_columns=("close",),
        lookback_days=20,
        observable_lag_days=0,
        status="implemented",
        data_source_status="implemented",
        description="Realized skewness: 20d rolling skew of daily returns",
    )

    # Technical indicators (inline from close prices)
    add(
        name="feature_rsi_14d",
        scope="technical",
        asset_type="both",
        source_family="technical",
        required_columns=("close",),
        lookback_days=14,
        observable_lag_days=0,
        status="implemented",
        data_source_status="implemented",
        description="RSI(14) computed inline from close prices",
    )
    add(
        name="feature_macd",
        scope="technical",
        asset_type="both",
        source_family="technical",
        required_columns=("close",),
        lookback_days=26,
        observable_lag_days=0,
        status="implemented",
        data_source_status="implemented",
        description="MACD histogram (12,26,9) computed inline from close prices",
    )

    # Turnover rate (external data source, merged into raw_df)
    add(
        name="feature_turnover_rate",
        scope="common",
        asset_type="both",
        source_family="price_volume",
        required_columns=("close", "volume"),
        lookback_days=0,
        observable_lag_days=0,
        status="implemented",
        data_source_status="implemented",
        description="Daily turnover rate from get_turnover_rate",
    )

    style_pairs = _style_pairs(config)
    etf_raw = [
        ("feature_excess_mom_20d", 20, ("feature_momentum_20d",)),
        ("feature_excess_mom_60d", 60, ("feature_momentum_60d",)),
        ("feature_volume_price_flow_20d", 20, ("feature_return_1d", "feature_dollar_volume")),
        ("feature_vol_transition_10_60d", 60, ("feature_volatility_10d", "feature_volatility_60d")),
    ]
    for name, lookback, depends_on in etf_raw:
        add(
            name=name,
            scope="etf_raw",
            asset_type="etf",
            source_family="price_volume",
            required_columns=("close", "volume"),
            lookback_days=lookback,
            observable_lag_days=0,
            depends_on=depends_on,
            status="implemented",
            data_source_status="implemented",
        )
    for pair in style_pairs:
        name = f"feature_style_spread_{pair['name']}_20d"
        add(
            name=name,
            scope="etf_raw",
            asset_type="etf",
            source_family="price_volume",
            required_columns=("close", "volume"),
            lookback_days=80,
            observable_lag_days=0,
            status="implemented",
            data_source_status="implemented",
        )
    add(
        name="feature_style_spread_composite_20d",
        scope="etf_raw",
        asset_type="etf",
        source_family="price_volume",
        required_columns=("close", "volume"),
        lookback_days=80,
        observable_lag_days=0,
        depends_on=tuple(f"feature_style_spread_{pair['name']}_20d" for pair in style_pairs),
        status="implemented",
    )

    zscore_sources = {
        "feature_z_excess_mom_20d": "feature_excess_mom_20d",
        "feature_z_volume_price_flow_20d": "feature_volume_price_flow_20d",
        "feature_z_vol_transition_10_60d": "feature_vol_transition_10_60d",
        "feature_z_style_spread_composite_20d": "feature_style_spread_composite_20d",
    }
    for name, source in zscore_sources.items():
        add(
            name=name,
            scope="etf_zscore",
            asset_type="etf",
            source_family="price_volume",
            required_columns=("close", "volume"),
            lookback_days=80,
            observable_lag_days=0,
            depends_on=(source,),
            status="implemented",
            data_source_status="implemented",
        )

    regime_columns = [
        "feature_regime_strength",
        "feature_regime_risk_on",
        "feature_regime_neutral",
        "feature_regime_risk_off",
        "feature_regime_rotation",
    ]
    for name in regime_columns:
        add(
            name=name,
            scope="regime",
            asset_type="both",
            source_family="regime_price_volume",
            required_columns=("close", "volume"),
            lookback_days=REGIME_FEATURE_LOOKBACK_DAYS,
            observable_lag_days=0,
            status="implemented",
            data_source_status="implemented",
        )
    for zscore_name in zscore_sources:
        zscore_key = zscore_name.removeprefix("feature_")
        for regime_name in regime_columns:
            regime_key = regime_name.removeprefix("feature_")
            add(
                name=f"feature_interaction_{zscore_key}_x_{regime_key}",
                scope="regime_interaction",
                asset_type="etf",
                source_family="derived",
                required_columns=(),
                lookback_days=REGIME_FEATURE_LOOKBACK_DAYS,
                observable_lag_days=0,
                depends_on=(zscore_name, regime_name),
                status="implemented",
                data_source_status="implemented",
            )

    add(
        name="feature_pe_ttm",
        scope="fundamental",
        asset_type="stock",
        source_family="fundamental",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
    )
    add(
        name="feature_pb_ratio",
        scope="fundamental",
        asset_type="stock",
        source_family="fundamental",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
    )
    add(
        name="feature_roe_ttm",
        scope="fundamental",
        asset_type="stock",
        source_family="fundamental",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
    )
    add(
        name="feature_dividend_yield",
        scope="fundamental",
        asset_type="stock",
        source_family="fundamental",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
        description="Dividend yield from get_factor(dividend_yield)",
    )
    add(
        name="feature_gross_profit_margin",
        scope="fundamental",
        asset_type="stock",
        source_family="fundamental",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
        description="Gross profit margin from get_factor(gross_profit_margin)",
    )
    add(
        name="feature_net_profit_growth_yoy",
        scope="fundamental",
        asset_type="stock",
        source_family="fundamental",
        observable_lag_days=1,
        lookback_days=365,
        status="implemented",
        data_source_status="implemented",
        description="Net profit YoY growth from PIT financials TTM comparison",
    )
    add(
        name="feature_margin_financing_balance",
        scope="flow",
        asset_type="both",
        source_family="flow",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
    )
    add(
        name="feature_institutional_holding_ratio",
        scope="flow",
        asset_type="stock",
        source_family="flow",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
        description="Northbound institutional holding proxy from stock_connect holding_ratio",
    )
    add(
        name="feature_macro_pmi",
        scope="macro",
        asset_type="both",
        source_family="macro",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
        description="Official manufacturing PMI with one-day observation lag",
    )
    add(
        name="feature_bond_yield_10y",
        scope="macro",
        asset_type="both",
        source_family="macro",
        observable_lag_days=0,
        status="implemented",
        data_source_status="implemented",
        description="10-year government bond yield from yield curve",
    )
    add(
        name="feature_northbound_aggregate_flow",
        scope="flow",
        asset_type="both",
        source_family="flow",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
        description="Market-level aggregate northbound net flow",
    )
    add(
        name="feature_northbound_net_flow",
        scope="flow",
        asset_type="stock",
        source_family="flow",
        observable_lag_days=1,
        status="implemented",
        data_source_status="implemented",
        description="Per-stock northbound net flow from get_stock_connect",
    )

    return FeatureCatalog(definitions)


def _style_pairs(config: dict) -> list[dict]:
    pairs = config.get("style_pairs")
    if pairs:
        return [dict(pair) for pair in pairs]
    return [
        {"name": "dividend_vs_growth"},
        {"name": "value_vs_growth"},
        {"name": "large_vs_small"},
    ]
