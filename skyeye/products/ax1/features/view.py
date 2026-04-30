"""Unified AX1 feature view."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from skyeye.products.ax1._common import require_columns
from skyeye.products.ax1.data_sources.technical import TECHNICAL_FEATURE_COLUMNS, build_technical_indicator_features

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureView:
    frame: pd.DataFrame
    columns_by_scope: dict[str, list[str]]
    feature_set: str = "ax1_unified_v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        columns: list[str] = []
        for scope_columns in self.columns_by_scope.values():
            columns.extend(scope_columns)
        return list(dict.fromkeys(columns))


class AX1FeatureViewBuilder:
    def __init__(self, config: dict | None = None) -> None:
        self.config = dict(config or {})

    def build(
        self,
        raw_df: pd.DataFrame,
        *,
        universe_metadata: pd.DataFrame,
        regime_state=None,
        regime_state_by_date=None,
        fundamental_df: pd.DataFrame | None = None,
        flow_df: pd.DataFrame | None = None,
        macro_df: pd.DataFrame | None = None,
    ) -> FeatureView:
        require_columns(raw_df, ["date", "order_book_id", "close"], entity="raw_df")
        metadata = _normalize_metadata(universe_metadata, self.config)
        frame = raw_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame = frame.merge(
            metadata,
            on="order_book_id",
            how="left",
            suffixes=("", "_meta"),
        )
        for column in ("asset_type", "universe_layer", "industry", "benchmark_id"):
            meta_column = f"{column}_meta"
            if meta_column in frame.columns:
                frame[column] = frame[meta_column].combine_first(frame[column] if column in frame.columns else None)
                frame = frame.drop(columns=[meta_column])
        frame["asset_type"] = frame.get("asset_type", "stock").fillna("stock").astype(str).str.lower()
        frame["universe_layer"] = frame.get("universe_layer", frame["asset_type"]).fillna(frame["asset_type"]).astype(str)
        frame["industry"] = frame.get("industry", "Unknown").fillna("Unknown").astype(str)
        configured_benchmark_id = _core_proxy_id(self.config)
        if _has_configured_core_proxy(self.config):
            frame["benchmark_id"] = configured_benchmark_id
        else:
            frame["benchmark_id"] = frame.get("benchmark_id", configured_benchmark_id).fillna(configured_benchmark_id).astype(str)
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce") if "volume" in frame.columns else np.nan
        frame = frame.sort_values(["order_book_id", "date"]).reset_index(drop=True)

        common_columns = _build_common_features(frame, self.config)
        etf_raw_columns = _build_etf_features(frame, self.config)
        _mask_scope_to_asset_type(frame, etf_raw_columns, asset_type="etf")
        regime_columns = _build_regime_features(
            frame,
            regime_state=regime_state,
            regime_state_by_date=regime_state_by_date,
        )
        etf_zscore_columns = [column for column in _etf_zscore_columns() if column in frame.columns]
        interaction_columns = (
            _build_regime_interactions(frame, etf_zscore_columns, regime_columns)
            if _should_build_regime_interactions(self.config)
            else []
        )
        _mask_scope_to_asset_type(frame, interaction_columns, asset_type="etf")
        technical_columns = _build_technical_features(frame)
        stock_columns: list[str] = []

        # Build fundamental features
        fundamental_columns: list[str] = []
        if fundamental_df is not None and not fundamental_df.empty:
            frame, fundamental_columns = self._build_fundamental_features(frame, fundamental_df)
            _mask_scope_to_asset_type(frame, fundamental_columns, asset_type="stock")

        # Build flow features
        flow_columns: list[str] = []
        if flow_df is not None and not flow_df.empty:
            frame, flow_columns = self._build_flow_features(frame, flow_df)

        # Build macro features
        macro_columns: list[str] = []
        if macro_df is not None and not macro_df.empty:
            frame, macro_columns = _build_macro_features(frame, macro_df)

        frame = frame.sort_values(["date", "order_book_id"]).reset_index(drop=True)
        columns_by_scope = {
            "common": common_columns,
            "etf_raw": etf_raw_columns,
            "etf_zscore": etf_zscore_columns,
            "regime": regime_columns,
            "regime_interaction": interaction_columns,
            "technical": technical_columns,
            "fundamental": fundamental_columns,
            "flow": flow_columns,
            "macro": macro_columns,
            "stock_specific": stock_columns,
        }
        return FeatureView(
            frame=frame,
            columns_by_scope=columns_by_scope,
            feature_set=str(self.config.get("feature_set", "ax1_unified_v1")),
            metadata={
                "regime_state": dict(regime_state or {}),
                "regime_state_by_date_count": int(len(regime_state_by_date or {})),
                "style_pairs": _style_pairs(self.config),
            },
        )

    def _build_fundamental_features(
        self,
        frame: pd.DataFrame,
        fundamental_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build fundamental features by merging fundamental data into the main frame.

        Args:
            frame: Main feature frame
            fundamental_df: Fundamental data with columns: date, order_book_id, feature_*

        Returns:
            Tuple of (modified frame, list of fundamental feature column names)
        """
        fundamental_df = fundamental_df.copy()
        fundamental_df["date"] = pd.to_datetime(fundamental_df["date"])
        fundamental_df["order_book_id"] = fundamental_df["order_book_id"].astype(str)

        # Left join fundamental data to main frame
        frame = frame.merge(
            fundamental_df,
            on=["date", "order_book_id"],
            how="left",
            suffixes=("", "_fund"),
        )

        # Return feature column names
        feature_cols = [col for col in fundamental_df.columns if col.startswith("feature_")]
        return frame, feature_cols

    def _build_flow_features(
        self,
        frame: pd.DataFrame,
        flow_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build flow features by merging flow data into the main frame.

        Args:
            frame: Main feature frame
            flow_df: Flow data with columns: date, order_book_id, feature_*

        Returns:
            Tuple of (modified frame, list of flow feature column names)
        """
        flow_df = flow_df.copy()
        flow_df["date"] = pd.to_datetime(flow_df["date"])
        flow_df["order_book_id"] = flow_df["order_book_id"].astype(str)

        # Left join flow data to main frame
        frame = frame.merge(
            flow_df,
            on=["date", "order_book_id"],
            how="left",
            suffixes=("", "_flow"),
        )

        # Return feature column names
        feature_cols = [col for col in flow_df.columns if col.startswith("feature_")]
        return frame, feature_cols


def resolve_feature_columns(config: dict, feature_view: FeatureView) -> list[str]:
    model = dict((config or {}).get("model") or {})
    features = dict((config or {}).get("features") or {})
    if "feature_columns" in model:
        raise ValueError("AX1 models must use feature_set/include_scopes, not legacy feature_columns")
    feature_set = model.get("feature_set") or features.get("feature_set")
    if not feature_set:
        raise ValueError("model.feature_set is required")
    if str(feature_set) != feature_view.feature_set:
        raise ValueError(f"feature_set mismatch: model={feature_set} view={feature_view.feature_set}")
    include_scopes = list(model.get("include_scopes") or features.get("include_scopes") or ["common"])
    columns: list[str] = []
    for scope in include_scopes:
        if str(scope) == "stock_specific":
            raise NotImplementedError("AX1 stock_specific feature scope is not implemented")
        try:
            scope_columns = feature_view.columns_by_scope[str(scope)]
        except KeyError as exc:
            raise ValueError(f"unknown feature scope: {scope}") from exc
        columns.extend(scope_columns)
    resolved = [column for column in dict.fromkeys(columns) if column in feature_view.frame.columns]
    if not resolved:
        raise ValueError("feature_set resolved no feature columns")
    return resolved


def _build_common_features(frame: pd.DataFrame, config: dict[str, Any] | None = None) -> list[str]:
    config = dict(config or {})
    grouped = frame.groupby("order_book_id", sort=False)
    frame["feature_return_1d"] = grouped["close"].pct_change()
    frame["feature_momentum_5d"] = grouped["close"].pct_change(5)
    frame["feature_volatility_5d"] = grouped["feature_return_1d"].transform(lambda s: s.rolling(5, min_periods=2).std())
    frame["feature_volatility_10d"] = grouped["feature_return_1d"].transform(lambda s: s.rolling(10, min_periods=3).std())
    frame["feature_volatility_20d"] = grouped["feature_return_1d"].transform(lambda s: s.rolling(20, min_periods=5).std())
    frame["feature_volatility_60d"] = grouped["feature_return_1d"].transform(lambda s: s.rolling(60, min_periods=10).std())
    raw_dollar_volume = (
        frame["close"] * pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    frame["_raw_dollar_volume"] = raw_dollar_volume
    rolling_dollar_volume = grouped["_raw_dollar_volume"].transform(lambda s: s.rolling(20, min_periods=3).mean())
    # Calibrate liquidity scale for ETF dollar volume (20d rolling mean).
    # 5e8 is a more realistic "fully liquid" threshold for typical broad/sector ETFs.
    full_liquidity = float(config.get("liquidity_full_dollar_volume", 500_000_000.0))
    frame["feature_liquidity_score"] = (rolling_dollar_volume / full_liquidity).clip(lower=0.0, upper=1.0).fillna(0.0)
    frame["feature_dollar_volume"] = np.log1p(raw_dollar_volume)
    frame["feature_risk_forecast"] = frame["feature_volatility_20d"].fillna(0.0).clip(lower=0.0)
    stock_cost_forecast = (0.0001 + (1.0 - frame["feature_liquidity_score"]) * 0.0015).clip(lower=0.0)
    etf_cost_bps = float(config.get("etf_cost_forecast_bps", 4.0))
    etf_cost_forecast = pd.Series(etf_cost_bps / 10000.0, index=frame.index, dtype=float)
    if "asset_type" in frame.columns:
        asset_type = frame["asset_type"].astype(str).str.lower()
        frame["feature_cost_forecast"] = np.where(asset_type.eq("etf"), etf_cost_forecast, stock_cost_forecast)
    else:
        frame["feature_cost_forecast"] = stock_cost_forecast
    # Amihud illiquidity: rolling mean of |return| / dollar_volume
    abs_return = frame["feature_return_1d"].abs().fillna(0.0)
    amihud_raw = (abs_return / raw_dollar_volume.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    frame["feature_amihud_illiquidity"] = grouped["feature_return_1d"].transform(
        lambda s: amihud_raw.loc[s.index].rolling(20, min_periods=5).mean()
    ).fillna(0.0)
    # Realized skewness: 20d rolling skew of daily returns
    frame["feature_realized_skew_20d"] = grouped["feature_return_1d"].transform(
        lambda s: s.rolling(20, min_periods=5).skew()
    ).fillna(0.0)
    # Turnover rate: use column from raw_df if available (loaded externally)
    turnover_in_scope = False
    if "turnover_rate" in frame.columns:
        frame["feature_turnover_rate"] = pd.to_numeric(frame["turnover_rate"], errors="coerce").fillna(0.0)
        if frame["feature_turnover_rate"].notna().any() and (frame["feature_turnover_rate"] != 0.0).any():
            turnover_in_scope = True
    elif "today" in frame.columns:
        frame["feature_turnover_rate"] = pd.to_numeric(frame["today"], errors="coerce").fillna(0.0)
        if frame["feature_turnover_rate"].notna().any() and (frame["feature_turnover_rate"] != 0.0).any():
            turnover_in_scope = True
    else:
        frame["feature_turnover_rate"] = np.nan
    common_features = [
        "feature_momentum_5d",
        "feature_volatility_5d",
        "feature_dollar_volume",
        "feature_liquidity_score",
        "feature_risk_forecast",
        "feature_cost_forecast",
        "feature_amihud_illiquidity",
        "feature_realized_skew_20d",
    ]
    if turnover_in_scope:
        common_features.append("feature_turnover_rate")
    return common_features


def _build_technical_features(frame: pd.DataFrame) -> list[str]:
    """Build technical indicator features from the shared technical data source logic."""
    build_technical_indicator_features(frame)
    return list(TECHNICAL_FEATURE_COLUMNS)


def _build_macro_features(frame: pd.DataFrame, macro_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Broadcast market-level macro features to all assets per date."""
    macro_df = macro_df.copy()
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    # Merge on date only (broadcasts to all order_book_ids)
    frame = frame.merge(macro_df, on="date", how="left", suffixes=("", "_macro"))
    feature_cols = [col for col in macro_df.columns if col.startswith("feature_")]
    return frame, feature_cols


def _build_etf_features(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    grouped = frame.groupby("order_book_id", sort=False)
    frame["feature_momentum_20d"] = grouped["close"].pct_change(20)
    frame["feature_momentum_60d"] = grouped["close"].pct_change(60)
    benchmark_momentum = _benchmark_momentum_by_date(frame)
    frame["feature_excess_mom_20d"] = [
        row.feature_momentum_20d - benchmark_momentum.get((row.benchmark_id, row.date, 20), np.nan)
        for row in frame[["benchmark_id", "date", "feature_momentum_20d"]].itertuples(index=False)
    ]
    frame["feature_excess_mom_60d"] = [
        row.feature_momentum_60d - benchmark_momentum.get((row.benchmark_id, row.date, 60), np.nan)
        for row in frame[["benchmark_id", "date", "feature_momentum_60d"]].itertuples(index=False)
    ]
    rolling_dollar_volume = grouped["_raw_dollar_volume"].transform(lambda s: s.rolling(20, min_periods=3).mean())
    volume_pulse = frame["_raw_dollar_volume"] / rolling_dollar_volume.replace(0.0, np.nan) - 1.0
    frame["feature_volume_price_flow_20d"] = (frame["feature_return_1d"].fillna(0.0) * volume_pulse.fillna(0.0)).replace(
        [np.inf, -np.inf],
        0.0,
    )
    frame.drop(columns=["_raw_dollar_volume"], inplace=True)
    frame["feature_vol_transition_10_60d"] = (
        frame["feature_volatility_10d"] / frame["feature_volatility_60d"].replace(0.0, np.nan) - 1.0
    ).replace([np.inf, -np.inf], np.nan)

    style_columns = _style_spread_columns(frame, config)
    normalized_sources = [
        "feature_excess_mom_20d",
        "feature_volume_price_flow_20d",
        "feature_vol_transition_10_60d",
        "feature_style_spread_composite_20d",
    ]
    for column in normalized_sources:
        if column in frame.columns:
            frame[f"feature_z_{column.removeprefix('feature_')}"] = _rolling_zscore(
                frame,
                column,
                window=_normalization_window(config),
                min_periods=_normalization_min_periods(config),
                winsorize_z=_normalization_winsorize_z(config),
            )
    return [
        "feature_excess_mom_20d",
        "feature_excess_mom_60d",
        "feature_volume_price_flow_20d",
        "feature_vol_transition_10_60d",
        *style_columns,
    ]


def _etf_zscore_columns() -> list[str]:
    return [
        "feature_z_excess_mom_20d",
        "feature_z_volume_price_flow_20d",
        "feature_z_vol_transition_10_60d",
        "feature_z_style_spread_composite_20d",
    ]


def _build_regime_features(
    frame: pd.DataFrame,
    *,
    regime_state: dict[str, Any] | None,
    regime_state_by_date: dict[Any, dict[str, Any]] | None,
) -> list[str]:
    fallback = _normalize_regime_state(regime_state)
    states_by_date = {
        pd.Timestamp(date): _normalize_regime_state(state)
        for date, state in (regime_state_by_date or {}).items()
    }
    states = [states_by_date.get(pd.Timestamp(date), fallback) for date in frame["date"]]
    risk_states = [str(state.get("risk_state", "neutral")) for state in states]
    rotation_states = [str(state.get("rotation_state", "co_move")) for state in states]
    frame["feature_regime_strength"] = [float(state.get("strength", 0.0) or 0.0) for state in states]
    frame["feature_regime_risk_on"] = [1.0 if value == "risk_on" else 0.0 for value in risk_states]
    frame["feature_regime_neutral"] = [1.0 if value == "neutral" else 0.0 for value in risk_states]
    frame["feature_regime_risk_off"] = [1.0 if value == "risk_off" else 0.0 for value in risk_states]
    frame["feature_regime_rotation"] = [1.0 if value == "rotation" else 0.0 for value in rotation_states]
    return [
        "feature_regime_strength",
        "feature_regime_risk_on",
        "feature_regime_neutral",
        "feature_regime_risk_off",
        "feature_regime_rotation",
    ]


def _normalize_regime_state(state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(state or {})
    market_regime = str(state.get("market_regime", "range_co_move"))
    risk_state = str(state.get("risk_state") or _risk_state_from_market_regime(market_regime))
    rotation_state = str(state.get("rotation_state") or ("rotation" if market_regime.endswith("_rotation") else "co_move"))
    try:
        strength = float(state.get("strength", 0.0))
    except (TypeError, ValueError):
        strength = 0.0
    if not np.isfinite(strength):
        strength = 0.0
    return {
        "market_regime": market_regime,
        "risk_state": risk_state,
        "rotation_state": rotation_state,
        "strength": min(max(strength, 0.0), 1.0),
    }


def _risk_state_from_market_regime(market_regime: str) -> str:
    if str(market_regime).startswith("bull_"):
        return "risk_on"
    if str(market_regime).startswith("bear_"):
        return "risk_off"
    return "neutral"


def _build_regime_interactions(frame: pd.DataFrame, zscore_columns: list[str], regime_columns: list[str]) -> list[str]:
    interaction_columns: list[str] = []
    active_regime_columns = [
        column
        for column in regime_columns
        if column
        in {
            "feature_regime_strength",
            "feature_regime_risk_on",
            "feature_regime_neutral",
            "feature_regime_risk_off",
            "feature_regime_rotation",
        }
    ]
    # Pre-compute regime means for demeaned interactions.
    # For 0/1 dummy variables (risk_on/neutral/risk_off/rotation), the mean
    # equals the in-sample frequency, so demeaning removes the sparsity problem
    # where 15/20 interaction features are constantly zero in any given regime.
    regime_means: dict[str, float] = {}
    for regime_column in active_regime_columns:
        values = pd.to_numeric(frame[regime_column], errors="coerce").fillna(0.0)
        regime_means[regime_column] = float(values.mean())
    for zscore_column in zscore_columns:
        zscore = pd.to_numeric(frame[zscore_column], errors="coerce")
        zscore_name = zscore_column.removeprefix("feature_")
        for regime_column in active_regime_columns:
            regime_name = regime_column.removeprefix("feature_")
            column = f"feature_interaction_{zscore_name}_x_{regime_name}"
            regime_values = pd.to_numeric(frame[regime_column], errors="coerce").fillna(0.0)
            frame[column] = zscore * (regime_values - regime_means[regime_column])
            interaction_columns.append(column)
    return interaction_columns


def _should_build_regime_interactions(config: dict[str, Any] | None) -> bool:
    config = dict(config or {})
    scopes = set()
    for key in ("include_scopes", "model_include_scopes", "active_include_scopes"):
        scopes.update(str(item) for item in (config.get(key) or []))
    return "regime_interaction" in scopes


def _style_spread_columns(frame: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    columns = []
    spread_values = []
    for pair in _style_pairs(config):
        name = str(pair["name"])
        long_id = str(pair["long"])
        short_id = str(pair["short"])
        column = f"feature_style_spread_{name}_20d"
        columns.append(column)
        spread = _pair_spread_by_date(frame, long_id=long_id, short_id=short_id)
        exposure = _pair_style_exposure(frame, long_id=long_id, short_id=short_id, config=config)
        frame[column] = frame["date"].map(spread).astype(float) * exposure
        spread_values.append(frame[column] * float(pair.get("weight", 1.0)))
    if spread_values:
        total_weight = sum(float(pair.get("weight", 1.0)) for pair in _style_pairs(config))
        if total_weight <= 0:
            total_weight = float(len(spread_values))
        frame["feature_style_spread_composite_20d"] = sum(spread_values) / float(total_weight)
    else:
        frame["feature_style_spread_composite_20d"] = np.nan
    columns.append("feature_style_spread_composite_20d")
    return columns


def _style_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = config.get("style_pairs")
    if pairs:
        return [dict(pair) for pair in pairs]
    return [
        {"name": "dividend_vs_growth", "long": "510880.XSHG", "short": "159915.XSHE", "weight": 1.0},
        {"name": "value_vs_growth", "long": "515180.XSHG", "short": "159915.XSHE", "weight": 1.0},
        {"name": "large_vs_small", "long": "515180.XSHG", "short": "159949.XSHE", "weight": 1.0},
    ]


def _pair_spread_by_date(frame: pd.DataFrame, *, long_id: str, short_id: str) -> pd.Series:
    rows = frame[frame["order_book_id"].isin([long_id, short_id])]
    index = pd.Index(frame["date"].unique())
    if rows.empty:
        return pd.Series(0.0, index=index)
    pivot = rows.pivot(index="date", columns="order_book_id", values="feature_momentum_20d")
    long_momentum = pivot.get(long_id, pd.Series(0.0, index=pivot.index)).fillna(0.0)
    short_momentum = pivot.get(short_id, pd.Series(0.0, index=pivot.index)).fillna(0.0)
    return (long_momentum - short_momentum).reindex(index, fill_value=0.0)


def _pair_style_exposure(
    frame: pd.DataFrame,
    *,
    long_id: str,
    short_id: str,
    config: dict[str, Any],
) -> pd.Series:
    signed_fallback = pd.Series(
        np.select(
            [
                frame["order_book_id"].astype(str).eq(long_id),
                frame["order_book_id"].astype(str).eq(short_id),
            ],
            [1.0, -1.0],
            default=0.0,
        ),
        index=frame.index,
        dtype=float,
    )

    # Detect degradation conditions and issue warnings
    fallback_reason = None
    if "feature_return_1d" not in frame.columns:
        fallback_reason = "feature_return_1d column missing"
    else:
        rows = frame.dropna(subset=["date", "order_book_id"]).copy()
        if rows.empty:
            fallback_reason = "no valid rows after dropping NA"
        else:
            pivot = rows.pivot_table(
                index="date",
                columns="order_book_id",
                values="feature_return_1d",
                aggfunc="last",
            ).sort_index()
            if long_id not in pivot.columns:
                fallback_reason = f"long ETF {long_id!r} not found in data"
            elif short_id not in pivot.columns:
                fallback_reason = f"short ETF {short_id!r} not found in data"

    if fallback_reason is not None:
        logger.warning(
            f"Style exposure calculation degraded to signed_fallback for pair ({long_id}, {short_id}): "
            f"{fallback_reason}. This may indicate ETF delisting or data availability issues."
        )
        return signed_fallback

    if "feature_return_1d" not in frame.columns:
        return signed_fallback
    rows = frame.dropna(subset=["date", "order_book_id"]).copy()
    if rows.empty:
        return signed_fallback
    pivot = rows.pivot_table(
        index="date",
        columns="order_book_id",
        values="feature_return_1d",
        aggfunc="last",
    ).sort_index()
    if long_id not in pivot.columns or short_id not in pivot.columns:
        return signed_fallback
    factor_return = (pivot[long_id] - pivot[short_id]).replace([np.inf, -np.inf], np.nan)
    window = _style_exposure_window(config)
    min_periods = _style_exposure_min_periods(config)
    # Align ddof between rolling var/cov to avoid systematic beta distortion.
    # Pandas rolling.cov defaults to ddof=1; we set both explicitly for clarity.
    factor_var = factor_return.rolling(window, min_periods=min_periods).var(ddof=1).replace(0.0, np.nan)
    beta_by_id: dict[str, pd.Series] = {}
    for order_book_id in pivot.columns:
        asset_return = pd.to_numeric(pivot[order_book_id], errors="coerce")
        beta_by_id[str(order_book_id)] = (
            asset_return.rolling(window, min_periods=min_periods).cov(factor_return, ddof=1) / factor_var
        )
    beta = pd.DataFrame(beta_by_id, index=pivot.index).replace([np.inf, -np.inf], np.nan)
    if long_id not in beta.columns or short_id not in beta.columns:
        return signed_fallback
    denominator = (beta[long_id] - beta[short_id]).replace(0.0, np.nan)
    exposure = beta.sub(beta[short_id], axis=0).mul(2.0).div(denominator, axis=0) - 1.0
    exposure = exposure.clip(lower=-_style_beta_clip(config), upper=_style_beta_clip(config))
    lookup = exposure.reset_index().melt(
        id_vars="date",
        var_name="order_book_id",
        value_name="_style_exposure",
    )
    lookup["date"] = pd.to_datetime(lookup["date"])
    lookup["order_book_id"] = lookup["order_book_id"].astype(str)
    row_keys = frame[["date", "order_book_id"]].copy()
    row_keys["_row_order"] = range(len(row_keys))
    row_keys["date"] = pd.to_datetime(row_keys["date"])
    row_keys["order_book_id"] = row_keys["order_book_id"].astype(str)
    matched = row_keys.merge(lookup, on=["date", "order_book_id"], how="left").sort_values("_row_order")
    result = pd.Series(matched["_style_exposure"].to_numpy(dtype=float), index=frame.index)
    return result.fillna(signed_fallback).astype(float)


def _benchmark_momentum_by_date(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp, int], float]:
    values: dict[tuple[str, pd.Timestamp, int], float] = {}
    for row in frame[["order_book_id", "date", "feature_momentum_20d", "feature_momentum_60d"]].itertuples(index=False):
        values[(row.order_book_id, row.date, 20)] = row.feature_momentum_20d
        values[(row.order_book_id, row.date, 60)] = row.feature_momentum_60d
    return values


def _rolling_zscore(frame: pd.DataFrame, column: str, *, window: int, min_periods: int, winsorize_z: float) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    grouped = values.groupby(frame["order_book_id"], sort=False)
    rolling_mean = grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).mean())
    rolling_std = grouped.transform(lambda series: series.rolling(window, min_periods=min_periods).std(ddof=0))
    zscore = ((values - rolling_mean) / rolling_std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return zscore.fillna(0.0).clip(lower=-float(winsorize_z), upper=float(winsorize_z))


def _mask_scope_to_asset_type(frame: pd.DataFrame, columns: list[str], *, asset_type: str) -> None:
    if not columns:
        return
    mask = frame["asset_type"].astype(str).str.lower().eq(asset_type)
    for column in columns:
        if column in frame.columns:
            frame.loc[~mask, column] = np.nan


def _normalize_metadata(universe_metadata: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    config = config or {}
    if universe_metadata is None or universe_metadata.empty:
        return pd.DataFrame(columns=["order_book_id", "asset_type", "universe_layer", "industry", "benchmark_id"])
    metadata = universe_metadata.copy()
    if "order_book_id" not in metadata.columns:
        raise ValueError("universe_metadata must include order_book_id")
    metadata["order_book_id"] = metadata["order_book_id"].astype(str)
    for column, default in (
        ("asset_type", "stock"),
        ("universe_layer", None),
        ("industry", "Unknown"),
        ("benchmark_id", _core_proxy_id(config)),
    ):
        if column not in metadata.columns:
            metadata[column] = default
    return metadata[["order_book_id", "asset_type", "universe_layer", "industry", "benchmark_id"]].drop_duplicates("order_book_id", keep="last")


def _core_proxy_id(config: dict[str, Any]) -> str:
    preferred = config.get("preferred_benchmark_ids")
    if preferred:
        return str(list(preferred)[0])
    return str(config.get("core_proxy_id", config.get("benchmark_id", "510300.XSHG")))


def _has_configured_core_proxy(config: dict[str, Any]) -> bool:
    return bool(config.get("core_proxy_id") or config.get("benchmark_id") or config.get("preferred_benchmark_ids"))


def _normalization_window(config: dict[str, Any]) -> int:
    normalization = dict(config.get("normalization") or {})
    windows = normalization.get("windows", [60])
    return int(normalization.get("window", windows[0] if isinstance(windows, list) and windows else 60))


def _normalization_min_periods(config: dict[str, Any]) -> int:
    return int((config.get("normalization") or {}).get("min_periods", 20))


def _normalization_winsorize_z(config: dict[str, Any]) -> float:
    return float((config.get("normalization") or {}).get("winsorize_z", 4.0))


def _style_exposure_window(config: dict[str, Any]) -> int:
    return int(config.get("style_exposure_window", 60))


def _style_exposure_min_periods(config: dict[str, Any]) -> int:
    return int(config.get("style_exposure_min_periods", 20))


def _style_beta_clip(config: dict[str, Any]) -> float:
    return float(config.get("style_beta_clip", 2.0))
