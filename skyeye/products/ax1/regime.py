"""AX1 market regime detector."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Mapping

import pandas as pd

from skyeye.market_regime_layer import (
    MarketRegimeConfig,
    compute_market_regime,
    required_market_regime_history_days,
)


DEFAULT_BENCHMARK_IDS = ("510300.XSHG",)
REQUIRED_RAW_COLUMNS = ("date", "order_book_id", "close")
OPTIONAL_BAR_COLUMNS = ("high", "low", "volume")


@dataclass(frozen=True)
class RegimeDetectorConfig:
    preferred_benchmark_ids: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BENCHMARK_IDS)
    benchmark_source: str = "raw_core_proxy"
    core_proxy_method: str = "preferred_id"
    industry_source: str = "universe_industry_etfs"
    fallback_regime: str = "range_co_move"
    lookback_days: int = 0
    market_regime_config: MarketRegimeConfig = field(default_factory=MarketRegimeConfig)


class RegimeDetector:
    """Build AX1 regime inputs from the raw ETF panel and delegate regime math."""

    def __init__(self, config: RegimeDetectorConfig | Mapping[str, Any] | None = None) -> None:
        self.config = _coerce_config(config)

    def detect(
        self,
        raw_df: pd.DataFrame,
        *,
        as_of_date: str | pd.Timestamp,
        universe_metadata: pd.DataFrame,
        data_provider: Any | None = None,
    ) -> dict[str, Any]:
        self._validate_data_sources(data_provider)
        as_of_ts = pd.to_datetime(as_of_date)
        missing = [column for column in REQUIRED_RAW_COLUMNS if column not in raw_df.columns]
        if missing:
            raise ValueError(f"raw_df requires columns: {missing}")
        meta_missing = [
            column for column in ("order_book_id", "universe_layer", "asset_type") if column not in universe_metadata.columns
        ]
        if meta_missing:
            raise ValueError(f"universe_metadata requires columns: {meta_missing}")

        panel = raw_df.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        panel = panel[panel["date"] <= as_of_ts].copy()
        if panel.empty:
            return self._fallback_result(as_of_ts, [], "no_rows_on_or_before_as_of")

        benchmark_bars, benchmark_ids, source = self._build_benchmark_proxy(panel, universe_metadata)
        if benchmark_bars.empty or not benchmark_ids:
            return self._fallback_result(as_of_ts, benchmark_ids, "missing_benchmark_proxy")

        industry_close = self._build_industry_close(panel, universe_metadata)
        regime = compute_market_regime(
            benchmark_bars,
            industry_close=industry_close,
            cfg=self.config.market_regime_config,
        )
        diagnostics = dict(regime.diagnostics)
        diagnostics["industry_count"] = int(industry_close.shape[1]) if industry_close is not None else 0
        diagnostics["benchmark_proxy_source"] = source
        diagnostics["configured_lookback_days"] = int(self.config.lookback_days)
        diagnostics["required_history_days"] = int(
            required_market_regime_history_days(self.config.market_regime_config)
        )
        diagnostics["history_days"] = int(pd.Index(benchmark_bars.index).nunique())

        if _is_insufficient_history(diagnostics):
            return self._fallback_result(as_of_ts, benchmark_ids, "insufficient_history", diagnostics)

        market_regime = regime.regime
        rotation_state = "rotation" if market_regime.endswith("_rotation") else "co_move"
        return {
            "as_of_date": as_of_ts,
            "market_regime": market_regime,
            "strength": float(regime.strength),
            "risk_state": _risk_state(market_regime),
            "style_state": "balanced",
            "volatility_state": _volatility_state(diagnostics),
            "rotation_state": rotation_state,
            "source": source,
            "benchmark_proxy_ids": benchmark_ids,
            "diagnostics": diagnostics,
        }

    def detect_by_date(
        self,
        raw_df: pd.DataFrame,
        *,
        universe_metadata: pd.DataFrame,
        as_of_dates: list[str | pd.Timestamp] | None = None,
        data_provider: Any | None = None,
    ) -> dict[pd.Timestamp, dict[str, Any]]:
        """Return point-in-time regime states keyed by date."""
        if raw_df is None or raw_df.empty or "date" not in raw_df.columns:
            return {}
        dates = pd.to_datetime(as_of_dates if as_of_dates is not None else raw_df["date"].dropna().unique())
        states: dict[pd.Timestamp, dict[str, Any]] = {}
        frame = raw_df.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        for date in sorted(pd.Timestamp(item) for item in dates):
            history_days = int(frame.loc[frame["date"] <= date, "date"].nunique())
            if self.config.lookback_days > 0 and history_days < self.config.lookback_days:
                states[date] = self._fallback_result(
                    date,
                    [],
                    "warmup_lookback_days",
                    {
                        "configured_lookback_days": int(self.config.lookback_days),
                        "required_history_days": int(
                            required_market_regime_history_days(self.config.market_regime_config)
                        ),
                        "history_days": history_days,
                    },
                )
            else:
                states[date] = self.detect(
                    raw_df,
                    as_of_date=date,
                    universe_metadata=universe_metadata,
                    data_provider=data_provider,
                )
        return states

    def _validate_data_sources(self, data_provider: Any | None) -> None:
        del data_provider
        if self.config.benchmark_source == "data_provider":
            raise NotImplementedError("regime benchmark_source=data_provider is not implemented")
        if self.config.industry_source == "data_provider_indices":
            raise NotImplementedError("regime industry_source=data_provider_indices is not implemented")

    def _build_benchmark_proxy(
        self,
        panel: pd.DataFrame,
        universe_metadata: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str], str]:
        ids_in_panel = set(panel["order_book_id"].dropna().astype(str).unique())
        preferred_ids = [order_book_id for order_book_id in self.config.preferred_benchmark_ids if order_book_id in ids_in_panel]
        if self.config.core_proxy_method == "preferred_id" and preferred_ids:
            return (_bars_for_single_or_equal_weight(panel, preferred_ids), preferred_ids, "raw_preferred_benchmark")

        core_ids = _metadata_ids(universe_metadata, layer="core", asset_type="etf")
        core_ids = [order_book_id for order_book_id in core_ids if order_book_id in ids_in_panel]
        if core_ids:
            return (_bars_for_single_or_equal_weight(panel, core_ids), core_ids, "raw_core_equal_weight")
        return (pd.DataFrame(), [], "fallback")

    def _build_industry_close(self, panel: pd.DataFrame, universe_metadata: pd.DataFrame) -> pd.DataFrame | None:
        if self.config.industry_source not in {"universe_industry_etfs", "auto"}:
            return None
        industry_ids = _metadata_ids(universe_metadata, layer="industry", asset_type="etf")
        if not industry_ids:
            return None
        industry_panel = panel[panel["order_book_id"].astype(str).isin(industry_ids)]
        if industry_panel.empty:
            return None
        wide = industry_panel.pivot_table(index="date", columns="order_book_id", values="close", aggfunc="last")
        return wide.sort_index()

    def _fallback_result(
        self,
        as_of_date: pd.Timestamp,
        benchmark_proxy_ids: list[str],
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return _fallback_result(
            as_of_date,
            benchmark_proxy_ids,
            reason,
            fallback_regime=self.config.fallback_regime,
            diagnostics=diagnostics,
        )


def _coerce_config(config: RegimeDetectorConfig | Mapping[str, Any] | None) -> RegimeDetectorConfig:
    if config is None:
        return RegimeDetectorConfig()
    if isinstance(config, RegimeDetectorConfig):
        return config
    preferred = config.get("preferred_benchmark_ids", DEFAULT_BENCHMARK_IDS)
    market_cfg = config.get("market_regime_config") or config.get("cfg") or MarketRegimeConfig()
    return RegimeDetectorConfig(
        preferred_benchmark_ids=tuple(str(order_book_id) for order_book_id in preferred),
        benchmark_source=str(config.get("benchmark_source", "raw_core_proxy")),
        core_proxy_method=str(config.get("core_proxy_method", "preferred_id")),
        industry_source=str(config.get("industry_source", "universe_industry_etfs")),
        fallback_regime=str(config.get("fallback_regime", "range_co_move")),
        lookback_days=int(config.get("lookback_days", 0) or 0),
        market_regime_config=_coerce_market_regime_config(market_cfg),
    )


def _coerce_market_regime_config(config: MarketRegimeConfig | Mapping[str, Any] | None) -> MarketRegimeConfig:
    if config is None:
        return MarketRegimeConfig()
    if isinstance(config, MarketRegimeConfig):
        return config
    if not isinstance(config, Mapping):
        raise TypeError("market_regime_config must be a mapping or MarketRegimeConfig")
    allowed_fields = {item.name for item in fields(MarketRegimeConfig)}
    unknown = sorted(str(key) for key in config if str(key) not in allowed_fields)
    if unknown:
        raise ValueError(f"market_regime_config contains unknown keys: {unknown}")
    return MarketRegimeConfig(**dict(config))


def _metadata_ids(universe_metadata: pd.DataFrame, *, layer: str, asset_type: str) -> list[str]:
    meta = universe_metadata.copy()
    mask = (meta["universe_layer"].astype(str).str.lower() == layer.lower()) & (
        meta["asset_type"].astype(str).str.lower() == asset_type.lower()
    )
    return sorted(meta.loc[mask, "order_book_id"].dropna().astype(str).unique().tolist())


def _bars_for_single_or_equal_weight(panel: pd.DataFrame, order_book_ids: list[str]) -> pd.DataFrame:
    selected = panel[panel["order_book_id"].astype(str).isin(order_book_ids)].copy()
    if selected.empty:
        return pd.DataFrame()
    if len(order_book_ids) == 1:
        columns = ["date", "close", *[column for column in OPTIONAL_BAR_COLUMNS if column in selected.columns]]
        bars = selected[columns].sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        return bars.sort_index()

    wide = selected.pivot_table(index="date", columns="order_book_id", values="close", aggfunc="last").sort_index()
    normalized = wide / wide.ffill().bfill().iloc[0]
    bars = pd.DataFrame({"close": normalized.mean(axis=1) * 100.0}, index=normalized.index)
    if "volume" in selected.columns:
        volume = selected.pivot_table(index="date", columns="order_book_id", values="volume", aggfunc="last")
        bars["volume"] = volume.sum(axis=1)
    return bars


def _is_insufficient_history(diagnostics: dict[str, Any]) -> bool:
    direction_diagnostics = diagnostics.get("direction_diagnostics", {})
    return direction_diagnostics.get("reason") == "insufficient_history"


def _risk_state(market_regime: str) -> str:
    if market_regime.startswith("bull_"):
        return "risk_on"
    if market_regime.startswith("bear_"):
        return "risk_off"
    return "neutral"


def _volatility_state(diagnostics: dict[str, Any]) -> str:
    trendiness = diagnostics.get("trendiness")
    if trendiness is None:
        return "unknown"
    return "high" if float(trendiness) >= 0.65 else "normal"


def _fallback_result(
    as_of_date: pd.Timestamp,
    benchmark_proxy_ids: list[str],
    reason: str,
    *,
    fallback_regime: str = "range_co_move",
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out_diagnostics = dict(diagnostics or {})
    out_diagnostics["reason"] = reason
    rotation_state = "rotation" if str(fallback_regime).endswith("_rotation") else "co_move"
    return {
        "as_of_date": as_of_date,
        "market_regime": str(fallback_regime),
        "strength": 0.0,
        "risk_state": _risk_state(str(fallback_regime)),
        "style_state": "balanced",
        "volatility_state": "unknown",
        "rotation_state": rotation_state,
        "source": "fallback",
        "benchmark_proxy_ids": benchmark_proxy_ids,
        "diagnostics": out_diagnostics,
    }
