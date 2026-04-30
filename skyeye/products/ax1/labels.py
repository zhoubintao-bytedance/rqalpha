"""AX1 label builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from skyeye.products.ax1._common import require_columns


@dataclass(frozen=True)
class MultiHorizonLabelBuilder:
    """按多个持有期生成 forward return、relative net return 和 forward realized volatility。"""

    horizons: Sequence[int] = (5, 10, 20)
    volatility_horizons: Sequence[int] = ()
    winsorize_quantiles: tuple[float, float] | None = None
    relative_return_enabled: bool = False
    relative_group_columns: Sequence[str] = ()
    relative_min_group_count: int = 2
    relative_fallback: str = "date"
    trading_days_per_year: int = 244
    cost_config: Mapping | None = None
    asset_type_column: str = "asset_type"
    entry_lag_days: int = 0
    price_column: str = "close"
    adjusted_price_column: str | None = None
    adjustment_factor_column: str | None = None

    def __post_init__(self) -> None:
        normalized = tuple(int(horizon) for horizon in self.horizons)
        if not normalized or any(horizon <= 0 for horizon in normalized):
            raise ValueError("horizons must be positive integers")
        object.__setattr__(self, "horizons", normalized)

        vol_horizons = tuple(int(horizon) for horizon in self.volatility_horizons)
        if any(horizon <= 0 for horizon in vol_horizons):
            raise ValueError("volatility_horizons must be positive integers")
        object.__setattr__(self, "volatility_horizons", vol_horizons)

        if self.winsorize_quantiles is not None:
            if len(self.winsorize_quantiles) != 2:
                raise ValueError("winsorize_quantiles must be a (lower, upper) pair")
            lo, hi = float(self.winsorize_quantiles[0]), float(self.winsorize_quantiles[1])
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError("winsorize_quantiles must satisfy 0 <= lower < upper <= 1")
            object.__setattr__(self, "winsorize_quantiles", (lo, hi))
        group_columns = tuple(str(column) for column in self.relative_group_columns)
        object.__setattr__(self, "relative_group_columns", group_columns)
        if self.relative_return_enabled and not group_columns:
            raise ValueError("relative_group_columns must be non-empty when relative return labels are enabled")
        if int(self.relative_min_group_count) < 2:
            raise ValueError("relative_min_group_count must be at least 2")
        object.__setattr__(self, "relative_min_group_count", int(self.relative_min_group_count))
        fallback = str(self.relative_fallback)
        if fallback not in {"date", "none"}:
            raise ValueError("relative_fallback must be 'date' or 'none'")
        object.__setattr__(self, "relative_fallback", fallback)

        if int(self.trading_days_per_year) <= 0:
            raise ValueError("trading_days_per_year must be positive")
        object.__setattr__(self, "trading_days_per_year", int(self.trading_days_per_year))
        if int(self.entry_lag_days) < 0:
            raise ValueError("entry_lag_days must be non-negative")
        object.__setattr__(self, "entry_lag_days", int(self.entry_lag_days))

    def build(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        require_columns(raw_df, ["date", "order_book_id"], entity="raw_df")
        if raw_df.empty:
            return raw_df.copy()

        working = raw_df.copy()
        working["_ax1_original_order"] = np.arange(len(working))
        working["_ax1_date"] = pd.to_datetime(working["date"])
        working = working.sort_values(["order_book_id", "_ax1_date"])

        label_price = _label_price_series(
            working,
            price_column=self.price_column,
            adjusted_price_column=self.adjusted_price_column,
            adjustment_factor_column=self.adjustment_factor_column,
        )
        grouped_label_price = label_price.groupby(working["order_book_id"])
        grouped_dates = working["_ax1_date"].groupby(working["order_book_id"])
        entry_price = grouped_label_price.shift(-self.entry_lag_days)
        entry_dates = grouped_dates.shift(-self.entry_lag_days)
        if self.entry_lag_days > 0:
            working["label_entry_date"] = entry_dates

        # 1. forward return label per horizon
        for horizon in self.horizons:
            future_price = grouped_label_price.shift(-(self.entry_lag_days + horizon))
            denominator = entry_price if self.entry_lag_days > 0 else label_price
            label = future_price / denominator - 1.0
            label = label.replace([np.inf, -np.inf], np.nan)
            working[f"label_return_{horizon}d"] = label
            if self.entry_lag_days > 0:
                working[f"label_exit_date_{horizon}d"] = grouped_dates.shift(-(self.entry_lag_days + horizon))

        # 2. cost-aware absolute net return labels. These remain realized-return labels.
        cost_enabled = bool((self.cost_config or {}).get("enabled", False))
        if cost_enabled or self.relative_return_enabled:
            cost_rates = _expected_cost_rates(
                working,
                cost_config=self.cost_config or {},
                asset_type_column=self.asset_type_column,
            )
            for horizon in self.horizons:
                gross_column = f"label_return_{horizon}d"
                working[f"label_net_return_{horizon}d"] = working[gross_column] - cost_rates

        # 3. peer-relative net return labels. These are the primary ETF rotation training labels.
        if self.relative_return_enabled:
            missing_groups = [column for column in self.relative_group_columns if column not in working.columns]
            if missing_groups:
                raise ValueError(f"raw_df missing relative group columns: {missing_groups}")
            for horizon in self.horizons:
                net_column = f"label_net_return_{horizon}d"
                working[f"label_relative_net_return_{horizon}d"] = _relative_net_return_series(
                    working,
                    working[net_column],
                    date_column="_ax1_date",
                    group_columns=self.relative_group_columns,
                    min_group_count=self.relative_min_group_count,
                    fallback=self.relative_fallback,
                    winsorize_quantiles=self.winsorize_quantiles,
                )

        # 4. forward realized volatility per horizon (optional)
        if self.volatility_horizons:
            daily_return = grouped_label_price.pct_change(fill_method=None)
            annualization = float(np.sqrt(self.trading_days_per_year))
            for horizon in self.volatility_horizons:
                working[f"label_volatility_{horizon}d"] = _forward_rolling_std(
                    daily_return,
                    working["order_book_id"],
                    horizon,
                    start_lag_days=self.entry_lag_days,
                ) * annualization

        return (
            working.sort_values("_ax1_original_order")
            .drop(columns=["_ax1_original_order", "_ax1_date"])
            .reset_index(drop=True)
        )


def _forward_rolling_std(
    daily_return: pd.Series,
    groups: pd.Series,
    horizon: int,
    *,
    start_lag_days: int = 0,
) -> pd.Series:
    """对每个资产计算 forward realized volatility；不足 horizon 个观测则为 NaN。"""
    shifted = daily_return.groupby(groups).shift(-(int(start_lag_days) + 1))
    result = pd.Series(np.nan, index=daily_return.index, dtype=float)
    for _, positions in groups.groupby(groups).groups.items():
        segment = shifted.loc[positions]
        reversed_std = segment[::-1].rolling(window=horizon, min_periods=horizon).std()
        result.loc[positions] = reversed_std[::-1].to_numpy()
    return result


def _relative_net_return_series(
    frame: pd.DataFrame,
    values: pd.Series,
    *,
    date_column: str,
    group_columns: Sequence[str],
    min_group_count: int,
    fallback: str,
    winsorize_quantiles: tuple[float, float] | None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    assigned = pd.Series(False, index=frame.index, dtype=bool)
    group_columns = tuple(group_columns)

    for _, day_index in frame.groupby(frame[date_column], sort=False).groups.items():
        day_positions = pd.Index(day_index)
        if group_columns:
            day_frame = frame.loc[day_positions, list(group_columns)]
            grouped = day_frame.groupby(list(group_columns), sort=False, dropna=False)
            for _, local_positions in grouped.groups.items():
                positions = pd.Index(local_positions)
                relative = _demeaned_peer_values(
                    numeric.loc[positions],
                    min_group_count=min_group_count,
                    winsorize_quantiles=winsorize_quantiles,
                )
                if relative.notna().any():
                    result.loc[positions] = relative
                    assigned.loc[positions] = True
        if fallback == "date":
            fallback_positions = day_positions[~assigned.loc[day_positions].to_numpy()]
            if len(fallback_positions) == 0:
                continue
            relative = _demeaned_peer_values(
                numeric.loc[day_positions],
                min_group_count=min_group_count,
                winsorize_quantiles=winsorize_quantiles,
            )
            result.loc[fallback_positions] = relative.loc[fallback_positions]
    return result


def _demeaned_peer_values(
    values: pd.Series,
    *,
    min_group_count: int,
    winsorize_quantiles: tuple[float, float] | None,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < int(min_group_count):
        return pd.Series(np.nan, index=values.index, dtype=float)
    if winsorize_quantiles is not None:
        lo, hi = winsorize_quantiles
        numeric = numeric.clip(lower=valid.quantile(lo), upper=valid.quantile(hi))
        valid = numeric.dropna()
    return numeric - float(valid.mean())


def _label_price_series(
    frame: pd.DataFrame,
    *,
    price_column: str,
    adjusted_price_column: str | None,
    adjustment_factor_column: str | None,
) -> pd.Series:
    price_column = str(price_column or "close")
    adjusted_price_column = str(adjusted_price_column) if adjusted_price_column else None
    adjustment_factor_column = str(adjustment_factor_column) if adjustment_factor_column else None

    if adjusted_price_column:
        if adjusted_price_column in frame.columns:
            return pd.to_numeric(frame[adjusted_price_column], errors="coerce")
        if not adjustment_factor_column or adjustment_factor_column not in frame.columns:
            require_columns(frame, [price_column], entity="raw_df")
            return pd.to_numeric(frame[price_column], errors="coerce")

    if adjustment_factor_column:
        if adjustment_factor_column not in frame.columns:
            raise ValueError(f"raw_df missing adjustment factor column: {adjustment_factor_column}")
        require_columns(frame, [price_column], entity="raw_df")
        price = pd.to_numeric(frame[price_column], errors="coerce")
        factor = pd.to_numeric(frame[adjustment_factor_column], errors="coerce")
        return price * factor

    require_columns(frame, [price_column], entity="raw_df")
    return pd.to_numeric(frame[price_column], errors="coerce")


def _expected_cost_rates(
    frame: pd.DataFrame,
    *,
    cost_config: Mapping,
    asset_type_column: str,
) -> pd.Series:
    asset_types = _asset_type_series(frame, asset_type_column, str(cost_config.get("default_asset_type", "stock")))
    stock_rate = _round_trip_asset_cost(cost_config.get("stock", {}))
    etf_rate = _round_trip_asset_cost(cost_config.get("etf", {}))
    rates = asset_types.map({"stock": stock_rate, "etf": etf_rate}).fillna(stock_rate)
    return rates.astype(float)


def _asset_type_series(frame: pd.DataFrame, asset_type_column: str, default_asset_type: str) -> pd.Series:
    if asset_type_column in frame.columns:
        raw = frame[asset_type_column]
    else:
        raw = pd.Series(default_asset_type, index=frame.index)
    normalized = raw.astype(str).str.strip().str.lower()
    normalized = normalized.replace(
        {
            "cs": "stock",
            "equity": "stock",
            "common_stock": "stock",
            "fund": "etf",
            "index_fund": "etf",
        }
    )
    return normalized.where(normalized.isin({"stock", "etf"}), str(default_asset_type).lower())


def _round_trip_asset_cost(config: Mapping) -> float:
    commission_rate = float(config.get("commission_rate", 0.0))
    stamp_tax_rate = float(config.get("stamp_tax_rate", 0.0))
    slippage_rate = float(config.get("slippage_bps", 0.0)) / 10000.0
    impact_rate = float(config.get("impact_bps", 0.0)) / 10000.0
    return (2.0 * commission_rate) + stamp_tax_rate + (2.0 * slippage_rate) + impact_rate
