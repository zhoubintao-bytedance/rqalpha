"""AX1 execution smoothing."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from skyeye.products.ax1._common import allocate_with_cap, apply_group_cap, require_columns

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutionSmoother:
    """执行层做权重聚合、基础过滤和最小执行约束。"""

    min_weight: float = 0.0
    max_turnover: float | None = None
    buffer_weight: float = 0.0
    no_trade_buffer_weight: float | None = None
    min_trade_value: float = 0.0
    portfolio_value: float | None = None
    target_gross_weight: float | None = None
    max_weight: float | None = None
    max_industry_weight: float | None = None
    net_alpha_threshold: float | None = None
    net_alpha_column: str = "net_alpha"

    # --- advanced execution constraints ---
    # capacity: max_trade_value_per_name = liquidity(dollar_volume) * participation_rate
    participation_rate: float | None = None
    liquidity_column: str = "dollar_volume"

    # t+1: today bought portion cannot be sold today.
    # requires current_weights as DataFrame with `today_buy_weight`.
    t_plus_one_lock: bool = False
    today_buy_weight_column: str = "today_buy_weight"

    def __post_init__(self) -> None:
        if self.no_trade_buffer_weight is not None and self.no_trade_buffer_weight < 0:
            raise ValueError("no_trade_buffer_weight must be non-negative")
        if self.no_trade_buffer_weight is not None:
            object.__setattr__(self, "buffer_weight", float(self.no_trade_buffer_weight))
        if self.min_weight < 0:
            raise ValueError("min_weight must be non-negative")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise ValueError("max_turnover must be non-negative")
        if self.buffer_weight < 0:
            raise ValueError("buffer_weight must be non-negative")
        if self.min_trade_value < 0:
            raise ValueError("min_trade_value must be non-negative")
        if self.portfolio_value is not None and self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        if self.target_gross_weight is not None and self.target_gross_weight < 0:
            raise ValueError("target_gross_weight must be non-negative")
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError("max_weight must be positive")
        if self.max_industry_weight is not None and self.max_industry_weight <= 0:
            raise ValueError("max_industry_weight must be positive")
        if self.net_alpha_threshold is not None and not self.net_alpha_column:
            raise ValueError("net_alpha_column must be provided when net_alpha_threshold is set")
        if self.participation_rate is not None:
            if self.participation_rate < 0:
                raise ValueError("participation_rate must be non-negative")
            if self.participation_rate > 1.0 + 1e-12:
                raise ValueError("participation_rate must be <= 1.0")
        if not isinstance(self.t_plus_one_lock, bool):
            raise ValueError("t_plus_one_lock must be a bool")
        if not str(self.liquidity_column or "").strip():
            raise ValueError("liquidity_column must be non-empty")
        if not str(self.today_buy_weight_column or "").strip():
            raise ValueError("today_buy_weight_column must be non-empty")

    def smooth(self, target_weights: pd.DataFrame, current_weights=None) -> pd.DataFrame:
        if target_weights is None or target_weights.empty:
            return _empty_smoothed()
        require_columns(target_weights, ["date", "order_book_id", "target_weight"], entity="target_weights")

        frame = target_weights.dropna(subset=["date", "order_book_id", "target_weight"]).copy()
        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce")
        frame = frame.dropna(subset=["target_weight"])

        aggregated = frame.groupby(["date", "order_book_id"], as_index=False, sort=True)["target_weight"].sum()
        metadata_candidates = ["industry", "asset_type", "universe_layer"]
        if self.net_alpha_column not in metadata_candidates:
            metadata_candidates.append(self.net_alpha_column)
        if self.participation_rate is not None and float(self.participation_rate) > 0:
            if self.liquidity_column not in metadata_candidates:
                metadata_candidates.append(self.liquidity_column)
        metadata_columns = [column for column in metadata_candidates if column in frame.columns]
        if metadata_columns:
            metadata = frame.drop_duplicates(["date", "order_book_id"], keep="last")[
                ["date", "order_book_id"] + metadata_columns
            ]
            aggregated = aggregated.merge(metadata, on=["date", "order_book_id"], how="left")
        if self.min_weight > 0:
            aggregated = aggregated[aggregated["target_weight"].abs() >= float(self.min_weight)]

        has_current_weights = current_weights is not None
        current_map = _coerce_current_weights(current_weights)
        today_buy_map = _coerce_today_buy_weights(
            current_weights,
            enabled=bool(self.t_plus_one_lock),
            column=str(self.today_buy_weight_column),
        )
        rows = []
        for date, day_df in aggregated.groupby("date", sort=True):
            normalized = _normalize_day(
                day_df.copy(),
                target_gross_weight=self._target_gross_weight(),
                max_weight=self.max_weight,
                max_industry_weight=self.max_industry_weight,
            )
            if normalized.empty:
                if not (has_current_weights and self.t_plus_one_lock and today_buy_map):
                    continue
                normalized = self._apply_min_execution_constraints(day_df.copy(), current_map, today_buy_map)
            elif has_current_weights:
                normalized = self._apply_min_execution_constraints(normalized, current_map, today_buy_map)
            normalized["component"] = "smoothed"
            normalized = normalized[normalized["target_weight"].abs() > 0]
            columns = ["date", "order_book_id", "target_weight", "component"]
            for metadata_column in ("industry", "asset_type", "universe_layer"):
                if metadata_column in normalized.columns:
                    columns.append(metadata_column)
            rows.extend(normalized[columns].to_dict("records"))
        columns = ["date", "order_book_id", "target_weight", "component"]
        for metadata_column in ("industry", "asset_type", "universe_layer"):
            if any(metadata_column in row for row in rows):
                columns.append(metadata_column)
        return pd.DataFrame(rows, columns=columns)

    def _target_gross_weight(self) -> float:
        if self.target_gross_weight is None:
            return 1.0
        return float(self.target_gross_weight)

    def _apply_min_execution_constraints(
        self,
        day_targets: pd.DataFrame,
        current_weights: dict[str, float],
        today_buy_weights: dict[str, float],
    ) -> pd.DataFrame:
        date = day_targets["date"].iloc[0]
        target_by_id = day_targets.set_index("order_book_id")["target_weight"].to_dict()
        industry_by_id = {}
        if "industry" in day_targets.columns:
            industry_by_id = day_targets.set_index("order_book_id")["industry"].to_dict()
        net_alpha_by_id = {}
        if self.net_alpha_column in day_targets.columns:
            net_alpha_by_id = day_targets.set_index("order_book_id")[self.net_alpha_column].to_dict()
        liquidity_by_id = {}
        if self.participation_rate is not None and self.participation_rate > 0:
            if self.liquidity_column not in day_targets.columns:
                raise ValueError(f"capacity requires liquidity column in targets: {self.liquidity_column}")
            liquidity_by_id = day_targets.set_index("order_book_id")[self.liquidity_column].to_dict()
        metadata_by_id = _metadata_by_id(day_targets)
        order_book_ids = sorted(set(target_by_id) | set(current_weights))
        rows = []
        for order_book_id in order_book_ids:
            target_weight = float(target_by_id.get(order_book_id, 0.0))
            current_weight = float(current_weights.get(order_book_id, 0.0))
            delta_weight = target_weight - current_weight
            if abs(delta_weight) < self.buffer_weight:
                target_weight = current_weight
                delta_weight = 0.0
            if self._is_blocked_by_net_alpha_gate(delta_weight, net_alpha_by_id.get(order_book_id)):
                target_weight = current_weight
                delta_weight = 0.0

            # T+1: today bought portion cannot be sold today.
            if self.t_plus_one_lock and delta_weight < 0:
                locked_weight = float(today_buy_weights.get(order_book_id, 0.0))
                if locked_weight < -1e-12:
                    raise ValueError("today_buy_weight must be non-negative")
                if locked_weight > current_weight + 1e-9:
                    raise ValueError(
                        "today_buy_weight cannot exceed current_weight for {}".format(order_book_id)
                    )
                min_target = max(0.0, locked_weight)
                if target_weight < min_target:
                    target_weight = min_target
                    delta_weight = target_weight - current_weight

            # Capacity: cap absolute trade value by liquidity * participation_rate.
            if self._capacity_enabled() and delta_weight != 0:
                liquidity_value = liquidity_by_id.get(order_book_id)
                delta_weight = self._cap_delta_by_capacity(delta_weight, liquidity_value)
                target_weight = current_weight + delta_weight

            if self._is_below_min_trade_value(delta_weight):
                target_weight = current_weight
            row = {
                "date": date,
                "order_book_id": order_book_id,
                "target_weight": target_weight,
            }
            if order_book_id in industry_by_id:
                row["industry"] = industry_by_id[order_book_id]
            if order_book_id in net_alpha_by_id:
                row[self.net_alpha_column] = net_alpha_by_id[order_book_id]
            row.update(metadata_by_id.get(order_book_id, {}))
            rows.append(row)

        columns = ["date", "order_book_id", "target_weight"]
        for metadata_column in ("industry", "asset_type", "universe_layer"):
            if any(metadata_column in row for row in rows):
                columns.append(metadata_column)
        if net_alpha_by_id:
            columns.append(self.net_alpha_column)
        constrained = pd.DataFrame(rows, columns=columns)
        constrained = self._scale_to_max_turnover(constrained, current_weights)
        if self.net_alpha_column in constrained.columns:
            constrained = constrained.drop(columns=[self.net_alpha_column])
        constrained_gross = float(
            pd.to_numeric(constrained["target_weight"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .sum()
        )
        normalized = _normalize_day(
            constrained,
            target_gross_weight=min(self._target_gross_weight(), constrained_gross),
            max_weight=self.max_weight,
            max_industry_weight=self.max_industry_weight,
        )

        # Post-enforce advanced execution constraints to avoid being undone by normalization.
        normalized_map = (
            normalized.set_index("order_book_id")["target_weight"].astype(float).to_dict()
            if normalized is not None and not normalized.empty
            else {}
        )
        final_rows = []
        final_ids = sorted(set(normalized_map) | set(current_weights) | set(today_buy_weights))
        for order_book_id in final_ids:
            target_weight = float(normalized_map.get(order_book_id, 0.0))
            current_weight = float(current_weights.get(order_book_id, 0.0))
            delta_weight = target_weight - current_weight

            # T+1 floor (no sell below today's bought portion).
            if self.t_plus_one_lock and delta_weight < 0:
                locked_weight = float(today_buy_weights.get(order_book_id, 0.0))
                if locked_weight < -1e-12:
                    raise ValueError("today_buy_weight must be non-negative")
                if locked_weight > current_weight + 1e-9:
                    raise ValueError(
                        "today_buy_weight cannot exceed current_weight for {}".format(order_book_id)
                    )
                min_target = max(0.0, locked_weight)
                if target_weight < min_target:
                    target_weight = min_target
                    delta_weight = target_weight - current_weight

            # Capacity cap (no trade above liquidity * participation_rate).
            if self._capacity_enabled() and delta_weight != 0:
                liquidity_value = liquidity_by_id.get(order_book_id)
                delta_weight = self._cap_delta_by_capacity(delta_weight, liquidity_value)
                target_weight = current_weight + delta_weight

            # Re-check min trade value after post adjustments.
            if self._is_below_min_trade_value(delta_weight):
                target_weight = current_weight

            row = {
                "date": date,
                "order_book_id": order_book_id,
                "target_weight": float(target_weight),
            }
            if order_book_id in industry_by_id:
                row["industry"] = industry_by_id[order_book_id]
            row.update(metadata_by_id.get(order_book_id, {}))
            final_rows.append(row)

        final = pd.DataFrame(final_rows, columns=columns)
        final = final[final["target_weight"].astype(float) > 0].copy()
        # Ensure max turnover is still respected after post-enforcement.
        final = self._scale_to_max_turnover(final, current_weights)
        return final

    def _is_blocked_by_net_alpha_gate(self, delta_weight: float, net_alpha) -> bool:
        if self.net_alpha_threshold is None or self.net_alpha_threshold <= 0 or delta_weight <= 0:
            return False
        if pd.isna(net_alpha):
            return True
        return float(net_alpha) < float(self.net_alpha_threshold)

    def _is_below_min_trade_value(self, delta_weight: float) -> bool:
        if self.min_trade_value <= 0 or self.portfolio_value is None:
            return False
        return abs(delta_weight) * self.portfolio_value < self.min_trade_value

    def _capacity_enabled(self) -> bool:
        return (
            self.participation_rate is not None
            and float(self.participation_rate) > 0
            and self.portfolio_value is not None
            and float(self.portfolio_value) > 0
        )

    def _cap_delta_by_capacity(self, delta_weight: float, liquidity_value) -> float:
        if not self._capacity_enabled() or delta_weight == 0:
            return delta_weight
        if liquidity_value is None or pd.isna(liquidity_value):
            raise ValueError("capacity requires non-null liquidity values")
        liquidity_value = float(liquidity_value)
        if liquidity_value < 0:
            raise ValueError("capacity liquidity values must be non-negative")
        max_trade_value = liquidity_value * float(self.participation_rate)
        if max_trade_value <= 0:
            return 0.0
        delta_value = abs(float(delta_weight)) * float(self.portfolio_value)
        if delta_value <= max_trade_value + 1e-12:
            return float(delta_weight)
        capped_delta = max_trade_value / float(self.portfolio_value)
        return float(capped_delta if delta_weight > 0 else -capped_delta)

    def _scale_to_max_turnover(
        self, day_targets: pd.DataFrame, current_weights: dict[str, float]
    ) -> pd.DataFrame:
        if self.max_turnover is None:
            return day_targets
        deltas = day_targets["target_weight"] - day_targets["order_book_id"].map(
            current_weights
        ).fillna(0.0)
        one_way_turnover = float(deltas.abs().sum() / 2.0)
        if one_way_turnover <= self.max_turnover or one_way_turnover <= 0:
            return day_targets

        # Alpha-aware prioritized reduction: sort trades by alpha descending
        # so highest-alpha trades are allocated budget first.
        scaled = day_targets.copy()
        current = scaled["order_book_id"].map(current_weights).fillna(0.0)
        scaled["_delta"] = scaled["target_weight"] - current

        # Compute alpha priority: |delta| * alpha, fallback to |delta| if no alpha.
        if self.net_alpha_column in scaled.columns:
            alpha = pd.to_numeric(scaled[self.net_alpha_column], errors="coerce").fillna(0.0)
            scaled["_alpha_priority"] = scaled["_delta"].abs() * alpha.abs()
        else:
            scaled["_alpha_priority"] = scaled["_delta"].abs()

        # Sort descending: highest-alpha trades get budget first.
        scaled = scaled.sort_values("_alpha_priority", ascending=False, kind="mergesort")

        budget = float(self.max_turnover)
        buy_budget = budget
        sell_budget = budget
        new_deltas = []
        for _, row in scaled.iterrows():
            delta = float(row["_delta"])
            if abs(delta) < 1e-15:
                new_deltas.append(0.0)
                continue
            if delta > 0:
                # Buy: cap delta to remaining buy budget.
                allowed = min(abs(delta), buy_budget)
                buy_budget -= allowed
                new_deltas.append(allowed if delta > 0 else 0.0)
            else:
                # Sell: cap delta to remaining sell budget.
                allowed = min(abs(delta), sell_budget)
                sell_budget -= allowed
                new_deltas.append(-allowed if delta < 0 else 0.0)

        scaled["_delta"] = new_deltas
        scaled["target_weight"] = current + scaled["_delta"]
        scaled = scaled.drop(columns=["_delta", "_alpha_priority"])
        return scaled

    def _apply_t_plus_one_and_capacity_rules(
        self, target_weights: pd.DataFrame, current_weights=None
    ) -> pd.DataFrame:
        """保留历史 API：advanced 规则已内联进 `_apply_min_execution_constraints`。"""
        return self.smooth(target_weights, current_weights)


def _empty_smoothed() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "order_book_id", "target_weight", "component"])


def _coerce_current_weights(current_weights) -> dict[str, float]:
    if current_weights is None:
        return {}
    if isinstance(current_weights, dict):
        items = current_weights.items()
        return {
            str(order_book_id): float(weight)
            for order_book_id, weight in items
            if pd.notna(weight)
        }
    if isinstance(current_weights, pd.DataFrame):
        if current_weights.empty:
            return {}
        require_columns(current_weights, ["order_book_id", "current_weight"], entity="target_weights")
        frame = current_weights.dropna(subset=["order_book_id", "current_weight"]).copy()
        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame["current_weight"] = pd.to_numeric(frame["current_weight"], errors="coerce")
        frame = frame.dropna(subset=["current_weight"])
        return frame.groupby("order_book_id")["current_weight"].sum().astype(float).to_dict()
    raise TypeError("current_weights must be a dict or DataFrame")


def _coerce_today_buy_weights(current_weights, *, enabled: bool, column: str) -> dict[str, float]:
    if not enabled:
        return {}
    if current_weights is None:
        return {}
    if isinstance(current_weights, dict):
        raise TypeError("t_plus_one_lock requires current_weights as a DataFrame")
    if isinstance(current_weights, pd.DataFrame):
        if current_weights.empty:
            return {}
        if column not in current_weights.columns:
            raise ValueError(f"t_plus_one_lock requires current_weights column: {column}")
        frame = current_weights.dropna(subset=["order_book_id", column]).copy()
        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=[column])
        values = frame.groupby("order_book_id")[column].sum().astype(float)
        return {str(order_book_id): float(value) for order_book_id, value in values.items() if pd.notna(value)}
    raise TypeError("current_weights must be a dict or DataFrame")


def _metadata_by_id(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    metadata_columns = [column for column in ("asset_type", "universe_layer") if column in frame.columns]
    if not metadata_columns:
        return {}
    values = frame.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id", keep="last")
    result: dict[str, dict[str, str]] = {}
    for _, row in values.iterrows():
        payload = {}
        for column in metadata_columns:
            if pd.notna(row[column]):
                payload[column] = str(row[column])
        result[str(row["order_book_id"])] = payload
    return result


def _normalize_day(
    day_targets: pd.DataFrame,
    target_gross_weight: float = 1.0,
    max_weight: float | None = None,
    max_industry_weight: float | None = None,
) -> pd.DataFrame:
    if day_targets.empty:
        return day_targets
    scores = day_targets.set_index("order_book_id")["target_weight"].astype(float).clip(lower=0.0)
    if scores.sum() <= 0 or target_gross_weight <= 0:
        return day_targets.iloc[0:0].copy()
    cap = float(max_weight) if max_weight is not None else float(target_gross_weight)
    weights = allocate_with_cap(scores, budget=float(target_gross_weight), max_weight=cap)
    if max_industry_weight is not None and "industry" in day_targets.columns:
        group_map = day_targets.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")["industry"]
        # Skip max_industry_weight when all ETFs are 'Unknown' (ETF industry classification is ambiguous)
        unique_industries = set(group_map.dropna().astype(str).unique())
        if unique_industries == {"Unknown"}:
            _logger.warning(
                "max_industry_weight constraint skipped: all instruments have industry='Unknown'. "
                "ETF industry classification is ambiguous; use exposure_group constraints for sector risk control."
            )
        else:
            weights = apply_group_cap(weights, group_map.to_dict(), float(target_gross_weight), float(max_industry_weight))
    normalized = day_targets.copy()
    normalized["target_weight"] = normalized["order_book_id"].map(weights).fillna(0.0)
    return normalized[normalized["target_weight"] > 0]
