"""Lot-aware executable portfolio conversion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from skyeye.products.ax1._common import coerce_cost_config, normalize_asset_type, require_columns

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExecutablePortfolioResult:
    portfolio: pd.DataFrame
    orders: pd.DataFrame
    summary: dict[str, float | int]


@dataclass(frozen=True)
class ExecutablePortfolioOptimizer:
    portfolio_value: float
    lot_size: int = 100
    min_trade_value: float = 0.0
    max_order_count: int | None = None
    cost_config: object | None = None
    max_weight: float | None = None
    max_industry_weight: float | None = None

    def __post_init__(self) -> None:
        if self.portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive")
        if self.lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if self.min_trade_value < 0:
            raise ValueError("min_trade_value must be non-negative")
        if self.max_order_count is not None and self.max_order_count < 0:
            raise ValueError("max_order_count must be non-negative")
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError("max_weight must be positive")
        if self.max_industry_weight is not None and self.max_industry_weight <= 0:
            raise ValueError("max_industry_weight must be positive")

    def optimize(self, target_weights: pd.DataFrame, current_shares) -> ExecutablePortfolioResult:
        require_columns(target_weights, ["order_book_id", "target_weight", "price"], entity="target_weights")
        current_share_map = _current_share_map(current_shares)
        frame = target_weights.copy()
        if frame.empty:
            return ExecutablePortfolioResult(
                portfolio=_empty_portfolio(),
                orders=_empty_orders(),
                summary=_summary(_empty_orders()),
            )

        frame["order_book_id"] = frame["order_book_id"].astype(str)
        frame["target_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0)
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        if frame["price"].isna().any() or (frame["price"] <= 0).any():
            raise ValueError("price must be positive for all targets")

        rows = []
        for _, row in frame.iterrows():
            order_book_id = str(row["order_book_id"])
            price = float(row["price"])
            current = int(current_share_map.get(order_book_id, 0))
            desired_shares = float(row["target_weight"]) * float(self.portfolio_value) / price
            order_shares = _round_delta_to_lot(desired_shares - current, self.lot_size)
            order_value = float(order_shares * price)
            target_shares = current + order_shares
            trade_reason = "trade" if order_shares != 0 else "unchanged"
            if order_shares != 0 and abs(order_value) < float(self.min_trade_value):
                target_shares = current
                order_shares = 0
                order_value = 0.0
                trade_reason = "below_min_trade_value"

            output = row.to_dict()
            output.update(
                {
                    "target_shares": int(target_shares),
                    "price": price,
                    "position_value": float(target_shares * price),
                    "order_shares": int(order_shares),
                    "order_value": float(order_value),
                    "side": _order_side(order_shares),
                    "estimated_cost": _estimate_order_cost(order_value, self.cost_config, row.get("asset_type")),
                    "trade_reason": trade_reason,
                }
            )
            output["target_weight"] = float(target_shares * price / float(self.portfolio_value))
            rows.append(output)

        portfolio = pd.DataFrame(rows)
        portfolio = _apply_max_order_count(
            portfolio,
            self.max_order_count,
            self.cost_config,
            float(self.portfolio_value),
        )
        portfolio = _repair_hard_caps(
            portfolio,
            current_share_map,
            lot_size=int(self.lot_size),
            portfolio_value=float(self.portfolio_value),
            cost_config=self.cost_config,
            max_weight=self.max_weight,
            max_industry_weight=self.max_industry_weight,
        )
        portfolio = _apply_max_order_count(
            portfolio,
            self.max_order_count,
            self.cost_config,
            float(self.portfolio_value),
        )
        orders = _orders_from_portfolio(portfolio)
        return ExecutablePortfolioResult(
            portfolio=_portfolio_columns(portfolio),
            orders=orders,
            summary=_summary(orders),
        )


def _current_share_map(current_shares) -> dict[str, int]:
    if current_shares is None:
        return {}
    if isinstance(current_shares, Mapping):
        return {str(key): int(value) for key, value in current_shares.items()}
    if isinstance(current_shares, pd.DataFrame):
        require_columns(current_shares, ["order_book_id", "current_shares"], entity="target_weights")
        return {
            str(row["order_book_id"]): int(row["current_shares"])
            for _, row in current_shares.iterrows()
        }
    raise TypeError("current_shares must be a mapping or DataFrame")


def _round_delta_to_lot(delta_shares: float, lot_size: int) -> int:
    if abs(delta_shares) <= 0:
        return 0
    sign = 1 if delta_shares > 0 else -1
    rounded_abs = int((abs(delta_shares) / lot_size) + 0.5) * lot_size
    return sign * rounded_abs


def _apply_max_order_count(
    portfolio: pd.DataFrame,
    max_order_count: int | None,
    cost_config: object | None,
    portfolio_value: float,
) -> pd.DataFrame:
    if max_order_count is None:
        return portfolio
    active = portfolio[portfolio["order_shares"] != 0].copy()
    if len(active) <= max_order_count:
        return portfolio

    keep_index = set(
        active.assign(
            _is_sell=pd.to_numeric(active["order_shares"], errors="coerce").fillna(0.0) < 0,
            _abs_order_value=active["order_value"].abs(),
        )
        .sort_values(["_is_sell", "_abs_order_value"], ascending=[False, False], kind="mergesort")
        .head(max_order_count)
        .index
    )
    result = portfolio.copy()
    drop_mask = (result["order_shares"] != 0) & ~result.index.isin(keep_index)
    result.loc[drop_mask, "target_shares"] = (
        result.loc[drop_mask, "target_shares"] - result.loc[drop_mask, "order_shares"]
    ).astype(int)
    result.loc[drop_mask, "order_shares"] = 0
    result.loc[drop_mask, "order_value"] = 0.0
    result.loc[drop_mask, "estimated_cost"] = 0.0
    result.loc[drop_mask, "side"] = "none"
    result.loc[drop_mask, "trade_reason"] = "max_order_count"
    result.loc[drop_mask, "position_value"] = (
        result.loc[drop_mask, "target_shares"] * result.loc[drop_mask, "price"]
    ).astype(float)
    result.loc[drop_mask, "target_weight"] = result.loc[drop_mask, "position_value"] / portfolio_value
    result.loc[result["order_shares"] != 0, "estimated_cost"] = result.loc[
        result["order_shares"] != 0
    ].apply(lambda row: _estimate_order_cost(float(row["order_value"]), cost_config, row.get("asset_type")), axis=1)
    return result


def _repair_hard_caps(
    portfolio: pd.DataFrame,
    current_share_map: dict[str, int],
    *,
    lot_size: int,
    portfolio_value: float,
    cost_config: object | None,
    max_weight: float | None,
    max_industry_weight: float | None,
) -> pd.DataFrame:
    if portfolio.empty or (max_weight is None and max_industry_weight is None):
        return portfolio
    result = portfolio.copy()
    if max_weight is not None:
        result = _repair_single_name_cap(
            result,
            current_share_map,
            lot_size=lot_size,
            portfolio_value=portfolio_value,
            cost_config=cost_config,
            max_weight=float(max_weight),
        )
    if max_industry_weight is not None:
        result = _repair_group_cap(
            result,
            current_share_map,
            lot_size=lot_size,
            portfolio_value=portfolio_value,
            cost_config=cost_config,
            max_group_weight=float(max_industry_weight),
        )
    return result


def _repair_single_name_cap(
    portfolio: pd.DataFrame,
    current_share_map: dict[str, int],
    *,
    lot_size: int,
    portfolio_value: float,
    cost_config: object | None,
    max_weight: float,
) -> pd.DataFrame:
    result = portfolio.copy()
    for index, row in result.iterrows():
        price = float(row["price"])
        max_shares = int((max_weight * portfolio_value / price) // lot_size) * lot_size
        target_shares = int(row["target_shares"])
        if target_shares > max_shares:
            _set_target_shares(
                result,
                index,
                max(0, max_shares),
                current_share_map,
                portfolio_value=portfolio_value,
                cost_config=cost_config,
                reason="hard_cap_repair",
            )
    return result


def _repair_group_cap(
    portfolio: pd.DataFrame,
    current_share_map: dict[str, int],
    *,
    lot_size: int,
    portfolio_value: float,
    cost_config: object | None,
    max_group_weight: float,
) -> pd.DataFrame:
    result = portfolio.copy()
    if "industry" not in result.columns:
        result["industry"] = "Unknown"
    result["industry"] = result["industry"].fillna("Unknown").astype(str)
    # Skip group cap when all industries are 'Unknown' (ETF industry classification is ambiguous)
    unique_industries = set(result["industry"].dropna().astype(str).unique())
    if unique_industries == {"Unknown"}:
        _logger.warning(
            "max_industry_weight constraint skipped: all instruments have industry='Unknown'. "
            "ETF industry classification is ambiguous; use exposure_group constraints for sector risk control."
        )
        return result
    max_value = max_group_weight * portfolio_value
    for industry in sorted(result["industry"].dropna().astype(str).unique()):
        group_index = list(result.index[result["industry"] == industry])
        while group_index:
            group_value = float(
                (pd.to_numeric(result.loc[group_index, "target_shares"], errors="coerce").fillna(0.0)
                 * pd.to_numeric(result.loc[group_index, "price"], errors="coerce").fillna(0.0)).sum()
            )
            if group_value <= max_value + 1e-12:
                break
            candidates = result.loc[group_index].copy()
            candidates = candidates[pd.to_numeric(candidates["target_shares"], errors="coerce").fillna(0).astype(int) > 0]
            if candidates.empty:
                break
            candidates["_lot_value"] = pd.to_numeric(candidates["price"], errors="coerce").fillna(float("inf")) * lot_size
            reduce_index = candidates.sort_values("_lot_value", ascending=True, kind="mergesort").index[0]
            current_target = int(result.loc[reduce_index, "target_shares"])
            _set_target_shares(
                result,
                reduce_index,
                max(0, current_target - lot_size),
                current_share_map,
                portfolio_value=portfolio_value,
                cost_config=cost_config,
                reason="hard_cap_repair",
            )
    return result


def _set_target_shares(
    portfolio: pd.DataFrame,
    index,
    target_shares: int,
    current_share_map: dict[str, int],
    *,
    portfolio_value: float,
    cost_config: object | None,
    reason: str,
) -> None:
    order_book_id = str(portfolio.loc[index, "order_book_id"])
    price = float(portfolio.loc[index, "price"])
    current_shares = int(current_share_map.get(order_book_id, 0))
    target_shares = int(max(0, target_shares))
    order_shares = int(target_shares - current_shares)
    order_value = float(order_shares * price)
    portfolio.loc[index, "target_shares"] = target_shares
    portfolio.loc[index, "position_value"] = float(target_shares * price)
    portfolio.loc[index, "target_weight"] = float(target_shares * price / portfolio_value)
    portfolio.loc[index, "order_shares"] = order_shares
    portfolio.loc[index, "order_value"] = order_value
    portfolio.loc[index, "side"] = _order_side(order_shares)
    portfolio.loc[index, "estimated_cost"] = _estimate_order_cost(
        order_value,
        cost_config,
        portfolio.loc[index].get("asset_type"),
    )
    portfolio.loc[index, "trade_reason"] = reason if order_shares != 0 else "unchanged"


def _order_side(order_shares: int) -> str:
    if order_shares > 0:
        return "buy"
    if order_shares < 0:
        return "sell"
    return "none"


def _estimate_order_cost(order_value: float, cost_config: object | None, asset_type: object = None) -> float:
    if abs(order_value) <= 0:
        return 0.0
    if isinstance(cost_config, dict) and ("stock" in cost_config or "etf" in cost_config):
        if cost_config.get("enabled") is False:
            return 0.0
        normalized_asset_type = normalize_asset_type(asset_type or cost_config.get("default_asset_type", "stock"))
        section = cost_config.get(normalized_asset_type) or cost_config.get(str(cost_config.get("default_asset_type", "stock")), {})
        return _estimate_order_cost_from_mapping(order_value, section)
    cfg = coerce_cost_config(cost_config)
    if cfg is None:
        return 0.0
    value = abs(float(order_value))
    commission = value * float(cfg.commission_rate)
    min_commission = float(getattr(cfg, "min_commission", 0.0))
    if commission > 0:
        commission = max(commission, min_commission)
    slippage = value * float(cfg.slippage_rate)
    stamp_tax = value * float(cfg.stamp_tax_rate) if order_value < 0 else 0.0
    return float(commission + slippage + stamp_tax)


def _estimate_order_cost_from_mapping(order_value: float, section: dict) -> float:
    value = abs(float(order_value))
    commission = value * float(section.get("commission_rate", 0.0))
    min_commission = float(section.get("min_commission", 0.0))
    if commission > 0:
        commission = max(commission, min_commission)
    slippage = value * float(section.get("slippage_bps", 0.0)) / 10000.0
    impact = value * float(section.get("impact_bps", 0.0)) / 10000.0
    stamp_tax = value * float(section.get("stamp_tax_rate", 0.0)) if order_value < 0 else 0.0
    return float(commission + slippage + impact + stamp_tax)


def _orders_from_portfolio(portfolio: pd.DataFrame) -> pd.DataFrame:
    orders = portfolio[portfolio["order_shares"] != 0].copy()
    return _order_columns(orders)


def _portfolio_columns(portfolio: pd.DataFrame) -> pd.DataFrame:
    base = [
        "date",
        "order_book_id",
        "target_weight",
        "target_shares",
        "price",
        "position_value",
        "order_shares",
        "order_value",
        "estimated_cost",
        "side",
        "trade_reason",
    ]
    columns = [column for column in base if column in portfolio.columns]
    extra = [column for column in portfolio.columns if column not in columns]
    return portfolio[columns + extra].reset_index(drop=True)


def _order_columns(orders: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "order_book_id",
        "asset_type",
        "universe_layer",
        "side",
        "order_shares",
        "price",
        "order_value",
        "estimated_cost",
        "trade_reason",
    ]
    return orders[[column for column in columns if column in orders.columns]].reset_index(drop=True)


def _empty_portfolio() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "order_book_id",
            "target_weight",
            "target_shares",
            "price",
            "position_value",
            "order_shares",
            "order_value",
        "estimated_cost",
        "side",
        "trade_reason",
    ]
    )


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "order_book_id",
            "order_shares",
            "price",
            "order_value",
            "estimated_cost",
            "side",
            "trade_reason",
        ]
    )


def _summary(orders: pd.DataFrame) -> dict[str, float | int]:
    if orders.empty:
        return {
            "order_count": 0,
            "buy_value": 0.0,
            "sell_value": 0.0,
            "gross_order_value": 0.0,
            "estimated_cost": 0.0,
        }
    order_values = pd.to_numeric(orders["order_value"], errors="coerce").fillna(0.0)
    return {
        "order_count": int(len(orders)),
        "buy_value": float(order_values[order_values > 0].sum()),
        "sell_value": float(-order_values[order_values < 0].sum()),
        "gross_order_value": float(order_values.abs().sum()),
        "estimated_cost": float(pd.to_numeric(orders["estimated_cost"], errors="coerce").fillna(0.0).sum()),
    }
