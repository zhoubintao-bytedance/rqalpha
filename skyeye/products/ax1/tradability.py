"""AX1 tradability contracts and alpha transfer attribution."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def build_tradable_outcome(
    *,
    target_weights: pd.DataFrame,
    labels: pd.DataFrame,
    orders: pd.DataFrame | None,
    portfolio_value: float,
    gross_label_column: str = "label_return_10d",
) -> dict[str, Any]:
    """Build the single tradable net-return outcome contract."""
    if float(portfolio_value) <= 0:
        raise ValueError("portfolio_value must be positive")
    weights = _normalize_panel(target_weights, value_columns=["target_weight"])
    label_frame = _normalize_panel(labels, value_columns=[gross_label_column])
    if weights.empty or label_frame.empty:
        return _empty_tradable_outcome(gross_label_column)
    if gross_label_column not in label_frame.columns:
        raise ValueError(f"labels missing gross_label_column: {gross_label_column}")

    merged = weights.merge(
        label_frame[["date", "order_book_id", gross_label_column]],
        on=["date", "order_book_id"],
        how="left",
    )
    merged["_target_weight"] = pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0.0)
    merged["_gross_label"] = pd.to_numeric(merged[gross_label_column], errors="coerce")
    merged = merged.dropna(subset=["date", "order_book_id", "_gross_label"])
    if merged.empty:
        return _empty_tradable_outcome(gross_label_column)

    gross_returns = (
        (merged["_target_weight"] * merged["_gross_label"])
        .groupby(merged["date"], sort=True)
        .sum()
        .astype(float)
    )
    dates = list(gross_returns.index)
    execution_costs = _execution_cost_by_date(orders, dates=dates, portfolio_value=float(portfolio_value))
    net_returns = gross_returns - execution_costs.reindex(dates, fill_value=0.0)
    turnover = _turnover_by_date(
        target_weights=weights,
        orders=orders,
        dates=dates,
        portfolio_value=float(portfolio_value),
    )
    net_curve = _equity_curve(net_returns)
    gross_curve = _equity_curve(gross_returns)
    return {
        "schema_version": 1,
        "return_column": gross_label_column,
        "cost_source": "execution_orders",
        "date_count": int(len(dates)),
        "gross_return_by_date": _series_to_date_dict(gross_returns),
        "execution_cost_by_date": _series_to_date_dict(execution_costs.reindex(dates, fill_value=0.0)),
        "net_return_by_date": _series_to_date_dict(net_returns),
        "turnover_by_date": _series_to_date_dict(turnover.reindex(dates, fill_value=0.0)),
        "net_equity_curve": net_curve["records"],
        "gross_equity_curve": gross_curve["records"],
        "mean_gross_return": _mean(gross_returns),
        "mean_net_return": _mean(net_returns),
        "mean_execution_cost": _mean(execution_costs.reindex(dates, fill_value=0.0)),
        "mean_turnover": _mean(turnover.reindex(dates, fill_value=0.0)),
        "total_execution_cost": float(execution_costs.reindex(dates, fill_value=0.0).sum()),
        "max_net_drawdown": net_curve["max_drawdown"],
        "gross_max_drawdown": gross_curve["max_drawdown"],
    }


def build_alpha_transfer_ledger(
    *,
    predictions: pd.DataFrame,
    target_weights: pd.DataFrame,
    executable_weights: pd.DataFrame,
    tradable_outcome: dict[str, Any],
    score_column: str = "expected_relative_net_return_10d",
) -> dict[str, Any]:
    """Attribute alpha retention from model score to executable weights."""
    prediction_frame = _normalize_panel(predictions, value_columns=[score_column])
    if prediction_frame.empty or score_column not in prediction_frame.columns:
        return _empty_alpha_transfer_ledger(score_column)
    target_frame = _normalize_panel(target_weights, value_columns=["target_weight"])
    executable_frame = _normalize_panel(executable_weights, value_columns=["target_weight"])

    base = prediction_frame[["date", "order_book_id", score_column]].copy()
    base["_score"] = pd.to_numeric(base[score_column], errors="coerce").fillna(0.0)
    target = _weight_payload(target_frame, "target_weight", "_target_weight")
    executable = _weight_payload(executable_frame, "target_weight", "_executable_weight")
    merged = base.merge(target, on=["date", "order_book_id"], how="left").merge(
        executable,
        on=["date", "order_book_id"],
        how="left",
    )
    for column in ("_target_weight", "_executable_weight"):
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)

    model_alpha = float(merged["_score"].sum())
    target_alpha = float((merged["_score"] * merged["_target_weight"]).sum())
    executable_alpha = float((merged["_score"] * merged["_executable_weight"]).sum())
    by_date = []
    for date, day_df in merged.groupby("date", sort=True):
        day_model = float(day_df["_score"].sum())
        day_target = float((day_df["_score"] * day_df["_target_weight"]).sum())
        day_executable = float((day_df["_score"] * day_df["_executable_weight"]).sum())
        by_date.append(
            {
                "date": _date_key(date),
                "model_alpha_weighted": day_model,
                "target_alpha_weighted": day_target,
                "executable_alpha_weighted": day_executable,
                "target_retention_ratio": _safe_ratio(day_target, day_model),
                "executable_retention_ratio": _safe_ratio(day_executable, day_model),
                "net_return": float((tradable_outcome.get("net_return_by_date") or {}).get(_date_key(date), 0.0)),
            }
        )

    by_asset = []
    for _, row in merged.sort_values(["date", "order_book_id"]).iterrows():
        by_asset.append(
            {
                "date": _date_key(row["date"]),
                "order_book_id": str(row["order_book_id"]),
                "score": float(row["_score"]),
                "target_weight": float(row["_target_weight"]),
                "executable_weight": float(row["_executable_weight"]),
                "target_alpha": float(row["_score"] * row["_target_weight"]),
                "executable_alpha": float(row["_score"] * row["_executable_weight"]),
                "trade_reason": str(row.get("trade_reason") or ""),
            }
        )

    return {
        "schema_version": 1,
        "score_column": score_column,
        "summary": {
            "model_alpha_weighted": model_alpha,
            "target_alpha_weighted": target_alpha,
            "executable_alpha_weighted": executable_alpha,
            "target_retention_ratio": _safe_ratio(target_alpha, model_alpha),
            "executable_retention_ratio": _safe_ratio(executable_alpha, model_alpha),
        },
        "blocker_counts": _blocker_counts(executable_frame),
        "by_date": by_date,
        "by_asset": by_asset,
    }


def _empty_tradable_outcome(return_column: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "return_column": str(return_column),
        "cost_source": "execution_orders",
        "date_count": 0,
        "gross_return_by_date": {},
        "execution_cost_by_date": {},
        "net_return_by_date": {},
        "turnover_by_date": {},
        "net_equity_curve": [],
        "gross_equity_curve": [],
        "mean_gross_return": 0.0,
        "mean_net_return": 0.0,
        "mean_execution_cost": 0.0,
        "mean_turnover": 0.0,
        "total_execution_cost": 0.0,
        "max_net_drawdown": 0.0,
        "gross_max_drawdown": 0.0,
    }


def _empty_alpha_transfer_ledger(score_column: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "score_column": str(score_column),
        "summary": {
            "model_alpha_weighted": 0.0,
            "target_alpha_weighted": 0.0,
            "executable_alpha_weighted": 0.0,
            "target_retention_ratio": 0.0,
            "executable_retention_ratio": 0.0,
        },
        "blocker_counts": {},
        "by_date": [],
        "by_asset": [],
    }


def _normalize_panel(frame: pd.DataFrame | None, *, value_columns: list[str]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["date", "order_book_id", *value_columns])
    result = frame.copy()
    missing = [column for column in ("date", "order_book_id") if column not in result.columns]
    if missing:
        raise ValueError(f"frame missing panel columns: {missing}")
    result["date"] = pd.to_datetime(result["date"])
    result["order_book_id"] = result["order_book_id"].astype(str)
    return result


def _execution_cost_by_date(
    orders: pd.DataFrame | None,
    *,
    dates: list[pd.Timestamp],
    portfolio_value: float,
) -> pd.Series:
    index = pd.Index(dates, name="date")
    if orders is None or orders.empty or "date" not in orders.columns:
        return pd.Series(0.0, index=index, dtype=float)
    frame = orders.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "estimated_cost" not in frame.columns:
        frame["estimated_cost"] = 0.0
    costs = pd.to_numeric(frame["estimated_cost"], errors="coerce").fillna(0.0)
    by_date = costs.groupby(frame["date"], sort=True).sum().astype(float) / float(portfolio_value)
    return by_date.reindex(index, fill_value=0.0)


def _turnover_by_date(
    *,
    target_weights: pd.DataFrame,
    orders: pd.DataFrame | None,
    dates: list[pd.Timestamp],
    portfolio_value: float,
) -> pd.Series:
    index = pd.Index(dates, name="date")
    if orders is not None and not orders.empty and "date" in orders.columns:
        frame = orders.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if "order_value" in frame.columns:
            values = pd.to_numeric(frame["order_value"], errors="coerce").fillna(0.0).abs()
            by_date = values.groupby(frame["date"], sort=True).sum().astype(float) / float(portfolio_value)
            return by_date.reindex(index, fill_value=0.0)
    return _weight_turnover_by_date(target_weights, dates)


def _weight_turnover_by_date(target_weights: pd.DataFrame, dates: list[pd.Timestamp]) -> pd.Series:
    index = pd.Index(dates, name="date")
    if target_weights is None or target_weights.empty:
        return pd.Series(0.0, index=index, dtype=float)
    previous: dict[str, float] | None = None
    values: dict[pd.Timestamp, float] = {}
    for date, day_df in target_weights.sort_values(["date", "order_book_id"]).groupby("date", sort=True):
        current = {
            str(row["order_book_id"]): float(row["target_weight"])
            for _, row in day_df.dropna(subset=["order_book_id", "target_weight"]).iterrows()
        }
        if previous is None:
            values[pd.Timestamp(date)] = 0.0
        else:
            universe = set(previous) | set(current)
            values[pd.Timestamp(date)] = 0.5 * sum(
                abs(float(current.get(order_book_id, 0.0)) - float(previous.get(order_book_id, 0.0)))
                for order_book_id in universe
            )
        previous = current
    return pd.Series(values, dtype=float).reindex(index, fill_value=0.0)


def _equity_curve(returns: pd.Series) -> dict[str, Any]:
    if returns is None or returns.empty:
        return {"records": [], "max_drawdown": 0.0}
    equity = (1.0 + pd.to_numeric(returns, errors="coerce").fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    records = [
        {
            "date": _date_key(date),
            "return": float(returns.loc[date]),
            "equity": float(equity.loc[date]),
            "drawdown": float(drawdown.loc[date]),
        }
        for date in returns.index
    ]
    return {"records": records, "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0}


def _series_to_date_dict(series: pd.Series) -> dict[str, float]:
    if series is None or series.empty:
        return {}
    return {_date_key(date): float(value) for date, value in series.items()}


def _date_key(date) -> str:
    return str(pd.Timestamp(date).date())


def _mean(series: pd.Series) -> float:
    if series is None or len(series) == 0:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.mean()) if len(values) else 0.0


def _weight_payload(frame: pd.DataFrame, source_column: str, output_column: str) -> pd.DataFrame:
    columns = ["date", "order_book_id", source_column]
    if "trade_reason" in frame.columns:
        columns.append("trade_reason")
    if frame is None or frame.empty or source_column not in frame.columns:
        return pd.DataFrame(columns=["date", "order_book_id", output_column])
    result = frame[columns].copy()
    result[output_column] = pd.to_numeric(result[source_column], errors="coerce").fillna(0.0)
    result = result.drop(columns=[source_column])
    return result.drop_duplicates(["date", "order_book_id"], keep="last")


def _blocker_counts(executable_weights: pd.DataFrame) -> dict[str, int]:
    if executable_weights is None or executable_weights.empty or "trade_reason" not in executable_weights.columns:
        return {}
    neutral = {"", "trade", "unchanged", "none"}
    reasons = [
        str(reason)
        for reason in executable_weights["trade_reason"].fillna("").astype(str)
        if str(reason).strip().lower() not in neutral
    ]
    return {reason: int(count) for reason, count in Counter(reasons).items()}


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if abs(denominator) <= 1e-12:
        return 0.0
    return float(numerator) / denominator
