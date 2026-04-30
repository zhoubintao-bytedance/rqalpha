"""AX1 research execution replay helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_portfolio_replay(
    *,
    config: dict[str, Any],
    fused_predictions: pd.DataFrame,
    scoped_raw: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
    industry_map: dict[str, str],
    universe_metadata: pd.DataFrame,
    regime_state: dict[str, Any],
    regime_state_by_date: dict[Any, dict[str, Any]],
    as_of_date,
) -> dict[str, Any]:
    import skyeye.products.ax1.run_experiment as runner
    from skyeye.products.ax1.evaluation.metrics import evaluate_portfolio_layer
    from skyeye.products.ax1.execution.smoother import ExecutionSmoother
    from skyeye.products.ax1.effective_breadth import summarize_effective_breadth_by_date
    from skyeye.products.ax1.optimizer.allocation import OpportunityPoolOptimizer
    from skyeye.products.ax1.optimizer.executable import ExecutablePortfolioOptimizer
    from skyeye.products.ax1.tradability import build_alpha_transfer_ledger, build_tradable_outcome

    risk_cfg = config.get("risk_model", {})
    risk_state_by_date = runner._fit_risk_model_by_date(scoped_raw, risk_cfg)
    risk_state = runner._risk_model_for_timestamp(risk_state_by_date, as_of_date)
    universe_ids = (
        universe_metadata["order_book_id"].astype(str).tolist()
        if universe_metadata is not None and not universe_metadata.empty and "order_book_id" in universe_metadata.columns
        else None
    )
    effective_breadth_summary = summarize_effective_breadth_by_date(
        risk_state_by_date,
        order_book_ids=universe_ids,
    )

    allocation_config = runner._opportunity_pool_config(config, regime_state)
    allocation_config_by_date = runner._opportunity_pool_config_by_date(
        config,
        dates=fused_predictions["date"].dropna().unique() if "date" in fused_predictions.columns else [],
        regime_state_by_date=regime_state_by_date,
        fallback_regime_state=regime_state,
    )
    target_weights = OpportunityPoolOptimizer().optimize(
        fused_predictions,
        constraints=config.get("constraints", {}),
        allocation_config=allocation_config_by_date,
        risk_model=risk_state_by_date,
    )

    # --- Portfolio stop-loss ---
    stop_loss_result = _apply_portfolio_stop_loss(
        target_weights,
        config=config,
        evaluation_labels=evaluation_labels,
        regime_state_by_date=regime_state_by_date,
    )
    target_weights = stop_loss_result["target_weights"]

    target_weights = runner._attach_execution_inputs(
        target_weights,
        predictions=fused_predictions,
        raw_frame=scoped_raw,
        price_column=str(config.get("execution", {}).get("price_column", "close")),
        alpha_column=str(config.get("execution", {}).get("net_alpha_column", "adjusted_expected_return")),
        execution_lag_days=int(config.get("execution", {}).get("execution_lag_days", 0)),
    )
    execution_reference = runner._build_execution_reference(
        scoped_raw,
        price_column=str(config.get("execution", {}).get("price_column", "close")),
        industry_map=industry_map,
        execution_lag_days=int(config.get("execution", {}).get("execution_lag_days", 0)),
    )
    execution_cfg = config.get("execution", {})
    constraints_cfg = config.get("constraints", {})
    target_gross_weight = max(
        0.0,
        float(constraints_cfg.get("target_gross_exposure", 1.0))
        - float(constraints_cfg.get("cash_buffer", 0.0)),
    )
    if target_weights is not None and not target_weights.empty and "target_weight" in target_weights.columns:
        realized_gross = float(pd.to_numeric(target_weights["target_weight"], errors="coerce").fillna(0.0).abs().groupby(target_weights["date"]).sum().max())
        target_gross_weight = min(target_gross_weight, realized_gross)
    smoother = ExecutionSmoother(
        min_weight=float(execution_cfg.get("min_weight", 0.0)),
        max_turnover=constraints_cfg.get("max_turnover"),
        buffer_weight=float(execution_cfg.get("buffer_weight", 0.0)),
        no_trade_buffer_weight=float(
            execution_cfg.get(
                "no_trade_buffer_weight",
                execution_cfg.get("buffer_weight", 0.0),
            )
        ),
        min_trade_value=float(execution_cfg.get("min_trade_value", 0.0)),
        portfolio_value=(
            float(execution_cfg["portfolio_value"])
            if execution_cfg.get("portfolio_value") is not None
            else None
        ),
        target_gross_weight=target_gross_weight,
        max_weight=(
            float(constraints_cfg["max_single_weight"])
            if constraints_cfg.get("max_single_weight") is not None
            else None
        ),
        max_industry_weight=(
            float(constraints_cfg["max_industry_weight"])
            if constraints_cfg.get("max_industry_weight") is not None
            else None
        ),
        net_alpha_threshold=float(execution_cfg.get("net_alpha_threshold", 0.0)),
        net_alpha_column=str(execution_cfg.get("net_alpha_column", "adjusted_expected_return")),
        participation_rate=(
            float(execution_cfg["participation_rate"])
            if execution_cfg.get("participation_rate") is not None
            else None
        ),
        liquidity_column=str(execution_cfg.get("liquidity_column", "dollar_volume")),
        t_plus_one_lock=bool(execution_cfg.get("t_plus_one_lock", False)),
        today_buy_weight_column=str(execution_cfg.get("today_buy_weight_column", "today_buy_weight")),
    )
    executable_optimizer = ExecutablePortfolioOptimizer(
        portfolio_value=float(execution_cfg.get("portfolio_value", 1_000_000)),
        lot_size=int(execution_cfg.get("lot_size", 100)),
        min_trade_value=float(execution_cfg.get("min_trade_value", 0.0)),
        max_order_count=int(execution_cfg["max_order_count"]) if execution_cfg.get("max_order_count") is not None else None,
        cost_config=config.get("costs", {}),
        max_weight=(
            float(constraints_cfg["max_single_weight"])
            if constraints_cfg.get("max_single_weight") is not None
            else None
        ),
        max_industry_weight=(
            float(constraints_cfg["max_industry_weight"])
            if constraints_cfg.get("max_industry_weight") is not None
            else None
        ),
    )
    smoothed_weights, orders, execution_summary = runner._execute_rolling_targets(
        target_weights,
        smoother=smoother,
        executable_optimizer=executable_optimizer,
        max_turnover=constraints_cfg.get("max_turnover"),
        rebalance_interval=int(execution_cfg.get("rebalance_interval", execution_cfg.get("rebalance_days_min", 1))),
        portfolio_value=float(execution_cfg.get("portfolio_value", 1_000_000)),
        execution_reference=execution_reference,
    )
    smoothed_weights = runner._attach_universe_metadata(smoothed_weights, universe_metadata)
    tradable_outcome = build_tradable_outcome(
        target_weights=smoothed_weights,
        labels=evaluation_labels,
        orders=orders,
        portfolio_value=float(execution_cfg.get("portfolio_value", 1_000_000)),
        gross_label_column=_select_tradable_return_column(evaluation_labels),
    )
    alpha_transfer_ledger = build_alpha_transfer_ledger(
        predictions=fused_predictions,
        target_weights=target_weights,
        executable_weights=smoothed_weights,
        tradable_outcome=tradable_outcome,
        score_column=str(config.get("allocation", {}).get("score_column", "expected_relative_net_return_10d")),
    )
    portfolio_metrics = evaluate_portfolio_layer(
        smoothed_weights,
        evaluation_labels,
        constraints=config.get("constraints", {}),
        initial_weights=runner._first_day_weights(smoothed_weights),
        cost_config=runner._enabled_cost_config(config.get("costs", {})),
        orders=orders,
        min_trade_value=float(execution_cfg.get("min_trade_value", 0.0)),
        portfolio_value=float(execution_cfg.get("portfolio_value", 1_000_000)),
        tradable_outcome=tradable_outcome,
    )
    portfolio_metrics.setdefault("portfolio", {})["effective_breadth"] = effective_breadth_summary
    return {
        "risk_state": risk_state,
        "risk_state_by_date": risk_state_by_date,
        "effective_breadth_summary": effective_breadth_summary,
        "allocation_config": allocation_config,
        "allocation_config_by_date": allocation_config_by_date,
        "target_weights": smoothed_weights,
        "orders": orders,
        "execution_summary": execution_summary,
        "tradable_outcome": tradable_outcome,
        "alpha_transfer_ledger": alpha_transfer_ledger,
        "portfolio_metrics": portfolio_metrics,
        "stop_loss_result": stop_loss_result,
    }


def _select_tradable_return_column(labels: pd.DataFrame) -> str:
    if labels is None or labels.empty:
        return "label_return_10d"
    for candidate in (
        "label_return_10d",
        "label_return_20d",
        "label_return_5d",
        "label_net_return_10d",
        "label_net_return_20d",
        "label_net_return_5d",
        "label_relative_net_return_10d",
        "label_relative_net_return_20d",
        "label_relative_net_return_5d",
    ):
        if candidate in labels.columns:
            return candidate
    label_columns = sorted(
        column
        for column in labels.columns
        if column.startswith("label_return_")
        or column.startswith("label_net_return_")
        or column.startswith("label_relative_net_return_")
    )
    return label_columns[0] if label_columns else "label_return_10d"


# ---------------------------------------------------------------------------
# Portfolio stop-loss integration
# ---------------------------------------------------------------------------

def _apply_portfolio_stop_loss(
    target_weights: pd.DataFrame,
    *,
    config: dict[str, Any],
    evaluation_labels: pd.DataFrame,
    regime_state_by_date: dict[Any, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply three-level portfolio stop-loss across all dates.

    Iterates day-by-day, maintaining equity curve and stop-loss state.
    Returns scaled target weights and stop-loss diagnostics.
    """
    from skyeye.products.ax1.risk.stop_loss import PortfolioStopLoss, StopLossLevel

    stop_loss_cfg = config.get("stop_loss", {})
    if not bool(stop_loss_cfg.get("enabled", False)):
        return {
            "target_weights": target_weights,
            "stop_loss_state": None,
            "stop_loss_trigger_count": 0,
            "stop_loss_log": [],
        }

    if target_weights is None or target_weights.empty:
        return {
            "target_weights": target_weights,
            "stop_loss_state": None,
            "stop_loss_trigger_count": 0,
            "stop_loss_log": [],
        }

    stop_loss = PortfolioStopLoss(stop_loss_cfg)

    # Pre-compute daily portfolio returns from target weights and evaluation labels.
    # Uses the previous day's weights × current day's realized returns.
    daily_returns = _compute_daily_portfolio_returns(target_weights, evaluation_labels)

    # Apply stop-loss day by day
    sorted_dates = sorted(target_weights["date"].unique())
    scaled_rows = []
    prev_level = StopLossLevel.NORMAL
    trigger_count = 0

    for date in sorted_dates:
        date_ts = pd.Timestamp(date)
        day_df = target_weights[target_weights["date"] == date]

        # Update equity from previous day's return
        portfolio_return = daily_returns.get(date_ts, 0.0)
        stop_loss.update_equity(date_ts, portfolio_return)

        # Get regime state for recovery gating
        regime_state = (regime_state_by_date or {}).get(date_ts)

        # Apply stop-loss scaling
        scaled_day = stop_loss.apply(day_df, date_ts, regime_state)
        scaled_rows.append(scaled_day)

        if stop_loss.state.active_level != prev_level:
            trigger_count += 1
            prev_level = stop_loss.state.active_level

    scaled_weights = pd.concat(scaled_rows, ignore_index=True) if scaled_rows else target_weights
    return {
        "target_weights": scaled_weights,
        "stop_loss_state": stop_loss.state,
        "stop_loss_trigger_count": trigger_count,
        "stop_loss_log": stop_loss.state.trigger_log,
    }


def _compute_daily_portfolio_returns(
    target_weights: pd.DataFrame,
    evaluation_labels: pd.DataFrame,
) -> dict[pd.Timestamp, float]:
    """Compute daily portfolio returns from target weights and realized returns.

    For each date, the portfolio return = sum(weight_on_date × realized_return_on_date).
    This uses the *same-day* weights and returns, which is a reasonable approximation
    for the stop-loss equity tracker (weights are set at the start of the period
    based on previous-day signals).
    """
    if target_weights is None or target_weights.empty:
        return {}
    if evaluation_labels is None or evaluation_labels.empty:
        return {}

    # Find the label column to use
    label_col = None
    for candidate in (
        "label_relative_net_return_10d",
        "label_net_return_10d",
        "label_relative_net_return_5d",
        "label_net_return_5d",
    ):
        if candidate in evaluation_labels.columns:
            label_col = candidate
            break
    if label_col is None:
        return {}

    # Merge target weights with labels on (date, order_book_id)
    tw = target_weights[["date", "order_book_id", "target_weight"]].copy()
    tw["date"] = pd.to_datetime(tw["date"])
    tw["target_weight"] = pd.to_numeric(tw["target_weight"], errors="coerce").fillna(0.0)

    labels = evaluation_labels[["date", "order_book_id", label_col]].copy()
    labels["date"] = pd.to_datetime(labels["date"])
    labels[label_col] = pd.to_numeric(labels[label_col], errors="coerce").fillna(0.0)

    merged = tw.merge(labels, on=["date", "order_book_id"], how="left")
    merged[label_col] = merged[label_col].fillna(0.0)

    # Weighted return per date
    result = {}
    for date, day_df in merged.groupby("date", sort=True):
        weighted_return = float((day_df["target_weight"] * day_df[label_col]).sum())
        result[pd.Timestamp(date)] = weighted_return
    return result


def ensure_replay_weights(weights: pd.DataFrame) -> pd.DataFrame:
    if weights is None or weights.empty:
        return pd.DataFrame(columns=["date", "order_book_id", "target_weight"])
    required = {"date", "order_book_id", "target_weight"}
    missing = required - set(weights.columns)
    if missing:
        raise ValueError(f"weights missing required columns: {sorted(missing)}")
    return weights.copy()
