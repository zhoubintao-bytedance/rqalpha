"""AX1 evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from skyeye.products.ax1._common import coerce_cost_config, normalize_asset_type


def evaluate_signal_layer(predictions: pd.DataFrame, labels: pd.DataFrame | None = None) -> dict:
    """评估 signal 层排序质量。"""
    predictions = predictions if predictions is not None else pd.DataFrame()
    signal = {
        "row_count": int(len(predictions)),
        "prediction_columns": [
            column
            for column in predictions.columns
            if column.startswith("expected_relative_net_return_")
        ],
        "group_backtest": {},
    }
    if labels is not None:
        label_columns = [
            column
            for column in labels.columns
            if column.startswith("label_relative_net_return_")
            or column.startswith("label_net_return_")
            or column.startswith("label_return_")
        ]
        signal["label_columns"] = label_columns
        signal["label_row_count"] = int(len(labels))
        metrics = _compute_signal_metrics(predictions, labels)
        signal.update(metrics)
    return {
        "signal": signal,
        "portfolio": {},
    }


def evaluate_portfolio_layer(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame | None = None,
    constraints: dict | None = None,
    initial_weights: dict[str, float] | None = None,
    cost_config=None,
    orders: pd.DataFrame | None = None,
    min_trade_value: float = 0.0,
    portfolio_value: float | None = None,
    risk_free_rate: float = 0.02 / 252,
    benchmark_weights_by_date: dict[str, dict[str, float]] | None = None,
    tradable_outcome: dict | None = None,
) -> dict:
    """评估组合层收益、风险、换手和基础约束。"""
    target_weights = target_weights if target_weights is not None else pd.DataFrame()
    portfolio = {
        "row_count": int(len(target_weights)),
        "date_count": int(target_weights["date"].nunique()) if "date" in target_weights.columns else 0,
        "normalized": _is_normalized(target_weights),
    }
    if returns is not None:
        portfolio["return_row_count"] = int(len(returns))
        portfolio.update(
            _compute_portfolio_metrics(
                target_weights,
                returns,
                initial_weights=initial_weights,
                cost_config=cost_config,
                risk_free_rate=risk_free_rate,
                benchmark_weights_by_date=benchmark_weights_by_date,
            )
        )
        if orders is not None and portfolio_value is not None and _is_cost_enabled(cost_config):
            portfolio.update(
                _compute_cost_metrics_from_orders(
                    orders,
                    target_weights=target_weights,
                    portfolio_return_mean=float(portfolio.get("portfolio_return_mean", 0.0)),
                    portfolio_value=float(portfolio_value),
                )
            )
        contribution = _contribution_by_asset_type(target_weights, returns, orders)
        if contribution:
            portfolio["contribution_by_asset_type"] = contribution
    if orders is not None and portfolio_value is not None:
        portfolio.update(
            _compute_turnover_metrics_from_orders(
                orders,
                target_weights=target_weights,
                portfolio_value=float(portfolio_value),
            )
        )
    if orders is not None:
        portfolio.update(_compute_order_execution_metrics(orders, min_trade_value=min_trade_value))
    portfolio.update(_compute_activity_metrics(target_weights, constraints or {}))
    industry_exposure = _industry_exposure_by_date(target_weights)
    if industry_exposure:
        portfolio["industry_exposure_by_date"] = industry_exposure
    allocation_metrics = _compute_allocation_metrics(target_weights, constraints or {})
    if allocation_metrics:
        portfolio["allocation"] = allocation_metrics
    portfolio["constraint_violations"] = _constraint_violations(target_weights, constraints or {})
    portfolio = _apply_tradable_outcome_metrics(portfolio, tradable_outcome)
    return {
        "signal": {},
        "portfolio": portfolio,
    }


def _is_normalized(target_weights: pd.DataFrame) -> bool:
    if target_weights.empty or not {"date", "target_weight"}.issubset(target_weights.columns):
        return False
    sums = target_weights.groupby("date")["target_weight"].sum()
    return bool(((sums - 1.0).abs() < 1e-8).all())


def _compute_signal_metrics(predictions: pd.DataFrame, labels: pd.DataFrame) -> dict:
    score_column = _select_prediction_column(predictions)
    label_column = _select_label_column(labels)
    if score_column is None or label_column is None:
        return {
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
            "rank_ic_significance": _empty_rank_ic_significance(),
            "top_bucket_spread_mean": 0.0,
            "top_k_hit_rate": 0.0,
            "group_backtest": {},
        }

    merged = _merge_on_panel_keys(predictions, labels, [score_column], [label_column])
    if merged.empty:
        return {
            "rank_ic_mean": 0.0,
            "rank_ic_ir": 0.0,
            "rank_ic_significance": _empty_rank_ic_significance(),
            "top_bucket_spread_mean": 0.0,
            "top_k_hit_rate": 0.0,
            "group_backtest": {},
        }

    rank_ics: list[float] = []
    spreads: list[float] = []
    hit_rates: list[float] = []
    # --- group backtest accumulators ---
    n_groups = 5
    group_daily_returns: dict[int, list[float]] = {g: [] for g in range(n_groups)}
    ls_spreads: list[float] = []

    for _, day_df in merged.groupby("date", sort=True):
        clean = day_df[[score_column, label_column]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(clean) < 2:
            continue
        pred_rank = clean[score_column].rank(method="average")
        label_rank = clean[label_column].rank(method="average")
        if pred_rank.nunique() > 1 and label_rank.nunique() > 1:
            rank_ic = pred_rank.corr(label_rank)
            if pd.notna(rank_ic) and np.isfinite(rank_ic):
                rank_ics.append(float(rank_ic))
        ranked = clean.sort_values(score_column, ascending=False)
        bucket_size = max(1, min(10, len(ranked) // 5 if len(ranked) >= 10 else 1))
        top = ranked.head(bucket_size)
        bottom = ranked.tail(bucket_size)
        spreads.append(float(top[label_column].mean() - bottom[label_column].mean()))
        hit_rates.append(float((top[label_column] > 0).mean()))

        # --- group backtest: assign to quintile groups ---
        ranked = clean.sort_values(score_column, ascending=False)
        n = len(ranked)
        group_labels = pd.Series(-1, index=ranked.index, dtype=int)
        for g in range(n_groups):
            start = int(n * g / n_groups)
            end = int(n * (g + 1) / n_groups)
            if g == n_groups - 1:
                end = n  # last group gets remainder
            group_labels.iloc[start:end] = g
        for g in range(n_groups):
            mask = group_labels == g
            if mask.any():
                group_daily_returns[g].append(float(ranked.loc[mask, label_column].mean()))
        # long-short: group 0 (top) minus group n-1 (bottom)
        if group_daily_returns[0] and group_daily_returns[n_groups - 1]:
            ls_spreads.append(group_daily_returns[0][-1] - group_daily_returns[n_groups - 1][-1])

    rank_ic_mean = float(np.mean(rank_ics)) if rank_ics else 0.0
    rank_ic_std = float(np.std(rank_ics, ddof=0)) if len(rank_ics) > 1 else 0.0
    rank_ic_significance = _compute_rank_ic_significance(rank_ics)

    # --- compute group backtest metrics ---
    group_bt = _compute_group_backtest_metrics(group_daily_returns, ls_spreads, n_groups)

    return {
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": float(rank_ic_mean / rank_ic_std) if rank_ic_std > 0 else 0.0,
        "rank_ic_significance": rank_ic_significance,
        "top_bucket_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "top_k_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
        "group_backtest": group_bt,
    }


def _empty_rank_ic_significance() -> dict[str, float | int | bool | str]:
    return {
        "method": "newey_west_mean_t_test",
        "fdr_method": "benjamini_hochberg",
        "n_observations": 0,
        "newey_west_lags": 0,
        "mean": 0.0,
        "t_stat": 0.0,
        "p_value": 1.0,
        "fdr_adjusted_p_value": 1.0,
        "significant_at_5pct": False,
    }


def _compute_rank_ic_significance(rank_ics: list[float], *, alpha: float = 0.05) -> dict[str, float | int | bool | str]:
    values = np.asarray(rank_ics, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return _empty_rank_ic_significance()

    lags = _newey_west_lags(len(values))
    mean = float(np.mean(values))
    se = _newey_west_standard_error_of_mean(values, lags)
    if se <= 1e-12:
        t_stat = math.copysign(math.inf, mean) if abs(mean) > 0.0 else 0.0
        p_value = 0.0 if abs(mean) > 0.0 else 1.0
    else:
        t_stat = float(mean / se)
        p_value = float(_two_sided_normal_p_value(t_stat))

    adjusted_p_value = float(_benjamini_hochberg_adjusted_pvalues([p_value])[0])
    return {
        "method": "newey_west_mean_t_test",
        "fdr_method": "benjamini_hochberg",
        "n_observations": int(len(values)),
        "newey_west_lags": int(lags),
        "mean": mean,
        "t_stat": t_stat,
        "p_value": p_value,
        "fdr_adjusted_p_value": adjusted_p_value,
        "significant_at_5pct": bool(adjusted_p_value <= float(alpha)),
    }


def _newey_west_lags(n_observations: int) -> int:
    if n_observations <= 2:
        return 0
    return max(1, int(round(4.0 * (float(n_observations) / 100.0) ** (2.0 / 9.0))))


def _newey_west_standard_error_of_mean(values: np.ndarray, lags: int) -> float:
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0
    centered = arr - float(np.mean(arr))
    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    max_lag = max(0, min(int(lags), n - 1))
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / float(max_lag + 1)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * weight * gamma
    long_run_variance = max(long_run_variance, 0.0)
    return float(math.sqrt(long_run_variance / n))


def _two_sided_normal_p_value(t_stat: float) -> float:
    if not math.isfinite(t_stat):
        return 0.0
    z = abs(float(t_stat)) / math.sqrt(2.0)
    return float(math.erfc(z))


def _benjamini_hochberg_adjusted_pvalues(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    indexed = sorted(enumerate(float(value) for value in p_values), key=lambda item: item[1])
    n = len(indexed)
    adjusted = [1.0] * n
    running = 1.0
    for rank, (original_index, p_value) in enumerate(reversed(indexed), start=1):
        order = n - rank + 1
        candidate = min(1.0, p_value * n / float(order))
        running = min(running, candidate)
        adjusted[original_index] = running
    return adjusted


def _compute_group_backtest_metrics(
    group_daily_returns: dict[int, list[float]],
    ls_spreads: list[float],
    n_groups: int,
) -> dict:
    """分组回测指标：每组累计收益、多空价差、单调性。

    解决 34 只 ETF 下 Rank IC 统计力不足的问题：通过观察信号排序后各组的
    实际收益是否单调递减来验证信号有效性，比单日相关性更稳健。
    """
    group_means: dict[str, float] = {}
    for g in range(n_groups):
        returns = group_daily_returns.get(g, [])
        group_means[f"group_{g}_return_mean"] = float(np.mean(returns)) if returns else 0.0

    ls_mean = float(np.mean(ls_spreads)) if ls_spreads else 0.0
    ls_std = float(np.std(ls_spreads, ddof=0)) if len(ls_spreads) > 1 else 0.0
    ls_ir = float(ls_mean / ls_std) if ls_std > 0 else 0.0

    # monotonicity: Spearman correlation between group rank (high rank = high signal) and group mean return
    # group_0 has strongest signal, so use reversed rank so that group_0 gets highest rank
    group_return_list = [group_means.get(f"group_{g}_return_mean", 0.0) for g in range(n_groups)]
    nonzero = [v for v in group_return_list if v != 0.0]
    if len(nonzero) >= 3:
        group_ranks = list(range(n_groups - 1, -1, -1))  # group_0 → rank 4, group_4 → rank 0
        monotonicity = float(pd.Series(group_ranks).corr(pd.Series(group_return_list), method="spearman"))
        if pd.isna(monotonicity):
            monotonicity = 0.0
    else:
        monotonicity = 0.0

    return {
        **group_means,
        "long_short_spread_mean": ls_mean,
        "long_short_ir": ls_ir,
        "monotonicity": monotonicity,
        "n_groups": n_groups,
        "date_count": len(ls_spreads),
    }


def _compute_portfolio_metrics(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    initial_weights: dict[str, float] | None = None,
    cost_config=None,
    risk_free_rate: float = 0.02 / 252,
    benchmark_weights_by_date: dict[str, dict[str, float]] | None = None,
) -> dict:
    label_column = _select_portfolio_label_column(returns)
    if label_column is None or target_weights.empty:
        result = {
            "portfolio_return_mean": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "information_ratio_proxy": 0.0,
            "mean_turnover": 0.0,
        }
        if cost_config is not None:
            result.update(_empty_cost_metrics())
        return result

    left_columns = ["target_weight"]
    right_columns = [label_column]
    if "asset_type" in target_weights.columns:
        left_columns.append("asset_type")
    elif "asset_type" in returns.columns:
        right_columns.append("asset_type")
    merged = _merge_on_panel_keys(target_weights, returns, left_columns, right_columns)
    if merged.empty:
        result = {
            "portfolio_return_mean": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "information_ratio_proxy": 0.0,
            "mean_turnover": 0.0,
            "evaluated_date_count": 0,
            "dropped_tail_date_count": 0,
        }
        if cost_config is not None:
            result.update(_empty_cost_metrics())
        return result

    merged["_label_value"] = pd.to_numeric(merged[label_column], errors="coerce")
    merged["_target_weight"] = pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0.0)
    all_dates = set(pd.to_datetime(merged["date"]).unique())
    complete_dates = merged.groupby("date")["_label_value"].apply(lambda values: values.notna().all())
    valid_dates = set(pd.to_datetime(complete_dates[complete_dates].index).unique())
    dropped_tail_date_count = len(all_dates - valid_dates)
    merged = merged[pd.to_datetime(merged["date"]).isin(valid_dates)].copy()
    if merged.empty:
        result = {
            "portfolio_return_mean": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "information_ratio_proxy": 0.0,
            "mean_turnover": 0.0,
            "evaluated_date_count": 0,
            "dropped_tail_date_count": int(dropped_tail_date_count),
        }
        if cost_config is not None:
            result.update(_empty_cost_metrics())
        return result

    rows = []
    prev_weights: dict[str, float] = dict(initial_weights or {})
    prev_asset_types: dict[str, str] = {}
    for date, day_df in merged.groupby("date", sort=True):
        weights = {
            str(row["order_book_id"]): float(row["target_weight"])
            for _, row in day_df.dropna(subset=["target_weight"]).iterrows()
        }
        asset_types = _asset_types_by_id(day_df)
        weighted_return = float((day_df["_target_weight"] * day_df["_label_value"]).sum())
        union = set(weights) | set(prev_weights)
        turnover = 0.5 * sum(abs(weights.get(item, 0.0) - prev_weights.get(item, 0.0)) for item in union)
        row = {"date": pd.Timestamp(date), "portfolio_return": weighted_return, "turnover": turnover}
        period_cost = _asset_specific_period_cost(
            weights,
            prev_weights,
            asset_types,
            prev_asset_types,
            cost_config,
        )
        if period_cost is not None:
            row["cost"] = period_cost
        rows.append(row)
        prev_weights = weights
        prev_asset_types = asset_types

    portfolio_returns = pd.DataFrame(rows)
    if portfolio_returns.empty:
        result = {
            "portfolio_return_mean": 0.0,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "information_ratio_proxy": 0.0,
            "mean_turnover": 0.0,
            "evaluated_date_count": 0,
            "dropped_tail_date_count": int(dropped_tail_date_count),
        }
        if cost_config is not None:
            result.update(_empty_cost_metrics())
        return result

    returns_series = portfolio_returns["portfolio_return"].astype(float)
    equity = (1.0 + returns_series.fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    std = float(returns_series.std(ddof=0)) if len(returns_series) > 1 else 0.0
    mean_return = float(returns_series.mean()) if len(returns_series) else 0.0
    result = {
        "portfolio_return_mean": mean_return,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "volatility": float(std * np.sqrt(252.0)),
        "information_ratio_proxy": float(mean_return / std) if std > 0 else 0.0,
        "mean_turnover": float(portfolio_returns["turnover"].mean()),
        "evaluated_date_count": int(portfolio_returns["date"].nunique()),
        "dropped_tail_date_count": int(dropped_tail_date_count),
    }
    if cost_config is not None:
        result.update(_compute_cost_metrics(portfolio_returns, cost_config))
    result.update(
        _compute_benchmark_skill_metrics(
            portfolio_returns,
            returns,
            label_column,
            risk_free_rate=risk_free_rate,
            target_weights=target_weights,
            benchmark_weights_by_date=benchmark_weights_by_date,
        )
    )
    return result


def _compute_benchmark_skill_metrics(
    portfolio_returns: pd.DataFrame,
    returns: pd.DataFrame,
    label_column: str,
    risk_free_rate: float = 0.02 / 252,
    target_weights: pd.DataFrame | None = None,
    benchmark_weights_by_date: dict[str, dict[str, float]] | None = None,
) -> dict[str, float | bool | int]:
    result: dict[str, float | bool | int] = {
        "opportunity_benchmark_available": False,
        "market_benchmark_available": False,
        "tracking_error_available": False,
        "tracking_error": 0.0,
        "information_ratio_available": False,
        "information_ratio_vs_benchmark": 0.0,
        "beta_available": False,
        "beta": 1.0,
        "jensen_alpha_available": False,
        "jensen_alpha": 0.0,
        "active_share_available": False,
        "active_share": 0.0,
    }
    if portfolio_returns is None or portfolio_returns.empty or returns is None or returns.empty:
        return result

    frame = portfolio_returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["_portfolio_return"] = pd.to_numeric(frame["portfolio_return"], errors="coerce").fillna(0.0)
    if "cost" in frame.columns:
        frame["_net_return"] = frame["_portfolio_return"] - pd.to_numeric(frame["cost"], errors="coerce").fillna(0.0)
    else:
        frame["_net_return"] = frame["_portfolio_return"]
    dates = list(frame["date"])

    opportunity = _opportunity_benchmark_by_date(returns, label_column).reindex(dates)
    valid_opportunity = opportunity.dropna()
    if len(valid_opportunity) > 0:
        valid_dates = valid_opportunity.index
        portfolio_returns_aligned = frame.set_index("date").loc[valid_dates, "_net_return"]
        excess = pd.Series(
            portfolio_returns_aligned.to_numpy(dtype=float) - valid_opportunity.to_numpy(dtype=float),
            index=valid_dates,
            dtype=float,
        )
        result.update(
            {
                "opportunity_benchmark_available": True,
                "opportunity_benchmark_date_count": int(len(excess)),
                "opportunity_benchmark_return_mean": float(valid_opportunity.mean()),
                "excess_net_mean_return": float(excess.mean()),
                "max_excess_drawdown": _max_drawdown_from_returns(excess),
                "alpha_hit_rate": float((excess > 0).mean()) if len(excess) else 0.0,
                "max_rolling_underperformance": _max_rolling_underperformance(excess),
            }
        )

    market = _market_benchmark_by_date(returns).reindex(dates)
    valid_market = market.dropna()
    if len(valid_market) > 0:
        valid_dates = valid_market.index
        portfolio_returns_aligned = frame.set_index("date").loc[valid_dates, "_net_return"]
        market_excess = pd.Series(
            portfolio_returns_aligned.to_numpy(dtype=float) - valid_market.to_numpy(dtype=float),
            index=valid_dates,
            dtype=float,
        )
        result.update(
            {
                "market_benchmark_available": True,
                "market_benchmark_return_mean": float(valid_market.mean()),
                "market_excess_mean_return": float(market_excess.mean()),
                "market_excess_drawdown": _max_drawdown_from_returns(market_excess),
            }
        )

        # 计算 Tracking Error 和 Information Ratio
        if len(market_excess) >= 2:
            tracking_error = float(market_excess.std(ddof=0) * np.sqrt(252.0))
            result["tracking_error"] = tracking_error
            result["tracking_error_available"] = True

            # Information Ratio: IR = mean_excess / std_excess * sqrt(252)
            excess_std = float(market_excess.std(ddof=0))
            if excess_std > 0:
                ir = float(market_excess.mean() / excess_std * np.sqrt(252.0))
                result["information_ratio_vs_benchmark"] = ir
                result["information_ratio_available"] = True

        # 计算 Beta 和 Jensen's Alpha
        if len(market_excess) >= 5:
            portfolio_series = frame["_net_return"].reset_index(drop=True).astype(float)
            benchmark_series = market.reset_index(drop=True).astype(float)

            # Beta = Cov(Rp, Rb) / Var(Rb)
            covariance = float(np.cov(portfolio_series, benchmark_series, ddof=0)[0, 1])
            variance = float(np.var(benchmark_series, ddof=0))

            if variance > 0:
                beta = covariance / variance
                result["beta"] = float(beta)
                result["beta_available"] = True

                # Jensen's Alpha: α = Rp - [Rf + β × (Rm - Rf)]
                rp_mean = float(portfolio_series.mean())
                rb_mean = float(benchmark_series.mean())
                alpha_daily = rp_mean - (risk_free_rate + beta * (rb_mean - risk_free_rate))
                result["jensen_alpha"] = float(alpha_daily * 252.0)
                result["jensen_alpha_available"] = True

    # 计算 Active Share
    if benchmark_weights_by_date and target_weights is not None:
        active_shares = []
        for date in dates:
            date_key = str(pd.Timestamp(date).date())
            # 提取当日组合权重
            day_weights = target_weights[target_weights["date"] == date]
            if day_weights.empty:
                continue
            portfolio_weights = {
                str(row["order_book_id"]): float(row["target_weight"])
                for _, row in day_weights.iterrows()
            }
            benchmark_weights = benchmark_weights_by_date.get(date_key)
            if benchmark_weights:
                as_ = _compute_active_share(portfolio_weights, benchmark_weights)
                active_shares.append(as_)

        if active_shares:
            result["active_share"] = float(np.mean(active_shares))
            result["active_share_available"] = True

    return result


def _opportunity_benchmark_by_date(returns: pd.DataFrame, label_column: str) -> pd.Series:
    if returns is None or returns.empty or "date" not in returns.columns:
        return pd.Series(dtype=float)
    frame = returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "opportunity_benchmark_return" in frame.columns:
        values = pd.to_numeric(frame["opportunity_benchmark_return"], errors="coerce")
        return values.groupby(frame["date"]).first().astype(float)
    if label_column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[label_column], errors="coerce")
    return values.groupby(frame["date"]).mean().astype(float)


def _market_benchmark_by_date(returns: pd.DataFrame) -> pd.Series:
    if returns is None or returns.empty or "date" not in returns.columns:
        return pd.Series(dtype=float)
    frame = returns.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    for column in ("market_benchmark_return", "benchmark_return", "market_return"):
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            return values.groupby(frame["date"]).first().astype(float)
    return pd.Series(dtype=float)


def _max_drawdown_from_returns(returns: pd.Series) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    series = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    equity = (1.0 + series).cumprod()
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    return float(drawdown.max()) if len(drawdown) else 0.0


def _max_rolling_underperformance(returns: pd.Series, window: int = 20) -> float:
    if returns is None or len(returns) == 0:
        return 0.0
    series = pd.to_numeric(returns, errors="coerce").fillna(0.0).astype(float)
    negative_period = float((-series).clip(lower=0.0).max()) if len(series) else 0.0
    rolling_window = max(1, min(int(window), len(series)))
    rolling_loss = float((-series.rolling(rolling_window, min_periods=1).sum()).clip(lower=0.0).max())
    return max(negative_period, rolling_loss)


def _compute_active_share(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
) -> float:
    """计算单日 Active Share

    Active Share = (1/2) × Σ |wi - bi|
    其中 wi 是组合权重，bi 是基准权重

    Args:
        portfolio_weights: 组合权重字典 {order_book_id: weight}
        benchmark_weights: 基准权重字典 {order_book_id: weight}

    Returns:
        Active Share ∈ [0, 1]
    """
    all_assets = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
    total_diff = sum(
        abs(portfolio_weights.get(asset, 0.0) - benchmark_weights.get(asset, 0.0))
        for asset in all_assets
    )
    return 0.5 * total_diff


def _constraint_violations(target_weights: pd.DataFrame, constraints: dict) -> dict:
    if target_weights.empty or "target_weight" not in target_weights.columns:
        return {
            "max_single_weight_count": 0,
            "max_industry_weight_count": 0,
            "gross_exposure_count": 0,
            "gross_exposure_shortfall_count": 0,
            "min_position_count": 0,
            "max_position_count": 0,
        }
    weights = pd.to_numeric(target_weights["target_weight"], errors="coerce").fillna(0.0)
    max_single_weight = constraints.get("max_single_weight")
    max_count = 0
    if max_single_weight is not None:
        max_count = int((weights > float(max_single_weight) + 1e-12).sum())

    max_industry_weight = constraints.get("max_industry_weight")
    max_industry_count = 0
    if max_industry_weight is not None and "date" in target_weights.columns:
        exposure = _industry_exposure_frame(target_weights)
        if not exposure.empty:
            max_industry_count = int((exposure["industry_weight"] > float(max_industry_weight) + 1e-12).sum())

    target_gross = constraints.get("target_gross_exposure")
    cash_buffer = float(constraints.get("cash_buffer", 0.0) or 0.0)
    gross_count = 0
    gross_shortfall_count = 0
    if target_gross is not None and "date" in target_weights.columns:
        target_budget = max(0.0, float(target_gross) - cash_buffer)
        gross = target_weights.assign(_weight=weights.abs()).groupby("date")["_weight"].sum()
        gross_count = int((gross > target_budget + 1e-8).sum())
        gross_shortfall_count = int((gross < target_budget - 1e-8).sum())

    min_position_count = constraints.get("min_position_count")
    max_position_count = constraints.get("max_position_count")
    min_position_violation = 0
    max_position_violation = 0
    if "date" in target_weights.columns and "order_book_id" in target_weights.columns:
        active = target_weights.assign(_weight=weights)
        active = active[active["_weight"] > 1e-12]
        position_counts = active.groupby("date")["order_book_id"].nunique()
        if min_position_count is not None:
            min_position_violation = int((position_counts < int(min_position_count)).sum())
        if max_position_count is not None:
            max_position_violation = int((position_counts > int(max_position_count)).sum())
    return {
        "max_single_weight_count": max_count,
        "max_industry_weight_count": max_industry_count,
        "gross_exposure_count": gross_count,
        "gross_exposure_shortfall_count": gross_shortfall_count,
        "min_position_count": min_position_violation,
        "max_position_count": max_position_violation,
    }


def _compute_activity_metrics(target_weights: pd.DataFrame, constraints: dict) -> dict[str, float | int]:
    if target_weights is None or target_weights.empty or not {"date", "target_weight"}.issubset(target_weights.columns):
        return {
            "active_gross_mean": 0.0,
            "active_day_ratio": 0.0,
            "cash_sitting_ratio": 0.0,
        }
    frame = target_weights.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0).abs()
    gross = frame.groupby("date", sort=True)["_weight"].sum()
    if gross.empty:
        return {
            "active_gross_mean": 0.0,
            "active_day_ratio": 0.0,
            "cash_sitting_ratio": 0.0,
        }
    target_gross = float(constraints.get("target_gross_exposure", 1.0) or 1.0)
    active = gross > 1e-12
    return {
        "active_gross_mean": float(gross.mean()),
        "active_gross_max": float(gross.max()),
        "active_day_ratio": float(active.mean()),
        "cash_sitting_ratio": float((~active).mean()),
        "mean_cash_weight": float((target_gross - gross).clip(lower=0.0).mean()),
    }


def _compute_allocation_metrics(target_weights: pd.DataFrame, constraints: dict) -> dict[str, float | int]:
    if target_weights is None or target_weights.empty or "date" not in target_weights.columns:
        return {
            "allocation_date_count": 0,
            "allocation_drift_mean": 0.0,
            "allocation_drift_max": 0.0,
        }
    if "universe_layer" not in target_weights.columns and "intended_weight" not in target_weights.columns:
        return {}

    frame = target_weights.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["_target_weight"] = pd.to_numeric(frame.get("target_weight", 0.0), errors="coerce").fillna(0.0)
    if "intended_weight" in frame.columns:
        frame["_intended_weight"] = pd.to_numeric(frame["intended_weight"], errors="coerce").fillna(frame["_target_weight"])
    else:
        frame["_intended_weight"] = frame["_target_weight"]

    result: dict[str, float | int] = {}
    if "universe_layer" in frame.columns:
        frame["_layer"] = frame["universe_layer"].fillna("unknown").astype(str)
        layer_exposure = frame.groupby(["date", "_layer"], sort=True)["_target_weight"].sum().unstack(fill_value=0.0)
        for layer in layer_exposure.columns:
            values = layer_exposure[layer]
            result[f"{layer}_weight_mean"] = float(values.mean()) if len(values) else 0.0
            result[f"{layer}_weight_max"] = float(values.max()) if len(values) else 0.0
    else:
        layer_exposure = pd.DataFrame(index=frame["date"].drop_duplicates().sort_values())

    drift = (frame["_target_weight"] - frame["_intended_weight"]).abs()
    result["allocation_drift_mean"] = float(drift.mean()) if len(drift) else 0.0
    result["allocation_drift_max"] = float(drift.max()) if len(drift) else 0.0

    for layer in layer_exposure.columns if "universe_layer" in frame.columns else []:
        budget_column = f"{layer}_budget"
        if budget_column not in frame.columns or "universe_layer" not in frame.columns:
            continue
        budgets = frame.groupby("date", sort=True)[budget_column].first().astype(float)
        exposure = layer_exposure[layer] if layer in layer_exposure.columns else pd.Series(0.0, index=budgets.index)
        exposure = exposure.reindex(budgets.index, fill_value=0.0)
        deviation = (exposure - budgets).abs()
        result[f"{layer}_budget_deviation_mean"] = float(deviation.mean()) if len(deviation) else 0.0
        result[f"{layer}_budget_deviation_max"] = float(deviation.max()) if len(deviation) else 0.0

    cash_budget = None
    if "cash_buffer" in frame.columns:
        cash_budget = frame.groupby("date", sort=True)["cash_buffer"].first().astype(float)
    elif constraints.get("cash_buffer") is not None:
        dates = frame["date"].drop_duplicates().sort_values()
        cash_budget = pd.Series(float(constraints.get("cash_buffer", 0.0) or 0.0), index=dates)
    if cash_budget is not None:
        gross = frame.groupby("date", sort=True)["_target_weight"].sum().reindex(cash_budget.index, fill_value=0.0)
        target_gross = float(constraints.get("target_gross_exposure", 1.0) or 1.0)
        realized_cash = target_gross - gross
        cash_deviation = (realized_cash - cash_budget).abs()
        result["cash_buffer_deviation_mean"] = float(cash_deviation.mean()) if len(cash_deviation) else 0.0
        result["cash_buffer_deviation_max"] = float(cash_deviation.max()) if len(cash_deviation) else 0.0

    if "rebalance_allowed" in frame.columns:
        allowed = frame.groupby("date", sort=True)["rebalance_allowed"].first().astype(bool)
        result["rebalance_allowed_date_count"] = int(allowed.sum())
        result["rebalance_blocked_date_count"] = int((~allowed).sum())
    result["allocation_date_count"] = int(frame["date"].nunique())
    return result


def _industry_exposure_frame(target_weights: pd.DataFrame) -> pd.DataFrame:
    if target_weights.empty or "date" not in target_weights.columns or "target_weight" not in target_weights.columns:
        return pd.DataFrame(columns=["date", "industry", "industry_weight"])
    frame = target_weights.copy()
    if "industry" not in frame.columns:
        frame["industry"] = "Unknown"
    frame["industry"] = frame["industry"].fillna("Unknown").astype(str)
    frame["_weight"] = pd.to_numeric(frame["target_weight"], errors="coerce").fillna(0.0).abs()
    return (
        frame.groupby(["date", "industry"], as_index=False, sort=True)["_weight"]
        .sum()
        .rename(columns={"_weight": "industry_weight"})
    )


def _industry_exposure_by_date(target_weights: pd.DataFrame) -> dict[str, dict[str, float]]:
    exposure = _industry_exposure_frame(target_weights)
    if exposure.empty:
        return {}
    result: dict[str, dict[str, float]] = {}
    for date, day_df in exposure.groupby("date", sort=True):
        date_key = str(pd.Timestamp(date).date())
        result[date_key] = {
            str(row["industry"]): float(row["industry_weight"])
            for _, row in day_df.iterrows()
        }
    return result


def _compute_cost_metrics(portfolio_returns: pd.DataFrame, cost_config) -> dict:
    if not _is_cost_enabled(cost_config):
        return {}
    if "cost" in portfolio_returns.columns:
        return _compute_cost_metrics_from_cost_column(portfolio_returns)
    cfg = coerce_cost_config(cost_config)
    if cfg is None:
        return {}
    from skyeye.products.tx1.cost_layer import compute_cost_metrics

    return compute_cost_metrics(portfolio_returns, cfg)


def _compute_cost_metrics_from_cost_column(portfolio_returns: pd.DataFrame) -> dict:
    if portfolio_returns is None or len(portfolio_returns) == 0:
        return _empty_cost_metrics()
    costed = portfolio_returns.copy()
    costed["cost"] = pd.to_numeric(costed["cost"], errors="coerce").fillna(0.0)
    costed["portfolio_return"] = pd.to_numeric(costed["portfolio_return"], errors="coerce").fillna(0.0)
    costed["turnover"] = pd.to_numeric(costed["turnover"], errors="coerce").fillna(0.0)
    costed["net_return"] = costed["portfolio_return"] - costed["cost"]

    mean_turnover = float(costed["turnover"].mean())
    gross_mean = float(costed["portfolio_return"].mean())
    net_mean = float(costed["net_return"].mean())
    mean_cost = float(costed["cost"].mean())
    if abs(gross_mean) > 1e-12:
        cost_erosion = 1.0 - (net_mean / gross_mean) if gross_mean > 0 else float("inf")
    else:
        cost_erosion = 0.0
    breakeven_bps = (gross_mean / mean_turnover * 10000.0) if mean_turnover > 1e-12 and gross_mean > 0 else 0.0
    return {
        "annual_turnover": float(np.clip(mean_turnover * 252.0, 0, None)),
        "mean_turnover_per_period": mean_turnover,
        "mean_period_cost": mean_cost,
        "cost_drag_annual": mean_cost * 252.0,
        "cost_erosion_ratio": float(np.clip(cost_erosion, 0, None)),
        "net_mean_return": net_mean,
        "breakeven_cost_bps": breakeven_bps,
    }


def _compute_cost_metrics_from_orders(
    orders: pd.DataFrame,
    *,
    target_weights: pd.DataFrame,
    portfolio_return_mean: float,
    portfolio_value: float,
) -> dict:
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    target_dates = _target_dates(target_weights)
    if not target_dates:
        return _empty_cost_metrics()
    daily_turnover = _order_turnover_values(
        orders,
        target_weights=target_weights,
        portfolio_value=portfolio_value,
    )
    normalized = _normalize_orders(orders)
    if normalized.empty:
        mean_cost = 0.0
    else:
        cost_by_date = normalized.groupby("date")["estimated_cost"].sum()
        daily_costs = []
        for date in target_dates:
            daily_costs.append(float(cost_by_date.get(date, 0.0)) / float(portfolio_value))
        mean_cost = float(np.mean(daily_costs)) if daily_costs else 0.0
    mean_turnover = float(np.mean(daily_turnover)) if daily_turnover else 0.0
    net_mean = float(portfolio_return_mean) - mean_cost
    if abs(float(portfolio_return_mean)) > 1e-12:
        cost_erosion = 1.0 - (net_mean / float(portfolio_return_mean)) if portfolio_return_mean > 0 else float("inf")
    else:
        cost_erosion = 0.0
    breakeven_bps = (
        float(portfolio_return_mean) / mean_turnover * 10000.0
        if mean_turnover > 1e-12 and portfolio_return_mean > 0
        else 0.0
    )
    return {
        "annual_turnover": float(np.clip(mean_turnover * 252.0, 0, None)),
        "mean_turnover_per_period": mean_turnover,
        "mean_period_cost": mean_cost,
        "cost_drag_annual": mean_cost * 252.0,
        "cost_erosion_ratio": float(np.clip(cost_erosion, 0, None)),
        "net_mean_return": net_mean,
        "breakeven_cost_bps": breakeven_bps,
    }


def _compute_turnover_metrics_from_orders(
    orders: pd.DataFrame,
    *,
    target_weights: pd.DataFrame,
    portfolio_value: float,
) -> dict:
    daily_turnover = _order_turnover_values(
        orders,
        target_weights=target_weights,
        portfolio_value=portfolio_value,
    )
    mean_turnover = float(np.mean(daily_turnover)) if daily_turnover else 0.0
    return {
        "mean_turnover": mean_turnover,
        "mean_turnover_per_period": mean_turnover,
        "annual_turnover": float(np.clip(mean_turnover * 252.0, 0, None)),
    }


def _apply_tradable_outcome_metrics(portfolio: dict, tradable_outcome: dict | None) -> dict:
    if not isinstance(tradable_outcome, dict) or not tradable_outcome:
        return portfolio

    result = dict(portfolio)
    if "max_drawdown" in result and "gross_max_drawdown" not in result:
        result["gross_max_drawdown"] = _finite_metric(result.get("max_drawdown"), 0.0)

    gross_mean = _finite_metric(tradable_outcome.get("mean_gross_return"), result.get("portfolio_return_mean", 0.0))
    net_mean = _finite_metric(tradable_outcome.get("mean_net_return"), result.get("net_mean_return", gross_mean))
    mean_cost = _finite_metric(
        tradable_outcome.get("mean_execution_cost"),
        result.get("mean_period_cost", max(0.0, gross_mean - net_mean)),
    )
    mean_turnover = _finite_metric(tradable_outcome.get("mean_turnover"), result.get("mean_turnover", 0.0))

    result.update(
        {
            "tradable_outcome_available": True,
            "tradable_outcome_date_count": int(_finite_metric(tradable_outcome.get("date_count"), 0.0)),
            "portfolio_return_mean": gross_mean,
            "net_mean_return": net_mean,
            "mean_period_cost": mean_cost,
            "cost_drag_annual": mean_cost * 252.0,
            "mean_turnover": mean_turnover,
            "mean_turnover_per_period": mean_turnover,
            "annual_turnover": float(np.clip(mean_turnover * 252.0, 0, None)),
            "max_drawdown": _finite_metric(tradable_outcome.get("max_net_drawdown"), result.get("max_drawdown", 0.0)),
            "gross_max_drawdown": _finite_metric(
                tradable_outcome.get("gross_max_drawdown"),
                result.get("gross_max_drawdown", 0.0),
            ),
            "tradable_cost_source": str(tradable_outcome.get("cost_source", "")),
        }
    )
    if abs(gross_mean) > 1e-12:
        cost_erosion = 1.0 - (net_mean / gross_mean) if gross_mean > 0 else float("inf")
    else:
        cost_erosion = 0.0
    result["cost_erosion_ratio"] = float(np.clip(cost_erosion, 0, None))
    result["breakeven_cost_bps"] = (
        gross_mean / mean_turnover * 10000.0 if mean_turnover > 1e-12 and gross_mean > 0 else 0.0
    )
    return result


def _finite_metric(value, fallback: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    if not math.isfinite(result):
        return float(fallback)
    return result


def _order_turnover_values(
    orders: pd.DataFrame,
    *,
    target_weights: pd.DataFrame,
    portfolio_value: float,
) -> list[float]:
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    target_dates = _target_dates(target_weights)
    if not target_dates:
        return []
    normalized = _normalize_orders(orders)
    if normalized.empty:
        return [0.0 for _ in target_dates]
    value_by_date = normalized.groupby("date")["order_value"].apply(lambda values: values.abs().sum())
    return [float(value_by_date.get(date, 0.0)) / float(portfolio_value) for date in target_dates]


def _target_dates(target_weights: pd.DataFrame) -> list[pd.Timestamp]:
    if target_weights is None or target_weights.empty or "date" not in target_weights.columns:
        return []
    dates = pd.to_datetime(target_weights["date"], errors="coerce").dropna().drop_duplicates()
    return [pd.Timestamp(date) for date in sorted(dates)]


def _asset_specific_period_cost(
    weights: dict[str, float],
    prev_weights: dict[str, float],
    asset_types: dict[str, str],
    prev_asset_types: dict[str, str],
    cost_config,
) -> float | None:
    if not _is_cost_enabled(cost_config):
        return None
    total_cost = 0.0
    for order_book_id in set(weights) | set(prev_weights):
        new_w = float(weights.get(order_book_id, 0.0))
        old_w = float(prev_weights.get(order_book_id, 0.0))
        delta = new_w - old_w
        if abs(delta) <= 0:
            continue
        asset_type = asset_types.get(order_book_id) or prev_asset_types.get(order_book_id)
        buy_rate, sell_rate = _one_way_cost_rates(cost_config, asset_type)
        if delta > 0:
            total_cost += delta * buy_rate
        else:
            total_cost += abs(delta) * sell_rate
    return float(total_cost)


def _asset_types_by_id(frame: pd.DataFrame) -> dict[str, str]:
    if "asset_type" not in frame.columns:
        return {}
    values = frame.dropna(subset=["order_book_id"]).drop_duplicates("order_book_id", keep="last")
    return {
        str(row["order_book_id"]): normalize_asset_type(row.get("asset_type"))
        for _, row in values.iterrows()
    }


def _is_cost_enabled(cost_config) -> bool:
    if cost_config is None:
        return False
    if isinstance(cost_config, dict):
        return bool(cost_config.get("enabled", True))
    return True


def _round_trip_cost_rate(cost_config, asset_type: str | None) -> float:
    from skyeye.products.tx1.cost_layer import CostConfig

    if isinstance(cost_config, CostConfig):
        return float(cost_config.round_trip_cost)
    if not isinstance(cost_config, dict):
        raise TypeError("cost_config must be a dict or CostConfig")

    if "stock" in cost_config or "etf" in cost_config:
        normalized = asset_type or str(cost_config.get("default_asset_type", "stock"))
        section = cost_config.get(normalized) or cost_config.get(str(cost_config.get("default_asset_type", "stock")), {})
        return _round_trip_cost_rate_from_mapping(section)

    return float(
        CostConfig(
            commission_rate=float(cost_config.get("commission_rate", 0.0008)),
            stamp_tax_rate=float(cost_config.get("stamp_tax_rate", 0.0005)),
            slippage_bps=float(cost_config.get("slippage_bps", 5.0)),
            min_commission=float(cost_config.get("min_commission", 0.0)),
        ).round_trip_cost
    )


def _one_way_cost_rates(cost_config, asset_type: str | None) -> tuple[float, float]:
    """Return (buy_rate, sell_rate) for a single asset.

    Buy side: commission + slippage (no stamp tax).
    Sell side: commission + slippage + stamp_tax.
    """
    from skyeye.products.tx1.cost_layer import CostConfig

    if isinstance(cost_config, CostConfig):
        buy = cost_config.commission_rate + cost_config.slippage_rate
        sell = buy + cost_config.stamp_tax_rate
        return (buy, sell)
    if not isinstance(cost_config, dict):
        raise TypeError("cost_config must be a dict or CostConfig")

    if "stock" in cost_config or "etf" in cost_config:
        normalized = asset_type or str(cost_config.get("default_asset_type", "stock"))
        section = cost_config.get(normalized) or cost_config.get(str(cost_config.get("default_asset_type", "stock")), {})
        return _one_way_cost_rates_from_mapping(section)

    commission_rate = float(cost_config.get("commission_rate", 0.0008))
    stamp_tax_rate = float(cost_config.get("stamp_tax_rate", 0.0005))
    slippage_rate = float(cost_config.get("slippage_bps", 5.0)) / 10000.0
    buy = commission_rate + slippage_rate
    sell = buy + stamp_tax_rate
    return (buy, sell)


def _one_way_cost_rates_from_mapping(config: dict) -> tuple[float, float]:
    """Return (buy_rate, sell_rate) from a per-asset-type config mapping."""
    commission_rate = float(config.get("commission_rate", 0.0))
    stamp_tax_rate = float(config.get("stamp_tax_rate", 0.0))
    slippage_rate = float(config.get("slippage_bps", 0.0)) / 10000.0
    impact_rate = float(config.get("impact_bps", 0.0)) / 10000.0
    buy = commission_rate + slippage_rate + 0.5 * impact_rate
    sell = commission_rate + slippage_rate + stamp_tax_rate + 0.5 * impact_rate
    return (buy, sell)


def _round_trip_cost_rate_from_mapping(config: dict) -> float:
    commission_rate = float(config.get("commission_rate", 0.0))
    stamp_tax_rate = float(config.get("stamp_tax_rate", 0.0))
    slippage_rate = float(config.get("slippage_bps", 0.0)) / 10000.0
    impact_rate = float(config.get("impact_bps", 0.0)) / 10000.0
    return (2.0 * commission_rate) + stamp_tax_rate + (2.0 * slippage_rate) + impact_rate


def _empty_cost_metrics() -> dict:
    return {
        "annual_turnover": 0.0,
        "mean_turnover_per_period": 0.0,
        "mean_period_cost": 0.0,
        "cost_drag_annual": 0.0,
        "cost_erosion_ratio": 0.0,
        "net_mean_return": 0.0,
        "breakeven_cost_bps": 0.0,
    }


def _select_prediction_column(predictions: pd.DataFrame) -> str | None:
    if "adjusted_expected_return" in predictions.columns:
        return "adjusted_expected_return"
    if "expected_relative_net_return_10d" in predictions.columns:
        return "expected_relative_net_return_10d"
    expected_relative_columns = sorted(column for column in predictions.columns if column.startswith("expected_relative_net_return_"))
    if expected_relative_columns:
        return expected_relative_columns[0]
    return None


def _select_label_column(labels: pd.DataFrame) -> str | None:
    if "label_relative_net_return_10d" in labels.columns:
        return "label_relative_net_return_10d"
    relative_columns = sorted(column for column in labels.columns if column.startswith("label_relative_net_return_"))
    if relative_columns:
        return relative_columns[0]
    if "label_net_return_10d" in labels.columns:
        return "label_net_return_10d"
    net_columns = sorted(column for column in labels.columns if column.startswith("label_net_return_"))
    if net_columns:
        return net_columns[0]
    if "label_return_10d" in labels.columns:
        return "label_return_10d"
    label_columns = sorted(column for column in labels.columns if column.startswith("label_return_"))
    return label_columns[0] if label_columns else None


def _select_portfolio_label_column(labels: pd.DataFrame) -> str | None:
    if "label_net_return_10d" in labels.columns:
        return "label_net_return_10d"
    net_columns = sorted(column for column in labels.columns if column.startswith("label_net_return_"))
    if net_columns:
        return net_columns[0]
    if "label_return_10d" in labels.columns:
        return "label_return_10d"
    label_columns = sorted(column for column in labels.columns if column.startswith("label_return_"))
    if label_columns:
        return label_columns[0]
    if "label_relative_net_return_10d" in labels.columns:
        return "label_relative_net_return_10d"
    relative_columns = sorted(column for column in labels.columns if column.startswith("label_relative_net_return_"))
    return relative_columns[0] if relative_columns else None


def _merge_on_panel_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_columns: list[str],
    right_columns: list[str],
) -> pd.DataFrame:
    required = ["date", "order_book_id"]
    if not set(required).issubset(left.columns) or not set(required).issubset(right.columns):
        return pd.DataFrame()
    join_keys = list(required)
    if "fold_id" in left.columns and "fold_id" in right.columns:
        join_keys.append("fold_id")
    left_payload = [column for column in left_columns if column not in join_keys]
    right_payload = [column for column in right_columns if column not in join_keys]
    left_frame = left[join_keys + left_payload].copy()
    right_frame = right[join_keys + right_payload].copy()
    left_frame["date"] = pd.to_datetime(left_frame["date"])
    right_frame["date"] = pd.to_datetime(right_frame["date"])
    left_frame["order_book_id"] = left_frame["order_book_id"].astype(str)
    right_frame["order_book_id"] = right_frame["order_book_id"].astype(str)
    return left_frame.merge(right_frame, on=join_keys, how="inner")


def _compute_order_execution_metrics(orders: pd.DataFrame, *, min_trade_value: float = 0.0) -> dict:
    normalized = _normalize_orders(orders)
    if normalized.empty:
        trading = {
            "trade_count": 0,
            "buy_count": 0,
            "sell_count": 0,
            "active_rebalance_date_count": 0,
            "avg_order_value": 0.0,
            "median_order_value": 0.0,
            "max_order_value": 0.0,
            "avg_orders_per_active_date": 0.0,
            "max_orders_single_date": 0,
        }
        return {
            "trading": trading,
            "manual_operation_burden": {
                "manual_trade_days": 0,
                "total_orders": 0,
                "max_orders_single_date": 0,
                "avg_order_value": 0.0,
                "small_order_count": 0,
            },
            "turnover_detail": {
                "total_order_value": 0.0,
                "estimated_cost_sum": 0.0,
                "by_date": {},
            },
        }

    order_values = normalized["order_value"]
    orders_by_date = normalized.groupby("date", sort=True).size()
    trading = {
        "trade_count": int(len(normalized)),
        "buy_count": int((normalized["side"] == "buy").sum()),
        "sell_count": int((normalized["side"] == "sell").sum()),
        "active_rebalance_date_count": int(orders_by_date.size),
        "avg_order_value": float(order_values.mean()),
        "median_order_value": float(order_values.median()),
        "max_order_value": float(order_values.max()),
        "avg_orders_per_active_date": float(orders_by_date.mean()),
        "max_orders_single_date": int(orders_by_date.max()),
    }
    min_trade_value = float(min_trade_value or 0.0)
    turnover_detail = {
        "total_order_value": float(order_values.sum()),
        "estimated_cost_sum": float(normalized["estimated_cost"].sum()),
        "by_date": _turnover_by_date(normalized),
    }
    return {
        "trading": trading,
        "manual_operation_burden": {
            "manual_trade_days": trading["active_rebalance_date_count"],
            "total_orders": trading["trade_count"],
            "max_orders_single_date": trading["max_orders_single_date"],
            "avg_order_value": trading["avg_order_value"],
            "small_order_count": int((order_values < min_trade_value).sum()) if min_trade_value > 0 else 0,
        },
        "turnover_detail": turnover_detail,
    }


def _normalize_orders(orders: pd.DataFrame) -> pd.DataFrame:
    if orders is None or orders.empty or "date" not in orders.columns:
        return pd.DataFrame(columns=["date", "order_book_id", "asset_type", "side", "order_value", "estimated_cost"])
    frame = orders.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if "order_book_id" not in frame.columns:
        frame["order_book_id"] = ""
    frame["order_book_id"] = frame["order_book_id"].astype(str)
    if "asset_type" in frame.columns:
        frame["asset_type"] = frame["asset_type"].map(normalize_asset_type)
    else:
        frame["asset_type"] = "unknown"
    frame["order_value"] = _order_value_series(frame)
    frame["estimated_cost"] = (
        pd.to_numeric(frame["estimated_cost"], errors="coerce").fillna(0.0)
        if "estimated_cost" in frame.columns
        else 0.0
    )
    frame["side"] = _order_side_series(frame)
    return frame[["date", "order_book_id", "asset_type", "side", "order_value", "estimated_cost"]]


def _order_value_series(orders: pd.DataFrame) -> pd.Series:
    if "order_value" in orders.columns:
        values = pd.to_numeric(orders["order_value"], errors="coerce").fillna(0.0)
    elif "value" in orders.columns:
        values = pd.to_numeric(orders["value"], errors="coerce").fillna(0.0)
    elif {"price", "quantity"}.issubset(orders.columns):
        values = (
            pd.to_numeric(orders["price"], errors="coerce").fillna(0.0)
            * pd.to_numeric(orders["quantity"], errors="coerce").fillna(0.0)
        )
    else:
        values = pd.Series(0.0, index=orders.index)
    return values.abs().astype(float)


def _order_side_series(orders: pd.DataFrame) -> pd.Series:
    side_column = next((column for column in ["side", "order_side", "direction"] if column in orders.columns), None)
    if side_column is None:
        return pd.Series("unknown", index=orders.index)
    raw = orders[side_column].astype(str).str.lower().str.strip()
    return raw.replace(
        {
            "b": "buy",
            "buy_open": "buy",
            "1": "buy",
            "s": "sell",
            "sell_close": "sell",
            "-1": "sell",
        }
    )


def _turnover_by_date(orders: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for date, day_df in orders.groupby("date", sort=True):
        result[str(pd.Timestamp(date).date())] = {
            "order_count": int(len(day_df)),
            "order_value_sum": float(day_df["order_value"].sum()),
            "estimated_cost_sum": float(day_df["estimated_cost"].sum()),
        }
    return result


def _contribution_by_asset_type(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    orders: pd.DataFrame | None = None,
) -> dict[str, dict[str, float | int]]:
    label_column = _select_portfolio_label_column(returns)
    if label_column is None or target_weights.empty:
        return {}

    left_columns = ["target_weight"]
    right_columns = [label_column]
    if "asset_type" in target_weights.columns:
        left_columns.append("asset_type")
    elif "asset_type" in returns.columns:
        right_columns.append("asset_type")
    merged = _merge_on_panel_keys(target_weights, returns, left_columns, right_columns)
    if merged.empty or "asset_type" not in merged.columns:
        return {}

    merged["asset_type"] = merged["asset_type"].map(normalize_asset_type)
    merged["_target_weight"] = pd.to_numeric(merged["target_weight"], errors="coerce").fillna(0.0)
    merged["_label_value"] = pd.to_numeric(merged[label_column], errors="coerce")
    merged = merged.dropna(subset=["_label_value"])
    if merged.empty:
        return {}
    merged["_gross_return"] = merged["_target_weight"] * merged["_label_value"]

    result: dict[str, dict[str, float | int]] = {}
    for asset_type, asset_df in merged.groupby("asset_type", sort=True):
        result[str(asset_type)] = {
            "gross_return_mean": float(asset_df["_gross_return"].mean()),
            "row_count": int(len(asset_df)),
        }

    order_aggregates = _order_aggregates_by_asset_type(orders)
    for asset_type, aggregates in order_aggregates.items():
        result.setdefault(asset_type, {"gross_return_mean": 0.0, "row_count": 0}).update(aggregates)
    return result


def _order_aggregates_by_asset_type(orders: pd.DataFrame | None) -> dict[str, dict[str, float | int]]:
    normalized = _normalize_orders(orders) if orders is not None else pd.DataFrame()
    if normalized.empty:
        return {}
    result: dict[str, dict[str, float | int]] = {}
    for asset_type, asset_df in normalized.groupby("asset_type", sort=True):
        result[str(asset_type)] = {
            "order_count": int(len(asset_df)),
            "order_value_sum": float(asset_df["order_value"].sum()),
            "estimated_cost_sum": float(asset_df["estimated_cost"].sum()),
        }
    return result
