import pandas as pd
import pytest

from skyeye.products.ax1.evaluation.metrics import evaluate_portfolio_layer, evaluate_signal_layer


def test_signal_layer_computes_rank_ic_spread_and_hit_rate():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "order_book_id": ["A", "B", "C", "A", "B", "C"],
            "adjusted_expected_return": [0.30, 0.20, 0.10, 0.10, 0.20, 0.30],
        }
    )
    labels = pd.DataFrame(
        {
            "date": predictions["date"],
            "order_book_id": predictions["order_book_id"],
            "label_return_10d": [0.03, 0.02, 0.01, 0.01, 0.02, 0.03],
        }
    )

    metrics = evaluate_signal_layer(predictions, labels)

    assert metrics["signal"]["rank_ic_mean"] == pytest.approx(1.0)
    assert metrics["signal"]["rank_ic_ir"] == pytest.approx(0.0)
    assert metrics["signal"]["rank_ic_significance"]["n_observations"] == 2
    assert metrics["signal"]["top_bucket_spread_mean"] == pytest.approx(0.02)
    assert metrics["signal"]["top_k_hit_rate"] == pytest.approx(1.0)


def test_signal_layer_reports_newey_west_rank_ic_significance_and_fdr():
    rows = []
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    assets = ["A", "B", "C", "D", "E"]
    signal = [3.0, 2.0, 1.0, -1.0, -2.0]
    for offset, date in enumerate(dates):
        noise = 0.0002 * ((offset % 5) - 2)
        for order_book_id, score, label in zip(assets, signal, [0.05, 0.03, 0.01, -0.01, -0.02], strict=True):
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "adjusted_expected_return": score + noise,
                    "label_return_10d": label + noise,
                }
            )
    frame = pd.DataFrame(rows)

    metrics = evaluate_signal_layer(
        frame[["date", "order_book_id", "adjusted_expected_return"]],
        frame[["date", "order_book_id", "label_return_10d"]],
    )

    significance = metrics["signal"]["rank_ic_significance"]
    assert significance["method"] == "newey_west_mean_t_test"
    assert significance["fdr_method"] == "benjamini_hochberg"
    assert significance["n_observations"] == len(dates)
    assert significance["t_stat"] > 0.0
    assert significance["p_value"] <= significance["fdr_adjusted_p_value"]
    assert significance["significant_at_5pct"] is True


def test_signal_layer_merge_respects_fold_id_for_overlapping_oos_windows():
    from skyeye.products.ax1.evaluation.metrics import _merge_on_panel_keys

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "order_book_id": ["A", "A"],
            "fold_id": [0, 1],
            "expected_relative_net_return_10d": [0.01, 0.02],
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "order_book_id": ["A", "A"],
            "fold_id": [0, 1],
            "label_relative_net_return_10d": [0.03, 0.04],
        }
    )

    merged = _merge_on_panel_keys(
        predictions,
        labels,
        ["fold_id", "expected_relative_net_return_10d"],
        ["fold_id", "label_relative_net_return_10d"],
    )

    assert len(merged) == 2
    assert merged["fold_id"].tolist() == [0, 1]


def test_portfolio_layer_computes_return_risk_turnover_and_constraint_summary():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["A", "B", "A", "B"],
            "target_weight": [0.60, 0.40, 0.20, 0.80],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.02, -0.005, -0.05, -0.0125],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        labels,
        constraints={"max_single_weight": 0.50, "target_gross_exposure": 1.0},
    )

    portfolio = metrics["portfolio"]
    assert portfolio["portfolio_return_mean"] == pytest.approx(-0.005)
    assert portfolio["max_drawdown"] == pytest.approx(0.02)
    assert portfolio["information_ratio_proxy"] == pytest.approx(-1.0 / 3.0)
    assert portfolio["mean_turnover"] == pytest.approx(0.45)
    assert portfolio["constraint_violations"]["max_single_weight_count"] == 2
    assert portfolio["constraint_violations"]["gross_exposure_count"] == 0
    assert portfolio["constraint_violations"]["gross_exposure_shortfall_count"] == 0


def test_portfolio_layer_checks_cash_buffer_adjusted_gross_budget():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["A", "B", "A", "B"],
            "target_weight": [0.60, 0.40, 0.30, 0.50],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        constraints={"target_gross_exposure": 1.0, "cash_buffer": 0.10},
    )

    violations = metrics["portfolio"]["constraint_violations"]
    assert violations["gross_exposure_count"] == 1
    assert violations["gross_exposure_shortfall_count"] == 1


def test_portfolio_layer_reports_opportunity_pool_allocation_metrics():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                ]
            ),
            "order_book_id": ["CORE", "IND", "STYLE", "CORE", "IND", "STYLE"],
            "universe_layer": ["core", "industry", "style", "core", "industry", "style"],
            "exposure_group": ["broad_beta", "sector", "style_factor", "broad_beta", "sector", "style_factor"],
            "target_weight": [0.45, 0.30, 0.15, 0.40, 0.35, 0.15],
            "intended_weight": [0.50, 0.25, 0.15, 0.42, 0.33, 0.15],
            "cash_buffer": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        constraints={"target_gross_exposure": 1.0, "cash_buffer": 0.10},
    )

    allocation = metrics["portfolio"]["allocation"]

    assert allocation["core_weight_mean"] == pytest.approx(0.425)
    assert allocation["industry_weight_mean"] == pytest.approx(0.325)
    assert allocation["style_weight_mean"] == pytest.approx(0.15)
    assert allocation["allocation_drift_mean"] > 0.0
    assert allocation["cash_buffer_deviation_mean"] == pytest.approx(0.0)
    assert "core_budget_deviation_mean" not in allocation


def test_portfolio_layer_reports_opportunity_benchmark_skill_metrics():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "order_book_id": ["A", "A", "A"],
            "target_weight": [1.0, 1.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "order_book_id": ["A", "B", "A", "B", "A", "B"],
            "label_return_10d": [0.03, 0.01, -0.02, -0.01, 0.04, 0.00],
            "market_benchmark_return": [0.02, 0.02, -0.10, -0.10, 0.03, 0.03],
        }
    )

    metrics = evaluate_portfolio_layer(target_weights, labels)

    portfolio = metrics["portfolio"]
    assert portfolio["opportunity_benchmark_available"] is True
    assert portfolio["opportunity_benchmark_return_mean"] == pytest.approx((0.02 - 0.015 + 0.02) / 3)
    assert portfolio["excess_net_mean_return"] == pytest.approx((0.01 - 0.005 + 0.02) / 3)
    assert portfolio["alpha_hit_rate"] == pytest.approx(2 / 3)
    assert portfolio["max_excess_drawdown"] > 0.0
    assert portfolio["max_rolling_underperformance"] == pytest.approx(0.005)
    assert portfolio["market_benchmark_available"] is True
    assert portfolio["market_excess_mean_return"] == pytest.approx((0.01 + 0.08 + 0.01) / 3)
    assert portfolio["active_gross_mean"] == pytest.approx(1.0)
    assert portfolio["active_day_ratio"] == pytest.approx(1.0)
    assert portfolio["cash_sitting_ratio"] == pytest.approx(0.0)


def test_portfolio_layer_reports_industry_constraint_violations():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "order_book_id": ["BANK_A", "BANK_B", "TECH_A"],
            "target_weight": [0.35, 0.30, 0.35],
            "industry": ["bank", "bank", "tech"],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        constraints={"max_industry_weight": 0.60},
    )

    violations = metrics["portfolio"]["constraint_violations"]
    assert violations["max_industry_weight_count"] == 1
    assert metrics["portfolio"]["industry_exposure_by_date"]["2024-01-01"]["bank"] == pytest.approx(0.65)


def test_portfolio_layer_computes_transaction_cost_and_net_return_metrics():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["A", "B", "A", "B"],
            "target_weight": [0.60, 0.40, 0.20, 0.80],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.02, -0.005, -0.05, -0.0125],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        labels,
        cost_config={
            "commission_rate": 0.001,
            "stamp_tax_rate": 0.0005,
            "slippage_bps": 5.0,
        },
    )

    portfolio = metrics["portfolio"]
    assert portfolio["cost_drag_annual"] > 0
    assert portfolio["net_mean_return"] < portfolio["portfolio_return_mean"]
    assert portfolio["cost_erosion_ratio"] >= 0
    assert portfolio["breakeven_cost_bps"] >= 0


def test_portfolio_layer_uses_order_ledger_for_cost_when_orders_are_provided():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "order_book_id": ["ETF_A", "ETF_A"],
            "asset_type": ["etf", "etf"],
            "target_weight": [0.50, 0.55],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.01, 0.01],
        }
    )
    orders = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "order_book_id": ["ETF_A"],
            "asset_type": ["etf"],
            "side": ["buy"],
            "order_value": [50_000.0],
            "estimated_cost": [12.0],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        labels,
        cost_config={"enabled": True, "etf": {"commission_rate": 0.01}},
        orders=orders,
        portfolio_value=100_000.0,
    )

    portfolio = metrics["portfolio"]
    expected_mean_cost = (12.0 / 100_000.0) / 2
    assert portfolio["mean_period_cost"] == pytest.approx(expected_mean_cost)
    assert portfolio["cost_drag_annual"] == pytest.approx(expected_mean_cost * 252)
    assert portfolio["net_mean_return"] == pytest.approx(portfolio["portfolio_return_mean"] - expected_mean_cost)


def test_portfolio_layer_uses_order_ledger_for_turnover_when_orders_are_provided():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "order_book_id": ["ETF_A", "ETF_A"],
            "asset_type": ["etf", "etf"],
            "target_weight": [0.50, 0.55],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.01, 0.01],
        }
    )
    empty_orders = pd.DataFrame(
        columns=["date", "order_book_id", "asset_type", "side", "order_value", "estimated_cost"]
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        labels,
        orders=empty_orders,
        portfolio_value=100_000.0,
    )

    portfolio = metrics["portfolio"]
    assert portfolio["mean_turnover"] == pytest.approx(0.0)
    assert portfolio["mean_turnover_per_period"] == pytest.approx(0.0)
    assert portfolio["annual_turnover"] == pytest.approx(0.0)


def test_portfolio_layer_uses_tradable_outcome_as_net_risk_truth():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "order_book_id": ["ETF_A", "ETF_A"],
            "target_weight": [1.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.02, 0.01],
        }
    )
    tradable_outcome = {
        "schema_version": 1,
        "mean_gross_return": 0.015,
        "mean_net_return": 0.007,
        "mean_execution_cost": 0.008,
        "mean_turnover": 0.40,
        "max_net_drawdown": 0.12,
        "gross_max_drawdown": 0.00,
        "date_count": 2,
    }

    metrics = evaluate_portfolio_layer(
        target_weights,
        labels,
        tradable_outcome=tradable_outcome,
    )

    portfolio = metrics["portfolio"]
    assert portfolio["tradable_outcome_available"] is True
    assert portfolio["portfolio_return_mean"] == pytest.approx(0.015)
    assert portfolio["net_mean_return"] == pytest.approx(0.007)
    assert portfolio["mean_period_cost"] == pytest.approx(0.008)
    assert portfolio["mean_turnover"] == pytest.approx(0.40)
    assert portfolio["max_drawdown"] == pytest.approx(0.12)
    assert portfolio["gross_max_drawdown"] == pytest.approx(0.00)


def test_portfolio_layer_reports_manual_trading_metrics_from_orders():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["STOCK_A", "ETF_A", "STOCK_A", "ETF_A"],
            "asset_type": ["stock", "etf", "stock", "etf"],
            "target_weight": [0.60, 0.40, 0.20, 0.80],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.02, 0.01, -0.05, 0.03],
        }
    )
    orders = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["STOCK_A", "ETF_A", "STOCK_A", "ETF_A"],
            "asset_type": ["stock", "etf", "stock", "etf"],
            "side": ["buy", "buy", "sell", "buy"],
            "order_value": [60000.0, 40000.0, 40000.0, 40000.0],
            "estimated_cost": [30.0, 4.0, 20.0, 4.0],
        }
    )

    metrics = evaluate_portfolio_layer(target_weights, labels, orders=orders, min_trade_value=45000.0)

    portfolio = metrics["portfolio"]
    assert portfolio["trading"] == {
        "trade_count": 4,
        "buy_count": 3,
        "sell_count": 1,
        "active_rebalance_date_count": 2,
        "avg_order_value": pytest.approx(45000.0),
        "median_order_value": pytest.approx(40000.0),
        "max_order_value": pytest.approx(60000.0),
        "avg_orders_per_active_date": pytest.approx(2.0),
        "max_orders_single_date": 2,
    }
    assert portfolio["manual_operation_burden"] == {
        "manual_trade_days": 2,
        "total_orders": 4,
        "max_orders_single_date": 2,
        "avg_order_value": pytest.approx(45000.0),
        "small_order_count": 3,
    }
    assert portfolio["contribution_by_asset_type"]["stock"]["gross_return_mean"] == pytest.approx(0.001)
    assert portfolio["contribution_by_asset_type"]["stock"]["order_value_sum"] == pytest.approx(100000.0)
    assert portfolio["contribution_by_asset_type"]["stock"]["estimated_cost_sum"] == pytest.approx(50.0)
    assert portfolio["contribution_by_asset_type"]["etf"]["gross_return_mean"] == pytest.approx(0.014)
    assert portfolio["contribution_by_asset_type"]["etf"]["order_value_sum"] == pytest.approx(80000.0)
    assert portfolio["contribution_by_asset_type"]["etf"]["estimated_cost_sum"] == pytest.approx(8.0)
    assert portfolio["turnover_detail"]["total_order_value"] == pytest.approx(180000.0)
    assert portfolio["turnover_detail"]["estimated_cost_sum"] == pytest.approx(58.0)
    assert portfolio["turnover_detail"]["by_date"]["2024-01-01"]["order_count"] == 2
    assert portfolio["turnover_detail"]["by_date"]["2024-01-02"]["order_value_sum"] == pytest.approx(80000.0)


def test_portfolio_layer_uses_asset_specific_stock_and_etf_costs():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["STOCK_A", "ETF_A", "STOCK_A", "ETF_A"],
            "asset_type": ["stock", "etf", "stock", "etf"],
            "target_weight": [0.50, 0.50, 0.00, 1.00],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.02, 0.02, 0.02, 0.02],
        }
    )

    metrics = evaluate_portfolio_layer(
        target_weights,
        labels,
        cost_config={
            "enabled": True,
            "stock": {"commission_rate": 0.004, "stamp_tax_rate": 0.001, "slippage_bps": 5.0, "impact_bps": 0.0},
            "etf": {"commission_rate": 0.0002, "stamp_tax_rate": 0.0, "slippage_bps": 1.0, "impact_bps": 0.0},
        },
    )

    portfolio = metrics["portfolio"]
    stock_round_trip = 0.004 * 2 + 0.001 + 0.0005 * 2
    etf_round_trip = 0.0002 * 2 + 0.0001 * 2
    expected_period_cost = 0.25 * stock_round_trip + 0.25 * etf_round_trip
    assert portfolio["mean_period_cost"] == pytest.approx(expected_period_cost)
    assert portfolio["net_mean_return"] == pytest.approx(0.02 - expected_period_cost)


def test_portfolio_layer_drops_dates_with_unmatured_forward_labels():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["A", "B", "A", "B"],
            "target_weight": [0.60, 0.40, 0.60, 0.40],
        }
    )
    labels = pd.DataFrame(
        {
            "date": target_weights["date"],
            "order_book_id": target_weights["order_book_id"],
            "label_return_10d": [0.02, 0.01, None, None],
        }
    )

    metrics = evaluate_portfolio_layer(target_weights, labels)

    portfolio = metrics["portfolio"]
    assert portfolio["portfolio_return_mean"] == pytest.approx(0.016)
    assert portfolio["evaluated_date_count"] == 1
    assert portfolio["dropped_tail_date_count"] == 1


def test_signal_layer_computes_group_backtest_metrics():
    """Group backtest: 10 ETFs per day, perfect signal → monotonicity = 1.0."""
    n_assets = 10
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    rows_pred, rows_label = [], []
    for date in dates:
        for i in range(n_assets):
            oid = f"ETF_{i:02d}"
            rows_pred.append({"date": date, "order_book_id": oid, "expected_relative_net_return_10d": float(n_assets - i) / n_assets})
            rows_label.append({"date": date, "order_book_id": oid, "label_relative_net_return_10d": float(n_assets - i) / 100.0})

    predictions = pd.DataFrame(rows_pred)
    labels = pd.DataFrame(rows_label)

    metrics = evaluate_signal_layer(predictions, labels)
    gb = metrics["signal"]["group_backtest"]

    assert gb["n_groups"] == 5
    assert gb["date_count"] == 3
    # Perfect signal: group_0 should have highest return, group_4 lowest
    assert gb["group_0_return_mean"] > gb["group_4_return_mean"]
    # Long-short spread should be positive
    assert gb["long_short_spread_mean"] > 0
    # Monotonicity should be 1.0 for perfect signal
    assert gb["monotonicity"] == pytest.approx(1.0)
    # Long-short IR: 0 when all spreads identical (ddof=0), which is expected for perfect signal + few dates
    assert gb["long_short_ir"] >= 0


def test_signal_layer_group_backtest_empty_labels():
    """Group backtest returns empty dict when no labels provided."""
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "order_book_id": ["A"],
            "expected_relative_net_return_10d": [0.1],
        }
    )
    metrics = evaluate_signal_layer(predictions, labels=None)
    assert metrics["signal"]["group_backtest"] == {}


def test_signal_layer_group_backtest_with_random_signal():
    """Random signal should have monotonicity near 0 over many dates."""
    import numpy as np

    rng = np.random.RandomState(42)
    n_assets, n_dates = 20, 100
    rows_pred, rows_label = [], []
    for d in range(n_dates):
        date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        scores = rng.randn(n_assets)
        actuals = rng.randn(n_assets) / 100.0
        for i in range(n_assets):
            rows_pred.append({"date": date, "order_book_id": f"S{i:03d}", "expected_relative_net_return_10d": scores[i]})
            rows_label.append({"date": date, "order_book_id": f"S{i:03d}", "label_relative_net_return_10d": actuals[i]})

    metrics = evaluate_signal_layer(pd.DataFrame(rows_pred), pd.DataFrame(rows_label))
    gb = metrics["signal"]["group_backtest"]
    assert abs(gb["monotonicity"]) < 0.5  # random signal should not show strong monotonicity


class TestTrackingError:
    """Tracking Error 测试"""

    def test_tracking_error_with_perfect_tracking(self):
        """完美跟踪：组合收益 = 基准收益 → TE = 0"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "order_book_id": ["A", "A", "A"],
                "target_weight": [1.0, 1.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
                "order_book_id": ["A", "A", "A", "B", "B", "B"],
                "label_return_10d": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
                "market_benchmark_return": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["tracking_error_available"] is True
        assert metrics["portfolio"]["tracking_error"] == pytest.approx(0.0, abs=1e-8)

    def test_tracking_error_with_deviation(self):
        """存在跟踪误差：组合与基准收益不同 → TE > 0"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "order_book_id": ["A", "A", "A"],
                "target_weight": [1.0, 1.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
                "order_book_id": ["A", "A", "A", "B", "B", "B"],
                "label_return_10d": [0.03, 0.02, 0.01, 0.02, 0.01, -0.01],
                "market_benchmark_return": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["tracking_error_available"] is True
        assert metrics["portfolio"]["tracking_error"] > 0.0

    def test_tracking_error_insufficient_data(self):
        """数据不足：少于2个交易日 → 不可用"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "order_book_id": ["A"],
                "target_weight": [1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "order_book_id": ["A", "B"],
                "label_return_10d": [0.02, 0.01],
                "market_benchmark_return": [0.02, 0.01],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["tracking_error_available"] is False
        assert metrics["portfolio"]["tracking_error"] == 0.0


class TestInformationRatio:
    """Information Ratio 测试"""

    def test_ir_positive_excess_return(self):
        """正超额收益 + 有限跟踪误差 → IR > 0"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "order_book_id": ["A", "A", "A"],
                "target_weight": [1.0, 1.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
                "order_book_id": ["A", "A", "A", "B", "B", "B"],
                "label_return_10d": [0.03, 0.02, 0.01, 0.02, 0.01, -0.01],
                "market_benchmark_return": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["information_ratio_available"] is True
        assert metrics["portfolio"]["information_ratio_vs_benchmark"] > 0

    def test_ir_negative_excess_return(self):
        """负超额收益 → IR < 0"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
                "order_book_id": ["A", "A", "A", "A"],
                "target_weight": [1.0, 1.0, 1.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"] * 2),
                "order_book_id": ["A", "A", "A", "A", "B", "B", "B", "B"],
                "label_return_10d": [0.01, -0.01, -0.02, 0.00, 0.02, 0.01, -0.01, 0.00],
                "market_benchmark_return": [0.02, 0.01, -0.01, 0.00, 0.02, 0.01, -0.01, 0.00],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["information_ratio_available"] is True
        assert metrics["portfolio"]["information_ratio_vs_benchmark"] < 0

    def test_ir_zero_tracking_error(self):
        """跟踪误差为 0 → IR = 0（避免除零）"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "order_book_id": ["A", "A", "A"],
                "target_weight": [1.0, 1.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"] * 2),
                "order_book_id": ["A", "A", "A", "B", "B", "B"],
                "label_return_10d": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
                "market_benchmark_return": [0.02, 0.01, -0.01, 0.02, 0.01, -0.01],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["information_ratio_available"] is False
        assert metrics["portfolio"]["information_ratio_vs_benchmark"] == 0.0


class TestBeta:
    """Beta 系数测试"""

    def test_beta_market_neutral(self):
        """组合收益完全跟踪基准 → β ≈ 1"""
        import numpy as np

        np.random.seed(42)
        n_dates = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

        target_weights = pd.DataFrame(
            {"date": dates, "order_book_id": ["A"] * n_dates, "target_weight": [1.0] * n_dates}
        )

        # 构造完全相关的收益
        returns_data = []
        for date in dates:
            r = np.random.randn() * 0.02
            returns_data.append({"date": date, "order_book_id": "A", "label_return_10d": r, "market_benchmark_return": r})
            returns_data.append({"date": date, "order_book_id": "B", "label_return_10d": r, "market_benchmark_return": r})

        labels = pd.DataFrame(returns_data)
        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["beta_available"] is True
        assert metrics["portfolio"]["beta"] == pytest.approx(1.0, abs=0.01)

    def test_beta_low_volatility(self):
        """低波动组合 → β < 1"""
        import numpy as np

        np.random.seed(42)
        n_dates = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

        target_weights = pd.DataFrame(
            {"date": dates, "order_book_id": ["A"] * n_dates, "target_weight": [1.0] * n_dates}
        )

        # 组合波动率低于基准
        returns_data = []
        for date in dates:
            benchmark_r = np.random.randn() * 0.02
            portfolio_r = benchmark_r * 0.5  # 波动减半
            returns_data.append(
                {"date": date, "order_book_id": "A", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )
            returns_data.append(
                {"date": date, "order_book_id": "B", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )

        labels = pd.DataFrame(returns_data)
        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["beta_available"] is True
        assert metrics["portfolio"]["beta"] < 1.0

    def test_beta_high_volatility(self):
        """高波动组合 → β > 1"""
        import numpy as np

        np.random.seed(42)
        n_dates = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

        target_weights = pd.DataFrame(
            {"date": dates, "order_book_id": ["A"] * n_dates, "target_weight": [1.0] * n_dates}
        )

        # 组合波动率高于基准
        returns_data = []
        for date in dates:
            benchmark_r = np.random.randn() * 0.02
            portfolio_r = benchmark_r * 1.5  # 波动增加
            returns_data.append(
                {"date": date, "order_book_id": "A", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )
            returns_data.append(
                {"date": date, "order_book_id": "B", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )

        labels = pd.DataFrame(returns_data)
        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["beta_available"] is True
        assert metrics["portfolio"]["beta"] > 1.0

    def test_beta_insufficient_data(self):
        """数据不足 → 返回默认值 1.0"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "order_book_id": ["A", "A"],
                "target_weight": [1.0, 1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
                "order_book_id": ["A", "A", "B", "B"],
                "label_return_10d": [0.02, 0.01, 0.02, 0.01],
                "market_benchmark_return": [0.02, 0.01, 0.02, 0.01],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["beta_available"] is False
        assert metrics["portfolio"]["beta"] == 1.0


class TestJensenAlpha:
    """Jensen's Alpha 测试"""

    def test_jensen_alpha_outperformance(self):
        """组合跑赢 CAPM 预期 → α > 0"""
        import numpy as np

        np.random.seed(42)
        n_dates = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

        target_weights = pd.DataFrame(
            {"date": dates, "order_book_id": ["A"] * n_dates, "target_weight": [1.0] * n_dates}
        )

        # 构造正Alpha：组合收益高于CAPM预期
        returns_data = []
        for date in dates:
            benchmark_r = np.random.randn() * 0.02
            portfolio_r = benchmark_r * 0.8 + 0.001  # 稳定超额收益
            returns_data.append(
                {"date": date, "order_book_id": "A", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )
            returns_data.append(
                {"date": date, "order_book_id": "B", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )

        labels = pd.DataFrame(returns_data)
        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["jensen_alpha_available"] is True
        assert metrics["portfolio"]["jensen_alpha"] > 0

    def test_jensen_alpha_underperformance(self):
        """组合跑输 CAPM 预期 → α < 0"""
        import numpy as np

        np.random.seed(42)
        n_dates = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

        target_weights = pd.DataFrame(
            {"date": dates, "order_book_id": ["A"] * n_dates, "target_weight": [1.0] * n_dates}
        )

        # 构造负Alpha：组合收益低于CAPM预期
        returns_data = []
        for date in dates:
            benchmark_r = np.random.randn() * 0.02
            portfolio_r = benchmark_r * 0.8 - 0.001  # 稳定负超额
            returns_data.append(
                {"date": date, "order_book_id": "A", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )
            returns_data.append(
                {"date": date, "order_book_id": "B", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )

        labels = pd.DataFrame(returns_data)
        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["jensen_alpha_available"] is True
        assert metrics["portfolio"]["jensen_alpha"] < 0

    def test_jensen_alpha_with_custom_risk_free_rate(self):
        """自定义无风险利率的 Jensen's Alpha"""
        import numpy as np

        np.random.seed(42)
        n_dates = 20
        dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")

        target_weights = pd.DataFrame(
            {"date": dates, "order_book_id": ["A"] * n_dates, "target_weight": [1.0] * n_dates}
        )

        returns_data = []
        for date in dates:
            benchmark_r = np.random.randn() * 0.02
            portfolio_r = benchmark_r * 0.9 + 0.0005
            returns_data.append(
                {"date": date, "order_book_id": "A", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )
            returns_data.append(
                {"date": date, "order_book_id": "B", "label_return_10d": portfolio_r, "market_benchmark_return": benchmark_r}
            )

        labels = pd.DataFrame(returns_data)

        # 使用自定义无风险利率 (年化 3%)
        custom_rf = 0.03 / 252
        metrics = evaluate_portfolio_layer(target_weights, labels, risk_free_rate=custom_rf)

        assert metrics["portfolio"]["jensen_alpha_available"] is True
        # 验证Jensen's Alpha被计算
        assert "jensen_alpha" in metrics["portfolio"]


class TestActiveShare:
    """Active Share 测试"""

    def test_active_share_identical_portfolio(self):
        """组合权重 = 基准权重 → AS = 0"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "order_book_id": ["A", "B"],
                "target_weight": [0.6, 0.4],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "order_book_id": ["A", "B"],
                "label_return_10d": [0.02, 0.01],
                "market_benchmark_return": [0.015, 0.015],
            }
        )
        benchmark_weights = {
            "2024-01-01": {"A": 0.6, "B": 0.4},
        }

        metrics = evaluate_portfolio_layer(target_weights, labels, benchmark_weights_by_date=benchmark_weights)

        assert metrics["portfolio"]["active_share_available"] is True
        assert metrics["portfolio"]["active_share"] == pytest.approx(0.0, abs=1e-8)

    def test_active_share_fully_active(self):
        """完全不同的权重 → AS = 1"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "order_book_id": ["A"],
                "target_weight": [1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "order_book_id": ["A"],
                "label_return_10d": [0.02],
                "market_benchmark_return": [0.015],
            }
        )
        benchmark_weights = {
            "2024-01-01": {"B": 1.0},  # 基准持有 B，组合持有 A
        }

        metrics = evaluate_portfolio_layer(target_weights, labels, benchmark_weights_by_date=benchmark_weights)

        assert metrics["portfolio"]["active_share_available"] is True
        assert metrics["portfolio"]["active_share"] == pytest.approx(1.0, abs=1e-8)

    def test_active_share_partial_overlap(self):
        """部分重叠 → 0 < AS < 1"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "order_book_id": ["A", "B"],
                "target_weight": [0.6, 0.4],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "order_book_id": ["A", "B"],
                "label_return_10d": [0.02, 0.01],
                "market_benchmark_return": [0.015, 0.015],
            }
        )
        benchmark_weights = {
            "2024-01-01": {"A": 0.4, "B": 0.6},
        }

        metrics = evaluate_portfolio_layer(target_weights, labels, benchmark_weights_by_date=benchmark_weights)

        assert metrics["portfolio"]["active_share_available"] is True
        # AS = 0.5 * (|0.6-0.4| + |0.4-0.6|) = 0.5 * (0.2 + 0.2) = 0.2
        assert metrics["portfolio"]["active_share"] == pytest.approx(0.2, abs=1e-8)

    def test_active_share_no_benchmark_weights(self):
        """无基准权重数据 → 不可用"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "order_book_id": ["A"],
                "target_weight": [1.0],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"]),
                "order_book_id": ["A"],
                "label_return_10d": [0.02],
                "market_benchmark_return": [0.015],
            }
        )

        metrics = evaluate_portfolio_layer(target_weights, labels)

        assert metrics["portfolio"]["active_share_available"] is False
        assert metrics["portfolio"]["active_share"] == 0.0

    def test_active_share_multiple_dates(self):
        """多日平均 Active Share"""
        target_weights = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                "order_book_id": ["A", "B", "A", "B"],
                "target_weight": [0.7, 0.3, 0.5, 0.5],
            }
        )
        labels = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"]),
                "order_book_id": ["A", "B", "A", "B"],
                "label_return_10d": [0.02, 0.01, 0.03, 0.02],
                "market_benchmark_return": [0.015, 0.015, 0.02, 0.02],
            }
        )
        benchmark_weights = {
            "2024-01-01": {"A": 0.5, "B": 0.5},
            "2024-01-02": {"A": 0.5, "B": 0.5},
        }

        metrics = evaluate_portfolio_layer(target_weights, labels, benchmark_weights_by_date=benchmark_weights)

        assert metrics["portfolio"]["active_share_available"] is True
        # Day 1: AS = 0.5 * (|0.7-0.5| + |0.3-0.5|) = 0.2
        # Day 2: AS = 0.5 * (|0.5-0.5| + |0.5-0.5|) = 0.0
        # Mean: (0.2 + 0.0) / 2 = 0.1
        assert metrics["portfolio"]["active_share"] == pytest.approx(0.1, abs=1e-8)
