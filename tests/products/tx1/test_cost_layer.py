# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from skyeye.products.tx1.cost_layer import (
    CONSERVATIVE_COST_CONFIG,
    DEFAULT_COST_CONFIG,
    STRESS_COST_CONFIG,
    CostConfig,
    apply_transaction_costs,
    compute_cost_metrics,
    estimate_annual_cost_drag,
)


@pytest.fixture
def portfolio_returns_df():
    dates = pd.bdate_range("2023-01-01", periods=20)
    return pd.DataFrame({
        "date": dates,
        "portfolio_return": np.random.default_rng(42).normal(0.001, 0.01, 20),
        "turnover": np.random.default_rng(42).uniform(0.0, 0.15, 20),
        "overlap": np.random.default_rng(42).uniform(0.7, 1.0, 20),
    })


class TestCostConfig:
    def test_default_values(self):
        cfg = CostConfig()
        assert cfg.commission_rate == 0.0008
        assert cfg.stamp_tax_rate == 0.0005
        assert cfg.slippage_bps == 5.0

    def test_slippage_rate(self):
        cfg = CostConfig(slippage_bps=10.0)
        assert cfg.slippage_rate == pytest.approx(0.001)

    def test_one_way_cost(self):
        cfg = CostConfig(commission_rate=0.001, slippage_bps=10.0)
        assert cfg.one_way_cost == pytest.approx(0.001 + 0.001)

    def test_round_trip_cost(self):
        cfg = CostConfig(commission_rate=0.001, stamp_tax_rate=0.0005, slippage_bps=0.0)
        # buy: 0.001, sell: 0.001 + 0.0005 = 0.0015
        assert cfg.round_trip_cost == pytest.approx(0.0025)

    def test_frozen(self):
        cfg = CostConfig()
        with pytest.raises(AttributeError):
            cfg.commission_rate = 0.01

    def test_preset_configs(self):
        assert CONSERVATIVE_COST_CONFIG.commission_rate > DEFAULT_COST_CONFIG.commission_rate
        assert STRESS_COST_CONFIG.slippage_bps > CONSERVATIVE_COST_CONFIG.slippage_bps


class TestApplyTransactionCosts:
    def test_apply_transaction_costs_uses_dailyized_portfolio_return_not_horizon_raw(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=1),
            "portfolio_return_horizon_raw": [0.20],
            "portfolio_return": [0.01],
            "turnover": [0.1],
            "overlap": [0.9],
        })

        result = apply_transaction_costs(df)

        expected_cost = 0.1 * DEFAULT_COST_CONFIG.round_trip_cost
        assert result.loc[0, "cost"] == pytest.approx(expected_cost)
        assert result.loc[0, "net_return"] == pytest.approx(0.01 - expected_cost)

    def test_adds_cost_and_net_return_columns(self, portfolio_returns_df):
        result = apply_transaction_costs(portfolio_returns_df)
        assert "cost" in result.columns
        assert "net_return" in result.columns
        assert len(result) == len(portfolio_returns_df)

    def test_net_return_less_than_gross(self, portfolio_returns_df):
        result = apply_transaction_costs(portfolio_returns_df)
        # Where turnover > 0, net_return should be < portfolio_return
        mask = result["turnover"] > 0
        assert (result.loc[mask, "net_return"] <= result.loc[mask, "portfolio_return"]).all()

    def test_zero_turnover_means_no_cost(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=3),
            "portfolio_return": [0.01, 0.02, -0.01],
            "turnover": [0.0, 0.0, 0.0],
            "overlap": [1.0, 1.0, 1.0],
        })
        result = apply_transaction_costs(df)
        assert (result["cost"] == 0.0).all()
        assert (result["net_return"] == result["portfolio_return"]).all()

    def test_custom_config(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=1),
            "portfolio_return": [0.01],
            "turnover": [0.1],
            "overlap": [0.9],
        })
        cfg = CostConfig(commission_rate=0.001, stamp_tax_rate=0.001, slippage_bps=10.0)
        result = apply_transaction_costs(df, cfg)
        expected_cost = 0.1 * cfg.round_trip_cost
        assert result["cost"].iloc[0] == pytest.approx(expected_cost)

    def test_empty_input(self):
        result = apply_transaction_costs(pd.DataFrame())
        assert len(result) == 0
        assert "cost" in result.columns

    def test_none_input(self):
        result = apply_transaction_costs(None)
        assert len(result) == 0


class TestEstimateAnnualCostDrag:
    def test_basic(self):
        drag = estimate_annual_cost_drag(0.05, DEFAULT_COST_CONFIG)
        expected = 0.05 * DEFAULT_COST_CONFIG.round_trip_cost * 252
        assert drag == pytest.approx(expected)

    def test_zero_turnover(self):
        assert estimate_annual_cost_drag(0.0) == 0.0

    def test_higher_cost_higher_drag(self):
        low = estimate_annual_cost_drag(0.05, DEFAULT_COST_CONFIG)
        high = estimate_annual_cost_drag(0.05, STRESS_COST_CONFIG)
        assert high > low


class TestComputeCostMetrics:
    def test_compute_cost_metrics_net_mean_return_uses_dailyized_portfolio_return(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=2),
            "portfolio_return_horizon_raw": [0.20, 0.10],
            "portfolio_return": [0.01, 0.005],
            "turnover": [0.1, 0.2],
            "overlap": [0.9, 0.8],
        })

        metrics = compute_cost_metrics(df)
        expected_costs = df["turnover"] * DEFAULT_COST_CONFIG.round_trip_cost
        expected_net_mean = float((df["portfolio_return"] - expected_costs).mean())
        assert metrics["net_mean_return"] == pytest.approx(expected_net_mean)

    def test_returns_all_keys(self, portfolio_returns_df):
        metrics = compute_cost_metrics(portfolio_returns_df)
        expected_keys = {
            "annual_turnover", "mean_turnover_per_period",
            "cost_drag_annual", "cost_erosion_ratio",
            "net_mean_return", "breakeven_cost_bps",
        }
        assert expected_keys == set(metrics.keys())

    def test_net_return_less_than_gross(self, portfolio_returns_df):
        metrics = compute_cost_metrics(portfolio_returns_df)
        gross = portfolio_returns_df["portfolio_return"].mean()
        assert metrics["net_mean_return"] <= gross

    def test_empty_input(self):
        metrics = compute_cost_metrics(pd.DataFrame())
        assert metrics["annual_turnover"] == 0.0
        assert metrics["net_mean_return"] == 0.0

    def test_conservative_vs_default(self, portfolio_returns_df):
        default = compute_cost_metrics(portfolio_returns_df, DEFAULT_COST_CONFIG)
        conservative = compute_cost_metrics(portfolio_returns_df, CONSERVATIVE_COST_CONFIG)
        assert conservative["cost_drag_annual"] > default["cost_drag_annual"]

    def test_breakeven_positive_when_gross_positive(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-01", periods=5),
            "portfolio_return": [0.01, 0.02, 0.005, 0.015, 0.01],
            "turnover": [0.05, 0.03, 0.02, 0.04, 0.01],
            "overlap": [0.9] * 5,
        })
        metrics = compute_cost_metrics(df)
        assert metrics["breakeven_cost_bps"] > 0
