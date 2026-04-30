import pandas as pd
import pytest

from skyeye.products.ax1.optimizer import ExecutablePortfolioOptimizer
from skyeye.products.tx1.cost_layer import CostConfig


def _targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["000001.XSHE", "000002.XSHE", "000003.XSHE"],
            "target_weight": [0.123, 0.103, 0.050],
            "price": [10.0, 20.0, 8.0],
        }
    )


def test_optimizer_rounds_orders_to_lots_and_keeps_weights_from_final_shares():
    result = ExecutablePortfolioOptimizer(
        portfolio_value=100_000,
        lot_size=100,
        min_trade_value=0,
        cost_config=CostConfig(commission_rate=0.001, stamp_tax_rate=0.002, slippage_bps=10.0),
    ).optimize(
        _targets(),
        current_shares={"000001.XSHE": 0, "000002.XSHE": 0, "000003.XSHE": 0},
    )

    portfolio = result.portfolio.set_index("order_book_id")
    orders = result.orders.set_index("order_book_id")

    assert orders["order_shares"].to_dict() == {
        "000001.XSHE": 1200,
        "000002.XSHE": 500,
        "000003.XSHE": 600,
    }
    assert portfolio.loc["000001.XSHE", "target_weight"] == pytest.approx(1200 * 10 / 100_000)
    assert portfolio.loc["000002.XSHE", "target_weight"] == pytest.approx(500 * 20 / 100_000)
    assert portfolio.loc["000003.XSHE", "target_weight"] == pytest.approx(600 * 8 / 100_000)
    assert portfolio["target_weight"].sum() == pytest.approx(0.268)
    assert orders.loc["000001.XSHE", "estimated_cost"] == pytest.approx(12000 * (0.001 + 0.001))
    assert result.summary["order_count"] == 3


def test_optimizer_skips_rounded_trade_below_min_trade_value_and_keeps_current_shares():
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "target_weight": [0.13, 0.20],
            "price": [10.0, 20.0],
        }
    )

    result = ExecutablePortfolioOptimizer(
        portfolio_value=100_000,
        lot_size=100,
        min_trade_value=5_000,
    ).optimize(targets, current_shares={"000001.XSHE": 1000, "000002.XSHE": 0})

    portfolio = result.portfolio.set_index("order_book_id")

    assert portfolio.loc["000001.XSHE", "target_shares"] == 1000
    assert portfolio.loc["000001.XSHE", "order_shares"] == 0
    assert portfolio.loc["000001.XSHE", "trade_reason"] == "below_min_trade_value"
    assert set(result.orders["order_book_id"]) == {"000002.XSHE"}


def test_optimizer_ignores_max_industry_weight_when_all_unknown():
    """When all instruments have industry='Unknown', max_industry_weight should be ignored.

    This prevents the constraint from becoming a global portfolio cap when industry data is missing.
    """
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["ETF_A", "ETF_B", "ETF_C"],
            "target_weight": [0.35, 0.30, 0.25],
            "price": [10.0, 20.0, 15.0],
            "industry": ["Unknown", "Unknown", "Unknown"],
        }
    )

    # With max_industry_weight=0.20, if applied it would cap portfolio at 20%
    result = ExecutablePortfolioOptimizer(
        portfolio_value=100_000,
        lot_size=100,
        min_trade_value=0,
        max_industry_weight=0.20,  # Should be ignored
    ).optimize(targets, current_shares={"ETF_A": 0, "ETF_B": 0, "ETF_C": 0})

    portfolio = result.portfolio.set_index("order_book_id")

    # Total weight should be ~0.90, not capped at 0.20
    assert portfolio["target_weight"].sum() > 0.80, "Portfolio should not be capped when all industry='Unknown'"

    # Individual weights should be allocated normally
    assert portfolio.loc["ETF_A", "target_weight"] > 0.30
    assert portfolio.loc["ETF_B", "target_weight"] > 0.25


def test_optimizer_keeps_largest_orders_when_max_order_count_is_exceeded():
    result = ExecutablePortfolioOptimizer(
        portfolio_value=100_000,
        lot_size=100,
        min_trade_value=0,
        max_order_count=2,
    ).optimize(
        _targets(),
        current_shares={"000001.XSHE": 0, "000002.XSHE": 0, "000003.XSHE": 0},
    )

    portfolio = result.portfolio.set_index("order_book_id")

    assert set(result.orders["order_book_id"]) == {"000001.XSHE", "000002.XSHE"}
    assert portfolio.loc["000003.XSHE", "target_shares"] == 0
    assert portfolio.loc["000003.XSHE", "trade_reason"] == "max_order_count"
    assert result.summary["order_count"] == 2


def test_optimizer_prioritizes_sells_when_max_order_count_is_exceeded():
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["SELL_A", "BUY_BIG", "BUY_SMALL"],
            "target_weight": [0.0, 0.50, 0.10],
            "price": [10.0, 100.0, 20.0],
        }
    )

    result = ExecutablePortfolioOptimizer(
        portfolio_value=100_000,
        lot_size=100,
        min_trade_value=0,
        max_order_count=1,
    ).optimize(targets, current_shares={"SELL_A": 1000, "BUY_BIG": 0, "BUY_SMALL": 0})

    portfolio = result.portfolio.set_index("order_book_id")

    assert list(result.orders["order_book_id"]) == ["SELL_A"]
    assert portfolio.loc["SELL_A", "order_shares"] == -1000
    assert portfolio.loc["BUY_BIG", "target_shares"] == 0
    assert portfolio.loc["BUY_BIG", "trade_reason"] == "max_order_count"


def test_optimizer_keeps_low_value_clearance_when_below_min_trade_value():
    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "order_book_id": ["000001.XSHE"],
            "target_weight": [0.0],
            "price": [10.0],
        }
    )

    result = ExecutablePortfolioOptimizer(
        portfolio_value=100_000,
        lot_size=100,
        min_trade_value=5_000,
    ).optimize(targets, current_shares={"000001.XSHE": 100})

    row = result.portfolio.iloc[0]

    assert row["target_shares"] == 100
    assert row["order_shares"] == 0
    assert row["target_weight"] == pytest.approx(0.01)
    assert row["trade_reason"] == "below_min_trade_value"
    assert result.orders.empty
