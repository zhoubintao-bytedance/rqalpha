import pandas as pd
import pytest

from skyeye.products.tx1.portfolio_proxy import PortfolioProxy


def _make_prediction_df(n_dates=30, n_stocks=50):
    """Create synthetic prediction data for testing."""
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    rows = []
    for date in dates:
        for i in range(n_stocks):
            rows.append({
                "date": date,
                "order_book_id": f"stock_{i:03d}",
                "prediction": float(n_stocks - i + (date.day % 5) * 0.1),
            })
    return pd.DataFrame(rows)


def test_first_day_always_rebalances():
    df = _make_prediction_df(n_dates=5, n_stocks=30)
    proxy = PortfolioProxy(buy_top_k=10, hold_top_k=15, rebalance_interval=20)

    result = proxy.build(df)

    # First day should have holdings
    first_date = result["date"].min()
    first_day = result[result["date"] == first_date]
    assert len(first_day) >= 10


def test_monthly_rebalance_holds_between_rebalances():
    df = _make_prediction_df(n_dates=25, n_stocks=30)
    proxy = PortfolioProxy(buy_top_k=10, hold_top_k=15, rebalance_interval=20)

    result = proxy.build(df)

    dates = sorted(result["date"].unique())
    # Holdings on day 2 should match day 1 (no rebalance yet)
    day1_stocks = set(result[result["date"] == dates[0]]["order_book_id"])
    day2_stocks = set(result[result["date"] == dates[1]]["order_book_id"])
    assert day1_stocks == day2_stocks


def test_rebalance_happens_at_interval():
    df = _make_prediction_df(n_dates=25, n_stocks=30)
    proxy = PortfolioProxy(buy_top_k=10, hold_top_k=15, rebalance_interval=5)

    result = proxy.build(df)

    dates = sorted(result["date"].unique())
    # Day 0: rebalance (first day)
    # Day 5: rebalance (interval=5)
    day0_stocks = set(result[result["date"] == dates[0]]["order_book_id"])
    day4_stocks = set(result[result["date"] == dates[4]]["order_book_id"])
    day5_stocks = set(result[result["date"] == dates[5]]["order_book_id"])
    # Between rebalances, holdings should be the same
    assert day0_stocks == day4_stocks
    # At rebalance, holdings may change (depends on predictions)
    assert len(day5_stocks) >= 10


def test_hold_buffer_retains_stocks():
    # Create data where stock rankings shift slightly
    dates = pd.bdate_range("2023-01-01", periods=3)
    rows = []
    # Day 1: stock_00 is rank 1, stock_09 is rank 10
    for i in range(20):
        rows.append({"date": dates[0], "order_book_id": f"stock_{i:03d}", "prediction": float(20 - i)})
    # Day 2 (rebalance): stock_09 drops to rank 12 (outside buy_top_k=10 but inside hold_top_k=15)
    for i in range(20):
        score = float(20 - i) if i != 9 else 8.5  # rank ~12
        rows.append({"date": dates[1], "order_book_id": f"stock_{i:03d}", "prediction": score})
    # Day 3 (same as day 2)
    for i in range(20):
        score = float(20 - i) if i != 9 else 8.5
        rows.append({"date": dates[2], "order_book_id": f"stock_{i:03d}", "prediction": score})

    df = pd.DataFrame(rows)
    proxy = PortfolioProxy(buy_top_k=10, hold_top_k=15, rebalance_interval=1)

    result = proxy.build(df)

    # stock_09 should be held on day 2 (within hold buffer)
    day2 = result[result["date"] == dates[1]]
    assert "stock_009" in set(day2["order_book_id"])


def test_equal_weight_on_active_holdings():
    df = _make_prediction_df(n_dates=3, n_stocks=30)
    proxy = PortfolioProxy(buy_top_k=10, hold_top_k=15, rebalance_interval=20)

    result = proxy.build(df)

    for _, day_df in result.groupby("date"):
        weights = day_df["weight"].values
        assert abs(weights.sum() - 1.0) < 1e-10
        assert all(abs(w - weights[0]) < 1e-10 for w in weights)


def test_empty_input_returns_empty():
    proxy = PortfolioProxy()
    result = proxy.build(pd.DataFrame(columns=["date", "order_book_id", "prediction"]))
    assert result.empty
