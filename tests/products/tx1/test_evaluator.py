import pandas as pd
import pytest

from skyeye.products.tx1.evaluator import build_portfolio_returns, evaluate_predictions, evaluate_portfolios


def test_evaluator_returns_prediction_metrics():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 4,
            "order_book_id": ["a", "b", "c", "d"],
            "prediction": [0.9, 0.7, 0.2, 0.1],
            "label_return_raw": [0.08, 0.05, -0.01, -0.03],
        }
    )

    metrics = evaluate_predictions(frame, top_k=2)

    assert metrics["rank_ic_mean"] > 0
    assert metrics["top_bucket_spread_mean"] > 0
    assert 0.0 <= metrics["top_k_hit_rate"] <= 1.0


def test_build_portfolio_returns_keeps_raw_horizon_return_and_dailyizes_portfolio_return():
    test_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "order_book_id": ["a", "b"],
            "label_return_raw": [0.20, 0.10],
        }
    )
    weights_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "order_book_id": ["a", "b"],
            "weight": [0.6, 0.4],
        }
    )

    result = build_portfolio_returns(test_df, weights_df, horizon_days=20)

    assert "portfolio_return_horizon_raw" in result.columns
    assert result.loc[0, "portfolio_return_horizon_raw"] == pytest.approx(0.16)
    assert result.loc[0, "portfolio_return"] == pytest.approx(0.008)


def test_build_portfolio_returns_rejects_non_positive_horizon_days():
    test_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "order_book_id": ["a"],
            "label_return_raw": [0.20],
        }
    )
    weights_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "order_book_id": ["a"],
            "weight": [1.0],
        }
    )

    with pytest.raises(ValueError):
        build_portfolio_returns(test_df, weights_df, horizon_days=0)


def test_build_portfolio_returns_rejects_missing_labels():
    test_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "order_book_id": ["a"],
            "label_return_raw": [0.20],
        }
    )
    weights_df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 2,
            "order_book_id": ["a", "b"],
            "weight": [0.5, 0.5],
        }
    )

    with pytest.raises(ValueError):
        build_portfolio_returns(test_df, weights_df, horizon_days=20)


def test_evaluate_portfolios_uses_dailyized_portfolio_return_for_metrics():
    portfolio_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "portfolio_return_horizon_raw": [0.20, -0.10],
            "portfolio_return": [0.01, -0.005],
            "turnover": [0.1, 0.2],
            "overlap": [0.8, 0.7],
        }
    )

    metrics = evaluate_portfolios(portfolio_returns)

    assert metrics["mean_return"] == pytest.approx(0.0025)
    assert metrics["volatility"] == pytest.approx(portfolio_returns["portfolio_return"].std(ddof=0) * (252.0 ** 0.5))


def test_evaluator_returns_portfolio_metrics():
    portfolio_returns = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "portfolio_return": [0.01, -0.02, 0.03],
            "turnover": [1.0, 0.5, 0.0],
            "overlap": [0.0, 0.5, 1.0],
        }
    )

    metrics = evaluate_portfolios(portfolio_returns)

    assert "mean_return" in metrics
    assert "max_drawdown" in metrics
    assert metrics["mean_turnover"] >= 0.0
