import pandas as pd
import pytest

from skyeye.products.ax1.tradability import (
    build_alpha_transfer_ledger,
    build_tradable_outcome,
)


def test_tradable_outcome_uses_execution_cost_once_for_net_curve():
    weights = pd.DataFrame(
        [
            {"date": "2024-01-02", "order_book_id": "510300.XSHG", "target_weight": 1.0},
            {"date": "2024-01-03", "order_book_id": "510300.XSHG", "target_weight": 1.0},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "order_book_id": "510300.XSHG",
                "label_return_10d": 0.010,
                "label_net_return_10d": 0.008,
            },
            {
                "date": "2024-01-03",
                "order_book_id": "510300.XSHG",
                "label_return_10d": -0.020,
                "label_net_return_10d": -0.022,
            },
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "order_book_id": "510300.XSHG",
                "side": "buy",
                "order_value": 1_000_000.0,
                "estimated_cost": 1000.0,
            }
        ]
    )

    outcome = build_tradable_outcome(
        target_weights=weights,
        labels=labels,
        orders=orders,
        portfolio_value=1_000_000.0,
        gross_label_column="label_return_10d",
    )

    assert outcome["schema_version"] == 1
    assert outcome["return_column"] == "label_return_10d"
    assert outcome["net_return_by_date"]["2024-01-02"] == pytest.approx(0.009)
    assert outcome["net_return_by_date"]["2024-01-03"] == pytest.approx(-0.020)
    assert outcome["net_equity_curve"][-1]["equity"] == pytest.approx(1.009 * 0.98)
    assert outcome["max_net_drawdown"] > 0.0


def test_alpha_transfer_ledger_reports_retention_and_blockers():
    predictions = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "order_book_id": "A",
                "expected_relative_net_return_10d": 0.020,
                "confidence": 0.50,
            },
            {
                "date": "2024-01-02",
                "order_book_id": "B",
                "expected_relative_net_return_10d": 0.010,
                "confidence": 1.00,
            },
        ]
    )
    target = pd.DataFrame(
        [
            {"date": "2024-01-02", "order_book_id": "A", "target_weight": 0.60},
            {"date": "2024-01-02", "order_book_id": "B", "target_weight": 0.40},
        ]
    )
    executable = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "order_book_id": "A",
                "target_weight": 0.30,
                "trade_reason": "capacity",
            },
            {
                "date": "2024-01-02",
                "order_book_id": "B",
                "target_weight": 0.40,
                "trade_reason": "trade",
            },
        ]
    )
    outcome = {"net_return_by_date": {"2024-01-02": 0.006}}

    ledger = build_alpha_transfer_ledger(
        predictions=predictions,
        target_weights=target,
        executable_weights=executable,
        tradable_outcome=outcome,
        score_column="expected_relative_net_return_10d",
    )

    assert ledger["schema_version"] == 1
    assert ledger["summary"]["model_alpha_weighted"] == pytest.approx(0.02 + 0.01)
    assert ledger["summary"]["executable_alpha_weighted"] == pytest.approx(0.30 * 0.02 + 0.40 * 0.01)
    assert ledger["blocker_counts"]["capacity"] == 1
