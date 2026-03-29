from pathlib import Path

import pytest

from skyeye.products.registry import find_strategy_spec_by_file


def test_aggregated_registry_finds_dividend_low_vol_strategy():
    strategy_path = Path(
        "/home/tiger/rqalpha/skyeye/products/dividend_low_vol/strategies/history_aware/strategy.py"
    )
    spec = find_strategy_spec_by_file(strategy_path)

    assert spec is not None
    assert spec.strategy_id == "dividend_low_vol.history_aware"


def test_aggregated_registry_finds_tx1_strategy():
    strategy_path = Path(
        "/home/tiger/rqalpha/skyeye/products/tx1/strategies/rolling_score/strategy.py"
    )
    spec = find_strategy_spec_by_file(strategy_path)

    assert spec is not None
    assert spec.strategy_id == "tx1.rolling_score"


def test_aggregated_registry_returns_none_for_unknown_file():
    spec = find_strategy_spec_by_file("/nonexistent/strategy.py")
    assert spec is None
