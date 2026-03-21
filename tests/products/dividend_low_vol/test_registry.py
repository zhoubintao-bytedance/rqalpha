from pathlib import Path

from skyeye.products.dividend_low_vol.registry import find_strategy_spec_by_file
from skyeye.products.dividend_low_vol.registry import get_strategy_spec
from skyeye.products.dividend_low_vol.registry import list_strategy_specs


def test_history_aware_strategy_is_registered():
    strategy_ids = {spec.strategy_id for spec in list_strategy_specs()}
    assert "dividend_low_vol.history_aware" in strategy_ids


def test_find_strategy_spec_by_entrypoint_resolves_current_strategy():
    strategy_path = Path(
        "/home/tiger/rqalpha/skyeye/products/dividend_low_vol/strategies/history_aware/strategy.py"
    )
    spec = find_strategy_spec_by_file(strategy_path)

    assert spec is not None
    assert spec.strategy_id == "dividend_low_vol.history_aware"
    assert spec.entrypoint_path == strategy_path


def test_get_strategy_spec_exposes_core_metadata():
    spec = get_strategy_spec("dividend_low_vol.history_aware")

    assert spec.name == "红利低波历史感知策略"
    assert spec.benchmark == "512890.XSHG"
    assert spec.raw["rebalance_frequency"] == "weekly"
    assert "heat_override" in spec.raw["risk_controls"]
