from pathlib import Path

import pytest

from skyeye.products.tx1.registry import (
    find_strategy_spec_by_file,
    get_strategy_spec,
    list_strategy_specs,
)


def test_tx1_rolling_score_strategy_is_registered():
    strategy_ids = {spec.strategy_id for spec in list_strategy_specs()}
    assert "tx1.rolling_score" in strategy_ids


def test_find_strategy_spec_by_entrypoint_resolves_tx1_strategy():
    strategy_path = Path(
        "/home/tiger/rqalpha/skyeye/products/tx1/strategies/rolling_score/strategy.py"
    )
    spec = find_strategy_spec_by_file(strategy_path)

    assert spec is not None
    assert spec.strategy_id == "tx1.rolling_score"
    assert spec.entrypoint_path == strategy_path


def test_get_strategy_spec_exposes_core_metadata():
    spec = get_strategy_spec("tx1.rolling_score")

    assert spec.benchmark == "000300.XSHG"
    assert spec.rolling_score_start_date is not None
    assert "artifact_line_id" in spec.raw


def test_tx1_spec_benchmark_is_frozen_at_strategy_level():
    spec = get_strategy_spec("tx1.rolling_score")

    assert spec.benchmark == "000300.XSHG"
