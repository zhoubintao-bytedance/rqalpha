import pytest

from skyeye.products.tx1.config import normalize_config


def test_normalize_config_fills_defaults():
    cfg = normalize_config({"model": {"kind": "linear"}})

    assert cfg["dataset"]["input_window"] == 60
    assert cfg["labels"]["horizon"] == 20
    assert cfg["splitter"]["train_years"] == 3
    assert cfg["splitter"]["val_months"] == 6
    assert cfg["splitter"]["test_months"] == 6
    assert cfg["splitter"]["embargo_days"] == 20
    assert cfg["portfolio"]["buy_top_k"] == 20
    assert cfg["portfolio"]["hold_top_k"] == 50
    assert cfg["portfolio"]["rebalance_interval"] == 20
    assert cfg["labels"]["transform"] == "rank"
    assert cfg["model"]["kind"] == "linear"


def test_normalize_config_rejects_non_frozen_research_knobs():
    with pytest.raises(ValueError):
        normalize_config({"dataset": {"input_window": 40}})

    with pytest.raises(ValueError):
        normalize_config({"labels": {"horizon": 10}})


def test_normalize_config_rejects_unknown_model_kind():
    with pytest.raises(ValueError):
        normalize_config({"model": {"kind": "unknown"}})
