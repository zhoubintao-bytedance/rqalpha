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
    assert cfg["portfolio"]["buy_top_k"] == 25
    assert cfg["portfolio"]["hold_top_k"] == 45
    assert cfg["portfolio"]["rebalance_interval"] == 20
    assert cfg["labels"]["transform"] == "rank"
    assert cfg["model"]["kind"] == "linear"
    assert not cfg["multi_output"]["enabled"]
    assert not cfg["multi_output"]["volatility"]["enabled"]
    assert not cfg["multi_output"]["max_drawdown"]["enabled"]


def test_normalize_config_rejects_non_frozen_research_knobs():
    with pytest.raises(ValueError):
        normalize_config({"dataset": {"input_window": 40}})

    with pytest.raises(ValueError):
        normalize_config({"labels": {"horizon": 10}})


def test_normalize_config_rejects_unknown_model_kind():
    with pytest.raises(ValueError):
        normalize_config({"model": {"kind": "unknown"}})


def test_normalize_config_accepts_multi_output_research_flags():
    cfg = normalize_config(
        {
            "model": {"kind": "linear"},
            "multi_output": {
                "enabled": True,
                "volatility": {"enabled": True, "transform": "log1p"},
                "max_drawdown": {"enabled": True, "transform": "robust"},
                "prediction": {
                    "combine_auxiliary": True,
                    "volatility_weight": 0.25,
                    "max_drawdown_weight": 0.1,
                },
                "reliability_score": {"enabled": True},
            },
        }
    )

    assert cfg["multi_output"]["volatility"]["transform"] == "log1p"
    assert cfg["multi_output"]["max_drawdown"]["transform"] == "robust"
    assert cfg["multi_output"]["prediction"]["combine_auxiliary"]
    assert cfg["multi_output"]["reliability_score"]["enabled"]


def test_normalize_config_rejects_auxiliary_flags_without_master_switch():
    with pytest.raises(ValueError):
        normalize_config(
            {
                "multi_output": {
                    "volatility": {"enabled": True},
                }
            }
        )


def test_normalize_config_rejects_invalid_multi_output_dependencies():
    with pytest.raises(ValueError):
        normalize_config(
            {
                "multi_output": {
                    "enabled": True,
                    "prediction": {
                        "combine_auxiliary": True,
                        "volatility_weight": 0.2,
                    },
                }
            }
        )

    with pytest.raises(ValueError):
        normalize_config(
            {
                "multi_output": {
                    "enabled": True,
                    "reliability_score": {"enabled": True},
                }
            }
        )
