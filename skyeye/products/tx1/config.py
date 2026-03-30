# -*- coding: utf-8 -*-

from copy import deepcopy


DEFAULT_CONFIG = {
    "dataset": {
        "input_window": 60,
    },
    "labels": {
        "horizon": 20,
        "transform": "rank",
        "winsorize": None,
    },
    "splitter": {
        "train_years": 3,
        "val_months": 6,
        "test_months": 6,
        "embargo_days": 20,
    },
    "model": {
        "kind": "linear",
    },
    "portfolio": {
        "buy_top_k": 20,
        "hold_top_k": 50,
        "rebalance_interval": 20,
        "holding_bonus": 0.5,
    },
    "evaluation": {
        "top_k": 20,
    },
    "costs": {
        "enabled": True,
        "commission_rate": 0.0008,
        "stamp_tax_rate": 0.0005,
        "slippage_bps": 5.0,
    },
    "preprocessing": {
        "enabled": False,
        "neutralize": True,
        "winsorize_scale": 5.0,
        "standardize": True,
    },
    "robustness": {
        "enabled": True,
        "stability_metric": "rank_ic_mean",
    },
}

VALID_MODEL_KINDS = {"linear", "tree", "lgbm"}
VALID_LABEL_TRANSFORMS = {"raw", "rank", "quantile"}
FROZEN_INPUT_WINDOW = 60
FROZEN_HORIZON = 20


def _deep_update(base, override):
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def normalize_config(config=None):
    cfg = _deep_update(DEFAULT_CONFIG, config or {})
    model_kind = cfg["model"]["kind"]
    if model_kind not in VALID_MODEL_KINDS:
        raise ValueError("unsupported model kind: {}".format(model_kind))
    transform = cfg["labels"]["transform"]
    if transform not in VALID_LABEL_TRANSFORMS:
        raise ValueError("unsupported label transform: {}".format(transform))
    if cfg["dataset"]["input_window"] <= 0:
        raise ValueError("input_window must be positive")
    if cfg["labels"]["horizon"] <= 0:
        raise ValueError("horizon must be positive")
    if cfg["dataset"]["input_window"] != FROZEN_INPUT_WINDOW:
        raise ValueError("phase-1 freezes input_window at {}".format(FROZEN_INPUT_WINDOW))
    if cfg["labels"]["horizon"] != FROZEN_HORIZON:
        if not cfg["labels"].get("allow_horizon_override", False):
            raise ValueError("phase-1 freezes horizon at {}".format(FROZEN_HORIZON))
    if cfg["portfolio"]["buy_top_k"] <= 0 or cfg["portfolio"]["hold_top_k"] <= 0:
        raise ValueError("top-k values must be positive")
    if cfg["portfolio"]["hold_top_k"] < cfg["portfolio"]["buy_top_k"]:
        raise ValueError("hold_top_k must be >= buy_top_k")
    return cfg
