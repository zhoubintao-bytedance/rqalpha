# -*- coding: utf-8 -*-

from copy import deepcopy


DEFAULT_CONFIG = {
    "dataset": {
        "input_window": 60,
    },
    "features": None,
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
    "multi_output": {
        "enabled": False,
        "volatility": {
            "enabled": False,
            "transform": "rank",
        },
        "max_drawdown": {
            "enabled": False,
            "transform": "rank",
        },
        "prediction": {
            "combine_auxiliary": False,
            "volatility_weight": 0.0,
            "max_drawdown_weight": 0.0,
        },
        "reliability_score": {
            "enabled": False,
            "method": "head_agreement",
        },
    },
    "portfolio": {
        "buy_top_k": 25,
        "hold_top_k": 45,
        "rebalance_interval": 20,
        "holding_bonus": 0.5,
        "single_stock_cap": 0.10,
        "turnover_threshold": 0.30,
        "ema_halflife": 5,
        "ema_min_weight": 0.005,
        "stop_loss_pct": 0,
        "stop_loss_cooldown_days": 0,
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
VALID_AUXILIARY_TRANSFORMS = {"raw", "rank", "quantile", "log1p", "robust"}
VALID_RELIABILITY_METHODS = {"head_agreement"}
FROZEN_INPUT_WINDOW = 60
FROZEN_HORIZON = 20


def _normalize_features(features):
    """规范化显式特征列表，保留顺序并移除重复项。"""
    if features is None:
        return None
    if not isinstance(features, (list, tuple)):
        raise ValueError("features must be a list or tuple of feature names")

    normalized = []
    for feature in features:
        if not isinstance(feature, str):
            raise ValueError("feature names must be strings")
        name = feature.strip()
        if not name:
            raise ValueError("feature names must not be empty")
        if name not in normalized:
            normalized.append(name)

    if not normalized:
        raise ValueError("features must not be empty")
    return normalized


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
    cfg["features"] = _normalize_features(cfg.get("features"))
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
    multi_output = cfg["multi_output"]
    for target_name in ("volatility", "max_drawdown"):
        target_transform = multi_output[target_name]["transform"]
        if target_transform not in VALID_AUXILIARY_TRANSFORMS:
            raise ValueError("unsupported auxiliary transform for {}: {}".format(target_name, target_transform))
    reliability_cfg = multi_output.get("reliability_score", {})
    if reliability_cfg.get("method") not in VALID_RELIABILITY_METHODS:
        raise ValueError("unsupported reliability method: {}".format(reliability_cfg.get("method")))
    combine_cfg = multi_output.get("prediction", {})
    if combine_cfg.get("volatility_weight", 0.0) < 0 or combine_cfg.get("max_drawdown_weight", 0.0) < 0:
        raise ValueError("auxiliary prediction weights must be non-negative")

    any_aux_enabled = any(
        multi_output[target_name].get("enabled", False)
        for target_name in ("volatility", "max_drawdown")
    )
    if any_aux_enabled and not multi_output.get("enabled", False):
        raise ValueError("multi_output.enabled must be true when auxiliary heads are enabled")
    if combine_cfg.get("combine_auxiliary", False):
        if not multi_output.get("enabled", False):
            raise ValueError("multi_output.enabled must be true when combine_auxiliary is enabled")
        if not any_aux_enabled:
            raise ValueError("combine_auxiliary requires at least one auxiliary head")
        if combine_cfg.get("volatility_weight", 0.0) <= 0 and combine_cfg.get("max_drawdown_weight", 0.0) <= 0:
            raise ValueError("combine_auxiliary requires at least one positive auxiliary weight")
        if combine_cfg.get("volatility_weight", 0.0) > 0 and not multi_output["volatility"].get("enabled", False):
            raise ValueError("volatility_weight requires volatility head to be enabled")
        if combine_cfg.get("max_drawdown_weight", 0.0) > 0 and not multi_output["max_drawdown"].get("enabled", False):
            raise ValueError("max_drawdown_weight requires max_drawdown head to be enabled")
    if reliability_cfg.get("enabled", False):
        if not multi_output.get("enabled", False):
            raise ValueError("multi_output.enabled must be true when reliability_score is enabled")
        if not any_aux_enabled:
            raise ValueError("reliability_score requires at least one auxiliary head")
    return cfg
