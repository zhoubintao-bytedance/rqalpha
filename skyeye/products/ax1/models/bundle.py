"""AX1 predictor bundle helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

from skyeye.products.ax1.models.lgbm_multi_target import LGBMMultiTargetPredictor
from skyeye.products.tx1.baseline_models import dump_model_bundle, load_model_bundle


SCHEMA_VERSION = 1


def dump_predictor_bundle(
    predictor: LGBMMultiTargetPredictor,
    preprocessor_bundle: dict[str, Any] | None,
    feature_columns: Sequence[str] | None,
) -> dict[str, Any]:
    """Serialize an AX1 LGBM predictor plus AX1 rule-layer config."""
    if not isinstance(predictor, LGBMMultiTargetPredictor):
        raise TypeError("only LGBMMultiTargetPredictor is supported")
    if predictor._multi_head is None:
        raise RuntimeError("predictor must be fit before dump")

    resolved_feature_columns = list(feature_columns or predictor._trained_feature_columns)
    if not resolved_feature_columns:
        raise ValueError("feature_columns must not be empty")

    model_bundle = dump_model_bundle(
        predictor._multi_head,
        model_kind="lgbm",
        feature_columns=resolved_feature_columns,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "product": "ax1",
        "predictor_kind": "lgbm_multi_target",
        "feature_columns": resolved_feature_columns,
        "preprocessor_bundle": deepcopy(preprocessor_bundle or {}),
        "model_bundle": model_bundle,
        "rule_config": {
            "horizons": [int(item) for item in predictor.horizons],
            "training_horizons": [int(item) for item in predictor.training_horizons],
            "risk_horizon": int(predictor.risk_horizon),
            "stability_horizon": int(predictor.stability_horizon),
            "liquidity_column": str(predictor.liquidity_column),
            "confidence_method": str(predictor.confidence_method),
            "params": deepcopy(dict(predictor.params or {})),
            "configured_feature_columns": list(predictor.feature_columns or resolved_feature_columns),
        },
    }


def load_predictor_bundle(bundle: dict[str, Any]) -> LGBMMultiTargetPredictor:
    """Restore an AX1 LGBM predictor from `dump_predictor_bundle` output."""
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be a dict")
    if bundle.get("product") != "ax1":
        raise ValueError("bundle product must be ax1")
    if bundle.get("predictor_kind") != "lgbm_multi_target":
        raise ValueError("only lgbm_multi_target predictor bundles are supported")

    model_bundle = bundle.get("model_bundle")
    _validate_lgbm_multi_head_bundle(model_bundle)
    rule_config = dict(bundle.get("rule_config") or {})
    feature_columns = tuple(bundle.get("feature_columns") or model_bundle.get("feature_columns") or ())
    if not feature_columns:
        raise ValueError("bundle feature_columns must not be empty")

    configured_feature_columns = tuple(rule_config.get("configured_feature_columns") or feature_columns)
    predictor = LGBMMultiTargetPredictor(
        horizons=tuple(int(item) for item in rule_config.get("horizons", (5, 10, 20))),
        training_horizons=tuple(int(item) for item in rule_config.get("training_horizons", (5, 10, 20))),
        risk_horizon=int(rule_config.get("risk_horizon", 10)),
        stability_horizon=int(rule_config.get("stability_horizon", 20)),
        feature_columns=configured_feature_columns,
        liquidity_column=str(rule_config.get("liquidity_column", "dollar_volume")),
        params=deepcopy(rule_config.get("params") or {}),
        confidence_method=str(rule_config.get("confidence_method", "sign_consistency")),
    )
    predictor._multi_head = load_model_bundle(model_bundle)
    predictor._trained_feature_columns = feature_columns
    predictor._head_configs = deepcopy(model_bundle["state"].get("head_configs", {}))
    return predictor


def _validate_lgbm_multi_head_bundle(model_bundle: Any) -> None:
    if not isinstance(model_bundle, dict):
        raise ValueError("model_bundle must be a dict")
    if model_bundle.get("model_kind") != "multi_head":
        raise ValueError("model_bundle must be a multi_head bundle")
    state = model_bundle.get("state")
    if not isinstance(state, dict):
        raise ValueError("model_bundle.state must be a dict")
    if state.get("base_model_kind") != "lgbm":
        raise ValueError("model_bundle base_model_kind must be lgbm")
    heads = state.get("heads")
    if not isinstance(heads, dict):
        raise ValueError("model_bundle.state.heads must be a dict")
    missing_heads = {"relative_net_return_5d", "relative_net_return_10d", "relative_net_return_20d", "risk"} - set(heads)
    if missing_heads:
        raise ValueError(f"model_bundle missing required heads: {sorted(missing_heads)}")
