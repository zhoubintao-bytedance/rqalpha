import pytest

from skyeye.products.ax1.config import build_component_manifest, normalize_config


def _lgbm_config(**overrides):
    config = {
        "model": {
            "kind": "lgbm_multi_target",
            "feature_set": "ax1_unified_v1",
            "include_scopes": ["common", "etf_zscore", "regime"],
            "training_horizons": [5, 10, 20],
            "risk_horizon": 10,
        },
        "labels": {
            "return_horizons": [5, 10, 20],
            "volatility_horizons": [10],
            "relative_return": {
                "enabled": True,
                "group_columns": ["universe_layer"],
                "min_group_count": 2,
                "fallback": "date",
            },
        },
        "splitter": {
            "train_end": "2024-06-28",
            "val_months": 2,
            "test_months": 2,
            "embargo_days": 5,
        },
    }
    for key, value in overrides.items():
        config[key] = value
    return config


def test_empty_normalize_config_uses_lgbm_training_contract():
    config = normalize_config({})

    params = config["model"]["params"]
    assert config["model"]["kind"] == "lgbm_multi_target"
    assert config["model"]["include_scopes"] == ["common", "etf_zscore", "regime"]
    assert config["experiment"]["seed"] == 20260430
    assert params["n_estimators"] == 150
    assert params["num_leaves"] == 12
    assert params["max_depth"] == 4
    assert params["min_child_samples"] == 80
    assert params["reg_alpha"] > 0
    assert params["reg_lambda"] == pytest.approx(2.0)
    assert 0 < params["subsample"] <= 1
    assert 0 < params["colsample_bytree"] <= 1
    assert params["seed"] == config["experiment"]["seed"]
    assert params["deterministic"] is True
    assert config["labels"]["volatility_horizons"] == [10]
    assert config["labels"]["entry_lag_days"] == 1
    assert "demean_cross_section" not in config["labels"]
    assert config["labels"]["relative_return"] == {
        "enabled": True,
        "group_columns": ["universe_layer"],
        "min_group_count": 2,
        "fallback": "date",
    }
    assert config["execution"]["execution_lag_days"] == 1
    assert config["splitter"]["kind"] == "walk_forward"
    assert config["splitter"]["train_end"]
    assert "return_scale_by_horizon" not in config["features"]
    assert "risk_confidence_scale" not in config["features"]
    assert config["view_fusion"]["kind"] == "noop_adjusted_return"
    assert config["view_fusion"]["enabled"] is False
    assert config["view_fusion"]["return_column"] == "expected_relative_net_return_10d"
    assert config["risk_model"]["kind"] == "statistical_factor"
    assert config["risk_model"]["lookback_days"] == 120
    assert config["risk_model"]["n_factors"] == 6
    assert config["risk_model"]["shrinkage"] == pytest.approx(0.30)
    assert config["allocation"]["kind"] == "opportunity_pool_optimizer"
    assert "layers" not in config["allocation"]
    assert config["allocation"]["allow_gross_underfill"] is True
    assert config["allocation"]["min_allocatable_score"] == pytest.approx(0.0)
    assert config["allocation"]["cash_fallback"]["enabled"] is True
    assert config["allocation"]["exposure_groups"]["sector"]["max_weight"] == pytest.approx(0.55)
    assert config["allocation"]["layer_exposure_groups"]["industry"] == "sector"
    assert config["allocation"]["execution_drift_buffer"] == pytest.approx(0.01)
    assert config["constraints"]["cash_buffer"] == pytest.approx(0.03)
    assert config["execution"]["max_order_count"] == 12
    assert config["execution"]["participation_rate"] == pytest.approx(0.05)
    assert config["data"]["price_adjustment"]["required"] is True
    assert config["data"]["price_adjustment"]["adjusted_price_column"] == "adjusted_close"
    assert config["model"]["param_policy"]["min_child_samples"]["default"] == 80
    assert config["model"]["param_policy"]["min_child_samples"]["candidates"] == [30, 50, 80]
    assert config["model"]["param_policy"]["learning_rate"]["candidates"] == [0.03, 0.05, 0.08]
    assert config["model"]["param_policy"]["reg_lambda"]["default"] == pytest.approx(2.0)
    assert config["model"]["param_policy"]["reg_lambda"]["candidates"] == [0.2, 0.5, 1.0, 2.0]


def test_lgbm_config_requires_unified_feature_set():
    config = _lgbm_config(model={"kind": "lgbm_multi_target", "feature_set": "", "include_scopes": ["common"]})

    with pytest.raises(ValueError, match="feature_set"):
        normalize_config(config)


def test_lgbm_config_requires_training_horizons_inside_return_horizons():
    config = _lgbm_config(
        model={
            "kind": "lgbm_multi_target",
            "feature_set": "ax1_unified_v1",
            "include_scopes": ["common"],
            "training_horizons": [5, 30],
            "risk_horizon": 10,
        }
    )

    with pytest.raises(ValueError, match="training_horizons"):
        normalize_config(config)


def test_lgbm_training_horizons_must_match_bundle_contract():
    config = _lgbm_config(
        model={
            "kind": "lgbm_multi_target",
            "feature_set": "ax1_unified_v1",
            "include_scopes": ["common"],
            "training_horizons": [5, 10],
            "risk_horizon": 10,
        }
    )

    with pytest.raises(ValueError, match="training_horizons.*5.*10.*20"):
        normalize_config(config)


def test_lgbm_config_requires_risk_horizon_volatility_label():
    config = _lgbm_config(labels={"volatility_horizons": [5]})

    with pytest.raises(ValueError, match="risk_horizon"):
        normalize_config(config)


def test_lgbm_config_requires_splitter_train_end():
    config = _lgbm_config(splitter={"kind": "single_split", "train_end": None})

    with pytest.raises(ValueError, match="splitter.train_end"):
        normalize_config(config)


def test_lgbm_walk_forward_requires_at_least_six_folds():
    config = _lgbm_config(
        splitter={
            "kind": "walk_forward",
            "train_end": "2024-03-29",
            "val_months": 1,
            "test_months": 1,
            "embargo_days": 5,
            "n_folds": 3,
            "step_months": 1,
        }
    )

    with pytest.raises(ValueError, match="walk_forward.*n_folds.*6"):
        normalize_config(config)


def test_lgbm_config_rejects_overfit_prone_params_and_missing_seed():
    config = _lgbm_config(
        model={
            "kind": "lgbm_multi_target",
            "feature_set": "ax1_unified_v1",
            "include_scopes": ["common"],
            "training_horizons": [5, 10, 20],
            "risk_horizon": 10,
            "params": {
                "n_estimators": 30,
                "num_leaves": 7,
                "learning_rate": 0.05,
                "min_child_samples": 5,
                "early_stopping_rounds": 5,
                "num_threads": 1,
                "verbose": -1,
            },
        }
    )

    with pytest.raises(ValueError, match="min_child_samples.*30|reg_alpha|experiment.seed"):
        normalize_config(config)


def test_manifest_marks_lgbm_training_as_implemented_for_lgbm_kind():
    config = normalize_config(
        _lgbm_config(
            splitter={
                "kind": "single_split",
                "train_end": "2024-06-28",
                "val_months": 2,
                "test_months": 2,
                "embargo_days": 5,
            }
        )
    )

    manifest = build_component_manifest(config)

    assert manifest["preprocessor"]["kind"] == "feature_preprocessor"
    assert manifest["splitter"]["kind"] == "single_split"
    assert manifest["implementation_status"]["model"] == "lgbm_multi_target"
    assert manifest["feature_schema"]["feature_set"] == "ax1_unified_v1"
    assert manifest["model_schema"]["include_scopes"] == ["common", "etf_zscore", "regime"]
    assert manifest["implementation_status"]["lightgbm_training"] != "not_implemented"
    assert manifest["implementation_status"]["labels"] == "etf_peer_relative_net_return_training_labels"
    assert manifest["implementation_status"]["risk_model"] == "statistical_factor_pca_covariance_penalty"
    assert manifest["implementation_status"]["factor_risk_model"] == "implemented_statistical_factor_pca"
    assert manifest["implementation_status"]["tradable_outcome"] == "implemented_net_equity_cost_once_contract"
    assert manifest["implementation_status"]["alpha_transfer_ledger"] == "implemented"
    assert manifest["implementation_status"]["confidence_diagnostic"] == "implemented_post_replay_tradable_net_success"
    assert manifest["implementation_status"]["promotion_gate_contract"] == "tradability_gate_plus_research_support_gate"
    assert manifest["implementation_status"]["black_litterman_posterior"] == "not_implemented"
    assert manifest["implementation_status"]["execution_capacity"] == "implemented_enabled"
    assert "rule_return_scale" not in manifest["implementation_status"]
    assert "rule_confidence" not in manifest["implementation_status"]


def test_rule_model_kind_is_rejected_from_ax1_main_runner_contract():
    with pytest.raises(ValueError, match="unsupported AX1 model kind.*rule_multi_target"):
        normalize_config(
            {
                "model": {
                    "kind": "rule_multi_target",
                    "feature_set": "ax1_unified_v1",
                    "include_scopes": ["common", "etf_zscore"],
                },
            }
        )


def test_manifest_marks_lgbm_walk_forward_as_default_training_path():
    config = normalize_config(
        _lgbm_config(
            splitter={
                "kind": "walk_forward",
                "train_end": "2024-03-29",
                "val_months": 1,
                "test_months": 1,
                "embargo_days": 5,
                "n_folds": 6,
                "step_months": 1,
            }
        )
    )

    manifest = build_component_manifest(config)

    assert manifest["splitter"]["kind"] == "walk_forward"
    assert manifest["implementation_status"]["splitter"] == "walk_forward_train_val_test"
    assert manifest["implementation_status"]["lightgbm_training"] == "walk_forward_multi_head_lgbm"


def test_lgbm_config_accepts_model_feature_scopes_for_regime_interactions():
    config = normalize_config(
        _lgbm_config(
            model={
                "kind": "lgbm_multi_target",
                "feature_set": "ax1_unified_v1",
                "include_scopes": ["common", "etf_zscore", "regime", "regime_interaction"],
                "training_horizons": [5, 10, 20],
                "risk_horizon": 10,
            },
            features={"include_scopes": ["common", "etf_zscore", "regime", "regime_interaction"]},
        )
    )

    assert config["model"]["include_scopes"] == ["common", "etf_zscore", "regime", "regime_interaction"]
