# -*- coding: utf-8 -*-

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from skyeye.market_regime_layer import MarketRegimeConfig, required_market_regime_history_days
from skyeye.products.ax1.config import DEFAULT_PROFILE_PATH, load_profile, normalize_config


PERSONAL_PROFILE_PATH = Path("skyeye/products/ax1/profiles/personal_etf_core.yaml")


def test_default_profile_regime_warmup_matches_market_regime_required_history():
    config = load_profile(DEFAULT_PROFILE_PATH)
    # AX1 使用 rsrs_m=90 来缩短 regime 盲跑期
    expected = required_market_regime_history_days(MarketRegimeConfig(rsrs_m=90))

    assert expected == 113
    assert config["regime"]["lookback_days"] == expected


def _merge_config(*configs):
    result = {}
    for config in configs:
        for key, value in (config or {}).items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                result[key] = _merge_config(result[key], value)
            else:
                result[key] = deepcopy(value)
    return result


def _fast_lgbm_params(n_estimators: int = 3):
    return {
        "n_estimators": max(int(n_estimators), 20),
        "num_leaves": 7,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "early_stopping_rounds": 10,
        "num_threads": 1,
        "verbose": -1,
    }


def _fast_walk_forward_override():
    return {
        "model": {"params": _fast_lgbm_params()},
        "universe": {
            "validate_pit_universe": False,  # Disable for tests without data_provider
        },
    }


def _fast_single_split_override():
    return {
        "splitter": {
            "kind": "single_split",
            "train_end": "2024-03-29",
            "val_months": 1,
            "test_months": 1,
            "embargo_days": 5,
        },
        "model": {"params": _fast_lgbm_params()},
        "universe": {
            "validate_pit_universe": False,  # Disable for tests without data_provider
        },
    }


def _fast_single_split_config(*overrides):
    return normalize_config(_merge_config(_fast_single_split_override(), *overrides))


def _sample_raw_frame():
    rows = []
    dates = pd.date_range("2024-01-02", periods=30, freq="B")
    order_book_ids = [f"{asset_idx:06d}.XSHE" for asset_idx in range(1, 16)]
    for asset_idx, order_book_id in enumerate(order_book_ids):
        for day_idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "close": 10.0 + asset_idx + day_idx * (0.05 + asset_idx * 0.01),
                    "adjusted_close": 10.0 + asset_idx + day_idx * (0.05 + asset_idx * 0.01),
                    "volume": 1000000 + asset_idx * 10000,
                }
            )
    return pd.DataFrame(rows)


def _sample_lgbm_raw_frame():
    rows = []
    dates = pd.date_range("2024-01-02", periods=320, freq="B")
    assets = [
        ("510050.XSHG", "core", "sse50"),
        ("510300.XSHG", "core", "broad"),
        ("510500.XSHG", "core", "broad"),
        ("512100.XSHG", "core", "csi1000"),
        ("588000.XSHG", "core", "star50"),
        ("512800.XSHG", "industry", "bank"),
        ("512880.XSHG", "industry", "brokerage"),
        ("512000.XSHG", "industry", "brokerage"),
        ("515000.XSHG", "industry", "technology"),
        ("512480.XSHG", "industry", "semiconductor"),
        ("510880.XSHG", "style", "dividend"),
        ("515180.XSHG", "style", "low_vol"),
    ]
    for asset_idx, (order_book_id, layer, industry) in enumerate(assets):
        base = 10.0 + asset_idx * 0.7
        drift = 0.001 + asset_idx * 0.0002
        for day_idx, date in enumerate(dates):
            seasonal = ((day_idx % 17) - 8) * 0.006
            close = base * (1.0 + drift * day_idx + seasonal)
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "asset_type": "etf",
                    "universe_layer": layer,
                    "industry": industry,
                    "close": close,
                    "adjusted_close": close,
                    "volume": 800000 + asset_idx * 25000 + (day_idx % 11) * 5000,
                }
            )
    return pd.DataFrame(rows)


def _sample_fast_lgbm_raw_frame():
    frame = _sample_lgbm_raw_frame()
    keep_ids = [
        "510050.XSHG",
        "510300.XSHG",
        "510500.XSHG",
        "512800.XSHG",
        "512880.XSHG",
        "510880.XSHG",
    ]
    return frame[frame["order_book_id"].isin(keep_ids)].reset_index(drop=True)


def _sample_personal_etf_raw_frame():
    rows = []
    dates = pd.date_range("2024-01-02", periods=230, freq="B")
    assets = [
        ("510050.XSHG", "etf", "core", "sse50"),
        ("510300.XSHG", "etf", "core", "broad"),
        ("510500.XSHG", "etf", "core", "broad"),
        ("512100.XSHG", "etf", "core", "csi1000"),
        ("588000.XSHG", "etf", "core", "star50"),
        ("512800.XSHG", "etf", "industry", "bank"),
        ("512880.XSHG", "etf", "industry", "brokerage"),
        ("512000.XSHG", "etf", "industry", "brokerage"),
        ("515000.XSHG", "etf", "industry", "technology"),
        ("512480.XSHG", "etf", "industry", "semiconductor"),
        ("515790.XSHG", "etf", "industry", "pv"),
        ("159928.XSHE", "etf", "industry", "consumer"),
        ("510880.XSHG", "etf", "style", "dividend"),
        ("515180.XSHG", "etf", "style", "low_vol"),
        ("159915.XSHE", "etf", "style", "growth"),
        ("000001.XSHE", "stock", "stock", "bank"),
        ("000002.XSHE", "stock", "stock", "real_estate"),
    ]
    for asset_idx, (order_book_id, asset_type, layer, industry) in enumerate(assets):
        for day_idx, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "asset_type": asset_type,
                    "universe_layer": layer,
                    "industry": industry,
                    "listed_date": pd.Timestamp("2020-01-01"),
                    "is_st": False,
                    "is_suspended": False,
                    "close": 3.0 + asset_idx * 0.4 + day_idx * (0.01 + asset_idx * 0.001),
                    "adjusted_close": 3.0 + asset_idx * 0.4 + day_idx * (0.01 + asset_idx * 0.001),
                    "volume": 3_000_000 + asset_idx * 300_000,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def personal_pipeline_result(tmp_path_factory):
    from skyeye.products.ax1.run_experiment import run_experiment

    return run_experiment(
        profile_path=PERSONAL_PROFILE_PATH,
        output_dir=tmp_path_factory.mktemp("ax1_personal_pipeline") / "experiments",
        raw_df=_sample_personal_etf_raw_frame(),
        experiment_name="ax1_personal_etf_core_test",
        config_override=_fast_single_split_override(),
    )


@pytest.fixture(scope="module")
def lgbm_walk_forward_result(tmp_path_factory):
    from skyeye.products.ax1.run_experiment import run_experiment

    monkeypatch = pytest.MonkeyPatch()
    _install_fast_lgbm_pipeline(monkeypatch)
    try:
        return run_experiment(
            profile_path=Path("skyeye/products/ax1/profiles/lgbm_multi_target.yaml"),
            output_dir=tmp_path_factory.mktemp("ax1_lgbm_pipeline") / "experiments",
            raw_df=_sample_fast_lgbm_raw_frame(),
            experiment_name="ax1_lgbm_walk_forward_test",
            config_override=_fast_walk_forward_override(),
        )
    finally:
        monkeypatch.undo()


def _install_fast_lgbm_pipeline(monkeypatch):
    import skyeye.products.ax1.run_experiment as ax1_runner

    monkeypatch.setattr(ax1_runner, "_run_lgbm_pipeline", _fast_run_lgbm_pipeline)


def _fast_run_lgbm_pipeline(config, labeled, *, feature_columns):
    from skyeye.products.ax1.run_experiment import _build_splitter

    target_columns = [column for column in labeled.columns if column.startswith("label_")]
    folds = _build_splitter(config).split(labeled)
    if not folds:
        raise ValueError("splitter produced no valid folds")

    fold_results = []
    prediction_frames = []
    label_frames = []
    for fold in folds:
        fold_id = int(fold.get("fold_id", 0))
        frame = fold["test_df"].dropna(subset=["date", "order_book_id"]).copy()
        if target_columns:
            frame = frame.dropna(subset=target_columns)
        frame = frame.sort_values(["date", "order_book_id"]).reset_index(drop=True)
        predictions = _fake_predictions_for_frame(frame)
        predictions["fold_id"] = fold_id
        evaluation_labels = frame[["date", "order_book_id", *target_columns]].copy()
        evaluation_labels["fold_id"] = fold_id
        research_predictions = predictions.merge(
            evaluation_labels,
            on=["date", "order_book_id", "fold_id"],
            how="left",
        )
        fold_results.append(
            {
                "fold_id": fold_id,
                "train_rows": int(len(fold["train_df"])),
                "val_rows": int(len(fold["val_df"])),
                "test_rows": int(len(frame)),
                "predictions_df": research_predictions,
                "validation_metrics": {"rank_ic_mean": 0.10, "top_bucket_spread_mean": 0.01},
                "prediction_metrics": {"rank_ic_mean": 0.10, "top_bucket_spread_mean": 0.01},
                "confidence_calibration": {"fold_id": fold_id, "status": "calibrated"},
            }
        )
        prediction_frames.append(predictions)
        label_frames.append(evaluation_labels)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    evaluation_labels = pd.concat(label_frames, ignore_index=True)
    aggregate_predictions = predictions.merge(
        evaluation_labels,
        on=["date", "order_book_id", "fold_id"],
        how="left",
    )
    n_folds = len(fold_results)
    summary = {
        "model_kind": "lgbm_multi_target",
        "feature_columns": list(feature_columns),
        "target_columns": target_columns,
        "fold_results": fold_results,
        "aggregate_predictions_df": aggregate_predictions,
        "aggregate_predictions_row_count": int(len(predictions)),
        "aggregate_labels_row_count": int(len(evaluation_labels)),
        "aggregate_metrics": {
            "n_folds": n_folds,
            "prediction": {
                "rank_ic_mean_mean": 0.10,
                "top_bucket_spread_mean_mean": 0.01,
            },
        },
        "stability": {"metric_key": "top_bucket_spread_mean", "cv": 0.0, "stability_score": 100.0},
        "positive_ratio": {"metric_key": "top_bucket_spread_mean", "positive_ratio": 1.0, "n_folds": n_folds},
        "overfit_flags": {
            "flag_ic_decay": False,
            "flag_spread_decay": False,
            "flag_val_dominant": False,
            "n_folds_compared": n_folds,
        },
        "feature_importance": {"schema_version": 1, "feature_columns": list(feature_columns), "heads": {}, "aggregate": {}},
        "feature_diagnostics": {},
        "feature_conflicts": {"pairwise": [], "high_corr_groups": [], "inverse_corr_groups": []},
        "feature_review_summary": {"schema_version": 1, "feature_count": 0, "warning_count": 0, "warnings": []},
        "confidence_calibration": {
            "status": "calibrated",
            "fold_count": n_folds,
            "fallback_count": 0,
            "folds": [{"fold_id": item["fold_id"], "status": "calibrated"} for item in fold_results],
        },
    }
    return predictions.drop(columns=["fold_id"], errors="ignore"), evaluation_labels, summary


def _fake_predictions_for_frame(frame):
    per_day_rank = frame.groupby("date", sort=False).cumcount().astype(float)
    day_size = frame.groupby("date", sort=False)["order_book_id"].transform("count").astype(float).clip(lower=1.0)
    score = (per_day_rank + 1.0) / day_size
    predictions = frame[["date", "order_book_id"]].copy()
    predictions["expected_relative_net_return_5d"] = 0.001 + score * 0.002
    predictions["expected_relative_net_return_10d"] = 0.002 + score * 0.003
    predictions["expected_relative_net_return_20d"] = 0.003 + score * 0.004
    predictions["risk_forecast"] = 0.05 + (1.0 - score) * 0.01
    predictions["liquidity_score"] = 1.0
    predictions["cost_forecast"] = 0.0001
    predictions["confidence_raw"] = 0.80
    predictions["confidence"] = 0.80
    return predictions


def test_default_profile_preserves_ax1_architecture_decisions():
    config = load_profile(DEFAULT_PROFILE_PATH)

    assert config["product"] == "ax1"
    assert config["experiment"]["name"] == "ax1_personal_etf_core"
    assert config["labels"]["return_horizons"] == [5, 10, 20]
    assert config["labels"]["stability_horizon"] == 20
    assert config["constraints"]["max_position_count"] == 15
    assert config["constraints"]["min_position_count"] == 1
    assert config["constraints"]["max_turnover"] == pytest.approx(0.15)
    assert config["constraints"]["cash_buffer"] == pytest.approx(0.03)
    assert config["execution"]["min_trade_value"] == 3000
    assert config["execution"]["rebalance_days_min"] == 10
    assert config["execution"]["rebalance_days_max"] == 20
    assert config["execution"]["rebalance_interval"] == 10
    assert config["execution"]["no_trade_buffer_weight"] == pytest.approx(0.002)
    assert config["execution"]["net_alpha_threshold"] == pytest.approx(0.0)
    assert config["execution"]["net_alpha_column"] == "adjusted_expected_return"
    assert config["execution"]["t_plus_one_lock"] is True
    assert config["execution"]["today_buy_weight_column"] == "today_buy_weight"
    assert config["execution"]["participation_rate"] == pytest.approx(0.05)
    assert config["execution"]["liquidity_column"] == "dollar_volume"
    assert config["execution"]["lot_size"] == 100
    assert config["execution"]["max_order_count"] == 12
    assert config["execution"]["price_column"] == "close"
    assert config["regime"]["enabled"] is True
    assert config["regime"]["preferred_benchmark_ids"][0] == "510300.XSHG"
    assert config["features"]["feature_set"] == "ax1_unified_v1"
    assert config["features"]["normalization"]["cross_sectional"] is False
    assert config["model"]["feature_set"] == "ax1_unified_v1"
    assert config["model"]["kind"] == "lgbm_multi_target"
    assert config["model"]["include_scopes"] == ["common", "etf_zscore", "regime"]
    assert config["labels"]["volatility_horizons"] == [10]
    assert config["labels"]["winsorize_quantiles"] == [0.01, 0.99]
    assert config["view_fusion"]["kind"] == "noop_adjusted_return"
    assert config["view_fusion"]["enabled"] is False
    assert config["risk_model"]["kind"] == "statistical_factor"
    assert config["risk_model"]["lookback_days"] == 120
    assert config["risk_model"]["n_factors"] == 6
    assert config["risk_model"]["shrinkage"] == pytest.approx(0.30)
    assert config["allocation"]["kind"] == "opportunity_pool_optimizer"
    assert "layers" not in config["allocation"]
    assert config["allocation"]["allow_gross_underfill"] is True
    assert config["allocation"]["cash_fallback"]["enabled"] is True
    assert config["allocation"]["exposure_groups"]["broad_beta"]["max_weight"] == pytest.approx(0.65)
    assert config["allocation"]["layer_exposure_groups"]["core"] == "broad_beta"
    assert config["allocation"]["execution_drift_buffer"] == pytest.approx(0.01)
    assert config["optimizer"]["kind"] == "opportunity_pool_optimizer"
    assert config["preprocessor"]["kind"] == "feature_preprocessor"
    assert config["splitter"]["kind"] == "walk_forward"
    assert config["splitter"]["train_end"] == "auto"
    assert config["splitter"]["n_folds"] == 6


def test_opportunity_pool_config_carries_exposure_contract_without_layer_budgets():
    from skyeye.products.ax1.run_experiment import _opportunity_pool_config

    config = {
        "allocation": {
            "kind": "opportunity_pool_optimizer",
            "score_column": "expected_relative_net_return_10d",
            "allow_gross_underfill": True,
            "min_allocatable_score": 0.0,
            "layer_exposure_groups": {"core": "broad_beta", "industry": "sector"},
            "exposure_groups": {
                "broad_beta": {"max_weight": 0.65, "score_multiplier": 1.0},
                "sector": {"max_weight": 0.55, "score_multiplier": 1.0},
            },
            "cash_fallback": {"enabled": True},
            "execution_drift_buffer": 0.01,
        },
        "constraints": {"cash_buffer": 0.03, "target_gross_exposure": 1.0},
    }

    result = _opportunity_pool_config(
        config,
        {"risk_state": "risk_off", "rotation_state": "rotation", "strength": 0.50},
    )

    assert result["kind"] == "opportunity_pool_optimizer"
    assert result["cash_buffer"] == pytest.approx(0.04)
    assert result["configured_cash_buffer"] == pytest.approx(0.03)
    assert result["target_gross_exposure"] == pytest.approx(1.0)
    assert result["layer_exposure_groups"]["industry"] == "sector"
    assert result["regime_state"] == {
        "risk_state": "risk_off",
        "rotation_state": "rotation",
        "strength": pytest.approx(0.50),
    }
    assert "layers" not in result
    assert not any(str(key).endswith("_budget") for key in result)


def test_opportunity_pool_config_by_date_does_not_use_future_regime():
    from skyeye.products.ax1.run_experiment import _opportunity_pool_config_by_date

    config = {
        "allocation": {
            "kind": "opportunity_pool_optimizer",
            "score_column": "expected_relative_net_return_10d",
            "allow_gross_underfill": True,
            "min_allocatable_score": 0.0,
            "layer_exposure_groups": {"core": "broad_beta", "industry": "sector"},
            "exposure_groups": {
                "broad_beta": {"max_weight": 0.65, "score_multiplier": 1.0},
                "sector": {"max_weight": 0.55, "score_multiplier": 1.0},
            },
            "cash_fallback": {"enabled": True},
            "execution_drift_buffer": 0.0,
        },
        "constraints": {"cash_buffer": 0.0, "target_gross_exposure": 1.0},
    }
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    target_date = pd.Timestamp("2024-01-03")
    states = {
        pd.Timestamp("2024-01-02"): {"risk_state": "neutral", "rotation_state": "co_move", "strength": 0.0},
        target_date: {"risk_state": "risk_off", "rotation_state": "co_move", "strength": 0.5},
        pd.Timestamp("2024-01-04"): {"risk_state": "risk_on", "rotation_state": "co_move", "strength": 1.0},
    }
    future_changed = dict(states)
    future_changed[pd.Timestamp("2024-01-04")] = {
        "risk_state": "risk_off",
        "rotation_state": "co_move",
        "strength": 1.0,
    }

    baseline = _opportunity_pool_config_by_date(config, dates=dates, regime_state_by_date=states)
    changed = _opportunity_pool_config_by_date(config, dates=dates, regime_state_by_date=future_changed)

    assert baseline[target_date]["regime_state"] == changed[target_date]["regime_state"]
    assert baseline[pd.Timestamp("2024-01-04")]["regime_state"] != changed[pd.Timestamp("2024-01-04")]["regime_state"]


def test_risk_model_by_date_does_not_use_future_prices():
    from skyeye.products.ax1.run_experiment import _fit_risk_model_by_date

    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    stable_rows = []
    for order_book_id, base in [("A", 100.0), ("B", 80.0)]:
        for step, date in enumerate(dates[:5]):
            stable_rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "close": base * (1.0 + step * 0.01),
                }
            )
    future_rows = []
    for date, a_close, b_close in zip(dates[5:], [300.0, 50.0, 400.0], [85.0, 85.5, 86.0]):
        future_rows.extend(
            [
                {"date": date, "order_book_id": "A", "close": a_close},
                {"date": date, "order_book_id": "B", "close": b_close},
            ]
        )
    stable = pd.DataFrame(stable_rows)
    expanded = pd.concat([stable, pd.DataFrame(future_rows)], ignore_index=True)
    target_date = pd.Timestamp(dates[4])
    risk_cfg = {"lookback_days": 4, "shrinkage": 0.0}

    stable_models = _fit_risk_model_by_date(stable, risk_cfg)
    expanded_models = _fit_risk_model_by_date(expanded, risk_cfg)
    stable_cov = stable_models[target_date].get_covariance_matrix()
    expanded_target_cov = expanded_models[target_date].get_covariance_matrix()
    expanded_final_cov = expanded_models[pd.Timestamp(dates[-1])].get_covariance_matrix()

    pd.testing.assert_frame_equal(stable_cov, expanded_target_cov)
    assert expanded_final_cov.loc["A", "A"] > expanded_target_cov.loc["A", "A"]


def test_personal_etf_core_profile_declares_etf_first_universe_and_costs():
    config = load_profile(PERSONAL_PROFILE_PATH)

    universe = config["universe"]
    assert universe["etf_first"] is True
    assert "core_etfs" not in universe
    assert "industry_etfs" not in universe
    assert "style_etfs" not in universe
    assert "stock_satellite" not in universe
    assert len(universe["layers"]["core"]["include"]) >= 5
    assert len(universe["layers"]["industry"]["include"]) >= 20
    assert len(universe["layers"]["style"]["include"]) >= 5
    assert universe["layers"]["stock_satellite"]["enabled"] is False
    assert config["constraints"]["target_gross_exposure"] == pytest.approx(1.0)
    assert config["constraints"]["cash_buffer"] == pytest.approx(0.03)
    assert config["allocation"]["execution_drift_buffer"] == pytest.approx(0.01)
    assert config["allocation"]["layer_exposure_groups"]["core"] == "broad_beta"
    assert config["allocation"]["exposure_groups"]["sector"]["max_weight"] == pytest.approx(0.55)
    assert config["costs"]["stock"]["stamp_tax_rate"] > config["costs"]["etf"]["stamp_tax_rate"]
    assert config["model"]["kind"] == "lgbm_multi_target"
    assert config["model"]["include_scopes"] == ["common", "etf_zscore", "regime"]


def test_ax1_profiles_expand_corrected_etf_pool_without_legacy_code_mismatches():
    profile_paths = [
        DEFAULT_PROFILE_PATH,
        PERSONAL_PROFILE_PATH,
        Path("skyeye/products/ax1/profiles/lgbm_multi_target.yaml"),
    ]
    required_industry = {
        "512070.XSHG",
        "515260.XSHG",
        "515880.XSHG",
        "512760.XSHG",
        "512720.XSHG",
        "515210.XSHG",
        "159865.XSHE",
        "159611.XSHE",
        "516110.XSHG",
        "562500.XSHG",
        "159639.XSHE",
        "159870.XSHE",
    }
    required_style = {
        "513050.XSHG",
        "513130.XSHG",
        "513100.XSHG",
        "513500.XSHG",
        "518880.XSHG",
        "159985.XSHE",
        "511010.XSHG",
        "511260.XSHG",
        "511220.XSHG",
    }

    for path in profile_paths:
        layers = load_profile(path)["universe"]["layers"]
        industry_ids = set(layers["industry"]["include"])
        style_ids = set(layers["style"]["include"])

        assert required_industry.issubset(industry_ids), path
        assert required_style.issubset(style_ids), path
        assert "516970.XSHG" in industry_ids
        assert "159766.XSHE" not in industry_ids
        assert "513060.XSHG" not in style_ids
        assert "159819.XSHE" in industry_ids


def test_ax1_profiles_share_training_timing_and_lgbm_contract():
    profile_paths = [
        DEFAULT_PROFILE_PATH,
        PERSONAL_PROFILE_PATH,
        Path("skyeye/products/ax1/profiles/lgbm_multi_target.yaml"),
    ]
    configs = [normalize_config(load_profile(path)) for path in profile_paths]

    signatures = [
            {
                "entry_lag_days": config["labels"]["entry_lag_days"],
                "execution_lag_days": config["execution"]["execution_lag_days"],
                "relative_return": config["labels"]["relative_return"],
                "view_return_column": config["view_fusion"]["return_column"],
                "winsorize_quantiles": config["labels"]["winsorize_quantiles"],
                "embargo_days": config["splitter"]["embargo_days"],
            "risk_lookback_days": config["risk_model"]["lookback_days"],
            "risk_n_factors": config["risk_model"]["n_factors"],
            "risk_shrinkage": config["risk_model"]["shrinkage"],
            "allocation_kind": config["allocation"]["kind"],
            "layer_exposure_groups": config["allocation"]["layer_exposure_groups"],
            "exposure_groups": config["allocation"]["exposure_groups"],
            "execution_drift_buffer": config["allocation"]["execution_drift_buffer"],
            "seed": config["experiment"]["seed"],
            "min_child_samples": config["model"]["params"]["min_child_samples"],
            "reg_alpha": config["model"]["params"]["reg_alpha"],
            "reg_lambda": config["model"]["params"]["reg_lambda"],
            "subsample": config["model"]["params"]["subsample"],
            "colsample_bytree": config["model"]["params"]["colsample_bytree"],
        }
        for config in configs
    ]

    assert signatures[0] == signatures[1] == signatures[2]
    assert configs[0]["constraints"]["cash_buffer"] == pytest.approx(0.03)
    assert configs[1]["constraints"]["cash_buffer"] == pytest.approx(0.03)
    assert configs[2]["constraints"]["cash_buffer"] == pytest.approx(0.02)


def test_config_rejects_invalid_executable_execution_settings():
    invalid_overrides = [
        ({"execution": {"rebalance_interval": 0}}, "execution.rebalance_interval"),
        ({"execution": {"no_trade_buffer_weight": -0.001}}, "no_trade_buffer_weight"),
        ({"execution": {"net_alpha_threshold": -0.001}}, "net_alpha_threshold"),
        ({"execution": {"lot_size": 0}}, "execution.lot_size"),
        ({"execution": {"max_order_count": 0}}, "execution.max_order_count"),
        ({"execution": {"price_column": ""}}, "execution.price_column"),
    ]
    for override, message in invalid_overrides:
        with pytest.raises(ValueError, match=message):
            normalize_config(override)


def test_config_rejects_aggressive_winsorize_quantiles():
    """小截面 ETF 下 winsorize 过猛应被拒绝。

    50 只 ETF 截面上 [0.02, 0.98] 只保留第 2~49 名，
    默认值已改为 [0.01, 0.99]（≈ 不裁剪），并禁止 lower>0.02 或 upper<0.98。
    """
    # 默认值应为 [0.01, 0.99]
    config = normalize_config({})
    assert config["labels"]["winsorize_quantiles"] == [0.01, 0.99]

    # 过激裁剪应被拒绝
    with pytest.raises(ValueError, match="clips too aggressively"):
        normalize_config({"labels": {"winsorize_quantiles": [0.05, 0.95]}})

    with pytest.raises(ValueError, match="clips too aggressively"):
        normalize_config({"labels": {"winsorize_quantiles": [0.03, 0.97]}})

    # 合理边界值应通过
    config_edge = normalize_config({"labels": {"winsorize_quantiles": [0.02, 0.98]}})
    assert config_edge["labels"]["winsorize_quantiles"] == [0.02, 0.98]


def test_rolling_execution_sells_holding_that_leaves_target_set():
    from skyeye.products.ax1.execution.smoother import ExecutionSmoother
    from skyeye.products.ax1.optimizer.executable import ExecutablePortfolioOptimizer
    from skyeye.products.ax1.run_experiment import _execute_rolling_targets

    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "order_book_id": ["A", "B"],
            "target_weight": [0.20, 0.20],
            "price": [10.0, 20.0],
            "adjusted_expected_return": [0.10, 0.10],
            "asset_type": ["stock", "stock"],
        }
    )
    execution_reference = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "order_book_id": ["A", "A", "B"],
            "price": [10.0, 11.0, 20.0],
            "adjusted_expected_return": [0.10, 0.0, 0.10],
            "asset_type": ["stock", "stock", "stock"],
        }
    )

    weights, orders, summary = _execute_rolling_targets(
        targets,
        smoother=ExecutionSmoother(
            target_gross_weight=1.0,
            net_alpha_threshold=0.05,
            net_alpha_column="adjusted_expected_return",
        ),
        executable_optimizer=ExecutablePortfolioOptimizer(
            portfolio_value=100_000,
            lot_size=100,
            min_trade_value=0,
            max_order_count=10,
        ),
        max_turnover=1.0,
        rebalance_interval=1,
        portfolio_value=100_000,
        execution_reference=execution_reference,
    )

    day2_orders = orders[orders["date"] == pd.Timestamp("2024-01-03")].set_index("order_book_id")

    assert day2_orders.loc["A", "order_shares"] < 0
    assert day2_orders.loc["A", "side"] == "sell"
    assert summary["total_order_count"] == 3
    assert "A" not in set(weights[weights["date"] == pd.Timestamp("2024-01-03")].query("target_shares > 0")["order_book_id"])


def test_run_portfolio_replay_uses_configured_net_alpha_threshold(monkeypatch):
    import skyeye.products.ax1.run_experiment as runner
    import skyeye.products.ax1.effective_breadth as breadth_module
    import skyeye.products.ax1.evaluation.metrics as metrics_module
    import skyeye.products.ax1.execution.smoother as smoother_module
    import skyeye.products.ax1.optimizer.allocation as allocation_module
    import skyeye.products.ax1.optimizer.executable as executable_module
    from skyeye.products.ax1.research.execution import run_portfolio_replay

    captured = {}

    class DummySmoother:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class DummyExecutableOptimizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class DummyOpportunityPoolOptimizer:
        def optimize(self, *args, **kwargs):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02"]),
                    "order_book_id": ["510300.XSHG"],
                    "target_weight": [0.2],
                }
            )

    monkeypatch.setattr(runner, "_fit_risk_model_by_date", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_risk_model_for_timestamp", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_opportunity_pool_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_opportunity_pool_config_by_date", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        runner,
        "_attach_execution_inputs",
        lambda frame, **kwargs: frame.assign(
            price=10.0,
            adjusted_expected_return=0.10,
            asset_type="etf",
        ),
    )
    monkeypatch.setattr(runner, "_build_execution_reference", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        runner,
        "_execute_rolling_targets",
        lambda target_weights, **kwargs: (
            target_weights.assign(target_shares=100),
            pd.DataFrame(columns=["date", "order_book_id", "order_shares", "side"]),
            {"total_order_count": 0},
        ),
    )
    monkeypatch.setattr(runner, "_attach_universe_metadata", lambda frame, metadata: frame)
    monkeypatch.setattr(runner, "_first_day_weights", lambda frame: {})
    monkeypatch.setattr(runner, "_enabled_cost_config", lambda costs: costs)
    monkeypatch.setattr(breadth_module, "summarize_effective_breadth_by_date", lambda *args, **kwargs: {})
    monkeypatch.setattr(metrics_module, "evaluate_portfolio_layer", lambda *args, **kwargs: {"portfolio": {}})
    monkeypatch.setattr(smoother_module, "ExecutionSmoother", DummySmoother)
    monkeypatch.setattr(executable_module, "ExecutablePortfolioOptimizer", DummyExecutableOptimizer)
    monkeypatch.setattr(allocation_module, "OpportunityPoolOptimizer", DummyOpportunityPoolOptimizer)

    run_portfolio_replay(
        config={
            "risk_model": {},
            "constraints": {
                "target_gross_exposure": 1.0,
                "cash_buffer": 0.0,
            },
            "execution": {
                "net_alpha_threshold": 0.07,
                "net_alpha_column": "adjusted_expected_return",
                "portfolio_value": 1_000_000,
                "min_trade_value": 3_000,
                "price_column": "close",
            },
            "stop_loss": {"enabled": False},
            "costs": {},
        },
        fused_predictions=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "order_book_id": ["510300.XSHG"],
            }
        ),
        scoped_raw=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "order_book_id": ["510300.XSHG"],
                "close": [10.0],
            }
        ),
        evaluation_labels=pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-02"]),
                "order_book_id": ["510300.XSHG"],
                "label_return_5d": [0.01],
            }
        ),
        industry_map={},
        universe_metadata=pd.DataFrame({"order_book_id": ["510300.XSHG"]}),
        regime_state={},
        regime_state_by_date={},
        as_of_date=pd.Timestamp("2024-01-02"),
    )

    assert captured["net_alpha_threshold"] == pytest.approx(0.07)


def test_attach_execution_inputs_uses_next_session_price_for_signal_date():
    from skyeye.products.ax1.run_experiment import _attach_execution_inputs

    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "order_book_id": ["A"],
            "target_weight": [0.25],
        }
    )
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "order_book_id": ["A"],
            "adjusted_expected_return": [0.01],
        }
    )
    raw_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "order_book_id": ["A", "A"],
            "close": [10.0, 11.5],
            "volume": [1000, 1200],
        }
    )

    attached = _attach_execution_inputs(
        targets,
        predictions=predictions,
        raw_frame=raw_frame,
        price_column="close",
        alpha_column="adjusted_expected_return",
        execution_lag_days=1,
    )

    assert attached.loc[0, "signal_date"] == pd.Timestamp("2024-01-02")
    assert attached.loc[0, "execution_date"] == pd.Timestamp("2024-01-03")
    assert attached.loc[0, "price"] == pytest.approx(11.5)
    assert attached.loc[0, "dollar_volume"] == pytest.approx(11.5 * 1200)


def test_rolling_execution_carries_holding_when_clearance_reference_is_missing():
    from skyeye.products.ax1.execution.smoother import ExecutionSmoother
    from skyeye.products.ax1.optimizer.executable import ExecutablePortfolioOptimizer
    from skyeye.products.ax1.run_experiment import _execute_rolling_targets

    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "order_book_id": ["A", "B"],
            "target_weight": [0.20, 0.20],
            "price": [10.0, 20.0],
            "adjusted_expected_return": [0.10, 0.10],
            "asset_type": ["stock", "stock"],
        }
    )
    execution_reference = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "order_book_id": ["A", "B"],
            "price": [10.0, 20.0],
            "adjusted_expected_return": [0.10, 0.10],
            "asset_type": ["stock", "stock"],
        }
    )

    weights, orders, summary = _execute_rolling_targets(
        targets,
        smoother=ExecutionSmoother(
            target_gross_weight=1.0,
            net_alpha_threshold=0.05,
            net_alpha_column="adjusted_expected_return",
        ),
        executable_optimizer=ExecutablePortfolioOptimizer(
            portfolio_value=100_000,
            lot_size=100,
            min_trade_value=0,
            max_order_count=10,
        ),
        max_turnover=1.0,
        rebalance_interval=1,
        portfolio_value=100_000,
        execution_reference=execution_reference,
    )

    day2_weights = weights[weights["date"] == pd.Timestamp("2024-01-03")].set_index("order_book_id")
    day2_orders = orders[orders["date"] == pd.Timestamp("2024-01-03")]

    assert "A" in day2_weights.index
    assert day2_weights.loc["A", "target_shares"] > 0
    assert day2_orders.empty
    assert summary["order_count_by_date"]["2024-01-03"] == 0


def test_rolling_execution_rebalances_when_carried_industry_cap_drifts_over_limit():
    from skyeye.products.ax1.execution.smoother import ExecutionSmoother
    from skyeye.products.ax1.optimizer.executable import ExecutablePortfolioOptimizer
    from skyeye.products.ax1.run_experiment import _execute_rolling_targets

    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-03",
                ]
            ),
            "order_book_id": ["BROAD_A", "BROAD_B", "TECH", "BROAD_A", "BROAD_B", "TECH"],
            "target_weight": [0.10, 0.10, 0.20, 0.10, 0.10, 0.20],
            "price": [10.0, 10.0, 10.0, 11.0, 11.0, 10.0],
            "industry": ["broad", "broad", "tech", "broad", "broad", "tech"],
            "adjusted_expected_return": [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
            "asset_type": ["etf", "etf", "etf", "etf", "etf", "etf"],
        }
    )

    weights, orders, summary = _execute_rolling_targets(
        targets,
        smoother=ExecutionSmoother(
            target_gross_weight=0.40,
            max_industry_weight=0.20,
            no_trade_buffer_weight=0.02,
            min_trade_value=5_000,
            portfolio_value=100_000,
        ),
        executable_optimizer=ExecutablePortfolioOptimizer(
            portfolio_value=100_000,
            lot_size=100,
            min_trade_value=5_000,
            max_order_count=10,
        ),
        max_turnover=1.0,
        rebalance_interval=10,
        portfolio_value=100_000,
    )

    day2 = weights[weights["date"] == pd.Timestamp("2024-01-03")]
    day2_orders = orders[orders["date"] == pd.Timestamp("2024-01-03")]

    assert day2.groupby("industry")["target_weight"].sum()["broad"] <= 0.20 + 1e-12
    assert not day2_orders.empty
    assert set(day2_orders["side"]) == {"sell"}
    assert summary["risk_rebalance_dates"] == 1
    assert summary["order_count_by_date"]["2024-01-03"] > 0


def test_rolling_execution_applies_net_alpha_gate_on_initial_build():
    from skyeye.products.ax1.execution.smoother import ExecutionSmoother
    from skyeye.products.ax1.optimizer.executable import ExecutablePortfolioOptimizer
    from skyeye.products.ax1.run_experiment import _execute_rolling_targets

    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["LOW_ALPHA", "HIGH_ALPHA"],
            "target_weight": [0.20, 0.20],
            "price": [10.0, 10.0],
            "adjusted_expected_return": [0.01, 0.10],
            "asset_type": ["stock", "stock"],
        }
    )

    weights, orders, _ = _execute_rolling_targets(
        targets,
        smoother=ExecutionSmoother(
            target_gross_weight=1.0,
            net_alpha_threshold=0.05,
            net_alpha_column="adjusted_expected_return",
        ),
        executable_optimizer=ExecutablePortfolioOptimizer(
            portfolio_value=100_000,
            lot_size=100,
            min_trade_value=0,
            max_order_count=10,
        ),
        max_turnover=1.0,
        rebalance_interval=1,
        portfolio_value=100_000,
    )

    assert set(orders["order_book_id"]) == {"HIGH_ALPHA"}
    assert set(weights[weights["target_shares"] > 0]["order_book_id"]) == {"HIGH_ALPHA"}


def test_rolling_execution_t_plus_one_lock_blocks_same_day_sell_after_buy():
    from skyeye.products.ax1.execution.smoother import ExecutionSmoother
    from skyeye.products.ax1.optimizer.executable import ExecutablePortfolioOptimizer
    from skyeye.products.ax1.run_experiment import _execute_rolling_targets

    targets = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02 10:00", "2024-01-02 14:00"]),
            "order_book_id": ["A", "A"],
            "target_weight": [0.50, 0.0],
            "price": [10.0, 10.0],
            "adjusted_expected_return": [0.10, 0.10],
            "dollar_volume": [10_000_000.0, 10_000_000.0],
            "asset_type": ["stock", "stock"],
        }
    )

    weights, orders, _ = _execute_rolling_targets(
        targets,
        smoother=ExecutionSmoother(
            target_gross_weight=1.0,
            t_plus_one_lock=True,
            today_buy_weight_column="today_buy_weight",
            portfolio_value=100_000,
        ),
        executable_optimizer=ExecutablePortfolioOptimizer(
            portfolio_value=100_000,
            lot_size=100,
            min_trade_value=0,
            max_order_count=10,
        ),
        max_turnover=1.0,
        rebalance_interval=1,
        portfolio_value=100_000,
    )

    same_day_late_orders = orders[orders["date"] == pd.Timestamp("2024-01-02 14:00")]
    late_weights = weights[weights["date"] == pd.Timestamp("2024-01-02 14:00")].set_index("order_book_id")

    assert same_day_late_orders.empty
    assert late_weights.loc["A", "target_shares"] > 0


def test_config_rejects_legacy_optimizer_base_overlay_weights():
    with pytest.raises(ValueError, match="legacy optimizer base/overlay"):
        normalize_config(
            {
                "optimizer": {
                    "base_weight": 0.70,
                    "overlay_weight": 0.30,
                }
            }
        )


def test_run_experiment_writes_lgbm_pipeline_contract(personal_pipeline_result):
    from skyeye.products.ax1.persistence import load_experiment

    result = personal_pipeline_result
    experiment_dir = Path(result["output_dir"])
    assert result["product"] == "ax1"
    assert result["status"] in {"ok", "constraint_warning"}
    assert result["constraint_status"] == "ok"
    assert result["evaluation"]["signal"]
    assert result["evaluation"]["portfolio"]
    assert result["regime_state"]["risk_state"] in {"risk_on", "neutral", "risk_off"}
    assert result["component_manifest"]["regime"]["enabled"] is True
    assert result["component_manifest"]["feature_schema"]["feature_set"] == "ax1_unified_v1"
    assert result["component_manifest"]["allocation"]["kind"] == "opportunity_pool_optimizer"
    assert result["data_audit"]["passed"] is True
    assert result["data_audit"]["hard_block_count"] == 0
    assert "feature_momentum_5d" in result["data_audit"]["features"]
    assert result["raw_data_quality"]["passed"] is True
    assert result["feature_data_quality"]["passed"] is True
    assert result["raw_data_quality"]["price_adjustment"]["method"] == "adjusted_price_column"
    assert result["data_version"]["data_hash"]
    assert result["universe_summary"]["pit_audit"]["passed"] is True
    readiness = result["training_readiness"]
    assert readiness["schema_version"] == 1
    assert readiness["blocker_count"] == 0
    assert readiness["checks"]["raw_data_quality"]["passed"] is True
    assert readiness["checks"]["feature_data_quality"]["passed"] is True
    assert readiness["checks"]["price_adjustment_contract"]["passed"] is True
    assert readiness["checks"]["lgbm_param_policy"]["passed"] is True
    assert readiness["checks"]["timing_contract"]["passed"] is True
    assert readiness["checks"]["lgbm_seed"]["passed"] is True
    assert readiness["checks"]["universe_pit_audit"]["passed"] is True
    assert result["implementation_status"]["constraints"] == "weight_turnover_industry_lot_order_constraints_enforced"
    assert result["implementation_status"]["regime_detector"] == "market_regime_etf_proxy"
    assert result["implementation_status"]["features"] == "unified_feature_view"
    assert result["implementation_status"]["optimizer"] == "opportunity_pool_optimizer_lot_aware_execution"
    assert result["implementation_status"]["execution_t_plus_one"] == "implemented_enabled"
    assert result["implementation_status"]["execution_capacity"] == "implemented_enabled"
    assert result["implementation_status"]["risk_model"] == "statistical_factor_pca_covariance_penalty"
    assert result["implementation_status"]["factor_risk_model"] == "implemented_statistical_factor_pca"
    assert result["component_manifest"]["costs"]["enabled"] is True
    assert result["implementation_status"]["model"] == "lgbm_multi_target"
    assert result["implementation_status"]["lightgbm_training"] == "single_split_multi_head_lgbm"
    assert result["training_summary"]["stability"]["metric_key"] == "top_bucket_spread_mean"
    assert result["training_summary"]["confidence_calibration"]["status"] in {"calibrated", "fallback"}
    assert result["parameter_validation_summary"]["schema_version"] == 1
    assert "current" in result["parameter_validation_summary"]
    assert "candidate_metrics" in result["parameter_validation_summary"]
    assert result["parameter_validation_summary"]["lgbm_param_policy"]["status"] == "within_policy"
    assert result["tradable_outcome"]["schema_version"] == 1
    assert result["alpha_transfer_ledger"]["schema_version"] == 1
    assert result["confidence_diagnostic"]["schema_version"] == 1
    assert result["confidence_diagnostic"]["outcome_column"] == "tradable_net_success"
    assert result["confidence_diagnostic"]["tradable_outcome_date_count"] == result["tradable_outcome"]["date_count"]
    assert result["evaluation"]["portfolio"]["tradable_outcome_available"] is True
    assert result["evaluation"]["portfolio"]["max_drawdown"] == pytest.approx(
        result["tradable_outcome"]["max_net_drawdown"]
    )
    assert "diagnostic_only" not in result["evaluation"]["signal"]
    assert result["evaluation"]["portfolio"]["constraint_violations"]
    assert result["evaluation"]["portfolio"]["allocation"]
    assert "net_mean_return" in result["evaluation"]["portfolio"]
    assert "cost_drag_annual" in result["evaluation"]["portfolio"]
    assert "trading" in result["evaluation"]["portfolio"]
    assert "manual_operation_burden" in result["evaluation"]["portfolio"]
    assert "turnover_detail" in result["evaluation"]["portfolio"]
    assert result["evaluation"]["portfolio"]["trading"]["trade_count"] >= 0
    violations = result["evaluation"]["portfolio"]["constraint_violations"]
    assert violations["max_single_weight_count"] == 0
    assert violations["max_industry_weight_count"] == 0
    assert violations["gross_exposure_count"] == 0
    weights = pd.DataFrame(result["target_weights"])
    constraints = result["config"]["constraints"]
    if weights.empty:
        assert result["evaluation"]["portfolio"]["active_gross_mean"] == pytest.approx(0.0)
        assert result["execution_summary"]["total_order_count"] == 0
    else:
        assert "contribution_by_asset_type" in result["evaluation"]["portfolio"]
        assert weights["target_weight"].max() <= constraints["max_single_weight"] + 1e-12
        assert weights.groupby(["date", "industry"])["target_weight"].sum().max() <= constraints["max_industry_weight"] + 1e-12
        expected_budget = constraints["target_gross_exposure"] - constraints["cash_buffer"]
        assert weights.groupby("date")["target_weight"].sum().max() <= expected_budget + 1e-12
        assert {"target_shares", "order_shares", "order_value", "price", "estimated_cost"}.issubset(weights.columns)
        assert (weights["target_shares"] % result["config"]["execution"]["lot_size"] == 0).all()
    assert result["execution_summary"]["max_order_count_observed"] <= result["config"]["execution"]["max_order_count"]
    assert (experiment_dir / "experiment.json").exists()
    first_fold = result["training_summary"]["fold_results"][0]
    assert first_fold["predictor_bundle"]["product"] == "ax1"
    assert first_fold["predictor_bundle"]["predictor_kind"] == "lgbm_multi_target"
    assert first_fold["preprocessor_bundle"]["kind"] == "feature_preprocessor"
    persisted = load_experiment(experiment_dir)
    persisted_fold = persisted["training_summary"]["fold_results"][0]
    assert persisted["artifact_schema_version"] == 1
    assert persisted_fold["model_bundle_ref"] == "folds/fold_000/model_bundle.json"
    assert persisted_fold["preprocessor_bundle_ref"] == "folds/fold_000/preprocessor_bundle.json"
    assert (experiment_dir / persisted_fold["model_bundle_ref"]).is_file()
    assert (experiment_dir / persisted_fold["preprocessor_bundle_ref"]).is_file()
    assert (experiment_dir / persisted_fold["predictions_ref"]).is_file()
    assert (experiment_dir / persisted_fold["weights_ref"]).is_file()


def test_run_experiment_default_lgbm_path_produces_non_constant_predictions(personal_pipeline_result):
    result = personal_pipeline_result
    prediction_summary = result["prediction_summary"]
    assert prediction_summary["row_count"] > 0
    assert prediction_summary["expected_relative_net_return_10d_std"] > 0.0
    assert result["implementation_status"]["features"] == "unified_feature_view"
    assert result["implementation_status"]["model"] == "lgbm_multi_target"
    assert result["training_summary"]["model_kind"] == "lgbm_multi_target"
    assert "feature_z_style_spread_composite_20d" in result["training_summary"]["feature_columns"]
    assert "feature_regime_strength" in result["training_summary"]["feature_columns"]
    assert "feature_interaction_z_excess_mom_20d_x_regime_risk_on" not in result["training_summary"]["feature_columns"]


def test_rule_model_override_is_rejected_from_main_pipeline_contract():
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


def test_enabled_stock_sleeve_requires_future_stock_alpha_module():
    with pytest.raises(NotImplementedError, match="stock alpha|stock_specific"):
        normalize_config(
            {
                "universe": {
                    "layers": {"stock_satellite": {"enabled": True}},
                },
            }
        )


def test_run_experiment_lgbm_profile_trains_walk_forward_by_default(lgbm_walk_forward_result):
    result = lgbm_walk_forward_result
    training_summary = result["training_summary"]
    fold_results = training_summary["fold_results"]

    assert result["status"] in {"ok", "constraint_warning"}
    assert result["implementation_status"]["model"] == "lgbm_multi_target"
    assert result["implementation_status"]["splitter"] == "walk_forward_train_val_test"
    assert result["implementation_status"]["lightgbm_training"] == "walk_forward_multi_head_lgbm"
    assert training_summary["model_kind"] == "lgbm_multi_target"
    assert len(fold_results) == 6
    assert result["gate_summary"]["metrics"]["n_folds"] == 6
    assert result["data_range"]["universe_size"] == 6
    assert result["prediction_summary"]["row_count"] == sum(fold["test_rows"] for fold in fold_results)
    assert training_summary["aggregate_labels_row_count"] == result["prediction_summary"]["row_count"]
    assert result["prediction_summary"]["expected_relative_net_return_10d_std"] > 0.0
    assert result["evaluation"]["signal"]["label_columns"]
    assert result["target_weights"]


def test_build_splitter_constructs_walk_forward_multi_fold():
    from skyeye.products.ax1.run_experiment import _build_splitter

    splitter = _build_splitter(
        {
            "splitter": {
                "kind": "walk_forward",
                "train_end": "2024-06-28",
                "val_months": 1,
                "test_months": 1,
                "embargo_days": 5,
                "n_folds": 3,
                "step_months": 1,
            }
        }
    )

    folds = splitter.split(_sample_lgbm_raw_frame())

    assert len(folds) == 3
    assert [fold["fold_id"] for fold in folds] == [0, 1, 2]
    assert all(len(fold["train_df"]) > 0 for fold in folds)
    assert all(len(fold["val_df"]) > 0 for fold in folds)
    assert all(len(fold["test_df"]) > 0 for fold in folds)


def test_build_splitter_uses_six_walk_forward_folds_when_n_folds_omitted():
    from skyeye.products.ax1.run_experiment import _build_splitter

    splitter = _build_splitter(
        {
            "splitter": {
                "kind": "walk_forward",
                "train_end": "2024-03-29",
                "val_months": 1,
                "test_months": 1,
                "embargo_days": 5,
                "step_months": 1,
            }
        }
    )

    folds = splitter.split(_sample_lgbm_raw_frame())

    assert len(folds) == 6
    assert [fold["fold_id"] for fold in folds] == list(range(6))


def test_run_experiment_lgbm_walk_forward_aggregates_oos_test_predictions(lgbm_walk_forward_result):
    result = lgbm_walk_forward_result
    training_summary = result["training_summary"]
    fold_results = training_summary["fold_results"]
    test_rows_by_fold = [fold["test_rows"] for fold in fold_results]

    assert result["status"] in {"ok", "constraint_warning"}
    assert training_summary["model_kind"] == "lgbm_multi_target"
    assert len(fold_results) > 1
    assert all(
        {"fold_id", "train_rows", "val_rows", "test_rows", "predictions_df"}.issubset(fold)
        for fold in fold_results
    )
    assert all(fold["train_rows"] > 0 and fold["val_rows"] > 0 and fold["test_rows"] > 0 for fold in fold_results)
    assert training_summary["aggregate_predictions_row_count"] == result["prediction_summary"]["row_count"]
    assert training_summary["aggregate_predictions_row_count"] == sum(test_rows_by_fold)
    assert training_summary["aggregate_labels_row_count"] == result["prediction_summary"]["row_count"]
    assert result["prediction_summary"]["expected_relative_net_return_10d_std"] > 0.0
    assert result["evaluation"]["signal"]["label_columns"]
    assert result["target_weights"]


def test_run_experiment_lgbm_walk_forward_wires_robustness_and_calibration_inputs(lgbm_walk_forward_result):
    from skyeye.products.ax1.calibration import build_calibration_bundle

    result = lgbm_walk_forward_result
    training_summary = result["training_summary"]
    fold_results = training_summary["fold_results"]

    assert training_summary["aggregate_metrics"]["prediction"]["rank_ic_mean_mean"] is not None
    assert training_summary["stability"]["metric_key"] == "top_bucket_spread_mean"
    assert training_summary["positive_ratio"]["n_folds"] == len(fold_results)
    assert training_summary["overfit_flags"]["n_folds_compared"] == len(fold_results)
    assert training_summary["confidence_calibration"]["fold_count"] == len(fold_results)
    assert training_summary["confidence_calibration"]["fallback_count"] == 0
    assert all("validation_metrics" in fold and "prediction_metrics" in fold for fold in fold_results)
    assert all("confidence_calibration" in fold for fold in fold_results)
    assert all("label_relative_net_return_10d" in fold["predictions_df"].columns for fold in fold_results)
    assert all("label_net_return_10d" in fold["predictions_df"].columns for fold in fold_results)
    assert "label_net_return_10d" not in pd.DataFrame(result["target_weights"]).columns

    bundle = build_calibration_bundle(result, bucket_count=10)
    assert 0 < bundle["summary"]["oos_rows"] <= training_summary["aggregate_predictions_row_count"]
    assert len(bundle["bucket_stats"]) == 10
    assert result["calibration_bundle"]["summary"]["oos_rows"] == bundle["summary"]["oos_rows"]
    assert result["gate_summary"]["gate_level"] == "canary_live"
    assert result["gate_summary"]["metrics"]["n_folds"] == len(fold_results)
    assert result["effective_breadth_summary"]["schema_version"] == 1
    assert result["effective_breadth_summary"]["latest"]["nominal_count"] > 0
    assert result["effective_breadth_summary"]["p5_effective_breadth"] <= result["effective_breadth_summary"]["mean_effective_breadth"]
    assert (
        training_summary["robustness"]["effective_breadth"]["latest"]["effective_breadth"]
        == result["effective_breadth_summary"]["latest"]["effective_breadth"]
    )
    assert result["gate_summary"]["metrics"]["effective_breadth"] == pytest.approx(
        result["effective_breadth_summary"]["p5_effective_breadth"]
    )
    assert result["evaluation"]["signal"]["rank_ic_significance"]["n_observations"] > 0
    assert result["data_version"]["feature_columns"] == training_summary["feature_columns"]
    assert result["implementation_status"]["walk_forward"] == "implemented"
    assert result["implementation_status"]["calibration"] == "implemented"
    assert result["implementation_status"]["promotion_gate"] == "implemented"
    assert result["implementation_status"]["splitter"] == "walk_forward_train_val_test"


def test_run_experiment_wires_capacity_participation_rate_to_execution_layer(tmp_path, monkeypatch):
    from skyeye.products.ax1.run_experiment import run_experiment

    _install_fast_lgbm_pipeline(monkeypatch)
    raw = _sample_personal_etf_raw_frame()
    config = _fast_single_split_config(
        {
            "constraints": {
                "max_turnover": 1.0,
                "max_single_weight": 0.20,
                "max_industry_weight": 1.0,
                "min_position_count": 1,
                "max_position_count": 10,
            },
            "execution": {
                "rebalance_interval": 1,
                "buffer_weight": 0.0,
                "no_trade_buffer_weight": 0.0,
                "net_alpha_threshold": 0.0,
                "participation_rate": 0.000001,
                "liquidity_column": "dollar_volume",
                "min_trade_value": 2000,
                "portfolio_value": 1_000_000,
                "lot_size": 100,
                "max_order_count": 10,
            },
        }
    )

    result = run_experiment(
        profile_path=None,
        output_dir=tmp_path / "experiments",
        raw_df=raw,
        experiment_name="ax1_capacity_wiring_test",
        config_override=config,
    )

    assert result["execution_summary"]["total_order_count"] == 0
    assert result["implementation_status"]["execution_capacity"] == "implemented_enabled"


def test_run_experiment_lgbm_raises_when_splitter_has_no_valid_fold(tmp_path):
    from skyeye.products.ax1.run_experiment import run_experiment

    with pytest.raises(ValueError, match="splitter produced no valid folds"):
        run_experiment(
            profile_path=None,
            output_dir=tmp_path / "experiments",
            raw_df=_sample_personal_etf_raw_frame().loc[
                lambda frame: frame["date"] <= pd.Timestamp("2024-02-15")
            ],
            experiment_name="ax1_lgbm_insufficient_data_test",
            config_override={
                "model": {
                    "kind": "lgbm_multi_target",
                    "feature_set": "ax1_unified_v1",
                    "include_scopes": ["common", "etf_zscore"],
                    "training_horizons": [5, 10, 20],
                    "risk_horizon": 10,
                    "params": {"n_estimators": 5, "num_threads": 1},
                },
                "labels": {
                    "return_horizons": [5, 10, 20],
                    "volatility_horizons": [10],
                },
                "splitter": {
                    "train_end": "2024-01-10",
                    "val_months": 6,
                    "test_months": 6,
                    "embargo_days": 20,
                },
                "universe": {
                    "validate_pit_universe": False,  # Disable for tests without data_provider
                },
            },
        )


def test_run_experiment_personal_profile_outputs_etf_first_summary(personal_pipeline_result):
    result = personal_pipeline_result
    assert result["status"] == "ok"
    assert result["experiment_name"] == "ax1_personal_etf_core_test"
    assert result["universe_summary"]["asset_type_counts"]["etf"] == 15
    assert result["universe_summary"]["layer_counts"] == {"core": 5, "industry": 7, "style": 3}
    assert "stock" not in result["universe_summary"]["asset_type_counts"]
    weights = pd.DataFrame(result["target_weights"])
    if weights.empty:
        assert result["evaluation"]["portfolio"]["active_gross_mean"] == pytest.approx(0.0)
        assert result["evaluation"]["portfolio"]["allocation"]["allocation_date_count"] == 0
    else:
        assert weights.groupby("date")["order_book_id"].nunique().max() <= 15
        assert set(weights["asset_type"]) == {"etf"}
        assert set(result["evaluation"]["portfolio"]["contribution_by_asset_type"]) == {"etf"}
        assert result["evaluation"]["portfolio"]["allocation"]["core_weight_mean"] >= 0.0
        assert result["evaluation"]["portfolio"]["allocation"]["industry_weight_mean"] >= 0.0
        assert result["evaluation"]["portfolio"]["allocation"]["style_weight_mean"] >= 0.0
    assert result["execution_summary"]["rebalance_allowed_dates"] >= 0
    assert result["prediction_summary"]["expected_relative_net_return_10d_std"] > 0.0
    assert result["implementation_status"]["model"] == "lgbm_multi_target"
    assert result["training_summary"]["stability"]["metric_key"] == "top_bucket_spread_mean"


def test_run_experiment_respects_rebalance_interval_and_max_order_count(tmp_path, monkeypatch):
    from skyeye.products.ax1.run_experiment import run_experiment

    _install_fast_lgbm_pipeline(monkeypatch)
    config = _fast_single_split_config(
        {
            "constraints": {
                "max_turnover": 1.0,
                "max_single_weight": 0.25,
                "max_industry_weight": 1.0,
                "min_position_count": 1,
                "max_position_count": 10,
            },
            "execution": {
                "rebalance_interval": 3,
                "no_trade_buffer_weight": 0.0,
                "net_alpha_threshold": 0.0,
                "min_trade_value": 2000,
                "portfolio_value": 1_000_000,
                "lot_size": 100,
                "max_order_count": 3,
            },
        }
    )

    result = run_experiment(
        profile_path=None,
        output_dir=tmp_path / "experiments",
        raw_df=_sample_personal_etf_raw_frame(),
        experiment_name="ax1_executable_rebalance_gate_test",
        config_override=config,
    )

    weights = pd.DataFrame(result["target_weights"])
    assert result["execution_summary"]["rebalance_interval"] == 3
    assert result["execution_summary"]["max_order_count_observed"] <= 3
    assert weights.groupby("date")["order_shares"].apply(lambda values: (values != 0).sum()).max() <= 3


def test_run_experiment_applies_rolling_current_weights_to_execution_layer(tmp_path, monkeypatch):
    from skyeye.products.ax1.run_experiment import run_experiment

    _install_fast_lgbm_pipeline(monkeypatch)
    config = _fast_single_split_config(
        {
            "constraints": {
                "max_turnover": 0.05,
                "max_single_weight": 0.20,
                "max_industry_weight": 1.0,
                "min_position_count": 1,
                "max_position_count": 10,
            },
            "execution": {
                "rebalance_interval": 1,
                "buffer_weight": 0.0,
                "no_trade_buffer_weight": 0.0,
                "net_alpha_threshold": 0.0,
                "min_trade_value": 2000,
            },
        }
    )
    result = run_experiment(
        profile_path=None,
        output_dir=tmp_path / "experiments",
        raw_df=_sample_personal_etf_raw_frame(),
        experiment_name="ax1_rolling_execution_test",
        config_override=config,
    )

    execution_summary = result["execution_summary"]
    weights = pd.DataFrame(result["target_weights"])
    lot_weight_tolerance = float(
        (
            pd.to_numeric(weights["price"], errors="coerce").fillna(0.0)
            * float(config["execution"]["lot_size"])
            / float(config["execution"]["portfolio_value"])
        ).max()
    )
    rounding_tolerance = lot_weight_tolerance * float(execution_summary["max_order_count_observed"]) / 2.0
    turnover_limit = float(config["constraints"]["max_turnover"]) + rounding_tolerance
    assert execution_summary["max_turnover_observed"] <= turnover_limit
    assert execution_summary["turnover_constrained_dates"] > 0
    assert result["evaluation"]["portfolio"]["mean_turnover"] <= turnover_limit


def test_run_experiment_wires_min_trade_filter_through_rolling_execution(tmp_path, monkeypatch):
    from skyeye.products.ax1.run_experiment import run_experiment

    _install_fast_lgbm_pipeline(monkeypatch)
    config = _fast_single_split_config(
        {
            "constraints": {
                "max_turnover": 1.0,
                "max_single_weight": 0.20,
                "max_industry_weight": 1.0,
                "min_position_count": 1,
                "max_position_count": 10,
            },
            "execution": {
                "buffer_weight": 0.0,
                "min_trade_value": 2_000_000,
                "portfolio_value": 1_000_000,
            },
        }
    )

    result = run_experiment(
        profile_path=None,
        output_dir=tmp_path / "experiments",
        raw_df=_sample_personal_etf_raw_frame(),
        experiment_name="ax1_min_trade_execution_test",
        config_override=config,
    )

    weights = pd.DataFrame(result["target_weights"])
    if not weights.empty:
        assert weights["target_weight"].sum() == pytest.approx(0.0)
        assert weights["target_shares"].sum() == 0
        assert weights["order_shares"].abs().sum() == 0
    assert result["execution_summary"]["total_order_count"] == 0
    assert result["evaluation"]["portfolio"]["mean_turnover"] == pytest.approx(0.0)


def test_promote_package_writes_research_manifest(tmp_path, personal_pipeline_result):
    from skyeye.products.ax1.package_io import load_package
    from skyeye.products.ax1.promote_package import promote_package

    package_dir = promote_package(
        experiment_dir=personal_pipeline_result["output_dir"],
        packages_root=tmp_path / "packages",
        package_id="ax1_research_test",
        skip_gate=True,
    )
    package_payload = load_package("ax1_research_test", packages_root=tmp_path / "packages")

    assert package_dir.name == "ax1_research_test"
    assert package_payload["manifest"]["product"] == "ax1"
    assert package_payload["manifest"]["status"] == "research_package"
    assert package_payload["manifest"]["gate_required"] is False
    assert package_payload["manifest"]["feature_schema"]["feature_set"] == "ax1_unified_v1"
    assert package_payload["manifest"]["view_fusion"]["kind"] == "noop_adjusted_return"
    assert package_payload["manifest"]["optimizer"]["kind"] == "opportunity_pool_optimizer"
    assert package_payload["manifest"]["execution"]["kind"] == "layered_smoother"
    assert package_payload["manifest"]["costs"]["enabled"] is True
    assert package_payload["manifest"]["allocation"]["kind"] == "opportunity_pool_optimizer"
    assert package_payload["manifest"]["raw_data_quality"]["passed"] is True
    assert package_payload["manifest"]["data_version"]["data_hash"]
    assert package_payload["manifest"]["parameter_validation_summary"]["lgbm_param_policy"]["status"] == "within_policy"
    assert (
        package_payload["manifest"]["implementation_status"]["optimizer"]
        == "opportunity_pool_optimizer_lot_aware_execution"
    )
    assert (
        package_payload["manifest"]["implementation_status"]["execution"]
        == "rebalance_interval_net_alpha_lot_aware_order_smoothing"
    )
    assert package_payload["manifest"]["implementation_status"]["lightgbm_training"] == "single_split_multi_head_lgbm"
    assert package_payload["manifest"]["implementation_status"]["black_litterman_posterior"] == "not_implemented"
    assert package_payload["manifest"]["artifact_schema_version"] == 1
    assert package_payload["manifest"]["model_bundle_refs"]
    assert package_payload["manifest"]["fold_artifacts"][0]["model_bundle_ref"] == "folds/fold_000/model_bundle.json"


def test_runner_delegates_training_and_config_merge_to_modules():
    import skyeye.products.ax1.run_experiment as ax1_runner
    from skyeye.products.ax1.research import training

    assert not hasattr(ax1_runner, "_deep_merge")
    assert ax1_runner._run_lgbm_pipeline is training.run_lgbm_pipeline


def test_promote_package_parser_exposes_require_gate():
    from skyeye.products.ax1.promote_package import build_parser

    help_text = build_parser().format_help()

    assert "--require-gate" in help_text
    assert "--gate-level" in help_text


class TestStylePairsValidation:
    """Test style_pairs configuration validation."""

    def _minimal_valid_config(self):
        """Return a minimal valid AX1 config for testing style_pairs validation."""
        return {
            "product": "ax1",
            "schema_version": 1,
            "experiment": {"seed": 20260430},
            "universe": {
                "layers": {
                    "core": {
                        "asset_type": "etf",
                        "exposure_group": "broad_beta",
                        "include": ["510300.XSHG"],
                    },
                    "style": {
                        "asset_type": "etf",
                        "exposure_group": "style_factor",
                        "include": ["510880.XSHG", "159915.XSHE"],
                    },
                }
            },
            "features": {
                "feature_set": "ax1_unified_v1",
                "include_scopes": ["common"],
                "normalization": {"window": 60, "min_periods": 20, "winsorize_z": 4.0},
                "style_pairs": [
                    {"name": "test_pair", "long": "510880.XSHG", "short": "159915.XSHE", "weight": 1.0}
                ],
            },
            "data": {
                "price_adjustment": {
                    "adjusted_price_column": "adjusted_close",
                },
                "quality": {
                    "suspicious_jump_abs_return": 0.15,
                }
            },
            "labels": {
                "return_horizons": [5, 10, 20],
                "stability_horizon": 20,
                "volatility_horizons": [10],
                "relative_return": {"enabled": True, "group_columns": ["universe_layer"], "min_group_count": 2},
            },
            "model": {
                "kind": "lgbm_multi_target",
                "feature_set": "ax1_unified_v1",
                "include_scopes": ["common"],
                "training_horizons": [5, 10, 20],
                "risk_horizon": 10,
                "params": _fast_lgbm_params(),
            },
            "splitter": {
                "kind": "single_split",
                "train_end": "2024-03-29",
                "val_months": 1,
                "test_months": 1,
            },
            "allocation": {
                "kind": "opportunity_pool_optimizer",
                "score_column": "expected_return",
                "layer_exposure_groups": {
                    "core": "broad_beta",
                    "style": "style_factor",
                },
                "exposure_groups": {
                    "broad_beta": {"max_weight": 0.65},
                    "style_factor": {"max_weight": 0.35},
                },
            },
        }

    def test_style_pairs_etf_in_style_layer_passes(self):
        """style_pairs ETF in style layer should pass validation."""
        from skyeye.products.ax1.config import normalize_config

        config = self._minimal_valid_config()
        # Should not raise - normalize_config will fill in missing params
        normalize_config(config)

    def test_style_pairs_etf_not_in_style_layer_fails(self):
        """style_pairs ETF not in style layer should fail validation."""
        from skyeye.products.ax1.config import normalize_config

        config = self._minimal_valid_config()
        # Remove 159915.XSHE from style layer
        config["universe"]["layers"]["style"]["include"] = ["510880.XSHG"]

        with pytest.raises(ValueError, match="style_pairs references ETF IDs not in universe.layers.style.include"):
            normalize_config(config)

    def test_empty_style_layer_skips_validation(self):
        """Empty style layer should skip validation for backward compatibility."""
        from skyeye.products.ax1.config import normalize_config

        config = self._minimal_valid_config()
        # Empty style layer
        config["universe"]["layers"]["style"]["include"] = []

        # Should not raise when style layer is empty
        normalize_config(config)

    def test_multiple_missing_etfs_reported(self):
        """All missing ETFs should be reported in error message."""
        from skyeye.products.ax1.config import normalize_config

        config = self._minimal_valid_config()
        # Only one ETF in style layer
        config["universe"]["layers"]["style"]["include"] = ["510880.XSHG"]
        # Multiple pairs with different ETFs
        config["features"]["style_pairs"] = [
            {"name": "pair1", "long": "510880.XSHG", "short": "159915.XSHE", "weight": 1.0},
            {"name": "pair2", "long": "515180.XSHG", "short": "159949.XSHE", "weight": 1.0},
        ]

        with pytest.raises(ValueError, match="159915.XSHE.*159949.XSHE.*515180.XSHG"):
            normalize_config(config)
