# -*- coding: utf-8 -*-
"""AX1 YAML profile loader and schema validator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from skyeye.market_regime_layer import MarketRegimeConfig, required_market_regime_history_days
from skyeye.products.ax1.layers import LayerRegistry


DEFAULT_PROFILE_PATH = Path(__file__).resolve().parent / "profiles" / "personal_etf_core.yaml"
DEFAULT_WALK_FORWARD_FOLDS = 6
DEFAULT_EXPERIMENT_SEED = 20260430
DEFAULT_REGIME_LOOKBACK_DAYS = required_market_regime_history_days(MarketRegimeConfig(rsrs_m=90))

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "objective": "mae",
    "metric": "mae",
    "boosting_type": "gbdt",
    "num_leaves": 12,
    "max_depth": 4,
    "learning_rate": 0.03,
    "n_estimators": 150,
    "subsample": 0.75,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_alpha": 0.3,
    "reg_lambda": 2.0,
    "min_child_samples": 80,
    "early_stopping_rounds": 20,
    "num_threads": 1,
    "verbose": -1,
    "deterministic": True,
    "force_col_wise": True,
}

DEFAULT_LGBM_PARAM_POLICY: dict[str, Any] = {
    "min_child_samples": {
        "default": 80,
        "candidates": [30, 50, 80],
        "warning_range": [30, 80],
        "hard_range": [20, 120],
        "reason": "Synthetic walk-forward audit in tmp/reg_lambda_experiment_v4.py favored 80 by improving OOS IC while materially shrinking IC gap versus the old 50 baseline.",
    },
    "learning_rate": {
        "default": 0.03,
        "candidates": [0.03, 0.05, 0.08],
        "warning_range": [0.03, 0.10],
        "hard_range": [0.005, 0.20],
        "reason": "Use 0.03 as stable baseline; audit 0.05/0.08 before changing defaults.",
    },
    "reg_lambda": {
        "default": 2.0,
        "candidates": [0.2, 0.5, 1.0, 2.0],
        "warning_range": [0.2, 2.0],
        "hard_range": [0.05, 4.0],
        "reason": "Synthetic walk-forward audit in tmp/reg_lambda_experiment_v4.py found λ=2.0 delivered the highest mean OOS IC among statistically improved candidates over the old λ=0.5 baseline.",
    },
}

DEFAULT_CONFIG: dict[str, Any] = {
    "product": "ax1",
    "schema_version": 1,
    "experiment": {
        "name": "ax1_personal_etf_core",
        "seed": DEFAULT_EXPERIMENT_SEED,
    },
    "universe": {
        "builder": "dynamic",
        "etf_first": True,
        "layers": {
            "core": {
                "asset_type": "etf",
                "exposure_group": "broad_beta",
                "include": [
                    "510050.XSHG",
                    "510300.XSHG",
                    "510500.XSHG",
                    "512100.XSHG",
                    "588000.XSHG",
                ],
            },
            "industry": {
                "asset_type": "etf",
                "exposure_group": "sector",
                "include": [
                    "512800.XSHG",
                    "512880.XSHG",
                    "512000.XSHG",
                    "512070.XSHG",
                    "515000.XSHG",
                    "515260.XSHG",
                    "512480.XSHG",
                    "159995.XSHE",
                    "512760.XSHG",
                    "512720.XSHG",
                    "515050.XSHG",
                    "515880.XSHG",
                    "515790.XSHG",
                    "515030.XSHG",
                    "516160.XSHG",
                    "512170.XSHG",
                    "512010.XSHG",
                    "159929.XSHE",
                    "159928.XSHE",
                    "512690.XSHG",
                    "512400.XSHG",
                    "515220.XSHG",
                    "515210.XSHG",
                    "512980.XSHG",
                    "512660.XSHG",
                    "516970.XSHG",
                    "159865.XSHE",
                    "159611.XSHE",
                    "516110.XSHG",
                    "159819.XSHE",
                    "562500.XSHG",
                    "159639.XSHE",
                    "159870.XSHE",
                    "512200.XSHG",
                ],
            },
            "style": {
                "asset_type": "etf",
                "exposure_group": "style_factor",
                "include": [
                    "510880.XSHG",
                    "515180.XSHG",
                    "512890.XSHG",
                    "159901.XSHE",
                    "159902.XSHE",
                    "159949.XSHE",
                    "159915.XSHE",
                    "513050.XSHG",
                    "513130.XSHG",
                    "513100.XSHG",
                    "513500.XSHG",
                    "518880.XSHG",
                    "159985.XSHE",
                    "511010.XSHG",
                    "511260.XSHG",
                    "511220.XSHG",
                ],
            },
            "stock_satellite": {
                "asset_type": "stock",
                "exposure_group": "stock_alpha",
                "enabled": False,
                "include": [],
                "max_count": 6,
            },
        },
        "min_listing_days": 120,
        "exclude_st": True,
        "exclude_suspended": True,
        "min_aum": 50_000_000,
        "min_daily_dollar_volume": 10_000_000,
        "liquidity_lookback_days": 20,
    },
    "features": {
        "feature_set": "ax1_unified_v1",
        "include_scopes": ["common", "etf_zscore", "regime"],
        "normalization": {
            "kind": "rolling_time_series",
            "windows": [60],
            "window": 60,
            "min_periods": 20,
            "winsorize_z": 4.0,
            "cross_sectional": False,
        },
        # Used by FeatureView to normalize rolling 20d dollar volume into [0, 1].
        # 5e8 is a more realistic "fully liquid" threshold for ETF turnover.
        "liquidity_full_dollar_volume": 500_000_000.0,
        # ETF transaction cost prior should stay in a realistic 3-5bp band.
        "etf_cost_forecast_bps": 4.0,
        "style_pairs": [
            {"name": "dividend_vs_growth", "long": "510880.XSHG", "short": "159915.XSHE", "weight": 1.0},
            {"name": "value_vs_growth", "long": "515180.XSHG", "short": "159915.XSHE", "weight": 1.0},
            {"name": "large_vs_small", "long": "515180.XSHG", "short": "159949.XSHE", "weight": 1.0},
        ],
        "style_exposure_window": 60,
        "style_exposure_min_periods": 20,
        "style_beta_clip": 2.0,
    },
    "data": {
        "price_adjustment": {
            "required": True,
            "price_column": "close",
            "adjusted_price_column": "adjusted_close",
            "adjustment_factor_column": "adjust_factor",
            "adjustment_status_column": "price_adjustment_status",
            "allow_declared_adjusted_close": False,
            "accepted_adjusted_statuses": ["adjusted", "forward_adjusted", "backward_adjusted"],
        },
        "quality": {
            "enabled": True,
            "enforce_hard_blocks": True,
            "required_raw_columns": ["date", "order_book_id", "close", "volume"],
            "optional_ohlc_columns": ["open", "high", "low"],
            # A股ETF受10%涨跌停限制，15%阈值可捕获复权错误和价格断点
            # 跨境ETF不受涨跌停限制，但15%单日波动仍属异常
            "suspicious_jump_abs_return": 0.15,
            "feature_missing_warning_ratio": 0.50,
            "min_asset_coverage_ratio": 0.50,
            "min_cross_section_count": 2,
        },
    },
    "regime": {
        "enabled": True,
        "benchmark_source": "raw_core_proxy",
        "core_proxy_method": "preferred_id",
        "preferred_benchmark_ids": [
            "510300.XSHG",
            "510050.XSHG",
        ],
        "industry_source": "universe_industry_etfs",
        "lookback_days": DEFAULT_REGIME_LOOKBACK_DAYS,
        "fallback_regime": "range_co_move",
        "market_regime_config": {
            "rsrs_m": 90,
        },
    },
    "labels": {
        "return_horizons": [5, 10, 20],
        "stability_horizon": 20,
        "volatility_horizons": [10],
        "winsorize_quantiles": [0.01, 0.99],
        "relative_return": {
            "enabled": True,
            "group_columns": ["universe_layer"],
            "min_group_count": 2,
            "fallback": "date",
        },
        "trading_days_per_year": 244,
        "entry_lag_days": 1,
    },
    "preprocessor": {
        "kind": "feature_preprocessor",
        "neutralize": True,
        "winsorize_scale": 3.5,
        "standardize": True,
        "min_obs": 5,
        "sector_optional": True,
    },
    "splitter": {
        "kind": "walk_forward",
        "train_end": "auto",
        "val_months": 1,
        "test_months": 1,
        "embargo_days": 20,
        "n_folds": DEFAULT_WALK_FORWARD_FOLDS,
        "step_months": 1,
    },
    "model": {
        "registry": "default",
        "kind": "lgbm_multi_target",
        "feature_set": "ax1_unified_v1",
        "include_scopes": ["common", "etf_zscore", "regime"],
        "training_horizons": [5, 10, 20],
        "risk_horizon": 10,
        "stability_horizon": 20,
        "liquidity_column": "feature_dollar_volume",
        "confidence_method": "sign_consistency",
        "params": deepcopy(DEFAULT_LGBM_PARAMS),
        "param_policy": deepcopy(DEFAULT_LGBM_PARAM_POLICY),
    },
    "view_fusion": {
        "kind": "noop_adjusted_return",
        "enabled": False,
        "return_column": "expected_relative_net_return_10d",
    },
    "risk_model": {
        "kind": "statistical_factor",
        "lookback_days": 120,
        "min_periods": 20,
        "n_factors": 6,
        "shrinkage": 0.30,
        "idiosyncratic_floor": 1e-8,
    },
    "optimizer": {
        "kind": "opportunity_pool_optimizer",
    },
    "allocation": {
        "kind": "opportunity_pool_optimizer",
        "score_column": "expected_relative_net_return_10d",
        "allow_gross_underfill": True,
        "min_allocatable_score": 0.0,
        "cash_fallback": {"enabled": True},
        "layer_exposure_groups": {
            "core": "broad_beta",
            "industry": "sector",
            "style": "style_factor",
            "stock_satellite": "stock_alpha",
        },
        "exposure_groups": {
            "broad_beta": {"max_weight": 0.65, "score_multiplier": 1.00},
            "sector": {"max_weight": 0.55, "score_multiplier": 1.00},
            "style_factor": {"max_weight": 0.35, "score_multiplier": 1.00},
            "defensive_cash": {"max_weight": 0.50, "score_multiplier": 0.80},
            "cross_border": {"max_weight": 0.25, "score_multiplier": 0.80},
            "commodity": {"max_weight": 0.20, "score_multiplier": 0.80},
            "stock_alpha": {"max_weight": 0.00, "score_multiplier": 0.00},
        },
        "execution_drift_buffer": 0.01,
    },
    "constraints": {
        "max_single_weight": 0.25,
        "target_gross_exposure": 1.0,
        "cash_buffer": 0.03,
        "max_turnover": 0.15,
        "min_trade_value": 3000,
        "max_industry_weight": 0.20,
        "max_position_count": 15,
        "min_position_count": 1,
        "max_portfolio_volatility": 0.30,
    },
    "stop_loss": {
        "enabled": True,
        "levels": [
            {"name": "yellow", "drawdown_threshold": 0.10, "target_exposure": 0.70},
            {"name": "orange", "drawdown_threshold": 0.15, "target_exposure": 0.40},
            {"name": "red", "drawdown_threshold": 0.20, "target_exposure": 0.00},
        ],
        "cooldown_trading_days": 10,
        "recovery_requires_regime_not_risk_off": True,
    },
    "costs": {
        "enabled": True,
        "stock": {
            "commission_rate": 0.0008,
            "stamp_tax_rate": 0.0005,
            "slippage_bps": 5.0,
            "impact_bps": 5.0,
            "min_commission": 0.0,
        },
        "etf": {
            "commission_rate": 0.0003,
            "stamp_tax_rate": 0.0,
            "slippage_bps": 2.0,
            "impact_bps": 2.0,
            "min_commission": 0.0,
        },
        "default_asset_type": "stock",
    },
    "execution": {
        "kind": "layered_smoother",
        "rebalance_days_min": 10,
        "rebalance_days_max": 20,
        "rebalance_interval": 10,
        "min_weight": 0.002,
        "min_trade_value": 3000,
        "buffer_weight": 0.002,
        "no_trade_buffer_weight": 0.002,
        "net_alpha_threshold": 0.0,
        "net_alpha_column": "adjusted_expected_return",
        "t_plus_one_lock": True,
        "today_buy_weight_column": "today_buy_weight",
        "participation_rate": 0.05,
        "liquidity_column": "dollar_volume",
        "portfolio_value": 1000000,
        "lot_size": 100,
        "max_order_count": 12,
        "price_column": "close",
        "execution_lag_days": 1,
    },
    "evaluation": {
        "signal_layer": True,
        "portfolio_layer": True,
    },
    "package": {
        "schema_version": 1,
        "status": "research_package",
    },
}


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    """读取 YAML profile，并合并默认配置。"""
    profile_path = Path(path) if path is not None else DEFAULT_PROFILE_PATH
    with profile_path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj) or {}
    return normalize_config(payload)


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """合并默认配置并做 schema 校验。"""
    merged = _deep_update(DEFAULT_CONFIG, config or {})
    _normalize_lgbm_params(merged)
    validate_config(merged)
    return merged


def merge_config(base: dict[str, Any], override: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a deep-merged config payload without schema validation."""
    return _deep_update(base, override or {})


def validate_config(config: dict[str, Any]) -> None:
    """校验 AX1 profile 的关键架构约束。"""
    if config.get("product") != "ax1":
        raise ValueError("AX1 profile product must be 'ax1'")
    if int(config.get("schema_version", 0)) < 1:
        raise ValueError("schema_version must be positive")
    experiment = config.get("experiment", {})
    if int(experiment.get("seed", 0)) <= 0:
        raise ValueError("experiment.seed must be a positive integer")
    layer_registry = LayerRegistry.from_config(config)
    for layer_name in layer_registry.layer_names():
        spec = layer_registry.spec(layer_name)
        if spec.asset_type == "stock" and spec.enabled:
            raise NotImplementedError("AX1 stock alpha/stock_specific feature scope is not implemented")

    labels = config.get("labels", {})
    return_horizons = [int(item) for item in labels.get("return_horizons", [])]
    if 10 not in return_horizons or 20 not in return_horizons:
        raise ValueError("AX1 Personal return_horizons must include 10 and 20")
    if int(labels.get("stability_horizon", 0)) != 20:
        raise ValueError("AX1 Personal stability_horizon must be 20")
    volatility_horizons = [int(item) for item in labels.get("volatility_horizons", [])]
    if any(horizon <= 0 for horizon in volatility_horizons):
        raise ValueError("volatility_horizons must be positive")
    winsorize_quantiles = labels.get("winsorize_quantiles")
    if winsorize_quantiles is not None:
        if len(winsorize_quantiles) != 2:
            raise ValueError("winsorize_quantiles must be a pair")
        lower, upper = float(winsorize_quantiles[0]), float(winsorize_quantiles[1])
        if not (0.0 <= lower < upper <= 1.0):
            raise ValueError("winsorize_quantiles must satisfy 0 <= lower < upper <= 1")
        # 50 只 ETF 截面上 [0.02, 0.98] 只保留第 2~49 名，裁剪过猛；
        # 小截面下分位数裁剪本身有离散跳跃问题，下界应 ≤ 0.01、上界应 ≥ 0.99。
        if lower > 0.02:
            raise ValueError(
                "winsorize_quantiles lower bound > 0.02 clips too aggressively for small ETF cross-sections"
            )
        if upper < 0.98:
            raise ValueError(
                "winsorize_quantiles upper bound < 0.98 clips too aggressively for small ETF cross-sections"
            )
    if int(labels.get("trading_days_per_year", 244)) <= 0:
        raise ValueError("trading_days_per_year must be positive")
    if int(labels.get("entry_lag_days", 0)) < 0:
        raise ValueError("labels.entry_lag_days must be non-negative")
    relative_return = labels.get("relative_return", {})
    if not isinstance(relative_return, dict):
        raise ValueError("labels.relative_return must be a mapping")
    if not bool(relative_return.get("enabled", False)):
        raise ValueError("AX1 ETF training requires labels.relative_return.enabled=true")
    group_columns = [str(item) for item in relative_return.get("group_columns", [])]
    if not group_columns:
        raise ValueError("labels.relative_return.group_columns must be non-empty")
    if int(relative_return.get("min_group_count", 0)) < 2:
        raise ValueError("labels.relative_return.min_group_count must be at least 2")
    if str(relative_return.get("fallback", "date")) not in {"date", "none"}:
        raise ValueError("labels.relative_return.fallback must be 'date' or 'none'")
    data = config.get("data", {})
    price_adjustment = data.get("price_adjustment", {})
    if bool(price_adjustment.get("required", True)):
        if not str(price_adjustment.get("price_column", "close") or "").strip():
            raise ValueError("data.price_adjustment.price_column must be non-empty")
        if (
            not price_adjustment.get("adjusted_price_column")
            and not price_adjustment.get("adjustment_factor_column")
            and not bool(price_adjustment.get("allow_declared_adjusted_close", False))
        ):
            raise ValueError("data.price_adjustment requires adjusted_price_column, adjustment_factor_column, or declared adjusted close")
    quality = data.get("quality", {})
    if bool(quality.get("enabled", True)):
        if float(quality.get("suspicious_jump_abs_return", 0.0)) <= 0:
            raise ValueError("data.quality.suspicious_jump_abs_return must be positive")
        if not 0.0 <= float(quality.get("feature_missing_warning_ratio", 0.0)) <= 1.0:
            raise ValueError("data.quality.feature_missing_warning_ratio must be between 0 and 1")
        if not 0.0 <= float(quality.get("min_asset_coverage_ratio", 0.0)) <= 1.0:
            raise ValueError("data.quality.min_asset_coverage_ratio must be between 0 and 1")
        if int(quality.get("min_cross_section_count", 1)) < 1:
            raise ValueError("data.quality.min_cross_section_count must be positive")

    preprocessor = config.get("preprocessor", {})
    if str(preprocessor.get("kind", "feature_preprocessor")) != "feature_preprocessor":
        raise ValueError("unsupported preprocessor.kind")
    if preprocessor.get("winsorize_scale") is not None and float(preprocessor["winsorize_scale"]) <= 0:
        raise ValueError("preprocessor.winsorize_scale must be positive or null")
    if int(preprocessor.get("min_obs", 5)) <= 0:
        raise ValueError("preprocessor.min_obs must be positive")

    splitter = config.get("splitter", {})
    splitter_kind = str(splitter.get("kind", "single_split"))
    if splitter_kind not in {"single_split", "walk_forward"}:
        raise ValueError("unsupported splitter.kind")
    if int(splitter.get("val_months", 1)) <= 0 or int(splitter.get("test_months", 1)) <= 0:
        raise ValueError("splitter val_months and test_months must be positive")
    if int(splitter.get("embargo_days", 0)) < 0:
        raise ValueError("splitter.embargo_days must be non-negative")
    if splitter_kind == "walk_forward":
        if int(splitter.get("n_folds", DEFAULT_WALK_FORWARD_FOLDS)) <= 0:
            raise ValueError("walk_forward splitter requires n_folds > 0")
        if int(splitter.get("step_months", 1)) <= 0:
            raise ValueError("walk_forward splitter requires step_months > 0")

    model = config.get("model", {})
    model_kind = str(model.get("kind", "lgbm_multi_target"))
    if model_kind != "lgbm_multi_target":
        raise ValueError(f"unsupported AX1 model kind: {model_kind}")
    features = config.get("features", {})
    feature_set = str(features.get("feature_set", ""))
    if feature_set != "ax1_unified_v1":
        raise ValueError("features.feature_set must be ax1_unified_v1")
    include_scopes = [str(item) for item in features.get("include_scopes", [])]
    if not include_scopes:
        raise ValueError("features.include_scopes must be non-empty")
    allowed_scopes = {"common", "etf_raw", "etf_zscore", "regime", "regime_interaction", "stock_specific", "fundamental", "flow", "technical", "macro"}
    unknown_scopes = sorted(set(include_scopes) - allowed_scopes)
    if unknown_scopes:
        raise ValueError(f"features.include_scopes contains unknown scopes: {unknown_scopes}")
    if "stock_specific" in set(include_scopes):
        raise NotImplementedError("AX1 stock_specific feature scope is not implemented")
    if bool((features.get("normalization") or {}).get("cross_sectional", False)):
        raise ValueError("features.normalization.cross_sectional must be false")
    normalization = features.get("normalization", {})
    normalization_windows = normalization.get("windows", [60])
    normalization_window = int(
        normalization.get(
            "window",
            normalization_windows[0] if isinstance(normalization_windows, list) and normalization_windows else 60,
        )
    )
    if normalization_window < 2:
        raise ValueError("features.normalization.window must be at least 2")
    if int(normalization.get("min_periods", 1)) < 1:
        raise ValueError("features.normalization.min_periods must be positive")
    if float(normalization.get("winsorize_z", 0.0)) <= 0:
        raise ValueError("features.normalization.winsorize_z must be positive")
    for pair in features.get("style_pairs", []) or []:
        if not pair.get("name") or not pair.get("long") or not pair.get("short"):
            raise ValueError("features.style_pairs require name/long/short")

    # Validate style_pairs ETF IDs are in style layer
    style_layer_include = set(
        config.get("universe", {})
        .get("layers", {})
        .get("style", {})
        .get("include", [])
    )
    style_pairs = features.get("style_pairs", []) or []
    all_style_pair_ids = set()
    for pair in style_pairs:
        for key in ("long", "short"):
            etf_id = str(pair.get(key, ""))
            if etf_id:
                all_style_pair_ids.add(etf_id)

    missing_ids = all_style_pair_ids - style_layer_include
    if missing_ids and style_layer_include:
        raise ValueError(
            f"features.style_pairs references ETF IDs not in universe.layers.style.include: "
            f"{sorted(missing_ids)}. Please add these ETFs to the style layer or update style_pairs configuration."
        )

    style_exposure_window = int(features.get("style_exposure_window", 60))
    style_exposure_min_periods = int(features.get("style_exposure_min_periods", 20))
    if style_exposure_window < 2:
        raise ValueError("features.style_exposure_window must be at least 2")
    if style_exposure_min_periods < 1:
        raise ValueError("features.style_exposure_min_periods must be positive")
    if style_exposure_min_periods > style_exposure_window:
        raise ValueError("features.style_exposure_min_periods cannot exceed style_exposure_window")
    if float(features.get("style_beta_clip", 2.0)) <= 0:
        raise ValueError("features.style_beta_clip must be positive")
    if model.get("feature_columns"):
        raise ValueError("lgbm_multi_target must use feature_set, not feature_columns")
    if str(model.get("feature_set", "")) != "ax1_unified_v1":
        raise ValueError("lgbm_multi_target feature_set must be ax1_unified_v1")
    model_scopes = [str(item) for item in model.get("include_scopes", [])]
    if not model_scopes:
        raise ValueError("lgbm_multi_target include_scopes must be non-empty")
    unknown_model_scopes = sorted(set(model_scopes) - allowed_scopes)
    if unknown_model_scopes:
        raise ValueError(f"lgbm_multi_target include_scopes contains unknown scopes: {unknown_model_scopes}")
    if "stock_specific" in set(model_scopes):
        raise NotImplementedError("AX1 stock_specific feature scope is not implemented")
    training_horizons = [int(item) for item in model.get("training_horizons", [])]
    if not training_horizons or not set(training_horizons).issubset(set(return_horizons)):
        raise ValueError("lgbm_multi_target training_horizons must be contained in labels.return_horizons")
    if training_horizons != [5, 10, 20]:
        raise ValueError("lgbm_multi_target training_horizons must be exactly [5, 10, 20]")
    risk_horizon = int(model.get("risk_horizon", 0))
    if risk_horizon not in set(volatility_horizons):
        raise ValueError("lgbm_multi_target risk_horizon must be present in labels.volatility_horizons")
    if not splitter.get("train_end"):
        raise ValueError("lgbm_multi_target requires splitter.train_end")
    if (
        splitter_kind == "walk_forward"
        and int(splitter.get("n_folds", DEFAULT_WALK_FORWARD_FOLDS)) < DEFAULT_WALK_FORWARD_FOLDS
    ):
        raise ValueError(
            f"lgbm_multi_target walk_forward splitter requires n_folds >= {DEFAULT_WALK_FORWARD_FOLDS}"
        )
    _validate_lgbm_params(model.get("params") or {})
    _validate_lgbm_param_policy(model.get("param_policy") or {})

    regime = config.get("regime", {})
    if bool(regime.get("enabled", False)):
        if str(regime.get("benchmark_source", "raw_core_proxy")) not in {"raw_core_proxy", "data_provider", "auto"}:
            raise ValueError("unsupported regime.benchmark_source")
        if str(regime.get("core_proxy_method", "preferred_id")) not in {"preferred_id", "equal_weight_core"}:
            raise ValueError("unsupported regime.core_proxy_method")
        if str(regime.get("industry_source", "universe_industry_etfs")) not in {"universe_industry_etfs", "data_provider_indices", "auto"}:
            raise ValueError("unsupported regime.industry_source")
        if int(regime.get("lookback_days", 1)) <= 0:
            raise ValueError("regime.lookback_days must be positive")
        if str(regime.get("fallback_regime", "range_co_move")) not in {
            "bull_co_move",
            "bull_rotation",
            "range_co_move",
            "range_rotation",
            "bear_co_move",
            "bear_rotation",
        }:
            raise ValueError("unsupported regime.fallback_regime")

    allocation = config.get("allocation", {})
    allocation_kind = str(allocation.get("kind", "opportunity_pool_optimizer"))
    if allocation_kind != "opportunity_pool_optimizer":
        raise ValueError("unsupported allocation.kind")
    legacy_allocation_keys = {
        "layers",
        "risk_on_tilt",
        "risk_off_tilt",
        "rotation_tilt",
        "min_tilt_strength",
    }
    present_legacy_allocation_keys = sorted(legacy_allocation_keys.intersection(allocation))
    if present_legacy_allocation_keys:
        raise ValueError(
            "legacy allocation budget/tilt keys are no longer supported: "
            + ", ".join(present_legacy_allocation_keys)
        )
    if not str(allocation.get("score_column", "") or "").strip():
        raise ValueError("allocation.score_column must be non-empty")
    if float(allocation.get("min_allocatable_score", 0.0)) < 0:
        raise ValueError("allocation.min_allocatable_score must be non-negative")
    cash_fallback = allocation.get("cash_fallback", {})
    if cash_fallback is not None and not isinstance(cash_fallback, dict):
        raise ValueError("allocation.cash_fallback must be a mapping")
    exposure_groups = allocation.get("exposure_groups")
    if not isinstance(exposure_groups, dict) or not exposure_groups:
        raise ValueError("allocation.exposure_groups must be a non-empty mapping")
    for group_name, payload in exposure_groups.items():
        if not isinstance(payload, dict):
            raise ValueError(f"allocation.exposure_groups.{group_name} must be a mapping")
        if payload.get("max_weight") is not None and float(payload.get("max_weight", 0.0)) < 0:
            raise ValueError(f"allocation.exposure_groups.{group_name}.max_weight must be non-negative")
        if float(payload.get("score_multiplier", 1.0)) < 0:
            raise ValueError(f"allocation.exposure_groups.{group_name}.score_multiplier must be non-negative")
    layer_exposure_groups = allocation.get("layer_exposure_groups", {})
    if layer_exposure_groups is not None and not isinstance(layer_exposure_groups, dict):
        raise ValueError("allocation.layer_exposure_groups must be a mapping")
    known_layers = set(layer_registry.layer_names())
    known_groups = {str(group_name) for group_name in exposure_groups}
    for layer_name, group_name in (layer_exposure_groups or {}).items():
        if str(layer_name) not in known_layers:
            raise ValueError(f"allocation.layer_exposure_groups.{layer_name} not found in universe.layers")
        if str(group_name) not in known_groups:
            raise ValueError(f"allocation.layer_exposure_groups.{layer_name} references unknown exposure group")
    for layer_name in layer_registry.layer_names():
        exposure_group = str((layer_exposure_groups or {}).get(layer_name) or layer_registry.exposure_group_for_layer(layer_name))
        if exposure_group not in known_groups:
            raise ValueError(f"universe.layers.{layer_name}.exposure_group references unknown allocation exposure group")
    if float(allocation.get("execution_drift_buffer", 0.0)) < 0:
        raise ValueError("allocation.execution_drift_buffer must be non-negative")

    risk_model = config.get("risk_model", {})
    risk_kind = str(risk_model.get("kind", "historical_covariance"))
    if risk_kind not in {"historical_covariance", "statistical_factor"}:
        raise ValueError("unsupported risk_model.kind")
    lookback_days = int(risk_model.get("lookback_days", 1))
    if lookback_days <= 0:
        raise ValueError("risk_model.lookback_days must be positive")
    min_periods = int(risk_model.get("min_periods", 2))
    if min_periods < 2:
        raise ValueError("risk_model.min_periods must be at least 2")
    if min_periods > lookback_days:
        raise ValueError("risk_model.min_periods cannot exceed risk_model.lookback_days")
    if risk_kind == "statistical_factor":
        if int(risk_model.get("n_factors", 1)) <= 0:
            raise ValueError("risk_model.n_factors must be positive")
        if float(risk_model.get("idiosyncratic_floor", 0.0)) < 0:
            raise ValueError("risk_model.idiosyncratic_floor must be non-negative")
    if float(risk_model.get("shrinkage", 0.0)) < 0 or float(risk_model.get("shrinkage", 0.0)) > 1:
        raise ValueError("risk_model.shrinkage must be between 0 and 1")

    optimizer = config.get("optimizer", {})
    optimizer_kind = str(optimizer.get("kind", "opportunity_pool_optimizer"))
    if optimizer_kind != "opportunity_pool_optimizer":
        raise ValueError("unsupported optimizer.kind")
    legacy_optimizer_keys = sorted({"base_weight", "overlay_weight"}.intersection(optimizer))
    if legacy_optimizer_keys:
        raise ValueError(
            "legacy optimizer base/overlay weights are no longer supported: "
            + ", ".join(legacy_optimizer_keys)
        )

    execution = config.get("execution", {})
    min_days = int(execution.get("rebalance_days_min", 0))
    max_days = int(execution.get("rebalance_days_max", 0))
    if min_days < 1 or max_days < 1 or min_days > max_days:
        raise ValueError("rebalance window must be positive and ordered")
    if int(execution.get("rebalance_interval", min_days)) < 1:
        raise ValueError("execution.rebalance_interval must be positive")

    constraints = config.get("constraints", {})
    if float(constraints.get("max_single_weight", 0.0)) <= 0:
        raise ValueError("max_single_weight must be positive")
    if float(constraints.get("target_gross_exposure", 0.0)) <= 0:
        raise ValueError("target_gross_exposure must be positive")
    if float(constraints.get("cash_buffer", 0.0)) < 0:
        raise ValueError("cash_buffer must be non-negative")
    if float(constraints.get("max_turnover", 0.0)) < 0:
        raise ValueError("max_turnover must be non-negative")
    min_position_count = int(constraints.get("min_position_count", 1))
    max_position_count = int(constraints.get("max_position_count", min_position_count))
    if min_position_count < 1 or max_position_count < 1:
        raise ValueError("position counts must be positive")
    if min_position_count > max_position_count:
        raise ValueError("min_position_count cannot exceed max_position_count")
    if float(execution.get("buffer_weight", 0.0)) < 0:
        raise ValueError("buffer_weight must be non-negative")
    if float(execution.get("no_trade_buffer_weight", execution.get("buffer_weight", 0.0))) < 0:
        raise ValueError("no_trade_buffer_weight must be non-negative")
    if float(execution.get("net_alpha_threshold", 0.0)) < 0:
        raise ValueError("net_alpha_threshold must be non-negative")
    participation_rate = execution.get("participation_rate")
    if participation_rate is not None:
        participation_rate = float(participation_rate)
        if participation_rate < 0:
            raise ValueError("participation_rate must be non-negative")
        if participation_rate > 1.0 + 1e-12:
            raise ValueError("participation_rate must be <= 1.0")
    if not str(execution.get("liquidity_column", "dollar_volume") or "").strip():
        raise ValueError("execution.liquidity_column must be non-empty")
    if bool(execution.get("t_plus_one_lock", False)):
        if not str(execution.get("today_buy_weight_column", "") or "").strip():
            raise ValueError("execution.today_buy_weight_column must be non-empty")
    if float(execution.get("min_trade_value", 0.0)) < 2000:
        raise ValueError("execution.min_trade_value must be at least 2000")
    if int(execution.get("lot_size", 0)) < 1:
        raise ValueError("execution.lot_size must be positive")
    if int(execution.get("max_order_count", 0)) < 1:
        raise ValueError("execution.max_order_count must be positive")
    if not str(execution.get("price_column", "")).strip():
        raise ValueError("execution.price_column must be non-empty")
    if int(execution.get("execution_lag_days", 0)) < 0:
        raise ValueError("execution.execution_lag_days must be non-negative")
    if int(execution.get("execution_lag_days", 0)) < int(labels.get("entry_lag_days", 0)):
        raise ValueError("execution.execution_lag_days must be >= labels.entry_lag_days")
    if constraints.get("max_industry_weight") is not None and float(constraints["max_industry_weight"]) <= 0:
        raise ValueError("max_industry_weight must be positive")

    # --- stop_loss validation ---
    stop_loss = config.get("stop_loss", {})
    if bool(stop_loss.get("enabled", False)):
        levels = stop_loss.get("levels", [])
        if not isinstance(levels, list) or len(levels) < 1:
            raise ValueError("stop_loss.levels must be a non-empty list when enabled")
        thresholds = []
        prev_exposure = float("inf")
        for lv in levels:
            if not isinstance(lv, dict):
                raise ValueError("each stop_loss level must be a mapping")
            threshold = float(lv.get("drawdown_threshold", 0.0))
            exposure = float(lv.get("target_exposure", 0.0))
            if threshold <= 0 or threshold >= 1:
                raise ValueError("stop_loss level drawdown_threshold must be in (0, 1)")
            if exposure < 0 or exposure > 1:
                raise ValueError("stop_loss level target_exposure must be in [0, 1]")
            thresholds.append(threshold)
            if exposure > prev_exposure + 1e-10:
                raise ValueError("stop_loss levels must have non-increasing target_exposure (higher severity = lower exposure)")
            prev_exposure = exposure
        if thresholds != sorted(thresholds):
            raise ValueError("stop_loss levels must have ascending drawdown_thresholds")
    cooldown = int(stop_loss.get("cooldown_trading_days", 10))
    if cooldown < 0:
        raise ValueError("stop_loss.cooldown_trading_days must be non-negative")

    costs = config.get("costs", {})
    for section_name in ("stock", "etf"):
        section = costs.get(section_name, {})
        for key in ("commission_rate", "stamp_tax_rate", "slippage_bps", "impact_bps", "min_commission"):
            if float(section.get(key, 0.0)) < 0:
                raise ValueError(f"costs.{section_name}.{key} must be non-negative")


def build_component_manifest(config: dict[str, Any]) -> dict[str, Any]:
    """生成 package manifest 需要的组件契约摘要。"""
    model_kind = str(config.get("model", {}).get("kind", "lgbm_multi_target"))
    splitter_kind = str(config.get("splitter", {}).get("kind", "single_split"))
    uses_lgbm = model_kind == "lgbm_multi_target"
    execution_cfg = config.get("execution", {})
    capacity_rate = execution_cfg.get("participation_rate")
    capacity_enabled = capacity_rate is not None and float(capacity_rate) > 0
    risk_model_kind = str(config.get("risk_model", {}).get("kind", "historical_covariance"))
    if risk_model_kind == "statistical_factor":
        risk_model_status = "statistical_factor_pca_covariance_penalty"
        factor_risk_status = "implemented_statistical_factor_pca"
    elif risk_model_kind == "historical_covariance":
        risk_model_status = "historical_covariance_full_covariance_penalty"
        factor_risk_status = "not_used_by_current_profile"
    else:
        risk_model_status = "not_implemented"
        factor_risk_status = "not_implemented"
    splitter_status = (
        "walk_forward_train_val_test"
        if uses_lgbm and splitter_kind == "walk_forward"
        else "single_split_train_val_test"
        if uses_lgbm
        else "configured_not_used_by_unsupervised_path"
    )
    lightgbm_status = (
        "walk_forward_multi_head_lgbm"
        if uses_lgbm and splitter_kind == "walk_forward"
        else "single_split_multi_head_lgbm"
        if uses_lgbm
        else "not_implemented"
    )
    return {
        "feature_schema": deepcopy(config.get("features", {})),
        "factor_schema": {},
        "model_schema": deepcopy(config.get("model", {})),
        "regime": deepcopy(config.get("regime", {})),
        "signals": {},
        "preprocessor": deepcopy(config.get("preprocessor", {})),
        "splitter": deepcopy(config.get("splitter", {})),
        "view_fusion": deepcopy(config.get("view_fusion", {})),
        "risk_model": deepcopy(config.get("risk_model", {})),
        "optimizer": deepcopy(config.get("optimizer", {})),
        "allocation": deepcopy(config.get("allocation", {})),
        "constraints": deepcopy(config.get("constraints", {})),
        "costs": deepcopy(config.get("costs", {})),
        "stop_loss": deepcopy(config.get("stop_loss", {})),
        "data": deepcopy(config.get("data", {})),
        "execution": deepcopy(config.get("execution", {})),
        "implementation_status": {
            "universe": "personal_etf_first_layered_universe",
            "regime_detector": "market_regime_etf_proxy" if config.get("regime", {}).get("enabled") else "disabled",
            "features": "unified_feature_view",
            "factors": "removed_unified_feature_view",
            "model": model_kind,
            "preprocessor": "external_cross_sectional_feature_preprocessor",
            "splitter": splitter_status,
            "walk_forward": "implemented" if splitter_kind == "walk_forward" else "not_used_by_current_profile",
            "view_fusion": "noop_adjusted_return",
            "risk_model": risk_model_status,
            "optimizer": "opportunity_pool_optimizer_lot_aware_execution",
            "constraints": "weight_turnover_industry_lot_order_constraints_enforced",
            "execution": "rebalance_interval_net_alpha_lot_aware_order_smoothing",
            "labels": "etf_peer_relative_net_return_training_labels",
            "price_adjustment": "explicit_adjusted_price_contract",
            "data_quality": "raw_panel_and_feature_matrix_quality_gate",
            "evaluation": "diagnostic_rank_ic_top_bucket_portfolio_risk_turnover_cost_order_burden_metrics",
            "calibration": "implemented" if uses_lgbm else "not_used_by_current_profile",
            "promotion_gate": "implemented" if uses_lgbm else "not_used_by_current_profile",
            "promotion_gate_contract": "tradability_gate_plus_research_support_gate" if uses_lgbm else "not_used_by_current_profile",
            "tradable_outcome": "implemented_net_equity_cost_once_contract",
            "alpha_transfer_ledger": "implemented",
            "confidence_diagnostic": "implemented_post_replay_tradable_net_success" if uses_lgbm else "not_used_by_current_profile",
            "lightgbm_training": lightgbm_status,
            "black_litterman_posterior": "not_implemented",
            "factor_risk_model": factor_risk_status,
            "execution_t_plus_one": (
                "implemented_enabled"
                if bool(execution_cfg.get("t_plus_one_lock", False))
                else "implemented_available"
            ),
            "execution_capacity": "implemented_enabled" if capacity_enabled else "implemented_available",
        },
    }


def _normalize_lgbm_params(config: dict[str, Any]) -> None:
    model = config.setdefault("model", {})
    if str(model.get("kind", "lgbm_multi_target")) != "lgbm_multi_target":
        return
    seed = int((config.get("experiment") or {}).get("seed", DEFAULT_EXPERIMENT_SEED))
    params = dict(DEFAULT_LGBM_PARAMS)
    params.update(model.get("params") or {})
    params.setdefault("seed", seed)
    params.setdefault("feature_fraction_seed", seed)
    params.setdefault("bagging_seed", seed)
    params.setdefault("data_random_seed", seed)
    params.setdefault("drop_seed", seed)
    params["seed"] = int(params["seed"])
    params["feature_fraction_seed"] = int(params["feature_fraction_seed"])
    params["bagging_seed"] = int(params["bagging_seed"])
    params["data_random_seed"] = int(params["data_random_seed"])
    params["drop_seed"] = int(params["drop_seed"])
    model["params"] = params
    policy = deepcopy(DEFAULT_LGBM_PARAM_POLICY)
    policy.update(model.get("param_policy") or {})
    model["param_policy"] = policy


def _validate_lgbm_params(params: dict[str, Any]) -> None:
    required = {
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "subsample_freq",
        "colsample_bytree",
        "max_depth",
        "seed",
        "feature_fraction_seed",
        "bagging_seed",
        "data_random_seed",
    }
    missing = sorted(key for key in required if key not in params)
    if missing:
        raise ValueError(f"lgbm_multi_target params missing required anti-overfit/reproducibility keys: {missing}")
    if int(params.get("min_child_samples", 0)) < 30:
        raise ValueError("lgbm_multi_target params.min_child_samples must be >= 30")
    if float(params.get("reg_alpha", 0.0)) <= 0:
        raise ValueError("lgbm_multi_target params.reg_alpha must be positive")
    if float(params.get("reg_lambda", 0.0)) <= 0:
        raise ValueError("lgbm_multi_target params.reg_lambda must be positive")
    if not 0.0 < float(params.get("subsample", 0.0)) <= 1.0:
        raise ValueError("lgbm_multi_target params.subsample must be in (0, 1]")
    if int(params.get("subsample_freq", 0)) < 1:
        raise ValueError("lgbm_multi_target params.subsample_freq must be positive")
    if not 0.0 < float(params.get("colsample_bytree", 0.0)) <= 1.0:
        raise ValueError("lgbm_multi_target params.colsample_bytree must be in (0, 1]")
    if int(params.get("max_depth", 0)) <= 0:
        raise ValueError("lgbm_multi_target params.max_depth must be positive")
    if int(params.get("early_stopping_rounds", 0)) < 10:
        raise ValueError("lgbm_multi_target params.early_stopping_rounds must be >= 10")
    if int(params.get("num_threads", 1)) != 1:
        raise ValueError("lgbm_multi_target params.num_threads must be 1 for reproducibility")
    if not bool(params.get("deterministic", False)):
        raise ValueError("lgbm_multi_target params.deterministic must be true")


def _validate_lgbm_param_policy(policy: dict[str, Any]) -> None:
    for name in ("min_child_samples", "learning_rate", "reg_lambda"):
        payload = policy.get(name)
        if not isinstance(payload, dict):
            raise ValueError(f"model.param_policy.{name} must be declared")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"model.param_policy.{name}.candidates must be non-empty")
        warning_range = payload.get("warning_range")
        hard_range = payload.get("hard_range")
        for range_name, values in (("warning_range", warning_range), ("hard_range", hard_range)):
            if not isinstance(values, list) or len(values) != 2:
                raise ValueError(f"model.param_policy.{name}.{range_name} must be a [low, high] pair")
            low, high = float(values[0]), float(values[1])
            if low > high:
                raise ValueError(f"model.param_policy.{name}.{range_name} must be ordered")


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
