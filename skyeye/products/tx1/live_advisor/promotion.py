# -*- coding: utf-8 -*-
"""TX1 live advisor 的 promotion gate 与产包提升逻辑。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from skyeye.products.tx1.artifacts import parse_artifact_line, resolve_artifact
from skyeye.products.tx1.baseline_models import (
    create_model,
    create_multi_head_model,
    dump_model_bundle,
    supports_validation,
)
from skyeye.products.tx1.config import normalize_config
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import FEATURE_COLUMNS
from skyeye.products.tx1.experiment_runner import ExperimentRunner
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.live_advisor.calibration import build_calibration_bundle
from skyeye.products.tx1.live_advisor.package_io import build_live_package_payload, save_live_package
from skyeye.products.tx1.persistence import load_experiment
from skyeye.products.tx1.preprocessor import FeaturePreprocessor
from skyeye.products.tx1.run_baseline_experiment import (
    build_data_dependency_summary,
    build_live_raw_df,
)
from skyeye.products.tx1.splitter import WalkForwardSplitter


PROMOTION_THRESHOLDS = {
    "min_folds": 12,
    "min_bucket_samples": 300,
    "min_rank_ic_mean": 0.03,
    "min_top_bucket_spread_mean": 0.005,
    "min_positive_ratio": 0.70,
    "max_cv_default_live": 0.60,
    "min_stability_score_default_live": 50.0,
    "max_drawdown": 0.12,
    "max_mean_turnover": 0.20,
}

DEFAULT_EXPERIMENTS_ROOT = (
    Path(__file__).resolve().parents[4] / "artifacts" / "experiments" / "tx1"
)


def evaluate_promotion_gate(experiment_result: dict, calibration_bundle: dict) -> dict:
    """根据研究实验与校准包结果判断 package 是否可提升。"""
    aggregate_metrics = experiment_result.get("aggregate_metrics", {})
    prediction_metrics = aggregate_metrics.get("prediction", {})
    portfolio_metrics = aggregate_metrics.get("portfolio", {})
    robustness = aggregate_metrics.get("robustness", {})
    stability = robustness.get("stability", {})
    overfit_flags = robustness.get("overfit_flags", {})
    regime_scores = robustness.get("regime_scores", {})
    metric_consistency = regime_scores.get("metric_consistency", {})
    num_folds = int(
        experiment_result.get("num_folds")
        or len(experiment_result.get("fold_results", []))
    )
    min_bucket_samples = min(
        [int(bucket.get("sample_count", 0)) for bucket in calibration_bundle.get("bucket_stats", [])] or [0]
    )

    checks = {
        "num_folds": _build_check(num_folds, ">=", PROMOTION_THRESHOLDS["min_folds"]),
        "bucket_sample_count": _build_check(
            min_bucket_samples,
            ">=",
            PROMOTION_THRESHOLDS["min_bucket_samples"],
        ),
        "rank_ic_mean": _build_check(
            float(prediction_metrics.get("rank_ic_mean", 0.0)),
            ">=",
            PROMOTION_THRESHOLDS["min_rank_ic_mean"],
        ),
        "top_bucket_spread_mean": _build_check(
            float(prediction_metrics.get("top_bucket_spread_mean", 0.0)),
            ">=",
            PROMOTION_THRESHOLDS["min_top_bucket_spread_mean"],
        ),
        "flag_ic_decay": _build_flag_check(overfit_flags.get("flag_ic_decay", False), expected=False),
        "flag_spread_decay": _build_flag_check(overfit_flags.get("flag_spread_decay", False), expected=False),
        "flag_val_dominant": _build_flag_check(overfit_flags.get("flag_val_dominant", False), expected=False),
        "positive_ratio": _build_check(
            float(metric_consistency.get("positive_ratio", 0.0)),
            ">=",
            PROMOTION_THRESHOLDS["min_positive_ratio"],
        ),
        "net_mean_return": _build_check(
            float(portfolio_metrics.get("net_mean_return", 0.0)),
            ">",
            0.0,
        ),
        "max_drawdown": _build_check(
            float(portfolio_metrics.get("max_drawdown", 1.0)),
            "<=",
            PROMOTION_THRESHOLDS["max_drawdown"],
        ),
        "mean_turnover": _build_check(
            float(portfolio_metrics.get("mean_turnover", 1.0)),
            "<=",
            PROMOTION_THRESHOLDS["max_mean_turnover"],
        ),
        "stability_score": _build_check(
            float(stability.get("stability_score", 0.0)),
            ">=",
            PROMOTION_THRESHOLDS["min_stability_score_default_live"],
        ),
        "cv": _build_check(
            float(stability.get("cv", 999.0)),
            "<=",
            PROMOTION_THRESHOLDS["max_cv_default_live"],
        ),
    }

    canary_required = (
        "num_folds",
        "bucket_sample_count",
        "rank_ic_mean",
        "top_bucket_spread_mean",
        "flag_ic_decay",
        "flag_spread_decay",
        "flag_val_dominant",
        "positive_ratio",
        "net_mean_return",
        "max_drawdown",
        "mean_turnover",
    )
    canary_passed = all(checks[check_name]["passed"] for check_name in canary_required)
    default_live_passed = canary_passed and checks["stability_score"]["passed"] and checks["cv"]["passed"]

    if default_live_passed:
        gate_level = "default_live"
    elif canary_passed:
        gate_level = "canary_live"
    else:
        gate_level = "blocked"

    return {
        "passed": bool(canary_passed),
        "gate_level": gate_level,
        "canary_live_passed": bool(canary_passed),
        "default_live_passed": bool(default_live_passed),
        "checks": checks,
        "thresholds": dict(PROMOTION_THRESHOLDS),
    }


def promote_experiment_to_live_package(
    *,
    experiment_ref: str | None = None,
    experiment_result: dict | None = None,
    raw_df=None,
    package_id: str | None = None,
    packages_root: str | Path | None = None,
    bucket_count: int = 10,
) -> dict:
    """把研究实验提升为 live advisor 可消费的 promoted package。"""
    experiment_result = experiment_result or load_experiment_from_ref(experiment_ref)
    calibration_bundle = build_calibration_bundle(experiment_result, bucket_count=bucket_count)
    gate_summary = evaluate_promotion_gate(experiment_result, calibration_bundle)
    if not gate_summary["passed"]:
        raise ValueError("experiment does not pass live promotion gate")

    config = normalize_config(experiment_result.get("config") or {})
    if raw_df is None:
        raw_df = build_live_raw_df()
    training = fit_live_model_bundle(config, raw_df)
    recent_canary_bundle = build_recent_canary_bundle(
        config,
        raw_df,
        bucket_count=bucket_count,
        preprocessor_bundle=training["preprocessor_bundle"],
    )

    feature_schema = {
        "feature_columns": list(training["feature_columns"]),
        "label_horizon": int(config["labels"]["horizon"]),
        "model_kind": str(config["model"]["kind"]),
        "multi_output_enabled": bool(config.get("multi_output", {}).get("enabled", False)),
    }
    portfolio_policy = dict(config.get("portfolio", {}))
    canary_reason = _build_canary_reason(gate_summary)
    evidence_end_date = training["fit_end_date"]
    if isinstance(recent_canary_bundle, dict):
        evidence_end_date = str(
            (recent_canary_bundle.get("window") or {}).get("end_date")
            or training["fit_end_date"]
        )
    data_dependency_summary = build_data_dependency_summary(training["feature_columns"])
    freshness_policy = _build_default_freshness_policy(int(config["labels"]["horizon"]))

    manifest = {
        "package_id": package_id or build_package_id(experiment_result, gate_summary, training["fit_end_date"]),
        "package_type": gate_summary["gate_level"],
        "source_experiment": _resolve_source_experiment_name(experiment_result),
        "horizon": int(config["labels"]["horizon"]),
        "fit_end_date": training["fit_end_date"],
        "label_end_date": training["fit_end_date"],
        "data_end_date": training["data_end_date"],
        "evidence_end_date": evidence_end_date,
        "created_at": datetime.now().isoformat(),
        "model_kind": str(config["model"]["kind"]),
        "required_features": list(training["feature_columns"]),
        "hashes": {},
        "gate_summary": gate_summary,
        "canary_reason": canary_reason,
        "data_dependency_summary": data_dependency_summary,
        "freshness_policy": freshness_policy,
        "universe_id": str(config.get("universe", {}).get("universe_id", "liquid_top_300")),
    }

    component_payloads = {
        "feature_schema": feature_schema,
        "preprocessor_bundle": training["preprocessor_bundle"],
        "model_bundle": training["model_bundle"],
        "calibration_bundle": calibration_bundle,
        "portfolio_policy": portfolio_policy,
    }
    if recent_canary_bundle is not None:
        component_payloads["recent_canary_bundle"] = recent_canary_bundle
    manifest["hashes"] = {
        component_name: _stable_hash(component_payloads[component_name])
        for component_name in component_payloads
    }

    payload = build_live_package_payload(
        manifest=manifest,
        feature_schema=feature_schema,
        preprocessor_bundle=training["preprocessor_bundle"],
        model_bundle=training["model_bundle"],
        calibration_bundle=calibration_bundle,
        portfolio_policy=portfolio_policy,
        recent_canary_bundle=recent_canary_bundle,
    )
    package_root = save_live_package(payload, packages_root=packages_root)
    payload["package_root"] = str(package_root)
    return payload


def load_experiment_from_ref(
    experiment_ref: str | None,
    *,
    artifacts_root: str | Path | None = None,
) -> dict:
    """兼容路径、完整 artifact ref 和 bare artifact line 的实验加载入口。"""
    if not experiment_ref:
        raise ValueError("experiment_ref must not be empty")
    path_ref = Path(experiment_ref)
    if path_ref.is_absolute() or "/" in str(experiment_ref) or str(experiment_ref).startswith("."):
        if path_ref.is_file():
            return load_experiment(str(path_ref.parent))
        return load_experiment(str(path_ref))

    if "@" in str(experiment_ref):
        parsed = parse_artifact_line(str(experiment_ref))
    else:
        parsed = parse_artifact_line("tx1.rolling_score@{}".format(experiment_ref))
    resolved = resolve_artifact(parsed, Path(artifacts_root or DEFAULT_EXPERIMENTS_ROOT))
    return load_experiment(str(resolved.artifact_root))


def fit_live_model_bundle(config: dict, raw_df) -> dict:
    """基于最新可得标签数据训练 final-fit live model，并导出 bundle。"""
    if raw_df is None or len(raw_df) == 0:
        raise ValueError("raw_df must not be empty")
    dataset_builder = DatasetBuilder(input_window=config["dataset"]["input_window"])
    label_builder = LabelBuilder(
        horizon=config["labels"]["horizon"],
        transform=config["labels"]["transform"],
        winsorize=config["labels"].get("winsorize"),
        target_config={
            "volatility": {
                "transform": config["multi_output"]["volatility"]["transform"],
            },
            "max_drawdown": {
                "transform": config["multi_output"]["max_drawdown"]["transform"],
            },
        },
    )
    dataset = dataset_builder.build(raw_df)
    labeled = label_builder.build(dataset)
    if labeled.empty:
        raise ValueError("labeled dataset must not be empty for live model fit")

    feature_columns = [column for column in FEATURE_COLUMNS if column in labeled.columns]
    preprocessor = _build_preprocessor_from_config(config)
    train_df = labeled.copy()
    if preprocessor is not None:
        train_df = preprocessor.transform(train_df, feature_columns)
        preprocessor_bundle = preprocessor.to_bundle(feature_columns)
    else:
        preprocessor_bundle = FeaturePreprocessor(
            neutralize=False,
            winsorize_scale=None,
            standardize=False,
        ).to_bundle(feature_columns)

    model = _fit_model(config, train_df, feature_columns)
    model_bundle = dump_model_bundle(
        model,
        model_kind=config["model"]["kind"],
        feature_columns=feature_columns,
    )
    model_bundle["prediction_config"] = {
        "multi_output_enabled": bool(config.get("multi_output", {}).get("enabled", False)),
        "combine_auxiliary": bool(config.get("multi_output", {}).get("prediction", {}).get("combine_auxiliary", False)),
        "volatility_weight": float(config.get("multi_output", {}).get("prediction", {}).get("volatility_weight", 0.0)),
        "max_drawdown_weight": float(config.get("multi_output", {}).get("prediction", {}).get("max_drawdown_weight", 0.0)),
        "reliability_enabled": bool(config.get("multi_output", {}).get("reliability_score", {}).get("enabled", False)),
    }

    return {
        "feature_columns": feature_columns,
        "preprocessor_bundle": preprocessor_bundle,
        "model_bundle": model_bundle,
        "fit_end_date": str(pd.Timestamp(labeled["date"].max()).date()),
        "data_end_date": str(pd.Timestamp(pd.to_datetime(raw_df["date"]).max()).date()),
    }


def build_package_id(experiment_result: dict, gate_summary: dict, fit_end_date: str) -> str:
    """构造稳定的 package_id，便于后续 registry 和审计。"""
    source = _resolve_source_experiment_name(experiment_result)
    suffix = str(fit_end_date).replace("-", "")
    return "tx1_{}_{}_{}".format(gate_summary["gate_level"], source, suffix)


def build_recent_canary_bundle(
    config: dict,
    raw_df,
    *,
    bucket_count: int = 10,
    recent_window_days: int = 60,
    min_recent_days: int = 20,
    preprocessor_bundle: dict | None = None,
) -> dict | None:
    """构建近端 canary 证据，避免只依赖旧 OOS 桶解释当前市场。"""
    dataset_builder = DatasetBuilder(input_window=config["dataset"]["input_window"])
    label_builder = LabelBuilder(
        horizon=config["labels"]["horizon"],
        transform=config["labels"]["transform"],
        winsorize=config["labels"].get("winsorize"),
        target_config={
            "volatility": {
                "transform": config["multi_output"]["volatility"]["transform"],
            },
            "max_drawdown": {
                "transform": config["multi_output"]["max_drawdown"]["transform"],
            },
        },
    )
    dataset = dataset_builder.build(raw_df)
    labeled = label_builder.build(dataset)
    if labeled.empty:
        return None

    unique_dates = sorted(pd.to_datetime(labeled["date"]).dt.normalize().unique())
    if len(unique_dates) < max(int(min_recent_days) + 1, 2):
        return None

    window_days = min(int(recent_window_days), len(unique_dates) - 1)
    if window_days < int(min_recent_days):
        window_days = min(int(min_recent_days), len(unique_dates) - 1)
    recent_start_date = pd.Timestamp(unique_dates[-window_days]).normalize()
    recent_end_date = pd.Timestamp(unique_dates[-1]).normalize()

    train_df = labeled[pd.to_datetime(labeled["date"]).dt.normalize() < recent_start_date].copy()
    canary_df = labeled[pd.to_datetime(labeled["date"]).dt.normalize() >= recent_start_date].copy()
    if train_df.empty or canary_df.empty:
        return None

    feature_columns = [column for column in FEATURE_COLUMNS if column in labeled.columns]
    preprocessor = (
        FeaturePreprocessor.from_bundle(preprocessor_bundle)
        if isinstance(preprocessor_bundle, dict)
        else _build_preprocessor_from_config(config)
    )
    train_features = train_df.copy()
    canary_features = canary_df.copy()
    if preprocessor is not None:
        train_features = preprocessor.transform(train_features, feature_columns)
        canary_features = preprocessor.transform(canary_features, feature_columns)

    model = _fit_model(config, train_features, feature_columns)
    predictions = model.predict(canary_features[feature_columns])
    canary_predictions = _build_scored_prediction_frame(canary_features, predictions, config)
    calibration_bundle = build_calibration_bundle(
        {"fold_results": [{"predictions_df": canary_predictions}]},
        bucket_count=bucket_count,
    )
    calibration_bundle["window"] = {
        "start_date": str(recent_start_date.date()),
        "end_date": str(recent_end_date.date()),
    }
    calibration_bundle["summary"].update(
        {
            "n_rows": int(len(canary_predictions)),
            "n_days": int(canary_predictions["date"].nunique()),
            "shadow_fit_end_date": str(pd.Timestamp(train_df["date"].max()).date()),
        }
    )
    return calibration_bundle


def _build_canary_reason(gate_summary: dict) -> list[str]:
    """把 canary 的阻塞原因沉淀到 manifest，便于 runtime 解释。"""
    if str(gate_summary.get("gate_level")) != "canary_live":
        return []
    checks = gate_summary.get("checks", {})
    reasons = [
        check_name
        for check_name in ("stability_score", "cv")
        if not checks.get(check_name, {}).get("passed", True)
    ]
    if reasons:
        return reasons
    return [
        check_name
        for check_name, check_result in checks.items()
        if not check_result.get("passed", True)
    ]


def _build_default_freshness_policy(horizon: int) -> dict:
    """给 live package 生成默认 freshness 策略。"""
    horizon_days = max(int(horizon), 1)
    return {
        "snapshot_max_delay_days": 1,
        "model_warning_days": horizon_days,
        "model_stop_days": horizon_days * 2,
        "evidence_warning_days": horizon_days,
        "evidence_stop_days": horizon_days * 2,
    }


def _build_check(actual: float, operator: str, threshold: float) -> dict:
    """构造统一的数值闸门检查结果。"""
    comparator = {
        ">=": lambda left, right: left >= right,
        ">": lambda left, right: left > right,
        "<=": lambda left, right: left <= right,
        "<": lambda left, right: left < right,
    }.get(operator)
    if comparator is None:
        raise ValueError("unsupported operator: {}".format(operator))
    return {
        "actual": actual,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(comparator(actual, threshold)),
    }


def _build_flag_check(actual: bool, *, expected: bool) -> dict:
    """构造布尔闸门检查结果。"""
    return {
        "actual": bool(actual),
        "operator": "==",
        "threshold": bool(expected),
        "passed": bool(actual) is bool(expected),
    }


def _resolve_source_experiment_name(experiment_result: dict) -> str:
    """解析实验名，优先复用已有 experiment_name。"""
    name = experiment_result.get("experiment_name")
    if name:
        return str(name)
    config = experiment_result.get("config") or {}
    if config.get("experiment_name"):
        return str(config["experiment_name"])
    return "ad_hoc"


def _build_preprocessor_from_config(config: dict):
    """按实验配置恢复训练期的预处理器。"""
    preproc_cfg = config.get("preprocessing", {})
    if not preproc_cfg.get("enabled", False):
        return None
    return FeaturePreprocessor(
        neutralize=preproc_cfg.get("neutralize", True),
        winsorize_scale=preproc_cfg.get("winsorize_scale", 5.0),
        standardize=preproc_cfg.get("standardize", True),
    )


def _fit_model(config: dict, train_df: pd.DataFrame, feature_columns: list[str]):
    """按实验配置训练 final-fit 模型。"""
    if config.get("multi_output", {}).get("enabled", False):
        head_specs = _build_head_specs(config)
        target_columns = [spec["target_column"] for spec in head_specs.values()]
        model = create_multi_head_model(
            config["model"]["kind"],
            head_specs,
            params=config["model"].get("params"),
        )
        model.fit(train_df[feature_columns], train_df[target_columns])
        return model
    model = create_model(config["model"]["kind"], params=config["model"].get("params"))
    fit_kwargs = {}
    if supports_validation(model):
        fit_kwargs = {}
    model.fit(train_df[feature_columns], train_df["target_label"], **fit_kwargs)
    return model


def _build_head_specs(config: dict) -> dict:
    """根据配置生成多头训练所需的目标映射。"""
    runner = ExperimentRunner(
        config=config,
        dataset_builder=DatasetBuilder(input_window=config["dataset"]["input_window"]),
        label_builder=LabelBuilder(horizon=config["labels"]["horizon"], transform=config["labels"]["transform"]),
        splitter=WalkForwardSplitter(
            train_years=config["splitter"]["train_years"],
            val_months=config["splitter"]["val_months"],
            test_months=config["splitter"]["test_months"],
            embargo_days=config["splitter"]["embargo_days"],
        ),
        preprocessor=None,
    )
    return runner._build_head_specs()


def _build_scored_prediction_frame(frame: pd.DataFrame, predictions, config: dict) -> pd.DataFrame:
    """把模型输出整理成校准函数可直接消费的预测明细。"""
    result = frame.copy()
    if isinstance(predictions, dict):
        if "return" in predictions:
            result["prediction_ret"] = predictions["return"]
            result["prediction"] = predictions["return"]
        if "volatility" in predictions:
            result["prediction_vol"] = predictions["volatility"]
        if "max_drawdown" in predictions:
            result["prediction_mdd"] = predictions["max_drawdown"]
        prediction_cfg = config.get("multi_output", {}).get("prediction", {})
        if prediction_cfg.get("combine_auxiliary", False):
            combined = result.groupby("date")["prediction_ret"].rank(method="average", pct=True)
            if prediction_cfg.get("volatility_weight", 0.0) > 0 and "prediction_vol" in result.columns:
                combined = combined - prediction_cfg["volatility_weight"] * result.groupby("date")["prediction_vol"].rank(method="average", pct=True)
            if prediction_cfg.get("max_drawdown_weight", 0.0) > 0 and "prediction_mdd" in result.columns:
                combined = combined - prediction_cfg["max_drawdown_weight"] * result.groupby("date")["prediction_mdd"].rank(method="average", pct=True)
            result["prediction"] = combined
    else:
        result["prediction_ret"] = predictions
        result["prediction"] = predictions
    required_columns = [
        "date",
        "order_book_id",
        "prediction",
        "label_return_raw",
        "label_volatility_horizon",
        "label_max_drawdown_horizon",
    ]
    return result.loc[:, required_columns].copy()


def _stable_hash(payload: dict) -> str:
    """对 JSON 组件生成稳定 hash。"""
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()
