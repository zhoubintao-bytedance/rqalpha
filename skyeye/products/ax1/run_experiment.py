# -*- coding: utf-8 -*-
"""AX1 experiment runner."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from skyeye.products.ax1.config import (
    DEFAULT_PROFILE_PATH,
    build_component_manifest,
    load_profile,
    merge_config,
    normalize_config,
)
from skyeye.products.ax1.research.training import (
    build_splitter as _build_splitter,
    prediction_horizons as _prediction_horizons,
    run_lgbm_pipeline as _run_lgbm_pipeline,
)


def run_experiment(
    *,
    profile_path: str | Path | None = None,
    output_dir: str | Path,
    raw_df: pd.DataFrame | None = None,
    raw_csv: str | Path | None = None,
    experiment_name: str | None = None,
    config_override: dict[str, Any] | None = None,
    data_provider=None,
) -> dict[str, Any]:
    """运行 AX1 research pipeline，并保存 experiment artifact。"""
    from skyeye.products.ax1.data_quality import build_data_version
    from skyeye.products.ax1.evaluation.metrics import evaluate_signal_layer
    from skyeye.products.ax1.features import AX1FeatureViewBuilder, resolve_feature_columns
    from skyeye.products.ax1.persistence import save_experiment
    from skyeye.products.ax1.regime import RegimeDetector
    from skyeye.products.ax1.universe import DynamicUniverseBuilder

    base_config = load_profile(profile_path or DEFAULT_PROFILE_PATH)
    config = normalize_config(merge_config(base_config, config_override)) if config_override is not None else base_config
    _set_global_seed(config)
    frame = _load_raw_frame(raw_df=raw_df, raw_csv=raw_csv)
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    raw_data_quality = _build_raw_data_quality(frame, config)
    _enforce_quality_if_enabled(raw_data_quality, config, context="raw_data_quality")

    as_of_date = frame["date"].max()
    universe_builder = DynamicUniverseBuilder()
    universe_metadata = universe_builder.build_with_metadata(
        frame,
        as_of_date=as_of_date,
        config=config.get("universe", {}),
        data_provider=data_provider,
        purpose="promotable_training",
    )
    universe_audit = _audit_universe_contract(universe_metadata, purpose="promotable_training")
    if not universe_audit.get("passed", False):
        reason_codes = [
            str(item.get("reason_code"))
            for item in universe_audit.get("hard_blocks", [])
        ]
        raise ValueError(f"AX1 universe PIT audit failed: {reason_codes}")
    universe = universe_metadata["order_book_id"].astype(str).tolist() if not universe_metadata.empty else []
    scoped = frame[frame["order_book_id"].isin(universe)].copy()
    scoped = _attach_universe_metadata(scoped, universe_metadata)
    industry_map = _build_industry_map(scoped, as_of_date=as_of_date, data_provider=data_provider)
    regime_state = _detect_regime(
        RegimeDetector,
        config,
        scoped,
        as_of_date=as_of_date,
        universe_metadata=universe_metadata,
        data_provider=data_provider,
    )
    regime_state_by_date = _detect_regime_by_date(
        RegimeDetector,
        config,
        scoped,
        universe_metadata=universe_metadata,
        data_provider=data_provider,
    )
    label_builder = _build_label_builder(config)
    model_kind = str(config.get("model", {}).get("kind", "lgbm_multi_target"))
    fundamental_df, flow_df, macro_df, scoped = _load_external_data_panels(
        scoped, universe_metadata, as_of_date=as_of_date, data_provider=data_provider,
    )
    feature_view = AX1FeatureViewBuilder(_feature_view_config(config)).build(
        scoped,
        universe_metadata=universe_metadata,
        regime_state=regime_state,
        regime_state_by_date=regime_state_by_date,
        fundamental_df=fundamental_df,
        flow_df=flow_df,
        macro_df=macro_df,
    )
    feature_frame = feature_view.frame
    labeled = label_builder.build(feature_frame)
    feature_columns = resolve_feature_columns(config, feature_view)
    data_version = build_data_version(frame, feature_columns=feature_columns)
    feature_catalog, data_audit = _audit_feature_contract(feature_columns, purpose="promotable_training")
    if not data_audit.get("passed", False):
        reason_codes = [
            str(item.get("reason_code"))
            for item in data_audit.get("hard_blocks", [])
        ]
        raise ValueError(f"AX1 feature data audit failed: {reason_codes}")
    feature_data_quality = _build_feature_data_quality(labeled, feature_columns, config)
    _enforce_quality_if_enabled(feature_data_quality, config, context="feature_data_quality")
    if model_kind != "lgbm_multi_target":
        raise ValueError(f"unsupported AX1 model kind: {model_kind}")
    predictions, evaluation_labels, training_summary = _run_lgbm_pipeline(
        config,
        labeled,
        feature_columns=feature_columns,
    )

    fused = _build_view_fusion(config).fuse(predictions)
    fused = _attach_industry(fused, industry_map)
    fused = _attach_universe_metadata(fused, universe_metadata)
    from skyeye.products.ax1.research.execution import run_portfolio_replay

    replay_result = run_portfolio_replay(
        config=config,
        fused_predictions=fused,
        scoped_raw=scoped,
        evaluation_labels=evaluation_labels,
        industry_map=industry_map,
        universe_metadata=universe_metadata,
        regime_state=regime_state,
        regime_state_by_date=regime_state_by_date,
        as_of_date=as_of_date,
    )
    allocation_config = replay_result["allocation_config"]
    allocation_config_by_date = replay_result["allocation_config_by_date"]
    smoothed_weights = replay_result["target_weights"]
    orders = replay_result["orders"]
    execution_summary = replay_result["execution_summary"]
    portfolio_metrics = replay_result["portfolio_metrics"]
    effective_breadth_summary = replay_result["effective_breadth_summary"]
    tradable_outcome = replay_result["tradable_outcome"]
    alpha_transfer_ledger = replay_result["alpha_transfer_ledger"]
    stop_loss_result = replay_result.get("stop_loss_result", {})

    signal_metrics = evaluate_signal_layer(predictions, evaluation_labels)
    from skyeye.products.ax1.parameter_validation import build_parameter_validation_summary

    parameter_validation_summary = build_parameter_validation_summary(config, predictions, evaluation_labels)
    training_summary = _attach_effective_breadth_to_robustness(training_summary, effective_breadth_summary)
    constraint_status = _constraint_status(portfolio_metrics["portfolio"].get("constraint_violations", {}))
    component_manifest = build_component_manifest(config)
    result = {
        "product": "ax1",
        "schema_version": config["schema_version"],
        "status": "ok" if constraint_status == "ok" else "constraint_warning",
        "constraint_status": constraint_status,
        "experiment_name": experiment_name or config.get("experiment", {}).get("name", "ax1_lgbm"),
        "config": config,
        "component_manifest": component_manifest,
        "feature_schema": component_manifest["feature_schema"],
        "implementation_status": component_manifest["implementation_status"],
        "factor_schema": component_manifest["factor_schema"],
        "feature_catalog": feature_catalog.to_dict(),
        "data_audit": data_audit,
        "raw_data_quality": raw_data_quality,
        "feature_data_quality": feature_data_quality,
        "data_version": data_version,
        "price_adjustment": raw_data_quality.get("price_adjustment", {}),
        "universe_audit": universe_audit,
        "model_schema": component_manifest["model_schema"],
        "risk_model": component_manifest["risk_model"],
        "optimizer": component_manifest["optimizer"],
        "constraints": component_manifest["constraints"],
        "costs": component_manifest["costs"],
        "data_range": {
            "start": str(frame["date"].min().date()),
            "end": str(frame["date"].max().date()),
            "rows": int(len(frame)),
            "universe_size": int(len(universe)),
        },
        "universe": universe,
        "universe_summary": _universe_summary(universe_metadata),
        "effective_breadth_summary": effective_breadth_summary,
        "regime_state": regime_state,
        "regime_state_by_date_count": int(len(regime_state_by_date)),
        "allocation_config": allocation_config,
        "allocation_config_by_date_count": int(len(allocation_config_by_date)) if allocation_config else 0,
        "training_summary": training_summary,
        "prediction_summary": _prediction_summary(predictions),
        "execution_summary": execution_summary,
        "parameter_validation_summary": parameter_validation_summary,
        "tradable_outcome": tradable_outcome,
        "alpha_transfer_ledger": alpha_transfer_ledger,
        "stop_loss": {
            "enabled": bool(config.get("stop_loss", {}).get("enabled", False)),
            "trigger_count": stop_loss_result.get("stop_loss_trigger_count", 0),
            "trigger_log": stop_loss_result.get("stop_loss_log", []),
        },
        "orders": orders.to_dict("records"),
        "target_weights": smoothed_weights.to_dict("records"),
        "evaluation": {
            "signal": signal_metrics["signal"],
            "portfolio": portfolio_metrics["portfolio"],
        },
    }
    from skyeye.products.ax1.calibration import build_calibration_bundle
    from skyeye.products.ax1.confidence import build_tradable_confidence_diagnostic
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result["confidence_diagnostic"] = build_tradable_confidence_diagnostic(
        predictions,
        tradable_outcome,
        bucket_count=int(config.get("model", {}).get("confidence_bucket_count", 5)),
        min_samples=int(config.get("model", {}).get("confidence_min_samples", 30)),
    )
    result["calibration_bundle"] = build_calibration_bundle(result, bucket_count=10)
    result["gate_summary"] = evaluate_promotion_gate(result)
    result["training_readiness"] = _build_training_readiness(result)
    saved_dir = save_experiment(result, output_dir, experiment_name=result["experiment_name"])
    result["output_dir"] = str(saved_dir)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AX1 experiment pipeline")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE_PATH), help="AX1 YAML profile path")
    parser.add_argument("--raw-csv", required=True, help="Raw OHLCV csv path")
    parser.add_argument("--output-dir", required=True, help="Experiment output root")
    parser.add_argument("--experiment-name", default=None, help="Optional experiment name")
    return parser


def _set_global_seed(config: dict[str, Any]) -> int:
    import random

    import numpy as np

    seed = int((config.get("experiment") or {}).get("seed", 0) or 0)
    if seed > 0:
        random.seed(seed)
        np.random.seed(seed)
    return seed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_experiment(
        profile_path=args.profile,
        raw_csv=args.raw_csv,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
    )
    print(result["output_dir"])
    return 0


def _load_raw_frame(*, raw_df: pd.DataFrame | None, raw_csv: str | Path | None) -> pd.DataFrame:
    if raw_df is not None:
        frame = raw_df.copy()
    elif raw_csv is not None:
        frame = pd.read_csv(raw_csv)
    else:
        raise ValueError("raw_df or raw_csv is required")
    missing = [column for column in ("date", "order_book_id", "close") if column not in frame.columns]
    if missing:
        raise ValueError("raw frame missing required columns: {}".format(", ".join(missing)))
    return frame


def _load_external_data_panels(
    scoped: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    data_provider=None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame]:
    """Load fundamental, flow, and macro data panels when data_provider is available."""
    if data_provider is None:
        return None, None, None, scoped

    from skyeye.data.facade import DataFacade
    from skyeye.products.ax1.data_sources.flow import FlowDataSource
    from skyeye.products.ax1.data_sources.fundamental import FundamentalDataSource
    from skyeye.products.ax1.data_sources.macro import MacroDataSource

    facade = data_provider if isinstance(data_provider, DataFacade) else DataFacade()
    start_date = scoped["date"].min()
    universe = universe_metadata["order_book_id"].astype(str).tolist() if not universe_metadata.empty else []
    stock_ids = [oid for oid in universe if FlowDataSource._is_stock(oid)]

    fundamental_df = None
    if stock_ids:
        try:
            fundamental_df = FundamentalDataSource().load_panel(
                order_book_ids=stock_ids,
                start_date=start_date,
                end_date=as_of_date,
                data_facade=facade,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load fundamental data: {exc}")

    flow_df = None
    if universe:
        try:
            flow_df = FlowDataSource().load_panel(
                order_book_ids=universe,
                start_date=start_date,
                end_date=as_of_date,
                data_facade=facade,
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load flow data: {exc}")

    macro_df = None
    try:
        macro_df = MacroDataSource().load_panel(
            order_book_ids=universe,
            start_date=start_date,
            end_date=as_of_date,
            data_facade=facade,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load macro data: {exc}")

    # Load turnover rate and merge into scoped (raw_df)
    try:
        turnover_df = facade.get_turnover_rate(
            order_book_ids=universe,
            start_date=start_date,
            end_date=as_of_date,
        )
        if turnover_df is not None and not turnover_df.empty:
            turnover_df["date"] = pd.to_datetime(turnover_df["date"])
            turnover_df["order_book_id"] = turnover_df["order_book_id"].astype(str)
            # Keep only relevant columns
            turnover_cols = ["date", "order_book_id"]
            for col in ["today", "turnover_rate"]:
                if col in turnover_df.columns:
                    turnover_cols.append(col)
            scoped = scoped.merge(
                turnover_df[turnover_cols].drop_duplicates(["date", "order_book_id"], keep="last"),
                on=["date", "order_book_id"],
                how="left",
            )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f"Failed to load turnover rate data: {exc}")

    return fundamental_df, flow_df, macro_df, scoped


def _detect_regime(
    detector_cls,
    config: dict[str, Any],
    raw_df: pd.DataFrame,
    *,
    as_of_date: pd.Timestamp,
    universe_metadata: pd.DataFrame,
    data_provider=None,
) -> dict[str, Any]:
    regime_cfg = config.get("regime", {})
    if not bool(regime_cfg.get("enabled", False)):
        return {
            "as_of_date": pd.Timestamp(as_of_date),
            "market_regime": str(regime_cfg.get("fallback_regime", "range_co_move")),
            "strength": 0.0,
            "risk_state": "neutral",
            "style_state": "balanced",
            "volatility_state": "unknown",
            "rotation_state": "co_move",
            "source": "disabled",
            "benchmark_proxy_ids": [],
            "diagnostics": {"reason": "disabled"},
        }
    return detector_cls(regime_cfg).detect(
        raw_df,
        as_of_date=as_of_date,
        universe_metadata=universe_metadata,
        data_provider=data_provider,
    )


def _detect_regime_by_date(
    detector_cls,
    config: dict[str, Any],
    raw_df: pd.DataFrame,
    *,
    universe_metadata: pd.DataFrame,
    data_provider=None,
) -> dict[pd.Timestamp, dict[str, Any]]:
    regime_cfg = config.get("regime", {})
    if raw_df is None or raw_df.empty or "date" not in raw_df.columns:
        return {}
    dates = sorted(pd.to_datetime(raw_df["date"].dropna().unique()))
    if not bool(regime_cfg.get("enabled", False)):
        return {
            pd.Timestamp(date): {
                "as_of_date": pd.Timestamp(date),
                "market_regime": str(regime_cfg.get("fallback_regime", "range_co_move")),
                "strength": 0.0,
                "risk_state": "neutral",
                "style_state": "balanced",
                "volatility_state": "unknown",
                "rotation_state": "co_move",
                "source": "disabled",
                "benchmark_proxy_ids": [],
                "diagnostics": {"reason": "disabled"},
            }
            for date in dates
        }
    return detector_cls(regime_cfg).detect_by_date(
        raw_df,
        as_of_dates=dates,
        universe_metadata=universe_metadata,
        data_provider=data_provider,
    )


def _feature_view_config(config: dict[str, Any]) -> dict[str, Any]:
    feature_config = dict(config.get("features", {}) or {})
    regime_config = dict(config.get("regime", {}) or {})
    model_config = dict(config.get("model", {}) or {})
    if model_config.get("include_scopes"):
        feature_config["model_include_scopes"] = list(model_config.get("include_scopes") or [])
    if not any(key in feature_config for key in ("core_proxy_id", "benchmark_id", "preferred_benchmark_ids")):
        preferred = regime_config.get("preferred_benchmark_ids")
        if preferred:
            feature_config["preferred_benchmark_ids"] = list(preferred)
    return feature_config


def _opportunity_pool_config(config: dict[str, Any], regime_state: dict[str, Any] | None = None) -> dict[str, Any]:
    allocation = dict(config.get("allocation", {}) or {})
    if allocation.get("kind") != "opportunity_pool_optimizer":
        raise ValueError("allocation.kind must be opportunity_pool_optimizer")
    if "layers" in allocation:
        raise ValueError("allocation.layers is no longer supported")
    constraints = config.get("constraints", {}) or {}
    configured_cash_buffer = float(allocation.get("cash_buffer", constraints.get("cash_buffer", 0.0)))
    execution_drift_buffer = float(allocation.get("execution_drift_buffer", 0.0))
    allocation["configured_cash_buffer"] = configured_cash_buffer
    allocation["execution_drift_buffer"] = execution_drift_buffer
    allocation["cash_buffer"] = configured_cash_buffer + execution_drift_buffer
    allocation["target_gross_exposure"] = float(
        allocation.get("target_gross_exposure", constraints.get("target_gross_exposure", 1.0))
    )
    allocation["regime_state"] = _opportunity_pool_regime_snapshot(regime_state)
    return allocation


def _opportunity_pool_config_by_date(
    config: dict[str, Any],
    *,
    dates,
    regime_state_by_date: dict[Any, dict[str, Any]] | None,
    fallback_regime_state: dict[str, Any] | None = None,
) -> dict[pd.Timestamp, dict[str, Any]]:
    if dates is None:
        return {}
    state_map = {
        pd.Timestamp(date): dict(state or {})
        for date, state in (regime_state_by_date or {}).items()
    }
    fallback = dict(fallback_regime_state or {})
    result: dict[pd.Timestamp, dict[str, Any]] = {}
    for date in sorted(pd.Timestamp(item) for item in pd.to_datetime(list(dates))):
        result[date] = _opportunity_pool_config(config, state_map.get(date, fallback))
    return result


def _fit_risk_model_by_date(raw_df: pd.DataFrame, risk_cfg: dict[str, Any] | None) -> dict[pd.Timestamp, Any]:
    from skyeye.products.ax1.risk.models import FactorRiskModel, HistoricalCovarianceRiskModel

    if raw_df is None or raw_df.empty or "date" not in raw_df.columns:
        return {}
    risk_cfg = dict(risk_cfg or {})
    frame = raw_df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    dates = sorted(pd.Timestamp(date) for date in frame["date"].dropna().unique())
    result: dict[pd.Timestamp, Any] = {}
    for date in dates:
        kind = str(risk_cfg.get("kind", "historical_covariance"))
        common_kwargs = {
            "shrinkage": float(risk_cfg.get("shrinkage", 0.0)),
            "min_periods": int(risk_cfg.get("min_periods", 2)),
            "lookback_days": int(risk_cfg["lookback_days"]) if risk_cfg.get("lookback_days") is not None else None,
        }
        if kind == "statistical_factor":
            model = FactorRiskModel(
                n_factors=int(risk_cfg.get("n_factors", 3)),
                idiosyncratic_floor=float(risk_cfg.get("idiosyncratic_floor", 1e-8)),
                **common_kwargs,
            )
        elif kind == "historical_covariance":
            model = HistoricalCovarianceRiskModel(**common_kwargs)
        else:
            raise ValueError(f"unsupported risk_model.kind: {kind}")
        result[date] = model.fit(frame[frame["date"] <= date])
    return result


def _risk_model_for_timestamp(risk_model_by_date: dict[pd.Timestamp, Any], date) -> Any:
    if not risk_model_by_date:
        return None
    target_date = pd.Timestamp(date)
    if target_date in risk_model_by_date:
        return risk_model_by_date[target_date]
    eligible_dates = [key for key in risk_model_by_date if key <= target_date]
    if not eligible_dates:
        return None
    return risk_model_by_date[max(eligible_dates)]


def _attach_effective_breadth_to_robustness(
    training_summary: dict[str, Any],
    effective_breadth_summary: dict[str, Any],
) -> dict[str, Any]:
    result = dict(training_summary or {})
    robustness = dict(result.get("robustness") or {})
    robustness["effective_breadth"] = dict(effective_breadth_summary or {})
    breadth_warnings = list((effective_breadth_summary or {}).get("warnings", []) or [])
    if breadth_warnings:
        existing_warnings = [str(item) for item in robustness.get("warnings", []) or []]
        for warning in breadth_warnings:
            if warning not in existing_warnings:
                existing_warnings.append(str(warning))
        robustness["warnings"] = existing_warnings
        robustness["warning_count"] = int(len(existing_warnings))
    result["robustness"] = robustness
    return result


def _opportunity_pool_regime_snapshot(regime_state: dict[str, Any] | None) -> dict[str, Any]:
    state = regime_state or {}
    return {
        "risk_state": str(state.get("risk_state", "neutral")),
        "rotation_state": str(state.get("rotation_state", "co_move")),
        "strength": _clip_float(state.get("strength", 0.0), lower=0.0, upper=1.0),
    }


def _clip_float(value: Any, *, lower: float, upper: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = lower
    if not pd.notna(numeric):
        numeric = lower
    return min(max(numeric, lower), upper)


def _build_label_builder(config: dict[str, Any]):
    from skyeye.products.ax1.labels import MultiHorizonLabelBuilder

    labels_cfg = config.get("labels", {})
    winsorize_quantiles = labels_cfg.get("winsorize_quantiles")
    if winsorize_quantiles is not None:
        winsorize_quantiles = (float(winsorize_quantiles[0]), float(winsorize_quantiles[1]))
    return MultiHorizonLabelBuilder(
        horizons=_prediction_horizons(config),
        volatility_horizons=[int(item) for item in labels_cfg.get("volatility_horizons", [])],
        winsorize_quantiles=winsorize_quantiles,
        relative_return_enabled=bool((labels_cfg.get("relative_return") or {}).get("enabled", False)),
        relative_group_columns=tuple((labels_cfg.get("relative_return") or {}).get("group_columns", [])),
        relative_min_group_count=int((labels_cfg.get("relative_return") or {}).get("min_group_count", 2)),
        relative_fallback=str((labels_cfg.get("relative_return") or {}).get("fallback", "date")),
        trading_days_per_year=int(labels_cfg.get("trading_days_per_year", 244)),
        cost_config=config.get("costs", {}),
        asset_type_column=str(labels_cfg.get("asset_type_column", "asset_type")),
        entry_lag_days=int(labels_cfg.get("entry_lag_days", 0)),
        price_column=str(labels_cfg.get("price_column", (config.get("data", {}).get("price_adjustment") or {}).get("price_column", "close"))),
        adjusted_price_column=labels_cfg.get("adjusted_price_column", (config.get("data", {}).get("price_adjustment") or {}).get("adjusted_price_column")),
        adjustment_factor_column=labels_cfg.get("adjustment_factor_column", (config.get("data", {}).get("price_adjustment") or {}).get("adjustment_factor_column")),
    )


def _audit_feature_contract(feature_columns: list[str], *, purpose: str):
    from skyeye.products.ax1.data_contract import audit_feature_set
    from skyeye.products.ax1.features.catalog import build_default_feature_catalog

    catalog = build_default_feature_catalog()
    report = audit_feature_set(feature_columns, catalog=catalog, purpose=purpose)
    return catalog, report


def _audit_universe_contract(universe_metadata: pd.DataFrame, *, purpose: str) -> dict[str, Any]:
    from skyeye.products.ax1.data_contract import audit_universe_metadata

    return audit_universe_metadata(universe_metadata, purpose=purpose)


def _build_raw_data_quality(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    from skyeye.products.ax1.data_quality import build_raw_data_quality_report

    return build_raw_data_quality_report(frame, config)


def _build_feature_data_quality(labeled: pd.DataFrame, feature_columns: list[str], config: dict[str, Any]) -> dict[str, Any]:
    from skyeye.products.ax1.data_quality import build_feature_matrix_quality_report

    return build_feature_matrix_quality_report(labeled, feature_columns, config=config)


def _enforce_quality_if_enabled(report: dict[str, Any], config: dict[str, Any], *, context: str) -> None:
    from skyeye.products.ax1.data_quality import enforce_data_quality

    quality_cfg = ((config.get("data") or {}).get("quality") or {})
    if bool(quality_cfg.get("enabled", True)) and bool(quality_cfg.get("enforce_hard_blocks", True)):
        enforce_data_quality(report, context=context)


def _run_rule_pipeline(
    config: dict[str, Any],
    labeled: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_view,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raise NotImplementedError("AX1 main runner only supports lgbm_multi_target")


def _build_view_fusion(config: dict[str, Any]):
    from skyeye.products.ax1.view_fusion.black_litterman import BlackLittermanViewFusion, NoOpViewFusion

    view_cfg = config.get("view_fusion", {})
    kind = str(view_cfg.get("kind", "black_litterman_noop"))
    if kind in {"black_litterman_noop", "noop", "noop_adjusted_return"}:
        return NoOpViewFusion(return_column=str(view_cfg.get("return_column", "expected_relative_net_return_10d")))
    if kind == "black_litterman":
        return BlackLittermanViewFusion()
    raise ValueError(f"unsupported AX1 view fusion kind: {kind}")


def _build_industry_map(scoped: pd.DataFrame, *, as_of_date: pd.Timestamp, data_provider=None) -> dict[str, str]:
    order_book_ids = sorted(scoped["order_book_id"].dropna().astype(str).unique()) if "order_book_id" in scoped.columns else []
    industry_map = _industry_map_from_frame(scoped)
    if data_provider is not None:
        provider_map = _industry_map_from_provider(data_provider, order_book_ids, as_of_date)
        provider_map.update(industry_map)
        industry_map = provider_map
    return {order_book_id: str(industry_map.get(order_book_id, "Unknown") or "Unknown") for order_book_id in order_book_ids}


def _industry_map_from_frame(frame: pd.DataFrame) -> dict[str, str]:
    industry_column = next(
        (
            column
            for column in ("industry", "sector", "industry_name", "sector_code", "sector_code_name", "industry_code")
            if column in frame.columns
        ),
        None,
    )
    if industry_column is None or frame.empty:
        return {}
    working = frame.dropna(subset=["order_book_id"]).copy()
    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.sort_values(["order_book_id", "date"])
    latest = working.drop_duplicates("order_book_id", keep="last")
    return {
        str(row["order_book_id"]): str(row[industry_column]) if pd.notna(row[industry_column]) else "Unknown"
        for _, row in latest.iterrows()
    }


def _industry_map_from_provider(data_provider, order_book_ids: list[str], as_of_date: pd.Timestamp) -> dict[str, str]:
    provider_method = getattr(data_provider, "get_industry", None)
    if provider_method is None or not order_book_ids:
        return {}
    try:
        payload = provider_method(order_book_ids, date=as_of_date)
    except TypeError:
        try:
            payload = provider_method(order_book_ids, "citics_2019", 1, as_of_date)
        except Exception:
            return {}
    except Exception:
        return {}
    frame = pd.DataFrame(payload) if payload is not None else pd.DataFrame()
    if frame.empty:
        return {}
    if "order_book_id" not in frame.columns and len(order_book_ids) == 1:
        frame.insert(0, "order_book_id", order_book_ids[0])
    if "order_book_id" not in frame.columns:
        return {}
    return _industry_map_from_frame(frame)


def _attach_industry(predictions: pd.DataFrame, industry_map: dict[str, str]) -> pd.DataFrame:
    if predictions is None or predictions.empty or "order_book_id" not in predictions.columns:
        return predictions
    result = predictions.copy()
    result["industry"] = result["order_book_id"].astype(str).map(industry_map).fillna("Unknown")
    return result


def _attach_universe_metadata(frame: pd.DataFrame, universe_metadata: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or universe_metadata is None or universe_metadata.empty:
        return frame
    if "order_book_id" not in frame.columns:
        return frame
    columns = [column for column in ("order_book_id", "asset_type", "universe_layer") if column in universe_metadata.columns]
    if len(columns) <= 1:
        return frame
    metadata = universe_metadata[columns].drop_duplicates("order_book_id", keep="last").copy()
    result = frame.drop(columns=[column for column in ("asset_type", "universe_layer") if column in frame.columns], errors="ignore")
    result["order_book_id"] = result["order_book_id"].astype(str)
    metadata["order_book_id"] = metadata["order_book_id"].astype(str)
    return result.merge(metadata, on="order_book_id", how="left")


def _universe_summary(universe_metadata: pd.DataFrame) -> dict[str, Any]:
    pit_audit = dict(getattr(universe_metadata, "attrs", {}).get("pit_audit") or {})
    if universe_metadata is None or universe_metadata.empty:
        return {
            "size": 0,
            "asset_type_counts": {},
            "layer_counts": {},
            "pit_audit": pit_audit,
        }
    return {
        "size": int(universe_metadata["order_book_id"].nunique()),
        "asset_type_counts": {
            str(key): int(value)
            for key, value in universe_metadata["asset_type"].fillna("unknown").astype(str).value_counts().sort_index().items()
        },
        "layer_counts": {
            str(key): int(value)
            for key, value in universe_metadata["universe_layer"].fillna("unknown").astype(str).value_counts().sort_index().items()
        },
        "pit_audit": pit_audit,
    }


def _build_training_readiness(result: dict[str, Any]) -> dict[str, Any]:
    config = result.get("config") or {}
    labels = config.get("labels") or {}
    execution = config.get("execution") or {}
    model = config.get("model") or {}
    params = model.get("params") or {}
    experiment_seed = int((config.get("experiment") or {}).get("seed", 0) or 0)
    data_audit = result.get("data_audit") or {}
    raw_data_quality = result.get("raw_data_quality") or {}
    feature_data_quality = result.get("feature_data_quality") or {}
    price_adjustment = result.get("price_adjustment") or {}
    universe_audit = result.get("universe_audit") or {}
    gate_summary = result.get("gate_summary") or {}
    robustness = (result.get("training_summary") or {}).get("robustness") or {}
    param_policy = (result.get("parameter_validation_summary") or {}).get("lgbm_param_policy") or {}

    checks = {
        "raw_data_quality": _readiness_check(bool(raw_data_quality.get("passed", False)), "raw panel data quality passed"),
        "feature_data_quality": _readiness_check(bool(feature_data_quality.get("passed", False)), "feature matrix quality passed before LGBM fillna"),
        "data_audit": _readiness_check(bool(data_audit.get("passed", False)), "feature data audit passed"),
        "universe_pit_audit": _readiness_check(bool(universe_audit.get("passed", False)), "universe PIT audit passed"),
        "price_adjustment_contract": _readiness_check(
            bool(raw_data_quality.get("passed", False))
            and str(price_adjustment.get("method", "")) not in {"", "missing", "not_required"},
            "training labels use an explicit adjusted price contract",
        ),
        "lgbm_param_policy": _readiness_check(
            str(param_policy.get("status", "blocked")) != "blocked",
            "LightGBM parameter policy has no hard block",
        ),
        "timing_contract": _readiness_check(
            int(labels.get("entry_lag_days", 0)) >= 1
            and int(execution.get("execution_lag_days", 0)) >= int(labels.get("entry_lag_days", 0)),
            "after-close signal uses next-session label entry and execution",
        ),
        "lgbm_seed": _readiness_check(
            experiment_seed > 0
            and all(int(params.get(key, 0) or 0) == experiment_seed for key in ("seed", "feature_fraction_seed", "bagging_seed", "data_random_seed")),
            "experiment seed is injected into LightGBM params",
        ),
        "lgbm_regularization": _readiness_check(
            int(params.get("min_child_samples", 0) or 0) >= 30
            and float(params.get("reg_alpha", 0.0) or 0.0) > 0.0
            and float(params.get("reg_lambda", 0.0) or 0.0) > 0.0
            and 0.0 < float(params.get("subsample", 0.0) or 0.0) <= 1.0
            and 0.0 < float(params.get("colsample_bytree", 0.0) or 0.0) <= 1.0,
            "LightGBM anti-overfit params are present",
        ),
    }
    blockers = [
        {"check": name, "message": check["message"]}
        for name, check in checks.items()
        if not check["passed"]
    ]
    warnings = []
    if gate_summary and not bool(gate_summary.get("passed", False)):
        warnings.append(
            {
                "reason_code": "promotion_gate_not_passed",
                "failed_checks": list(gate_summary.get("failed_checks", []) or []),
            }
        )
    if (robustness.get("bootstrap_ci") or {}).get("ci_crosses_zero"):
        warnings.append({"reason_code": "bootstrap_ci_crosses_zero"})
    if (robustness.get("sample_decay") or {}).get("flag_late_decay"):
        warnings.append({"reason_code": "late_sample_decay"})
    effective_breadth = robustness.get("effective_breadth") or result.get("effective_breadth_summary") or {}
    for warning in effective_breadth.get("warnings", []) or []:
        warnings.append({"reason_code": str(warning)})
    status = "blocked" if blockers else "warning" if warnings else "go"
    return {
        "schema_version": 1,
        "status": status,
        "blocker_count": int(len(blockers)),
        "warning_count": int(len(warnings)),
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
    }


def _readiness_check(passed: bool, message: str) -> dict[str, Any]:
    return {"passed": bool(passed), "message": str(message)}


def _enabled_cost_config(costs_config: dict[str, Any] | None):
    costs = dict(costs_config or {})
    return costs if costs.get("enabled", False) else None


def _attach_execution_inputs(
    target_weights: pd.DataFrame,
    *,
    predictions: pd.DataFrame,
    raw_frame: pd.DataFrame,
    price_column: str,
    alpha_column: str,
    execution_lag_days: int = 0,
) -> pd.DataFrame:
    if target_weights is None or target_weights.empty:
        return target_weights
    result = target_weights.copy()
    keys = ["date", "order_book_id"]
    result["date"] = pd.to_datetime(result["date"])
    result["order_book_id"] = result["order_book_id"].astype(str)
    result["signal_date"] = result["date"]

    if price_column not in raw_frame.columns:
        raise ValueError(f"execution.price_column not found in raw frame: {price_column}")
    price_frame = _execution_price_frame(
        raw_frame,
        price_column=price_column,
        execution_lag_days=int(execution_lag_days),
    )
    result = result.drop(columns=["price", "execution_date"], errors="ignore").merge(price_frame, on=keys, how="left")
    if result["price"].isna().any():
        result = result[result["price"].notna()].reset_index(drop=True)
        if result.empty:
            raise ValueError("missing execution price for all target weights")

    # Attach liquidity proxy for capacity controls.
    if "dollar_volume" not in result.columns:
        if "volume" in raw_frame.columns and price_column in raw_frame.columns:
            liquidity = raw_frame[keys + [price_column, "volume"]].copy()
            liquidity["date"] = pd.to_datetime(liquidity["date"])
            liquidity["order_book_id"] = liquidity["order_book_id"].astype(str)
            liquidity["dollar_volume"] = (
                pd.to_numeric(liquidity[price_column], errors="coerce")
                * pd.to_numeric(liquidity["volume"], errors="coerce")
            ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
            liquidity["feature_dollar_volume"] = np.log1p(liquidity["dollar_volume"])
            liquidity = _shift_execution_frame(
                liquidity.drop(columns=[price_column, "volume"]),
                execution_lag_days=int(execution_lag_days),
            )
            result = result.merge(liquidity[keys + ["dollar_volume", "feature_dollar_volume"]], on=keys, how="left")

    if alpha_column in predictions.columns:
        alpha_frame = predictions[keys + [alpha_column]].copy()
        alpha_frame["date"] = pd.to_datetime(alpha_frame["date"])
        alpha_frame["order_book_id"] = alpha_frame["order_book_id"].astype(str)
        alpha_frame = alpha_frame.drop_duplicates(keys, keep="last")
        result = result.drop(columns=[alpha_column], errors="ignore").merge(alpha_frame, on=keys, how="left")
    return result


def _build_execution_reference(
    raw_frame: pd.DataFrame,
    *,
    price_column: str,
    industry_map: dict[str, str],
    execution_lag_days: int = 0,
) -> pd.DataFrame:
    keys = ["date", "order_book_id"]
    columns = [column for column in keys + [price_column, "asset_type", "universe_layer", "industry"] if column in raw_frame.columns]
    reference = raw_frame[columns].copy()
    reference["date"] = pd.to_datetime(reference["date"])
    reference["order_book_id"] = reference["order_book_id"].astype(str)
    if price_column != "price":
        reference = reference.rename(columns={price_column: "price"})
    if "industry" not in reference.columns:
        reference["industry"] = reference["order_book_id"].map(industry_map).fillna("Unknown")
    return _shift_execution_frame(reference.drop_duplicates(keys, keep="last"), execution_lag_days=int(execution_lag_days))


def _execution_price_frame(raw_frame: pd.DataFrame, *, price_column: str, execution_lag_days: int) -> pd.DataFrame:
    keys = ["date", "order_book_id"]
    price_frame = raw_frame[keys + [price_column]].copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame["order_book_id"] = price_frame["order_book_id"].astype(str)
    price_frame = price_frame.drop_duplicates(keys, keep="last").rename(columns={price_column: "price"})
    return _shift_execution_frame(price_frame, execution_lag_days=execution_lag_days)


def _shift_execution_frame(frame: pd.DataFrame, *, execution_lag_days: int) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    if int(execution_lag_days) <= 0:
        result = frame.copy()
        result["execution_date"] = pd.to_datetime(result["date"])
        return result
    result = frame.copy()
    result["execution_date"] = pd.to_datetime(result["date"])
    result = result.sort_values(["order_book_id", "execution_date"]).reset_index(drop=True)
    result["date"] = result.groupby("order_book_id", sort=False)["execution_date"].shift(int(execution_lag_days))
    result = result.dropna(subset=["date"]).reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def _execute_rolling_targets(
    target_weights: pd.DataFrame,
    *,
    smoother,
    executable_optimizer,
    max_turnover: Any = None,
    rebalance_interval: int = 1,
    portfolio_value: float = 1_000_000,
    execution_reference: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if target_weights is None or target_weights.empty:
        empty_weights = smoother.smooth(target_weights)
        empty_orders = _empty_orders()
        return empty_weights, empty_orders, {
            "date_count": 0,
            "max_turnover_observed": 0.0,
            "turnover_constrained_dates": 0,
            "rebalance_interval": int(rebalance_interval),
            "rebalance_skipped_dates": 0,
            "rebalance_allowed_dates": 0,
            "rebalance_blocked_dates": 0,
            "risk_rebalance_dates": 0,
            "max_order_count_observed": 0,
            "total_order_count": 0,
            "gross_order_value": 0.0,
            "estimated_cost": 0.0,
            "order_count_by_date": {},
        }

    rows = []
    order_rows = []
    previous_weights: dict[str, float] | None = None
    current_shares: dict[str, int] = {}
    current_positions: dict[str, dict[str, Any]] = {}
    today_buy_shares: dict[str, int] = {}
    current_session_date = None
    max_observed = 0.0
    constrained_dates = 0
    skipped_dates = 0
    rebalance_allowed_dates = 0
    rebalance_blocked_dates = 0
    risk_rebalance_dates = 0
    missing_reference_dates = 0
    max_order_count_observed = 0
    max_turnover_value = float(max_turnover) if max_turnover is not None else None
    rebalance_interval = max(1, int(rebalance_interval))
    last_rebalance_index: int | None = None
    order_count_by_date: dict[str, int] = {}

    grouped = list(target_weights.sort_values(["date", "order_book_id"]).groupby("date", sort=True))
    for date_index, (date, day_targets) in enumerate(grouped):
        date_key = str(pd.Timestamp(date).date())
        session_date = pd.Timestamp(date).date()
        if current_session_date != session_date:
            today_buy_shares = {}
            current_session_date = session_date
        day_reference = _execution_reference_for_date(execution_reference, date)
        missing_current_reference = _has_missing_current_execution_reference(
            day_targets,
            current_shares,
            day_reference,
        )
        day_targets = _augment_day_targets_with_reference(day_targets, current_shares, day_reference)
        should_rebalance = (
            previous_weights is None
            or last_rebalance_index is None
            or (date_index - last_rebalance_index) >= rebalance_interval
        )
        if "rebalance_allowed" in day_targets.columns:
            allowed_values = day_targets["rebalance_allowed"].dropna()
            if not allowed_values.empty:
                rebalance_allowed = bool(allowed_values.astype(bool).any())
                if rebalance_allowed:
                    rebalance_allowed_dates += 1
                else:
                    rebalance_blocked_dates += 1
                    should_rebalance = False
        if missing_current_reference:
            should_rebalance = False
            missing_reference_dates += 1
        risk_rebalance = False
        if not should_rebalance and not missing_current_reference:
            risk_rebalance = _would_carry_hard_constraint_violation(
                date,
                day_targets,
                current_shares=current_shares,
                portfolio_value=portfolio_value,
                execution_reference=day_reference,
                fallback_positions=current_positions,
                max_gross_weight=getattr(smoother, "target_gross_weight", None),
                max_weight=getattr(smoother, "max_weight", None),
                max_industry_weight=getattr(smoother, "max_industry_weight", None),
            )
            if risk_rebalance:
                should_rebalance = True
                risk_rebalance_dates += 1
        before_weights = dict(previous_weights or {})
        if not should_rebalance:
            executed_day = _carry_executable_positions(
                date,
                day_targets,
                current_shares=current_shares,
                portfolio_value=portfolio_value,
                execution_reference=day_reference,
                fallback_positions=current_positions,
            )
            orders_day = _empty_orders()
            skipped_dates += 1
            observed_turnover = 0.0
        else:
            active_smoother = _smoother_for_risk_rebalance(smoother) if risk_rebalance else smoother
            active_executable_optimizer = _executable_optimizer_with_hard_caps(
                executable_optimizer,
                active_smoother,
            )
            if risk_rebalance:
                active_executable_optimizer = _executable_optimizer_for_risk_rebalance(active_executable_optimizer)
            raw_weights = _weights_by_id(day_targets, column="target_weight")
            raw_turnover = _one_way_turnover(previous_weights or raw_weights, raw_weights)
            current_weights_for_smoother = _current_weights_for_smoother(
                previous_weights=previous_weights,
                current_shares=current_shares,
                today_buy_shares=today_buy_shares,
                day_targets=day_targets,
                portfolio_value=portfolio_value,
                t_plus_one_enabled=bool(getattr(active_smoother, "t_plus_one_lock", False)),
                today_buy_weight_column=str(
                    getattr(active_smoother, "today_buy_weight_column", "today_buy_weight")
                ),
            )
            smoothed_day = active_smoother.smooth(day_targets, current_weights_for_smoother)
            smoothed_day = _restore_execution_columns(smoothed_day, day_targets)
            smoothed_day = _append_current_share_targets(smoothed_day, day_targets, current_shares)
            if smoothed_day is None or smoothed_day.empty:
                executed_day = _carry_executable_positions(
                    date,
                    day_targets,
                    current_shares=current_shares,
                    portfolio_value=portfolio_value,
                    execution_reference=day_reference,
                    fallback_positions=current_positions,
                )
                orders_day = _empty_orders()
            else:
                executable_result = active_executable_optimizer.optimize(smoothed_day, current_shares)
                executed_day = _mark_risk_rebalance_reason(executable_result.portfolio, risk_rebalance)
                orders_day = _mark_risk_rebalance_reason(executable_result.orders, risk_rebalance)
            observed_turnover = _one_way_turnover(before_weights or _weights_by_id(executed_day, column="target_weight"), _weights_by_id(executed_day, column="target_weight"))
            if max_turnover_value is not None and raw_turnover > max_turnover_value + 1e-12:
                constrained_dates += 1
            last_rebalance_index = date_index

        if not executed_day.empty:
            executed_day = executed_day.copy()
            executed_day["date"] = pd.Timestamp(date)
            executed_day["component"] = "executed"
            rows.extend(executed_day.to_dict("records"))
            previous_weights = _weights_by_id(executed_day, column="target_weight")
            current_shares = {
                str(row["order_book_id"]): int(row["target_shares"])
                for _, row in executed_day.dropna(subset=["order_book_id", "target_shares"]).iterrows()
            }
            current_positions = _position_rows_by_id(executed_day)
            today_buy_shares = _update_today_buy_shares(today_buy_shares, orders_day, current_shares)
        else:
            previous_weights = {}
            current_shares = {}
            current_positions = {}
            today_buy_shares = {}

        if not orders_day.empty:
            orders_day = orders_day.copy()
            orders_day["date"] = pd.Timestamp(date)
            order_rows.extend(orders_day.to_dict("records"))
        order_count = int(len(orders_day))
        order_count_by_date[date_key] = order_count
        max_order_count_observed = max(max_order_count_observed, order_count)
        max_observed = max(max_observed, observed_turnover)

    weight_columns = ["date", "order_book_id", "target_weight", "component"]
    metadata_columns = [
        "industry",
        "asset_type",
        "universe_layer",
        "exposure_group",
        "target_shares",
        "price",
        "position_value",
        "order_shares",
        "order_value",
        "estimated_cost",
        "trade_reason",
        "intended_weight",
        "cash_buffer",
        "rebalance_allowed",
        "expected_relative_net_return_10d",
        "adjusted_expected_return",
    ]
    for metadata_column in metadata_columns:
        if any(metadata_column in row for row in rows):
            weight_columns.append(metadata_column)
    executed_weights = pd.DataFrame(rows, columns=weight_columns)

    order_columns = [
        "date",
        "order_book_id",
        "asset_type",
        "universe_layer",
        "side",
        "order_shares",
        "price",
        "order_value",
        "estimated_cost",
        "trade_reason",
    ]
    orders = pd.DataFrame(order_rows, columns=[column for column in order_columns if any(column in row for row in order_rows)])
    orders = _fill_order_metadata_from_weights(orders, executed_weights)
    gross_order_value = float(pd.to_numeric(orders.get("order_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0).abs().sum()) if not orders.empty else 0.0
    estimated_cost = float(pd.to_numeric(orders.get("estimated_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not orders.empty else 0.0
    return executed_weights, orders, {
        "date_count": int(executed_weights["date"].nunique()) if not executed_weights.empty else 0,
        "max_turnover_observed": float(max_observed),
        "turnover_constrained_dates": int(constrained_dates),
        "rebalance_interval": int(rebalance_interval),
        "rebalance_skipped_dates": int(skipped_dates),
        "rebalance_allowed_dates": int(rebalance_allowed_dates),
        "rebalance_blocked_dates": int(rebalance_blocked_dates),
        "risk_rebalance_dates": int(risk_rebalance_dates),
        "max_order_count_observed": int(max_order_count_observed),
        "total_order_count": int(len(orders)),
        "gross_order_value": gross_order_value,
        "estimated_cost": estimated_cost,
        "order_count_by_date": order_count_by_date,
        "missing_execution_reference_dates": int(missing_reference_dates),
    }


def _would_carry_hard_constraint_violation(
    date,
    day_targets: pd.DataFrame,
    *,
    current_shares: dict[str, int],
    portfolio_value: float,
    execution_reference: pd.DataFrame | None,
    fallback_positions: dict[str, dict[str, Any]] | None,
    max_gross_weight: float | None,
    max_weight: float | None,
    max_industry_weight: float | None,
) -> bool:
    if not current_shares or (max_gross_weight is None and max_weight is None and max_industry_weight is None):
        return False
    carried = _carry_executable_positions(
        date,
        day_targets,
        current_shares=current_shares,
        portfolio_value=portfolio_value,
        execution_reference=execution_reference,
        fallback_positions=fallback_positions,
    )
    if carried is None or carried.empty or "target_weight" not in carried.columns:
        return False
    weights = pd.to_numeric(carried["target_weight"], errors="coerce").fillna(0.0).abs()
    if max_gross_weight is not None and float(weights.sum()) > float(max_gross_weight) + 1e-12:
        return True
    if max_weight is not None and float(weights.max() if not weights.empty else 0.0) > float(max_weight) + 1e-12:
        return True
    if max_industry_weight is None:
        return False
    frame = carried.copy()
    if "industry" not in frame.columns:
        frame["industry"] = "Unknown"
    frame["industry"] = frame["industry"].fillna("Unknown").astype(str)
    frame["_weight"] = weights
    exposure = frame.groupby("industry")["_weight"].sum()
    return bool((exposure > float(max_industry_weight) + 1e-12).any())


def _smoother_for_risk_rebalance(smoother):
    return replace(
        smoother,
        buffer_weight=0.0,
        no_trade_buffer_weight=0.0,
        min_trade_value=0.0,
    )


def _executable_optimizer_for_risk_rebalance(executable_optimizer):
    return replace(executable_optimizer, min_trade_value=0.0)


def _executable_optimizer_with_hard_caps(executable_optimizer, smoother):
    updates = {}
    if getattr(executable_optimizer, "max_weight", None) is None:
        updates["max_weight"] = getattr(smoother, "max_weight", None)
    if getattr(executable_optimizer, "max_industry_weight", None) is None:
        updates["max_industry_weight"] = getattr(smoother, "max_industry_weight", None)
    return replace(executable_optimizer, **updates) if updates else executable_optimizer


def _mark_risk_rebalance_reason(frame: pd.DataFrame, enabled: bool) -> pd.DataFrame:
    if not enabled or frame is None or frame.empty or "order_shares" not in frame.columns:
        return frame
    result = frame.copy()
    order_shares = pd.to_numeric(result["order_shares"], errors="coerce").fillna(0.0)
    result.loc[order_shares != 0, "trade_reason"] = "risk_cap_rebalance"
    return result


_MISSING_METADATA_STRINGS = {"", "nan", "none", "null", "<na>"}


def _missing_metadata_mask(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.lower()
    return values.isna() | normalized.isin(_MISSING_METADATA_STRINGS)


def _fill_order_metadata_from_weights(orders: pd.DataFrame, executed_weights: pd.DataFrame) -> pd.DataFrame:
    if orders is None or orders.empty or executed_weights is None or executed_weights.empty:
        return orders
    result = orders.copy()
    if "order_book_id" not in result.columns or "order_book_id" not in executed_weights.columns:
        return result
    weight_metadata = executed_weights.dropna(subset=["order_book_id"]).copy()
    weight_metadata["order_book_id"] = weight_metadata["order_book_id"].astype(str)
    result["order_book_id"] = result["order_book_id"].astype(str)
    for column in ("asset_type", "universe_layer"):
        if column not in weight_metadata.columns:
            continue
        column_metadata = weight_metadata[["order_book_id", column]].copy()
        column_metadata = column_metadata[~_missing_metadata_mask(column_metadata[column])]
        if column_metadata.empty:
            continue
        column_metadata = column_metadata.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")
        mapped = result["order_book_id"].map(column_metadata[column])
        if column not in result.columns:
            result[column] = mapped
        else:
            missing = _missing_metadata_mask(result[column])
            result[column] = result[column].where(~missing, mapped)
    return result


def _current_weights_for_smoother(
    *,
    previous_weights: dict[str, float] | None,
    current_shares: dict[str, int],
    today_buy_shares: dict[str, int],
    day_targets: pd.DataFrame,
    portfolio_value: float,
    t_plus_one_enabled: bool,
    today_buy_weight_column: str,
):
    if not t_plus_one_enabled:
        return previous_weights if previous_weights is not None else {}

    previous_weights = previous_weights or {}
    price_by_id = {}
    if day_targets is not None and not day_targets.empty and "price" in day_targets.columns:
        price_frame = day_targets.dropna(subset=["order_book_id", "price"]).drop_duplicates(
            "order_book_id",
            keep="last",
        )
        price_by_id = {
            str(row["order_book_id"]): float(row["price"])
            for _, row in price_frame.iterrows()
        }
    order_book_ids = sorted(set(previous_weights) | set(current_shares) | set(today_buy_shares))
    rows = []
    for order_book_id in order_book_ids:
        shares = int(current_shares.get(order_book_id, 0))
        price = price_by_id.get(order_book_id)
        if price is not None and shares > 0:
            current_weight = float(shares) * float(price) / float(portfolio_value)
        else:
            current_weight = float(previous_weights.get(order_book_id, 0.0))
        bought_shares = min(max(int(today_buy_shares.get(order_book_id, 0)), 0), max(shares, 0))
        today_buy_weight = 0.0
        if price is not None and bought_shares > 0:
            today_buy_weight = float(bought_shares) * float(price) / float(portfolio_value)
            today_buy_weight = min(float(today_buy_weight), float(current_weight))
        rows.append(
            {
                "order_book_id": order_book_id,
                "current_weight": float(current_weight),
                today_buy_weight_column: float(today_buy_weight),
            }
        )
    return pd.DataFrame(rows, columns=["order_book_id", "current_weight", today_buy_weight_column])


def _update_today_buy_shares(
    today_buy_shares: dict[str, int],
    orders_day: pd.DataFrame,
    current_shares: dict[str, int],
) -> dict[str, int]:
    result = dict(today_buy_shares or {})
    if orders_day is not None and not orders_day.empty and "order_shares" in orders_day.columns:
        for _, row in orders_day.dropna(subset=["order_book_id", "order_shares"]).iterrows():
            order_shares = int(row["order_shares"])
            if order_shares > 0:
                order_book_id = str(row["order_book_id"])
                result[order_book_id] = int(result.get(order_book_id, 0)) + order_shares
    capped = {}
    for order_book_id, bought_shares in result.items():
        remaining = int(current_shares.get(order_book_id, 0))
        locked = min(max(int(bought_shares), 0), max(remaining, 0))
        if locked > 0:
            capped[str(order_book_id)] = locked
    return capped


def _carry_executable_positions(
    date,
    day_targets: pd.DataFrame,
    *,
    current_shares: dict[str, int],
    portfolio_value: float,
    execution_reference: pd.DataFrame | None = None,
    fallback_positions: dict[str, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    if not current_shares:
        return pd.DataFrame(columns=["date", "order_book_id", "target_weight", "target_shares", "price"])
    reference_frames = []
    if fallback_positions:
        reference_frames.append(pd.DataFrame(fallback_positions.values()))
    reference_frames.append(day_targets)
    if execution_reference is not None and not execution_reference.empty:
        reference_frames.append(execution_reference)
    reference = pd.concat(reference_frames, ignore_index=True, sort=False)
    metadata = reference.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")
    rows = []
    for order_book_id, shares in sorted(current_shares.items()):
        if order_book_id not in metadata.index:
            continue
        source = metadata.loc[order_book_id]
        price = float(source["price"])
        row = source.to_dict()
        position_value = float(int(shares) * price)
        row.update(
            {
                "date": pd.Timestamp(date),
                "order_book_id": order_book_id,
                "target_shares": int(shares),
                "price": price,
                "position_value": position_value,
                "target_weight": position_value / float(portfolio_value),
                "order_shares": 0,
                "order_value": 0.0,
                "estimated_cost": 0.0,
                "trade_reason": "rebalance_interval_skip",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _has_missing_current_execution_reference(
    day_targets: pd.DataFrame,
    current_shares: dict[str, int],
    day_reference: pd.DataFrame,
) -> bool:
    if not current_shares:
        return False
    target_ids = set(day_targets["order_book_id"].astype(str)) if "order_book_id" in day_targets.columns else set()
    reference_ids = set(day_reference["order_book_id"].astype(str)) if day_reference is not None and not day_reference.empty else set()
    executable_ids = target_ids | reference_ids
    return any(int(shares) > 0 and order_book_id not in executable_ids for order_book_id, shares in current_shares.items())


def _position_rows_by_id(executed_day: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if executed_day is None or executed_day.empty or not {"order_book_id", "target_shares", "price"}.issubset(executed_day.columns):
        return {}
    frame = executed_day.dropna(subset=["order_book_id", "target_shares", "price"]).copy()
    frame["target_shares"] = pd.to_numeric(frame["target_shares"], errors="coerce").fillna(0).astype(int)
    frame = frame[frame["target_shares"] > 0]
    return {
        str(row["order_book_id"]): row.to_dict()
        for _, row in frame.drop_duplicates("order_book_id", keep="last").iterrows()
    }


def _execution_reference_for_date(execution_reference: pd.DataFrame | None, date) -> pd.DataFrame:
    if execution_reference is None or execution_reference.empty:
        return pd.DataFrame()
    reference = execution_reference.copy()
    reference["date"] = pd.to_datetime(reference["date"])
    reference["order_book_id"] = reference["order_book_id"].astype(str)
    current = reference[reference["date"] == pd.Timestamp(date)]
    return current.drop_duplicates(["date", "order_book_id"], keep="last")


def _augment_day_targets_with_reference(
    day_targets: pd.DataFrame,
    current_shares: dict[str, int],
    day_reference: pd.DataFrame,
) -> pd.DataFrame:
    if not current_shares or day_reference is None or day_reference.empty:
        return day_targets
    existing = set(day_targets["order_book_id"].astype(str)) if "order_book_id" in day_targets.columns else set()
    rows = []
    reference = day_reference.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")
    for order_book_id, shares in current_shares.items():
        if int(shares) == 0 or order_book_id in existing or order_book_id not in reference.index:
            continue
        row = reference.loc[order_book_id].to_dict()
        row.update(
            {
                "date": reference.loc[order_book_id]["date"],
                "order_book_id": order_book_id,
                "target_weight": 0.0,
                "component": "clearance",
                "adjusted_expected_return": row.get("adjusted_expected_return", 0.0),
            }
        )
        rows.append(row)
    if not rows:
        return day_targets
    return pd.concat([day_targets, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _append_current_share_targets(
    smoothed_day: pd.DataFrame,
    day_targets: pd.DataFrame,
    current_shares: dict[str, int],
) -> pd.DataFrame:
    if not current_shares:
        return smoothed_day
    smoothed = smoothed_day.copy() if smoothed_day is not None else pd.DataFrame()
    existing = set(smoothed["order_book_id"].astype(str)) if "order_book_id" in smoothed.columns else set()
    reference = day_targets.drop_duplicates("order_book_id", keep="last").set_index("order_book_id")
    rows = []
    for order_book_id, shares in current_shares.items():
        if int(shares) == 0 or order_book_id in existing or order_book_id not in reference.index:
            continue
        row = reference.loc[order_book_id].to_dict()
        row.update(
            {
                "date": reference.loc[order_book_id]["date"],
                "order_book_id": order_book_id,
                "target_weight": 0.0,
                "component": "clearance",
            }
        )
        rows.append(row)
    if not rows:
        return smoothed_day
    return pd.concat([smoothed, pd.DataFrame(rows)], ignore_index=True, sort=False)


def _restore_execution_columns(smoothed_day: pd.DataFrame, day_targets: pd.DataFrame) -> pd.DataFrame:
    if smoothed_day is None or smoothed_day.empty or day_targets is None or day_targets.empty:
        return smoothed_day
    keys = ["date", "order_book_id"]
    extra_columns = [
        column
        for column in day_targets.columns
        if column not in set(keys + ["target_weight", "component"]) and column not in smoothed_day.columns
    ]
    if not extra_columns:
        return smoothed_day
    extras = day_targets[keys + extra_columns].drop_duplicates(keys, keep="last").copy()
    result = smoothed_day.copy()
    result["date"] = pd.to_datetime(result["date"])
    result["order_book_id"] = result["order_book_id"].astype(str)
    extras["date"] = pd.to_datetime(extras["date"])
    extras["order_book_id"] = extras["order_book_id"].astype(str)
    return result.merge(extras, on=keys, how="left")


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "order_book_id",
            "asset_type",
            "universe_layer",
            "side",
            "order_shares",
            "price",
            "order_value",
            "estimated_cost",
            "trade_reason",
        ]
    )


def _smooth_rolling_targets(
    target_weights: pd.DataFrame,
    *,
    smoother,
    max_turnover: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if target_weights is None or target_weights.empty:
        return smoother.smooth(target_weights), {
            "date_count": 0,
            "max_turnover_observed": 0.0,
            "turnover_constrained_dates": 0,
        }

    rows = []
    previous_weights: dict[str, float] | None = None
    max_observed = 0.0
    constrained_dates = 0
    max_turnover_value = float(max_turnover) if max_turnover is not None else None
    for _, day_targets in target_weights.sort_values(["date", "order_book_id"]).groupby("date", sort=True):
        raw_weights = _weights_by_id(day_targets, column="target_weight")
        raw_turnover = _one_way_turnover(previous_weights or raw_weights, raw_weights)
        if previous_weights is None:
            smoothed_day = smoother.smooth(day_targets)
            observed_turnover = 0.0
        else:
            smoothed_day = smoother.smooth(day_targets, previous_weights)
            observed_turnover = _one_way_turnover(previous_weights, _weights_by_id(smoothed_day, column="target_weight"))
        if max_turnover_value is not None and raw_turnover > max_turnover_value + 1e-12:
            constrained_dates += 1
        max_observed = max(max_observed, observed_turnover)
        previous_weights = _weights_by_id(smoothed_day, column="target_weight")
        rows.extend(smoothed_day.to_dict("records"))

    columns = ["date", "order_book_id", "target_weight", "component"]
    for metadata_column in ("industry", "asset_type", "universe_layer"):
        if any(metadata_column in row for row in rows):
            columns.append(metadata_column)
    smoothed = pd.DataFrame(rows, columns=columns)
    return smoothed, {
        "date_count": int(smoothed["date"].nunique()) if not smoothed.empty else 0,
        "max_turnover_observed": float(max_observed),
        "turnover_constrained_dates": int(constrained_dates),
    }


def _weights_by_id(frame: pd.DataFrame, *, column: str) -> dict[str, float]:
    if frame is None or frame.empty:
        return {}
    return {
        str(row["order_book_id"]): float(row[column])
        for _, row in frame.dropna(subset=["order_book_id", column]).iterrows()
    }


def _one_way_turnover(current: dict[str, float], target: dict[str, float]) -> float:
    universe = set(current) | set(target)
    return float(0.5 * sum(abs(float(target.get(item, 0.0)) - float(current.get(item, 0.0))) for item in universe))


def _first_day_weights(target_weights: pd.DataFrame) -> dict[str, float]:
    if target_weights is None or target_weights.empty or "date" not in target_weights.columns:
        return {}
    first_date = target_weights["date"].min()
    return _weights_by_id(target_weights[target_weights["date"] == first_date], column="target_weight")


def _prediction_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    if predictions is None or predictions.empty:
        return {
            "row_count": 0,
            "expected_relative_net_return_10d_std": 0.0,
        }
    summary: dict[str, Any] = {
        "row_count": int(len(predictions)),
        "prediction_columns": [
            column
            for column in predictions.columns
            if column.startswith("expected_relative_net_return_")
        ],
    }
    score_column = "expected_relative_net_return_10d"
    if score_column in predictions.columns:
        values = pd.to_numeric(predictions[score_column], errors="coerce")
        summary[f"{score_column}_std"] = float(values.std(ddof=0)) if len(values.dropna()) > 1 else 0.0
        summary[f"{score_column}_min"] = float(values.min()) if len(values.dropna()) else 0.0
        summary[f"{score_column}_max"] = float(values.max()) if len(values.dropna()) else 0.0
    return summary


def _constraint_status(violations: dict[str, Any]) -> str:
    hard_keys = ("max_single_weight_count", "max_industry_weight_count", "gross_exposure_count", "max_position_count")
    hard_violation_count = sum(int(violations.get(key, 0) or 0) for key in hard_keys)
    return "hard_violation" if hard_violation_count > 0 else "ok"


if __name__ == "__main__":
    raise SystemExit(main())
