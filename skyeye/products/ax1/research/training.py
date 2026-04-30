"""AX1 LGBM research training pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from skyeye.products.ax1.config import DEFAULT_WALK_FORWARD_FOLDS


def run_lgbm_pipeline(
    config: dict[str, Any],
    labeled: pd.DataFrame,
    *,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from skyeye.products.ax1.confidence import aggregate_confidence_calibration
    from skyeye.products.ax1.feature_diagnostics import analyze_fold_feature_diagnostics
    from skyeye.products.ax1.robustness import (
        aggregate_fold_metrics,
        build_robustness_summary,
        compute_positive_ratio,
        compute_stability_score,
        detect_overfit_flags,
    )

    splitter = build_splitter(config)
    folds = splitter.split(labeled)
    if not folds:
        raise ValueError("splitter produced no valid folds")
    targets = target_columns(config)

    fold_results: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    for fold in folds:
        fold_predictions, fold_labels, fold_summary = run_lgbm_fold(
            config,
            fold,
            feature_columns=feature_columns,
            target_columns=targets,
        )
        fold_results.append(fold_summary)
        prediction_frames.append(fold_predictions)
        label_frames.append(fold_labels)

    aggregate_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    aggregate_labels = pd.concat(label_frames, ignore_index=True) if label_frames else pd.DataFrame()
    feature_report = analyze_fold_feature_diagnostics(
        fold_results=fold_results,
        feature_columns=feature_columns,
        label_column=_feature_diagnostics_label_column(config),
    )
    persisted_fold_results = [_strip_fold_diagnostic_frames(fold_result) for fold_result in fold_results]
    summary: dict[str, Any] = {
        "model_kind": "lgbm_multi_target",
        "seed": int((config.get("experiment") or {}).get("seed", 0) or 0),
        "model_params": dict((config.get("model") or {}).get("params") or {}),
        "feature_columns": list(feature_columns),
        "target_columns": targets,
        "fold_results": persisted_fold_results,
        "aggregate_predictions_df": aggregate_predictions,
        "aggregate_predictions_row_count": int(len(aggregate_predictions)),
        "aggregate_labels_row_count": int(len(aggregate_labels)),
        "aggregate_metrics": aggregate_fold_metrics(fold_results),
        "stability": compute_stability_score(fold_results),
        "positive_ratio": compute_positive_ratio(fold_results),
        "overfit_flags": detect_overfit_flags(fold_results),
        "robustness": build_robustness_summary(
            fold_results,
            seed=int((config.get("experiment") or {}).get("seed", 0) or 0),
        ),
        "feature_importance": _aggregate_lgbm_feature_importance(fold_results, feature_columns),
        "confidence_calibration": aggregate_confidence_calibration(fold_results),
        "feature_diagnostics": feature_report["feature_diagnostics"],
        "feature_conflicts": feature_report["feature_conflicts"],
        "feature_review_summary": feature_report["feature_review_summary"],
    }
    if len(persisted_fold_results) == 1:
        summary.update(
            {
                key: value
                for key, value in persisted_fold_results[0].items()
                if key not in {"predictions_df", "confidence_calibration"}
            }
        )
    return aggregate_predictions, aggregate_labels, summary


def run_lgbm_fold(
    config: dict[str, Any],
    fold: dict[str, Any],
    *,
    feature_columns: list[str],
    target_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from skyeye.products.ax1.confidence import apply_confidence_calibration, fit_confidence_calibrator
    from skyeye.products.ax1.evaluation.metrics import evaluate_signal_layer
    from skyeye.products.ax1.models.bundle import dump_predictor_bundle

    fold_id = int(fold.get("fold_id", 0))
    train_labeled = _drop_missing_targets(fold["train_df"], target_columns, split_name=f"fold {fold_id} train")
    val_labeled = _drop_missing_targets(fold["val_df"], target_columns, split_name=f"fold {fold_id} val")
    test_labeled = fold["test_df"].reset_index(drop=True)

    preprocessor = build_preprocessor(config)
    train_features = preprocessor.transform(train_labeled, feature_columns)
    val_features = preprocessor.transform(val_labeled, feature_columns)
    test_features = preprocessor.transform(test_labeled, feature_columns)

    # Feature diagnostics should run on the pre-standardization panel.
    # Otherwise, cross-sectional z-score makes per-date std ~ 1 and the
    # low-variance check becomes ineffective.
    diagnostic_preprocessor = build_feature_diagnostics_preprocessor(config)
    diagnostic_features = diagnostic_preprocessor.transform(test_labeled, feature_columns)
    predictor = build_predictor(config, horizons=prediction_horizons(config), feature_columns=feature_columns)
    predictor.fit(
        train_features,
        train_labeled,
        val_features=val_features,
        val_labels=val_labeled,
    )
    preprocessor_bundle = preprocessor.to_bundle(feature_columns)
    predictor_bundle = dump_predictor_bundle(predictor, preprocessor_bundle, feature_columns)
    validation_predictions = predictor.predict(val_features)
    validation_predictions = validation_predictions.copy()
    validation_predictions["fold_id"] = fold_id
    val_labeled = val_labeled.copy()
    val_labeled["fold_id"] = fold_id
    confidence_calibrator, confidence_summary = fit_confidence_calibrator(
        validation_predictions,
        val_labeled,
        label_column=_feature_diagnostics_label_column(config),
        outcome_column=None,
        bucket_count=int(config.get("model", {}).get("confidence_bucket_count", 5)),
        min_samples=int(config.get("model", {}).get("confidence_min_samples", 30)),
    )
    confidence_summary["fold_id"] = fold_id
    validation_predictions = apply_confidence_calibration(validation_predictions, confidence_calibrator)
    predictions = predictor.predict(test_features)
    predictions = predictions.copy()
    predictions["fold_id"] = fold_id
    predictions = apply_confidence_calibration(predictions, confidence_calibrator)
    test_labeled = test_labeled.copy()
    test_labeled["fold_id"] = fold_id
    validation_metrics = evaluate_signal_layer(validation_predictions, val_labeled)["signal"]
    prediction_metrics = evaluate_signal_layer(predictions, test_labeled)["signal"]
    research_predictions = _attach_research_labels(predictions, test_labeled)
    summary = {
        "fold_id": fold_id,
        "train_rows": int(len(train_labeled)),
        "val_rows": int(len(val_labeled)),
        "test_rows": int(len(test_labeled)),
        "predictions_df": research_predictions,
        "features_df": _feature_diagnostics_frame(diagnostic_features, feature_columns, fold_id=fold_id),
        "labels": _feature_diagnostics_labels(test_labeled, target_columns, fold_id=fold_id),
        "feature_importance": predictor.feature_importance(),
        "predictor_bundle": predictor_bundle,
        "preprocessor_bundle": preprocessor_bundle,
        "validation_metrics": validation_metrics,
        "prediction_metrics": prediction_metrics,
        "confidence_calibration": confidence_summary,
        "train_end": fold.get("train_end"),
        "val_start": fold.get("val_start"),
        "val_end": fold.get("val_end"),
        "test_start": fold.get("test_start"),
        "test_end": fold.get("test_end"),
    }
    return predictions, test_labeled, summary


def build_preprocessor(config: dict[str, Any]):
    from skyeye.products.ax1.preprocessing import FeaturePreprocessor

    preprocessor_cfg = config.get("preprocessor", {})
    return FeaturePreprocessor(
        neutralize=bool(preprocessor_cfg.get("neutralize", True)),
        winsorize_scale=preprocessor_cfg.get("winsorize_scale", 3.5),
        standardize=bool(preprocessor_cfg.get("standardize", True)),
        min_obs=int(preprocessor_cfg.get("min_obs", 5)),
        sector_optional=bool(preprocessor_cfg.get("sector_optional", True)),
    )


def build_feature_diagnostics_preprocessor(config: dict[str, Any]):
    """Build a preprocessor for feature diagnostics.

    We intentionally disable cross-sectional standardization so that diagnostics
    like mean_cross_sectional_std reflect the raw/winsorized/neutralized signal
    amplitude rather than a post-zscore artifact.
    """

    from skyeye.products.ax1.preprocessing import FeaturePreprocessor

    preprocessor_cfg = config.get("preprocessor", {})
    return FeaturePreprocessor(
        neutralize=bool(preprocessor_cfg.get("neutralize", True)),
        winsorize_scale=preprocessor_cfg.get("winsorize_scale", 3.5),
        standardize=False,
        min_obs=int(preprocessor_cfg.get("min_obs", 5)),
        sector_optional=bool(preprocessor_cfg.get("sector_optional", True)),
    )


def build_splitter(config: dict[str, Any]):
    from skyeye.products.ax1.training import SingleSplitSplitter, WalkForwardSplitter

    splitter_cfg = config.get("splitter", {})
    kind = str(splitter_cfg.get("kind", "single_split"))
    common_kwargs = {
        "train_end": splitter_cfg.get("train_end"),
        "val_months": int(splitter_cfg.get("val_months", 6)),
        "test_months": int(splitter_cfg.get("test_months", 6)),
        "embargo_days": int(splitter_cfg.get("embargo_days", 20)),
    }
    if kind == "single_split":
        return SingleSplitSplitter(**common_kwargs)
    if kind == "walk_forward":
        return WalkForwardSplitter(
            **common_kwargs,
            n_folds=int(splitter_cfg.get("n_folds", DEFAULT_WALK_FORWARD_FOLDS)),
            step_months=int(splitter_cfg.get("step_months", 1)),
        )
    raise ValueError(f"unsupported splitter.kind: {kind}")


def build_predictor(
    config: dict[str, Any],
    *,
    horizons: list[int],
    feature_columns: list[str] | None = None,
):
    model_cfg = config.get("model", {})
    kind = str(model_cfg.get("kind", "lgbm_multi_target"))
    if kind == "lgbm_multi_target":
        from skyeye.products.ax1.models.lgbm_multi_target import LGBMMultiTargetPredictor

        return LGBMMultiTargetPredictor(
            horizons=horizons,
            training_horizons=model_cfg.get("training_horizons", [5, 10, 20]),
            risk_horizon=int(model_cfg.get("risk_horizon", 10)),
            stability_horizon=int(model_cfg.get("stability_horizon", config.get("labels", {}).get("stability_horizon", 20))),
            feature_columns=feature_columns,
            liquidity_column=str(model_cfg.get("liquidity_column", "feature_dollar_volume")),
            params=dict(model_cfg.get("params") or {}),
            confidence_method=str(model_cfg.get("confidence_method", "sign_consistency")),
        )
    raise ValueError(f"unsupported AX1 model kind: {kind}")


def target_columns(config: dict[str, Any]) -> list[str]:
    model_cfg = config.get("model", {})
    training_horizons = [int(item) for item in model_cfg.get("training_horizons", [5, 10, 20])]
    risk_horizon = int(model_cfg.get("risk_horizon", 10))
    return [f"label_relative_net_return_{horizon}d" for horizon in training_horizons] + [
        f"label_volatility_{risk_horizon}d"
    ]


def prediction_horizons(config: dict[str, Any]) -> list[int]:
    labels_cfg = config.get("labels", {})
    horizons = [int(item) for item in labels_cfg.get("return_horizons", [5, 10, 20])]
    stability_horizon = int(labels_cfg.get("stability_horizon", 20))
    if stability_horizon not in horizons:
        horizons.append(stability_horizon)
    return horizons


def _feature_diagnostics_label_column(config: dict[str, Any]) -> str:
    horizon = int(config.get("labels", {}).get("stability_horizon", 20))
    return f"label_relative_net_return_{horizon}d"


def _feature_diagnostics_frame(features: pd.DataFrame, feature_columns: list[str], *, fold_id: int) -> pd.DataFrame:
    keys = [column for column in ("date", "order_book_id") if column in features.columns]
    columns = keys + [column for column in feature_columns if column in features.columns]
    frame = features.loc[:, columns].copy()
    frame["fold_id"] = int(fold_id)
    return frame


def _feature_diagnostics_labels(labels: pd.DataFrame, target_columns: list[str], *, fold_id: int) -> pd.DataFrame:
    keys = [column for column in ("date", "order_book_id") if column in labels.columns]
    label_columns = [
        column
        for column in labels.columns
        if column in set(target_columns)
        or column.startswith("label_relative_net_return_")
        or column.startswith("label_net_return_")
        or column.startswith("label_return_")
    ]
    frame = labels.loc[:, keys + list(dict.fromkeys(label_columns))].copy()
    frame["fold_id"] = int(fold_id)
    return frame


def _strip_fold_diagnostic_frames(fold_result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fold_result.items()
        if key not in {"features_df", "labels"}
    }


def _aggregate_lgbm_feature_importance(
    fold_results: list[dict[str, Any]],
    feature_columns: list[str],
) -> dict[str, Any]:
    values_by_kind: dict[str, dict[str, list[float]]] = {
        "gain": {feature: [] for feature in feature_columns},
        "split": {feature: [] for feature in feature_columns},
    }
    fold_count = 0
    for fold_result in fold_results:
        importance = fold_result.get("feature_importance") or {}
        aggregate = importance.get("aggregate") or {}
        if aggregate:
            fold_count += 1
        for kind in ("gain", "split"):
            for item in aggregate.get(kind, []) or []:
                feature = str(item.get("feature"))
                if feature in values_by_kind[kind]:
                    values_by_kind[kind][feature].append(float(item.get("importance", 0.0) or 0.0))
    return {
        "schema_version": 1,
        "feature_columns": list(feature_columns),
        "fold_count": int(fold_count),
        "aggregate": {
            kind: _format_aggregate_importance(values_by_kind[kind])
            for kind in ("gain", "split")
        },
    }


def _format_aggregate_importance(values_by_feature: dict[str, list[float]]) -> list[dict[str, Any]]:
    pairs = [
        (feature, float(sum(values) / len(values)) if values else 0.0)
        for feature, values in values_by_feature.items()
    ]
    total = sum(max(value, 0.0) for _, value in pairs)
    return [
        {
            "feature": feature,
            "importance": value,
            "normalized_importance": float(max(value, 0.0) / total) if total > 0 else 0.0,
            "rank": rank,
        }
        for rank, (feature, value) in enumerate(
            sorted(pairs, key=lambda item: (-item[1], item[0])),
            start=1,
        )
    ]


def _attach_research_labels(predictions: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if predictions is None or predictions.empty or labels is None or labels.empty:
        return predictions
    keys = [column for column in ("date", "order_book_id", "fold_id") if column in predictions.columns and column in labels.columns]
    if len(keys) < 2:
        return predictions
    label_columns = [
        column
        for column in labels.columns
        if column.startswith("label_relative_net_return_")
        or column.startswith("label_net_return_")
        or column.startswith("label_return_")
        or column.startswith("label_volatility_")
    ]
    if not label_columns:
        return predictions
    left = predictions.copy()
    right = labels[keys + label_columns].copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    left["order_book_id"] = left["order_book_id"].astype(str)
    right["order_book_id"] = right["order_book_id"].astype(str)
    return left.drop(columns=label_columns, errors="ignore").merge(right, on=keys, how="left")


def _drop_missing_targets(frame: pd.DataFrame, target_columns: list[str], *, split_name: str) -> pd.DataFrame:
    missing = [column for column in target_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{split_name} split missing target columns: {missing}")
    cleaned = frame.dropna(subset=target_columns).reset_index(drop=True)
    if cleaned.empty:
        raise ValueError(f"{split_name} split has no rows after dropping missing targets")
    return cleaned
