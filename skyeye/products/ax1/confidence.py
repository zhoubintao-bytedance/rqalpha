"""AX1 validation-fold confidence calibration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_confidence_calibrator(
    validation_predictions: pd.DataFrame,
    validation_labels: pd.DataFrame,
    *,
    label_column: str | None = None,
    outcome_column: str | None = None,
    bucket_count: int = 5,
    min_samples: int = 30,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit a small bucket calibrator from validation-fold predictions."""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    raw_column = _raw_confidence_column(validation_predictions)
    merged = _merge_predictions_and_labels(
        validation_predictions,
        validation_labels,
        extra_columns=[label_column, outcome_column],
    )
    label_column = label_column or _select_label_column(merged)
    if outcome_column is not None and outcome_column not in merged.columns:
        return _fallback_calibrator(raw_column, label_column, "missing_outcome_column", outcome_column=outcome_column)
    if label_column is None or label_column not in merged.columns:
        return _fallback_calibrator(raw_column, label_column, "missing_label_column", outcome_column=outcome_column)
    success_column = outcome_column or label_column
    working = merged.copy()
    working[raw_column] = pd.to_numeric(working[raw_column], errors="coerce")
    working[label_column] = pd.to_numeric(working[label_column], errors="coerce")
    working[success_column] = pd.to_numeric(working[success_column], errors="coerce")
    working = working.dropna(subset=[raw_column, label_column, success_column]).reset_index(drop=True)
    if len(working) < int(min_samples):
        return _fallback_calibrator(
            raw_column,
            label_column,
            "insufficient_validation_samples",
            outcome_column=outcome_column,
            sample_count=len(working),
        )
    if working[raw_column].nunique(dropna=True) < 2:
        return _fallback_calibrator(
            raw_column,
            label_column,
            "constant_raw_confidence",
            outcome_column=outcome_column,
            sample_count=len(working),
        )

    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    edges = np.unique(working[raw_column].quantile(quantiles).to_numpy(dtype=float))
    if len(edges) < 2:
        return _fallback_calibrator(
            raw_column,
            label_column,
            "degenerate_bucket_edges",
            outcome_column=outcome_column,
            sample_count=len(working),
        )
    edges[0] = min(edges[0], float(working[raw_column].min()))
    edges[-1] = max(edges[-1], float(working[raw_column].max()))

    bucket_ids = [_locate_bucket(value, edges) for value in working[raw_column]]
    working["confidence_bucket"] = bucket_ids
    global_hit_rate = float((working[success_column] > 0.0).mean())
    buckets = []
    for bucket_index in range(len(edges) - 1):
        bucket_df = working[working["confidence_bucket"] == bucket_index]
        if bucket_df.empty:
            calibrated_confidence = global_hit_rate
            mean_return = 0.0
        else:
            calibrated_confidence = float((bucket_df[success_column] > 0.0).mean())
            mean_return = float(bucket_df[label_column].mean())
        buckets.append(
            {
                "bucket_index": int(bucket_index),
                "lower": float(edges[bucket_index]),
                "upper": float(edges[bucket_index + 1]),
                "sample_count": int(len(bucket_df)),
                "hit_rate": calibrated_confidence,
                "mean_return": mean_return,
                "calibrated_confidence": calibrated_confidence,
            }
        )

    calibrator = {
        "schema_version": 1,
        "method": "validation_bucket_hit_rate",
        "status": "calibrated",
        "raw_column": raw_column,
        "label_column": label_column,
        "outcome_column": outcome_column,
        "bucket_edges": [float(edge) for edge in edges],
        "buckets": buckets,
        "default_confidence": global_hit_rate,
    }
    summary = _summary_from_calibrator(calibrator, sample_count=len(working))
    return calibrator, summary


def apply_confidence_calibration(predictions: pd.DataFrame, calibrator: dict[str, Any]) -> pd.DataFrame:
    """Apply a fitted confidence calibrator while preserving confidence_raw."""
    if predictions is None or predictions.empty:
        return predictions
    result = predictions.copy()
    raw_column = str((calibrator or {}).get("raw_column") or _raw_confidence_column(result))
    if raw_column not in result.columns and "confidence" in result.columns:
        result[raw_column] = result["confidence"]
    if raw_column not in result.columns:
        result[raw_column] = 0.0
    if "confidence_raw" not in result.columns:
        result["confidence_raw"] = pd.to_numeric(result[raw_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        raw_column = "confidence_raw"
    if (calibrator or {}).get("status") != "calibrated":
        result["confidence"] = pd.to_numeric(result[raw_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        return result

    edges = np.asarray(calibrator.get("bucket_edges") or [], dtype=float)
    buckets = {int(item.get("bucket_index", idx)): item for idx, item in enumerate(calibrator.get("buckets") or [])}
    default_confidence = float(calibrator.get("default_confidence", 0.0) or 0.0)
    raw_values = pd.to_numeric(result[raw_column], errors="coerce").fillna(0.0)
    calibrated = []
    for value in raw_values:
        bucket_index = _locate_bucket(float(value), edges)
        bucket = buckets.get(bucket_index, {})
        calibrated.append(float(bucket.get("calibrated_confidence", default_confidence)))
    result["confidence"] = pd.Series(calibrated, index=result.index).clip(0.0, 1.0)
    return result


def aggregate_confidence_calibration(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    folds = [dict(item.get("confidence_calibration") or {}) for item in fold_results]
    folds = [item for item in folds if item]
    calibrated_count = sum(1 for item in folds if item.get("status") == "calibrated")
    fallback_count = sum(1 for item in folds if item.get("status") == "fallback")
    sample_count = sum(int(item.get("sample_count", 0) or 0) for item in folds)
    return {
        "schema_version": 1,
        "status": "calibrated" if calibrated_count else "fallback",
        "fold_count": int(len(folds)),
        "calibrated_count": int(calibrated_count),
        "fallback_count": int(fallback_count),
        "sample_count": int(sample_count),
        "folds": folds,
    }


def build_tradable_confidence_diagnostic(
    predictions: pd.DataFrame,
    tradable_outcome: dict[str, Any] | None,
    *,
    bucket_count: int = 5,
    min_samples: int = 30,
) -> dict[str, Any]:
    """Build a post-replay diagnostic that maps confidence to tradable net success."""
    base = {
        "schema_version": 1,
        "method": "tradable_net_success_bucket_hit_rate",
        "outcome_column": "tradable_net_success",
        "status": "fallback",
        "sample_count": 0,
        "tradable_outcome_date_count": int((tradable_outcome or {}).get("date_count", 0) or 0),
        "outcome_success_rate": 0.0,
        "mean_tradable_net_return": 0.0,
        "buckets": [],
    }
    net_by_date = (tradable_outcome or {}).get("net_return_by_date") or {}
    if predictions is None or predictions.empty or not net_by_date:
        return {**base, "fallback_reason": "missing_predictions_or_tradable_outcome"}
    if not {"date", "order_book_id"}.issubset(predictions.columns):
        return {**base, "fallback_reason": "missing_panel_keys"}

    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    labels = frame[["date", "order_book_id"]].copy()
    labels["tradable_net_return"] = labels["date"].map(lambda value: net_by_date.get(str(pd.Timestamp(value).date())))
    labels["tradable_net_success"] = labels["tradable_net_return"]
    labels = labels.dropna(subset=["tradable_net_return"]).reset_index(drop=True)
    if labels.empty:
        return {**base, "fallback_reason": "no_prediction_dates_in_tradable_outcome"}

    calibrator, summary = fit_confidence_calibrator(
        frame,
        labels,
        label_column="tradable_net_return",
        outcome_column="tradable_net_success",
        bucket_count=bucket_count,
        min_samples=min_samples,
    )
    returns = pd.to_numeric(labels["tradable_net_return"], errors="coerce").dropna()
    return {
        **base,
        "status": summary["status"],
        "raw_column": summary["raw_column"],
        "label_column": summary["label_column"],
        "sample_count": summary["sample_count"],
        "bucket_count": summary["bucket_count"],
        "buckets": summary["buckets"],
        "fallback_reason": summary.get("fallback_reason"),
        "outcome_success_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "mean_tradable_net_return": float(returns.mean()) if len(returns) else 0.0,
        "calibrator_status": calibrator.get("status"),
    }


def _merge_predictions_and_labels(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    extra_columns: list[str | None] | None = None,
) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()
    if labels is None or labels.empty:
        return predictions.copy()
    keys = [column for column in ("date", "order_book_id", "fold_id") if column in predictions.columns and column in labels.columns]
    if len(keys) < 2:
        return predictions.copy()
    left = predictions.copy()
    right = labels.copy()
    left["date"] = pd.to_datetime(left["date"])
    right["date"] = pd.to_datetime(right["date"])
    left["order_book_id"] = left["order_book_id"].astype(str)
    right["order_book_id"] = right["order_book_id"].astype(str)
    label_columns = [
        column
        for column in right.columns
        if column.startswith("label_relative_net_return_")
        or column.startswith("label_net_return_")
        or column.startswith("label_return_")
    ]
    payload_columns = list(dict.fromkeys([*label_columns, *(column for column in (extra_columns or []) if column)]))
    payload_columns = [column for column in payload_columns if column in right.columns and column not in keys]
    return left.merge(right[keys + payload_columns].drop_duplicates(keys, keep="last"), on=keys, how="left")


def _raw_confidence_column(frame: pd.DataFrame) -> str:
    if frame is not None and "confidence_raw" in frame.columns:
        return "confidence_raw"
    return "confidence"


def _select_label_column(frame: pd.DataFrame) -> str | None:
    for column in (
        "label_relative_net_return_20d",
        "label_relative_net_return_10d",
        "label_net_return_20d",
        "label_net_return_10d",
        "label_return_20d",
        "label_return_10d",
    ):
        if column in frame.columns:
            return column
    columns = sorted(
        column
        for column in frame.columns
        if column.startswith("label_relative_net_return_")
        or column.startswith("label_net_return_")
        or column.startswith("label_return_")
    )
    return columns[0] if columns else None


def _fallback_calibrator(
    raw_column: str,
    label_column: str | None,
    reason: str,
    *,
    outcome_column: str | None = None,
    sample_count: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    calibrator = {
        "schema_version": 1,
        "method": "identity",
        "status": "fallback",
        "raw_column": raw_column,
        "label_column": label_column,
        "outcome_column": outcome_column,
        "bucket_edges": [0.0, 1.0],
        "buckets": [],
        "default_confidence": None,
        "fallback_reason": reason,
    }
    return calibrator, _summary_from_calibrator(calibrator, sample_count=sample_count)


def _summary_from_calibrator(calibrator: dict[str, Any], *, sample_count: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": str(calibrator.get("status")),
        "method": str(calibrator.get("method")),
        "raw_column": str(calibrator.get("raw_column")),
        "label_column": calibrator.get("label_column"),
        "outcome_column": calibrator.get("outcome_column"),
        "sample_count": int(sample_count),
        "bucket_count": int(max(0, len(calibrator.get("bucket_edges") or []) - 1)),
        "buckets": list(calibrator.get("buckets") or []),
        "fallback_reason": calibrator.get("fallback_reason"),
    }


def _locate_bucket(value: float, edges: np.ndarray) -> int:
    if len(edges) < 2:
        return 0
    clipped = min(max(float(value), float(edges[0])), float(edges[-1]))
    bucket_index = int(np.searchsorted(edges[1:], clipped, side="right"))
    return min(max(bucket_index, 0), len(edges) - 2)
