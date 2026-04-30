# -*- coding: utf-8 -*-
"""AX1 experiment JSON persistence helpers."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_FILENAME = "experiment.json"
ARTIFACT_SCHEMA_VERSION = 1


def save_experiment(
    result: dict,
    output_dir: str | Path,
    experiment_name: str | None = None,
) -> Path:
    """保存 AX1 实验结果到 `experiment.json`，并返回实验目录。"""
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")

    experiment_dir = Path(output_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    fold_artifacts = _write_fold_artifacts(result, experiment_dir)
    payload = make_json_compatible(_build_experiment_payload(result, fold_artifacts))
    if experiment_name is not None:
        payload["experiment_name"] = experiment_name

    with (experiment_dir / EXPERIMENT_FILENAME).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)

    return experiment_dir


def load_experiment(experiment_dir: str | Path) -> dict:
    """从实验目录读取 `experiment.json`。"""
    experiment_path = Path(experiment_dir) / EXPERIMENT_FILENAME
    with experiment_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_json_compatible(value: Any) -> Any:
    """递归转换 pandas / pathlib / numpy 类型，确保可安全写入 JSON。"""
    if isinstance(value, pd.DataFrame):
        return make_json_compatible(value.to_dict(orient="records"))

    if isinstance(value, dict):
        return {
            _json_key(key): make_json_compatible(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_compatible(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    if isinstance(value, np.generic):
        return make_json_compatible(value.item())

    if isinstance(value, np.ndarray):
        return make_json_compatible(value.tolist())

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if _is_missing_scalar(value):
        return None

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def _build_experiment_payload(result: dict, fold_artifacts: list[dict[str, Any]]) -> dict:
    payload = dict(result)
    if fold_artifacts:
        payload["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION
        payload["folds"] = fold_artifacts
        payload["training_summary"] = _training_summary_with_refs(
            result.get("training_summary") or {},
            fold_artifacts,
        )
    return payload


def _training_summary_with_refs(
    training_summary: dict[str, Any],
    fold_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = dict(training_summary)
    artifact_by_fold = {
        int(item.get("fold_id", item.get("index", 0))): item
        for item in fold_artifacts
    }
    fold_results = []
    for raw_fold in training_summary.get("fold_results") or []:
        if not isinstance(raw_fold, dict):
            continue
        fold_id = int(raw_fold.get("fold_id", len(fold_results)))
        fold_payload = {
            key: value
            for key, value in raw_fold.items()
            if key not in {
                "features_df",
                "labels",
                "predictions_df",
                "predictor_bundle",
                "preprocessor_bundle",
            }
        }
        fold_payload.update(_fold_ref_fields(artifact_by_fold.get(fold_id, {})))
        fold_results.append(fold_payload)
    summary["fold_results"] = fold_results
    if isinstance(summary.get("aggregate_predictions_df"), pd.DataFrame):
        summary.pop("aggregate_predictions_df", None)
    return summary


def _write_fold_artifacts(result: dict, experiment_dir: Path) -> list[dict[str, Any]]:
    training_summary = result.get("training_summary")
    if not isinstance(training_summary, dict):
        return []
    raw_fold_results = training_summary.get("fold_results")
    if not isinstance(raw_fold_results, list) or not raw_fold_results:
        return []

    weights = _records_to_frame(result.get("target_weights"))
    orders = _records_to_frame(result.get("orders"))
    fold_artifacts: list[dict[str, Any]] = []
    fold_count = len(raw_fold_results)
    for index, raw_fold in enumerate(raw_fold_results):
        if not isinstance(raw_fold, dict):
            continue
        fold_id = int(raw_fold.get("fold_id", index))
        fold_dir = experiment_dir / "folds" / f"fold_{fold_id:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        refs: dict[str, str] = {}
        predictions = raw_fold.get("predictions_df")
        if isinstance(predictions, pd.DataFrame):
            refs["predictions_ref"] = _write_parquet(
                _normalize_frame_dates(predictions),
                fold_dir / "predictions.parquet",
                experiment_dir,
            )
        labels = raw_fold.get("labels")
        if isinstance(labels, pd.DataFrame):
            refs["labels_ref"] = _write_parquet(
                _normalize_frame_dates(labels),
                fold_dir / "labels.parquet",
                experiment_dir,
            )

        fold_weights = _select_fold_frame(weights, fold_id=fold_id, fold=raw_fold, fold_count=fold_count)
        refs["weights_ref"] = _write_parquet(
            _normalize_frame_dates(fold_weights),
            fold_dir / "weights.parquet",
            experiment_dir,
        )
        fold_orders = _select_fold_frame(orders, fold_id=fold_id, fold=raw_fold, fold_count=fold_count)
        refs["orders_ref"] = _write_parquet(
            _normalize_frame_dates(fold_orders),
            fold_dir / "orders.parquet",
            experiment_dir,
        )

        predictor_bundle = raw_fold.get("predictor_bundle") or raw_fold.get("model_bundle")
        if isinstance(predictor_bundle, dict) and predictor_bundle:
            refs["model_bundle_ref"] = _write_json_artifact(
                predictor_bundle,
                fold_dir / "model_bundle.json",
                experiment_dir,
            )
        preprocessor_bundle = raw_fold.get("preprocessor_bundle")
        if not isinstance(preprocessor_bundle, dict) or not preprocessor_bundle:
            preprocessor_bundle = (
                predictor_bundle.get("preprocessor_bundle")
                if isinstance(predictor_bundle, dict)
                else None
            )
        if isinstance(preprocessor_bundle, dict) and preprocessor_bundle:
            refs["preprocessor_bundle_ref"] = _write_json_artifact(
                preprocessor_bundle,
                fold_dir / "preprocessor_bundle.json",
                experiment_dir,
            )

        fold_meta = {
            "fold_id": fold_id,
            "index": fold_id,
            "path": _relative_path(fold_dir, experiment_dir),
            "date_range": {
                "train_end": make_json_compatible(raw_fold.get("train_end")),
                "val_start": make_json_compatible(raw_fold.get("val_start")),
                "val_end": make_json_compatible(raw_fold.get("val_end")),
                "test_start": make_json_compatible(raw_fold.get("test_start")),
                "test_end": make_json_compatible(raw_fold.get("test_end")),
            },
            "row_counts": {
                "train": int(raw_fold.get("train_rows", 0) or 0),
                "val": int(raw_fold.get("val_rows", 0) or 0),
                "test": int(raw_fold.get("test_rows", 0) or 0),
                "weights": int(len(fold_weights)),
                "orders": int(len(fold_orders)),
            },
        }
        fold_meta.update(refs)
        _write_json_payload(fold_dir / "fold_metadata.json", fold_meta)
        fold_meta["fold_metadata_ref"] = _relative_path(fold_dir / "fold_metadata.json", experiment_dir)
        fold_artifacts.append(fold_meta)
    return fold_artifacts


def _fold_ref_fields(fold_artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fold_artifact.items()
        if key.endswith("_ref") or key in {"path", "date_range", "row_counts"}
    }


def _records_to_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, list):
        return pd.DataFrame(value)
    if isinstance(value, tuple):
        return pd.DataFrame(list(value))
    return pd.DataFrame()


def _select_fold_frame(
    frame: pd.DataFrame,
    *,
    fold_id: int,
    fold: dict[str, Any],
    fold_count: int,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "fold_id" in frame.columns:
        return frame.loc[frame["fold_id"] == fold_id].copy()
    if fold_count == 1:
        return frame.copy()
    if "date" in frame.columns:
        start = _optional_timestamp(fold.get("test_start"))
        end = _optional_timestamp(fold.get("test_end"))
        if start is not None or end is not None:
            dates = pd.to_datetime(frame["date"], errors="coerce")
            mask = pd.Series(True, index=frame.index)
            if start is not None:
                mask &= dates >= start
            if end is not None:
                mask &= dates <= end
            return frame.loc[mask].copy()
    return pd.DataFrame(columns=frame.columns)


def _optional_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or _is_missing_scalar(value):
        return None
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return None


def _normalize_frame_dates(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    payload = frame.copy()
    if "date" in payload.columns:
        payload["date"] = pd.to_datetime(payload["date"], errors="coerce")
    return payload


def _write_parquet(frame: pd.DataFrame, path: Path, experiment_dir: Path) -> str:
    frame.to_parquet(path, index=False)
    return _relative_path(path, experiment_dir)


def _write_json_artifact(payload: dict[str, Any], path: Path, experiment_dir: Path) -> str:
    _write_json_payload(path, payload)
    return _relative_path(path, experiment_dir)


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_compatible(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)


def _relative_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _json_key(key: Any) -> str:
    """JSON object key 统一落成字符串，避免 Path/Timestamp 等类型写盘失败。"""
    return str(make_json_compatible(key))


def _is_missing_scalar(value: Any) -> bool:
    """识别 pandas/numpy 的缺失标量；非标量容器不在这里处理。"""
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)
