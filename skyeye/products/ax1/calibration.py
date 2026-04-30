# -*- coding: utf-8 -*-
"""AX1 OOS score calibration helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_calibration_bundle(
    experiment_result: dict[str, Any],
    bucket_count: int = 10,
    score_column: str = "expected_relative_net_return_10d",
) -> dict[str, Any]:
    """Build fixed-width per-date pct-rank calibration buckets from OOS folds."""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    frame = _collect_oos_predictions(experiment_result, score_column=score_column)
    if frame.empty:
        raise ValueError("experiment_result contains no OOS predictions for calibration")

    return_label_column = _select_return_label_column(frame)
    if return_label_column is None:
        raise ValueError("OOS predictions missing AX1 return label column")
    volatility_label_column = _select_volatility_label_column(frame)

    working = frame.copy()
    working[score_column] = pd.to_numeric(working[score_column], errors="coerce")
    working[return_label_column] = pd.to_numeric(working[return_label_column], errors="coerce")
    working = working.dropna(subset=[score_column, return_label_column]).reset_index(drop=True)
    if working.empty:
        raise ValueError("experiment_result contains no finite OOS scores and return labels for calibration")
    working["score_rank_pct"] = working.groupby("date")[score_column].rank(method="average", pct=True)
    bucket_edges = np.linspace(0.0, 1.0, bucket_count + 1)
    working["bucket_id"] = working["score_rank_pct"].map(lambda value: _locate_bucket_id(value, bucket_edges))

    bucket_stats = []
    for bucket_index in range(bucket_count):
        lower = float(bucket_edges[bucket_index])
        upper = float(bucket_edges[bucket_index + 1])
        bucket_id = _format_bucket_id(bucket_index)
        bucket_df = working[working["bucket_id"] == bucket_id].copy()
        if bucket_df.empty:
            bucket_stats.append(_empty_bucket(bucket_id, lower, upper, volatility_label_column is not None))
            continue

        returns = pd.to_numeric(bucket_df[return_label_column], errors="coerce")
        bucket_stats.append(
            {
                "bucket_id": bucket_id,
                "lower": lower,
                "upper": upper,
                "sample_count": int(len(bucket_df)),
                "win_rate": float((returns > 0).mean()),
                "mean_return": float(returns.mean()) if returns.notna().any() else 0.0,
                "median_return": float(returns.median()) if returns.notna().any() else 0.0,
                "return_quantiles": _quantile_dict(returns),
                "volatility_quantiles": (
                    _quantile_dict(bucket_df[volatility_label_column])
                    if volatility_label_column is not None
                    else {}
                ),
            }
        )

    return {
        "score_column": score_column,
        "return_label_column": return_label_column,
        "volatility_label_column": volatility_label_column,
        "bucket_edges": [float(edge) for edge in bucket_edges],
        "bucket_stats": bucket_stats,
        "summary": {
            "oos_rows": int(len(working)),
            "oos_dates": int(working["date"].nunique()),
            "bucket_count": int(bucket_count),
        },
        "score_sanity_reference": _build_score_sanity_reference(working, score_column=score_column),
    }


def lookup_calibration_bucket(calibration_bundle: dict[str, Any], score_rank_pct: float) -> dict[str, Any]:
    """Lookup calibration stats by per-date score pct-rank."""
    bucket_id = _locate_bucket_id(score_rank_pct, calibration_bundle.get("bucket_edges", []))
    for bucket_stat in calibration_bundle.get("bucket_stats", []):
        if bucket_stat.get("bucket_id") == bucket_id:
            return bucket_stat
    raise KeyError("no calibration bucket found for score_rank_pct={}".format(score_rank_pct))


def _collect_oos_predictions(experiment_result: dict[str, Any], *, score_column: str) -> pd.DataFrame:
    fold_results = (
        (experiment_result.get("training_summary") or {}).get("fold_results")
        or experiment_result.get("fold_results")
        or []
    )
    frames = []
    for fold_index, fold_result in enumerate(fold_results, start=1):
        frame = fold_result.get("predictions_df")
        if frame is None or len(frame) == 0:
            continue
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame)
        if score_column not in frame.columns:
            raise ValueError(f"fold {fold_index} predictions_df missing score column: {score_column}")
        if "date" not in frame.columns or "order_book_id" not in frame.columns:
            raise ValueError(f"fold {fold_index} predictions_df missing panel keys")
        current = frame.copy()
        current["date"] = pd.to_datetime(current["date"])
        current["fold_index"] = int(fold_index)
        frames.append(current)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["date", "order_book_id"]).reset_index(drop=True)


def _select_return_label_column(frame: pd.DataFrame) -> str | None:
    for column in ("label_net_return_10d", "label_return_10d"):
        if column in frame.columns:
            return column
    net_columns = sorted(column for column in frame.columns if column.startswith("label_net_return_"))
    if net_columns:
        return net_columns[0]
    return_columns = sorted(column for column in frame.columns if column.startswith("label_return_"))
    return return_columns[0] if return_columns else None


def _select_volatility_label_column(frame: pd.DataFrame) -> str | None:
    columns = sorted(column for column in frame.columns if column.startswith("label_volatility_"))
    return columns[0] if columns else None


def _locate_bucket_id(score_rank_pct: float, bucket_edges: list[float] | np.ndarray) -> str:
    if score_rank_pct is None or not np.isfinite(score_rank_pct):
        raise ValueError("score_rank_pct must be finite")
    if len(bucket_edges) < 2:
        raise ValueError("bucket_edges must contain at least 2 values")
    clipped = min(max(float(score_rank_pct), 0.0), 1.0)
    bucket_index = int(np.searchsorted(np.asarray(bucket_edges[1:], dtype=float), clipped, side="left"))
    bucket_index = min(max(bucket_index, 0), len(bucket_edges) - 2)
    return _format_bucket_id(bucket_index)


def _format_bucket_id(bucket_index: int) -> str:
    return "b{:02d}".format(int(bucket_index))


def _empty_bucket(bucket_id: str, lower: float, upper: float, has_volatility: bool) -> dict[str, Any]:
    return {
        "bucket_id": bucket_id,
        "lower": lower,
        "upper": upper,
        "sample_count": 0,
        "win_rate": 0.0,
        "mean_return": 0.0,
        "median_return": 0.0,
        "return_quantiles": {},
        "volatility_quantiles": _empty_quantile_dict() if has_volatility else {},
    }


def _quantile_dict(series: pd.Series) -> dict[str, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def _empty_quantile_dict() -> dict[str, float]:
    return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}


def _build_score_sanity_reference(frame: pd.DataFrame, *, score_column: str) -> dict[str, Any]:
    daily_stats = []
    for _, day_df in frame.groupby("date", sort=True):
        scores = pd.to_numeric(day_df[score_column], errors="coerce").dropna()
        if scores.empty:
            continue
        daily_stats.append(
            {
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "score_mean": float(scores.mean()),
                "score_std": float(scores.std(ddof=0)),
                "score_p05": float(scores.quantile(0.05)),
                "score_p50": float(scores.quantile(0.50)),
                "score_p95": float(scores.quantile(0.95)),
                "score_top_spread": float(scores.quantile(0.95) - scores.quantile(0.50)),
            }
        )
    stats = pd.DataFrame(daily_stats)
    if stats.empty:
        return {
            "score_min_p50": 0.0,
            "score_max_p50": 0.0,
            "score_mean_p50": 0.0,
            "score_std_p05": 0.0,
            "score_top_spread_p05": 0.0,
            "n_days": 0,
        }
    return {
        "score_min_p50": float(stats["score_min"].quantile(0.50)),
        "score_max_p50": float(stats["score_max"].quantile(0.50)),
        "score_mean_p50": float(stats["score_mean"].quantile(0.50)),
        "score_std_p01": float(stats["score_std"].quantile(0.01)),
        "score_std_p05": float(stats["score_std"].quantile(0.05)),
        "score_top_spread_p01": float(stats["score_top_spread"].quantile(0.01)),
        "score_top_spread_p05": float(stats["score_top_spread"].quantile(0.05)),
        "score_p05_p50": float(stats["score_p05"].quantile(0.50)),
        "score_p50_p50": float(stats["score_p50"].quantile(0.50)),
        "score_p95_p50": float(stats["score_p95"].quantile(0.50)),
        "n_days": int(len(stats)),
    }
