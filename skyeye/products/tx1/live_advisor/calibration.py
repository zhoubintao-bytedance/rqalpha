# -*- coding: utf-8 -*-
"""TX1 live advisor 的 OOS 校准统计。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_calibration_bundle(
    experiment_result: dict,
    *,
    bucket_count: int = 10,
    score_column: str = "prediction",
) -> dict:
    """从研究侧 OOS predictions_df 构建校准包。"""
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")
    frame = _collect_oos_predictions(experiment_result, score_column=score_column)
    if frame.empty:
        raise ValueError("experiment_result contains no OOS predictions for calibration")

    frame = frame.copy()
    frame["score_rank_pct"] = frame.groupby("date")[score_column].rank(
        method="average",
        pct=True,
    )
    bucket_edges = np.linspace(0.0, 1.0, bucket_count + 1)
    frame["bucket_id"] = frame["score_rank_pct"].map(
        lambda value: locate_bucket_id(value, bucket_edges)
    )

    bucket_stats = []
    for bucket_index in range(bucket_count):
        lower = float(bucket_edges[bucket_index])
        upper = float(bucket_edges[bucket_index + 1])
        bucket_id = _format_bucket_id(bucket_index)
        bucket_df = frame[frame["bucket_id"] == bucket_id].copy()
        if bucket_df.empty:
            bucket_stats.append(
                {
                    "bucket_id": bucket_id,
                    "lower": lower,
                    "upper": upper,
                    "sample_count": 0,
                    "win_rate": 0.0,
                    "mean_return": 0.0,
                    "median_return": 0.0,
                    "return_quantiles": _empty_quantile_dict(),
                    "volatility_quantiles": _empty_quantile_dict(),
                    "max_drawdown_quantiles": _empty_quantile_dict(),
                }
            )
            continue
        bucket_stats.append(
            {
                "bucket_id": bucket_id,
                "lower": lower,
                "upper": upper,
                "sample_count": int(len(bucket_df)),
                "win_rate": float((bucket_df["label_return_raw"] > 0).mean()),
                "mean_return": float(bucket_df["label_return_raw"].mean()),
                "median_return": float(bucket_df["label_return_raw"].median()),
                "return_quantiles": _quantile_dict(bucket_df["label_return_raw"]),
                "volatility_quantiles": _quantile_dict(bucket_df["label_volatility_horizon"]),
                "max_drawdown_quantiles": _quantile_dict(bucket_df["label_max_drawdown_horizon"]),
            }
        )

    return {
        "score_column": score_column,
        "bucket_edges": [float(edge) for edge in bucket_edges],
        "bucket_stats": bucket_stats,
        "summary": {
            "oos_rows": int(len(frame)),
            "oos_dates": int(frame["date"].nunique()),
            "bucket_count": int(bucket_count),
        },
        "score_sanity_reference": _build_score_sanity_reference(frame, score_column=score_column),
    }


def lookup_calibration_bucket(calibration_bundle: dict, score_rank_pct: float) -> dict:
    """按分位数找到对应 bucket 的校准统计。"""
    bucket_edges = calibration_bundle.get("bucket_edges", [])
    bucket_id = locate_bucket_id(score_rank_pct, bucket_edges)
    for bucket_stat in calibration_bundle.get("bucket_stats", []):
        if bucket_stat.get("bucket_id") == bucket_id:
            return bucket_stat
    raise KeyError("no calibration bucket found for score_rank_pct={}".format(score_rank_pct))


def locate_bucket_id(score_rank_pct: float, bucket_edges: list[float] | np.ndarray) -> str:
    """把分位数映射到 bucket_id。"""
    if score_rank_pct is None or not np.isfinite(score_rank_pct):
        raise ValueError("score_rank_pct must be finite")
    if len(bucket_edges) < 2:
        raise ValueError("bucket_edges must contain at least 2 values")
    clipped = min(max(float(score_rank_pct), 0.0), 1.0)
    bucket_index = int(np.searchsorted(np.asarray(bucket_edges[1:], dtype=float), clipped, side="left"))
    bucket_index = min(max(bucket_index, 0), len(bucket_edges) - 2)
    return _format_bucket_id(bucket_index)


def _collect_oos_predictions(experiment_result: dict, score_column: str) -> pd.DataFrame:
    """收集所有 fold 的 OOS predictions_df，并校验必需列。"""
    frames = []
    required_columns = {
        "date",
        "order_book_id",
        score_column,
        "label_return_raw",
        "label_volatility_horizon",
        "label_max_drawdown_horizon",
    }
    for fold_index, fold_result in enumerate(experiment_result.get("fold_results", []), start=1):
        frame = fold_result.get("predictions_df")
        if frame is None or len(frame) == 0:
            continue
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                "fold {} predictions_df missing required columns: {}".format(
                    fold_index,
                    ", ".join(sorted(missing_columns)),
                )
            )
        current = frame.loc[:, sorted(required_columns)].copy()
        current["date"] = pd.to_datetime(current["date"])
        current["fold_index"] = int(fold_index)
        frames.append(current)
    if not frames:
        return pd.DataFrame(columns=sorted(required_columns))
    return pd.concat(frames, ignore_index=True).sort_values(["date", "order_book_id"]).reset_index(drop=True)


def _quantile_dict(series: pd.Series) -> dict:
    """把序列压成固定分位数字典，便于对外展示风险区间。"""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return _empty_quantile_dict()
    return {
        "p10": float(values.quantile(0.10)),
        "p25": float(values.quantile(0.25)),
        "p50": float(values.quantile(0.50)),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
    }


def _empty_quantile_dict() -> dict:
    """返回空分位数结构，避免调用方再做键存在判断。"""
    return {
        "p10": 0.0,
        "p25": 0.0,
        "p50": 0.0,
        "p75": 0.0,
        "p90": 0.0,
    }


def _build_score_sanity_reference(frame: pd.DataFrame, *, score_column: str) -> dict:
    """提取运行时 stop-serve 需要的分布参考值。"""
    daily_stats = []
    for _, day_df in frame.groupby("date", sort=True):
        valid_scores = pd.to_numeric(day_df[score_column], errors="coerce").dropna()
        if valid_scores.empty:
            continue
        daily_stats.append(
            {
                "prediction_std": float(valid_scores.std(ddof=0)),
                "top_spread": float(valid_scores.quantile(0.95) - valid_scores.quantile(0.50)),
            }
        )
    stats_frame = pd.DataFrame(daily_stats)
    if stats_frame.empty:
        return {
            "prediction_std_p01": 0.0,
            "prediction_std_p05": 0.0,
            "top_spread_p01": 0.0,
            "top_spread_p05": 0.0,
            "n_days": 0,
        }
    return {
        "prediction_std_p01": float(stats_frame["prediction_std"].quantile(0.01)),
        "prediction_std_p05": float(stats_frame["prediction_std"].quantile(0.05)),
        "top_spread_p01": float(stats_frame["top_spread"].quantile(0.01)),
        "top_spread_p05": float(stats_frame["top_spread"].quantile(0.05)),
        "n_days": int(len(stats_frame)),
    }


def _format_bucket_id(bucket_index: int) -> str:
    """统一 bucket 标识格式，便于 runtime 查找。"""
    return "b{:02d}".format(int(bucket_index))
