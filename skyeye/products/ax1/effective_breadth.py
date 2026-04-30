# -*- coding: utf-8 -*-
"""Correlation-adjusted ETF universe breadth helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_MIN_EFFECTIVE_BREADTH = 15.0
DEFAULT_MIN_BREADTH_RATIO = 0.50


def effective_breadth_from_covariance(
    covariance: pd.DataFrame,
    *,
    as_of_date: Any | None = None,
    order_book_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Compute eigenvalue effective breadth from a covariance matrix."""

    matrix = _prepare_covariance(covariance, order_book_ids=order_book_ids)
    nominal_count = int(len(matrix))
    if nominal_count == 0:
        return _single_summary(
            as_of_date=as_of_date,
            nominal_count=0,
            effective_breadth=0.0,
            eigenvalue_count=0,
            dropped_zero_variance_count=0,
        )

    diagonal = np.diag(matrix.to_numpy(dtype=float))
    positive_mask = diagonal > 0.0
    dropped_zero_variance_count = int((~positive_mask).sum())
    if not positive_mask.any():
        return _single_summary(
            as_of_date=as_of_date,
            nominal_count=nominal_count,
            effective_breadth=0.0,
            eigenvalue_count=0,
            dropped_zero_variance_count=dropped_zero_variance_count,
        )

    usable = matrix.iloc[positive_mask, positive_mask]
    std = np.sqrt(np.diag(usable.to_numpy(dtype=float)))
    corr = usable.to_numpy(dtype=float) / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    numerator = float(np.sum(eigenvalues) ** 2)
    denominator = float(np.sum(eigenvalues**2))
    effective_breadth = numerator / denominator if denominator > 0.0 else 0.0
    return _single_summary(
        as_of_date=as_of_date,
        nominal_count=nominal_count,
        effective_breadth=effective_breadth,
        eigenvalue_count=int(len(eigenvalues)),
        dropped_zero_variance_count=dropped_zero_variance_count,
    )


def summarize_effective_breadth_by_date(
    risk_model_by_date: dict[Any, Any],
    *,
    order_book_ids: Iterable[str] | None = None,
    min_effective_breadth: float = DEFAULT_MIN_EFFECTIVE_BREADTH,
    min_breadth_ratio: float = DEFAULT_MIN_BREADTH_RATIO,
) -> dict[str, Any]:
    """Summarize per-date risk models into a JSON-friendly breadth report."""

    daily: list[dict[str, Any]] = []
    for date, risk_model in sorted((risk_model_by_date or {}).items(), key=lambda item: pd.Timestamp(item[0])):
        if risk_model is None or not hasattr(risk_model, "get_covariance_matrix"):
            continue
        covariance = risk_model.get_covariance_matrix()
        if covariance is None:
            continue
        daily.append(
            effective_breadth_from_covariance(
                covariance,
                as_of_date=pd.Timestamp(date),
                order_book_ids=order_book_ids,
            )
        )

    if not daily:
        return {
            "schema_version": 1,
            "date_count": 0,
            "latest": {},
            "mean_effective_breadth": 0.0,
            "mean_nominal_count": 0.0,
            "mean_breadth_ratio": 0.0,
            "p5_effective_breadth": 0.0,
            "p5_breadth_ratio": 0.0,
            "min_effective_breadth": 0.0,
            "warning_count": 0,
            "warnings": [],
            "daily": [],
        }

    latest = daily[-1]
    effective_values = [float(item["effective_breadth"]) for item in daily]
    nominal_values = [float(item["nominal_count"]) for item in daily]
    ratio_values = [float(item["breadth_ratio"]) for item in daily]
    warnings = _breadth_warnings(
        latest=latest,
        mean_effective_breadth=float(np.mean(effective_values)),
        mean_breadth_ratio=float(np.mean(ratio_values)),
        p5_effective_breadth=float(np.percentile(effective_values, 5)),
        p5_breadth_ratio=float(np.percentile(ratio_values, 5)),
        min_effective_breadth=float(min_effective_breadth),
        min_breadth_ratio=float(min_breadth_ratio),
    )
    return {
        "schema_version": 1,
        "date_count": int(len(daily)),
        "latest": latest,
        "mean_effective_breadth": float(np.mean(effective_values)),
        "mean_nominal_count": float(np.mean(nominal_values)),
        "mean_breadth_ratio": float(np.mean(ratio_values)),
        "p5_effective_breadth": float(np.percentile(effective_values, 5)),
        "p5_breadth_ratio": float(np.percentile(ratio_values, 5)),
        "min_effective_breadth": float(np.min(effective_values)),
        "warning_count": int(len(warnings)),
        "warnings": warnings,
        "daily": daily,
    }


def _prepare_covariance(covariance: pd.DataFrame, *, order_book_ids: Iterable[str] | None) -> pd.DataFrame:
    if covariance is None or covariance.empty:
        return pd.DataFrame()
    frame = covariance.copy()
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if order_book_ids is not None:
        requested = [str(item) for item in dict.fromkeys(order_book_ids)]
        available = [item for item in requested if item in frame.index and item in frame.columns]
        frame = frame.reindex(index=available, columns=available)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    matrix = frame.to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    return pd.DataFrame(matrix, index=frame.index, columns=frame.columns)


def _single_summary(
    *,
    as_of_date: Any | None,
    nominal_count: int,
    effective_breadth: float,
    eigenvalue_count: int,
    dropped_zero_variance_count: int,
) -> dict[str, Any]:
    ratio = effective_breadth / float(nominal_count) if nominal_count > 0 else 0.0
    result = {
        "nominal_count": int(nominal_count),
        "effective_breadth": float(effective_breadth),
        "breadth_ratio": float(ratio),
        "eigenvalue_count": int(eigenvalue_count),
        "dropped_zero_variance_count": int(dropped_zero_variance_count),
        "method": "eigenvalue_participation_ratio",
        "matrix": "correlation_from_covariance",
    }
    if as_of_date is not None:
        result["as_of_date"] = str(pd.Timestamp(as_of_date).date())
    return result


def _breadth_warnings(
    *,
    latest: dict[str, Any],
    mean_effective_breadth: float,
    mean_breadth_ratio: float,
    p5_effective_breadth: float,
    p5_breadth_ratio: float,
    min_effective_breadth: float,
    min_breadth_ratio: float,
) -> list[str]:
    warnings: list[str] = []
    if p5_effective_breadth < min_effective_breadth or mean_effective_breadth < min_effective_breadth:
        warnings.append("low_effective_breadth")
    if p5_breadth_ratio < min_breadth_ratio or mean_breadth_ratio < min_breadth_ratio:
        warnings.append("low_effective_breadth_ratio")
    if int(latest.get("dropped_zero_variance_count", 0)) > 0:
        warnings.append("zero_variance_assets_excluded")
    return warnings
