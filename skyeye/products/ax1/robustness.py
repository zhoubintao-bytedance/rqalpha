# -*- coding: utf-8 -*-
"""AX1 fold-level robustness helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def aggregate_fold_metrics(fold_results: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
    """Aggregate per-fold prediction and validation metrics."""
    folds = list(fold_results or [])
    return {
        "n_folds": len(folds),
        "prediction": _aggregate_metric_group(folds, "prediction_metrics"),
        "validation": _aggregate_metric_group(folds, "validation_metrics"),
    }


def compute_stability_score(fold_results, metric_key: str = "top_bucket_spread_mean") -> dict[str, Any]:
    """Compute cross-fold stability for a prediction metric."""
    values = _extract_metric_values(fold_results, metric_key)
    if not values:
        return _empty_stability(metric_key)

    arr = np.asarray(values, dtype=float)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    worst_val = float(np.min(arr))
    median_val = float(np.median(arr))

    cv = std_val / abs(mean_val) if abs(mean_val) > 1e-12 else float("inf")
    cv_score = _score_cv(cv)
    worst_score = _score_worst(worst_val, median_val)
    consecutive_low = _max_consecutive_below(arr, median_val)
    consecutive_score = _score_consecutive(consecutive_low, len(arr))
    stability = 0.5 * cv_score + 0.3 * worst_score + 0.2 * consecutive_score

    return {
        "stability_score": float(np.clip(stability, 0.0, 100.0)),
        "cv": cv,
        "worst_value": worst_val,
        "max_consecutive_low": int(consecutive_low),
        "cv_score": float(cv_score),
        "worst_score": float(worst_score),
        "consecutive_score": float(consecutive_score),
        "n_folds": int(len(arr)),
        "mean": mean_val,
        "std": std_val,
        "metric_key": metric_key,
    }


def compute_positive_ratio(
    fold_results,
    metric_key: str = "top_bucket_spread_mean",
    *,
    min_effect: float = 0.0002,
) -> dict[str, Any]:
    """Return the share of folds whose prediction metric exceeds *min_effect*."""
    values = _extract_metric_values(fold_results, metric_key)
    if not values:
        return {"positive_ratio": 0.0, "n_folds": 0, "metric_key": metric_key}
    return {
        "positive_ratio": float(sum(value > min_effect for value in values) / len(values)),
        "n_folds": int(len(values)),
        "metric_key": metric_key,
    }


def detect_overfit_flags(fold_results) -> dict[str, Any]:
    """Detect validation-vs-OOS metric decay."""
    ic_decays: list[float] = []
    spread_decays: list[float] = []
    val_wins = 0

    for fold in fold_results or []:
        val_m = fold.get("validation_metrics") or {}
        test_m = fold.get("prediction_metrics") or {}
        if not val_m or not test_m:
            continue

        val_ic = _coerce_float(val_m.get("rank_ic_mean"), 0.0)
        test_ic = _coerce_float(test_m.get("rank_ic_mean"), 0.0)
        val_spread = _coerce_float(val_m.get("top_bucket_spread_mean"), 0.0)
        test_spread = _coerce_float(test_m.get("top_bucket_spread_mean"), 0.0)

        ic_decays.append(val_ic - test_ic)
        spread_decays.append(val_spread - test_spread)
        if (val_ic + val_spread) > (test_ic + test_spread):
            val_wins += 1

    if not ic_decays:
        return _empty_overfit()

    n_folds = len(ic_decays)
    mean_ic_decay = float(np.mean(ic_decays))
    mean_spread_decay = float(np.mean(spread_decays))
    val_dominant_ratio = float(val_wins / n_folds)

    # Relative thresholds: scale with OOS benchmark, floor to avoid over-sensitivity
    mean_oos_ic = float(np.mean([_coerce_float(f.get("prediction_metrics", {}).get("rank_ic_mean"), 0.0)
                                  for f in fold_results or []]))
    mean_oos_spread = float(np.mean([_coerce_float(f.get("prediction_metrics", {}).get("top_bucket_spread_mean"), 0.0)
                                      for f in fold_results or []]))
    ic_decay_threshold = max(0.01, 0.3 * abs(mean_oos_ic))
    spread_decay_threshold = max(0.002, 0.25 * abs(mean_oos_spread))

    return {
        "val_test_ic_decay": mean_ic_decay,
        "val_test_spread_decay": mean_spread_decay,
        "val_dominant_ratio": val_dominant_ratio,
        "flag_ic_decay": bool(mean_ic_decay > ic_decay_threshold),
        "flag_spread_decay": bool(mean_spread_decay > spread_decay_threshold),
        "flag_val_dominant": bool(val_dominant_ratio > 0.8),
        "n_folds_compared": int(n_folds),
    }


def bootstrap_metric_ci(
    fold_results,
    metric_key: str = "top_bucket_spread_mean",
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260430,
    confidence: float = 0.80,
) -> dict[str, Any]:
    values = _extract_metric_values(fold_results, metric_key)
    confidence = max(0.0, min(1.0, float(confidence)))
    alpha = 1.0 - confidence
    if not values:
        return {
            "metric_key": metric_key,
            "n_observations": 0,
            "n_bootstrap": int(n_bootstrap),
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "confidence": confidence,
            "alpha": alpha,
            "test_side": "one_sided_positive",
            "ci_crosses_zero": True,
        }
    arr = np.asarray(values, dtype=float)
    n_bootstrap = max(1, int(n_bootstrap))
    rng = np.random.default_rng(int(seed))
    sample_indices = rng.integers(0, len(arr), size=(n_bootstrap, len(arr)))
    boot_means = arr[sample_indices].mean(axis=1)
    ci_low = np.quantile(boot_means, alpha)
    ci_high = np.quantile(boot_means, 1.0)
    return {
        "metric_key": metric_key,
        "n_observations": int(len(arr)),
        "n_bootstrap": int(n_bootstrap),
        "mean": float(np.mean(arr)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "confidence": confidence,
        "alpha": alpha,
        "test_side": "one_sided_positive",
        "ci_crosses_zero": bool(float(ci_low) <= 0.0),
    }


def compute_sample_decay(fold_results, metric_key: str = "top_bucket_spread_mean") -> dict[str, Any]:
    """Compute sample decay by comparing early vs late fold performance.

    Requires at least 4 folds total (2 per side) for decay flag detection.
    With fewer folds, flag_late_decay is always False due to insufficient statistical power.
    """
    values = _extract_metric_values(fold_results, metric_key)
    if len(values) < 2:
        return {
            "metric_key": metric_key,
            "n_observations": int(len(values)),
            "early_mean": float(values[0]) if values else 0.0,
            "late_mean": float(values[-1]) if values else 0.0,
            "late_minus_early": 0.0,
            "flag_late_decay": False,
            "insufficient_observations": True,
        }
    split = max(1, len(values) // 2)
    early = np.asarray(values[:split], dtype=float)
    late = np.asarray(values[split:], dtype=float)
    early_mean = float(np.mean(early))
    late_mean = float(np.mean(late))
    late_minus_early = late_mean - early_mean

    # Require at least 2 observations per side (>= 4 total) for reliable decay detection
    min_obs_per_side = 2
    insufficient_folds = len(early) < min_obs_per_side or len(late) < min_obs_per_side

    return {
        "metric_key": metric_key,
        "n_observations": int(len(values)),
        "early_mean": early_mean,
        "late_mean": late_mean,
        "late_minus_early": float(late_minus_early),
        "flag_late_decay": bool(not insufficient_folds and late_minus_early < -max(0.001, abs(early_mean) * 0.25)),
        "insufficient_observations": bool(insufficient_folds),
    }


def build_robustness_summary(
    fold_results,
    *,
    metric_key: str = "top_bucket_spread_mean",
    parameter_validation_summary: dict[str, Any] | None = None,
    seed: int = 20260430,
    n_bootstrap: int = 1000,
    bootstrap_confidence: float = 0.80,
) -> dict[str, Any]:
    bootstrap = bootstrap_metric_ci(
        fold_results,
        metric_key=metric_key,
        n_bootstrap=n_bootstrap,
        seed=seed,
        confidence=bootstrap_confidence,
    )
    sample_decay = compute_sample_decay(fold_results, metric_key=metric_key)
    overfit = detect_overfit_flags(fold_results)
    parameter_sensitivity = {
        "warning_count": int((parameter_validation_summary or {}).get("warning_count", 0) or 0),
        "warnings": list((parameter_validation_summary or {}).get("warnings", []) or []),
        "fragile": bool((parameter_validation_summary or {}).get("fragile", False)),
    }
    warnings: list[str] = []
    if bootstrap["ci_crosses_zero"]:
        warnings.append("bootstrap_ci_crosses_zero")
    if sample_decay["flag_late_decay"]:
        warnings.append("late_sample_decay")
    if overfit.get("flag_ic_decay"):
        warnings.append("validation_test_ic_decay")
    if overfit.get("flag_spread_decay"):
        warnings.append("validation_test_spread_decay")
    warnings.extend(str(item) for item in parameter_sensitivity["warnings"])
    return {
        "schema_version": 1,
        "metric_key": metric_key,
        "bootstrap_ci": bootstrap,
        "sample_decay": sample_decay,
        "overfit_flags": overfit,
        "parameter_sensitivity": parameter_sensitivity,
        "warning_count": int(len(warnings)),
        "warnings": warnings,
    }


def _aggregate_metric_group(folds: list[dict[str, Any]], group_key: str) -> dict[str, float]:
    values_by_key: dict[str, list[float]] = {}
    for fold in folds:
        metrics = fold.get(group_key) or {}
        for key, value in metrics.items():
            numeric = _coerce_float(value)
            if numeric is not None:
                values_by_key.setdefault(str(key), []).append(numeric)
    return {
        f"{key}_mean": float(np.mean(values))
        for key, values in sorted(values_by_key.items())
        if values
    }


def _extract_metric_values(fold_results, metric_key: str) -> list[float]:
    values: list[float] = []
    for fold in fold_results or []:
        metrics = fold.get("prediction_metrics") or fold
        value = _coerce_float(metrics.get(metric_key))
        if value is not None:
            values.append(value)
    return values


def _empty_stability(metric_key: str) -> dict[str, Any]:
    return {
        "stability_score": 0.0,
        "cv": 0.0,
        "worst_value": 0.0,
        "max_consecutive_low": 0,
        "cv_score": 0.0,
        "worst_score": 0.0,
        "consecutive_score": 0.0,
        "n_folds": 0,
        "mean": 0.0,
        "std": 0.0,
        "metric_key": metric_key,
    }


def _empty_overfit() -> dict[str, Any]:
    return {
        "val_test_ic_decay": 0.0,
        "val_test_spread_decay": 0.0,
        "val_dominant_ratio": 0.0,
        "flag_ic_decay": False,
        "flag_spread_decay": False,
        "flag_val_dominant": False,
        "n_folds_compared": 0,
    }


def _score_cv(cv: float) -> float:
    if not np.isfinite(cv):
        return 0.0
    if cv <= 0.1:
        return 100.0
    if cv >= 1.0:
        return 0.0
    return float(100.0 * (1.0 - (cv - 0.1) / 0.9))


def _score_worst(worst_value: float, median_value: float) -> float:
    if median_value <= 0:
        return 100.0 if worst_value >= median_value else 0.0
    ratio = worst_value / median_value
    if ratio >= 0.8:
        return 100.0
    if ratio <= 0.0:
        return 0.0
    return float(100.0 * ratio / 0.8)


def _score_consecutive(consecutive_low: int, n_values: int) -> float:
    if n_values <= 1 or consecutive_low <= 1:
        return 100.0
    ratio = consecutive_low / n_values
    if ratio >= 0.5:
        return 0.0
    return float(100.0 * (1.0 - ratio / 0.5))


def _max_consecutive_below(values: np.ndarray, threshold: float) -> int:
    max_run = 0
    current = 0
    for value in values:
        if value < threshold:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return int(max_run)


def _coerce_float(value, default=None):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(numeric):
        return default
    return numeric
