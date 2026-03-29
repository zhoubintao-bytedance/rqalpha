# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from skyeye.products.tx1.robustness import (
    compute_regime_scores,
    compute_stability_score,
    detect_overfit_flags,
)


def _make_fold_results(
    ic_values,
    spread_values=None,
    val_ic_values=None,
    val_spread_values=None,
    portfolio_returns=None,
):
    """Helper to create fold results for testing."""
    results = []
    for i, ic in enumerate(ic_values):
        spread = (spread_values or [0.01] * len(ic_values))[i]
        val_ic = (val_ic_values or ic_values)[i]
        val_spread = (val_spread_values or [0.01] * len(ic_values))[i]

        fold = {
            "fold_index": i + 1,
            "prediction_metrics": {
                "rank_ic_mean": ic,
                "top_bucket_spread_mean": spread,
            },
            "validation_metrics": {
                "rank_ic_mean": val_ic,
                "top_bucket_spread_mean": val_spread,
            },
            "portfolio_metrics": {
                "mean_return": ic * 0.5,
                "max_drawdown": abs(ic) * 0.2,
            },
        }

        if portfolio_returns is not None and i < len(portfolio_returns):
            fold["portfolio_returns_df"] = portfolio_returns[i]
        else:
            dates = pd.bdate_range("2023-01-01", periods=10)
            fold["portfolio_returns_df"] = pd.DataFrame({
                "date": dates,
                "portfolio_return": np.random.default_rng(42 + i).normal(0.001, 0.01, 10),
            })

        results.append(fold)
    return results


class TestComputeStabilityScore:
    def test_perfect_stability(self):
        folds = _make_fold_results([0.05, 0.05, 0.05, 0.05])
        result = compute_stability_score(folds, "rank_ic_mean")
        assert result["stability_score"] == pytest.approx(100.0)
        assert result["cv"] == pytest.approx(0.0)
        assert result["n_folds"] == 4

    def test_high_variance_low_stability(self):
        folds = _make_fold_results([0.1, -0.1, 0.1, -0.1])
        result = compute_stability_score(folds, "rank_ic_mean")
        assert result["stability_score"] < 50.0
        assert result["cv"] > 0.5

    def test_empty_input(self):
        result = compute_stability_score([], "rank_ic_mean")
        assert result["stability_score"] == 0.0
        assert result["n_folds"] == 0

    def test_single_fold(self):
        folds = _make_fold_results([0.05])
        result = compute_stability_score(folds, "rank_ic_mean")
        assert result["n_folds"] == 1

    def test_metric_key_portfolio(self):
        folds = _make_fold_results([0.04, 0.06, 0.05, 0.05])
        result = compute_stability_score(folds, "mean_return")
        assert result["metric_key"] == "mean_return"
        assert result["n_folds"] == 4

    def test_worst_value_tracked(self):
        folds = _make_fold_results([0.05, 0.03, 0.07, 0.06])
        result = compute_stability_score(folds, "rank_ic_mean")
        assert result["worst_value"] == pytest.approx(0.03)

    def test_consecutive_low_detection(self):
        # First 3 clearly below median (median=0.05), then 2 above
        folds = _make_fold_results([0.01, 0.02, 0.01, 0.10, 0.12])
        result = compute_stability_score(folds, "rank_ic_mean")
        # median=0.02, values below: 0.01, _, 0.01 → max_consecutive=1
        # Use clearer data: 3 low, 2 high → median ~ 0.03
        folds2 = _make_fold_results([0.01, 0.02, 0.025, 0.10, 0.12])
        result2 = compute_stability_score(folds2, "rank_ic_mean")
        assert result2["max_consecutive_low"] >= 2

    def test_score_between_0_and_100(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            values = rng.normal(0.05, 0.03, 6).tolist()
            folds = _make_fold_results(values)
            result = compute_stability_score(folds, "rank_ic_mean")
            assert 0.0 <= result["stability_score"] <= 100.0


class TestDetectOverfitFlags:
    def test_no_overfit(self):
        # Validation and test metrics roughly equal
        folds = _make_fold_results(
            ic_values=[0.05, 0.05, 0.05],
            val_ic_values=[0.05, 0.05, 0.05],
        )
        result = detect_overfit_flags(folds)
        assert not result["flag_ic_decay"]
        assert not result["flag_val_dominant"]

    def test_strong_overfit(self):
        # Validation much better than test
        folds = _make_fold_results(
            ic_values=[0.02, 0.01, 0.01],
            val_ic_values=[0.10, 0.09, 0.08],
        )
        result = detect_overfit_flags(folds)
        assert result["flag_ic_decay"]
        assert result["val_dominant_ratio"] > 0.8

    def test_empty_input(self):
        result = detect_overfit_flags([])
        assert result["n_folds_compared"] == 0
        assert not result["flag_ic_decay"]

    def test_test_better_than_val(self):
        # Test outperforms validation → no overfit
        folds = _make_fold_results(
            ic_values=[0.08, 0.09, 0.07],
            val_ic_values=[0.03, 0.04, 0.02],
        )
        result = detect_overfit_flags(folds)
        assert not result["flag_ic_decay"]
        assert result["val_test_ic_decay"] < 0

    def test_n_folds_counted(self):
        folds = _make_fold_results([0.05] * 5, val_ic_values=[0.05] * 5)
        result = detect_overfit_flags(folds)
        assert result["n_folds_compared"] == 5


class TestComputeRegimeScores:
    def test_basic_regime_split(self):
        dates = pd.bdate_range("2023-01-01", periods=10)
        returns = [0.01, -0.02, 0.015, -0.005, 0.008, -0.01, 0.02, -0.003, 0.005, -0.015]
        port_df = pd.DataFrame({"date": dates, "portfolio_return": returns})
        folds = _make_fold_results([0.05, 0.06], portfolio_returns=[port_df, port_df])

        result = compute_regime_scores(folds, "rank_ic_mean")
        assert result["up_regime"]["n_periods"] > 0
        assert result["down_regime"]["n_periods"] > 0
        assert result["up_regime"]["mean_return"] > 0
        assert result["down_regime"]["mean_return"] <= 0

    def test_metric_consistency(self):
        folds = _make_fold_results([0.05, 0.06, 0.04, 0.07])
        result = compute_regime_scores(folds, "rank_ic_mean")
        assert result["metric_consistency"]["positive_folds"] == 4
        assert result["metric_consistency"]["positive_ratio"] == 1.0

    def test_mixed_folds(self):
        folds = _make_fold_results([0.05, -0.02, 0.03])
        result = compute_regime_scores(folds, "rank_ic_mean")
        assert result["metric_consistency"]["positive_folds"] == 2
        assert result["metric_consistency"]["total_folds"] == 3

    def test_empty_input(self):
        result = compute_regime_scores([], "rank_ic_mean")
        assert result["up_regime"]["n_periods"] == 0
        assert result["metric_consistency"]["total_folds"] == 0
