# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from skyeye.products.tx1.compare_experiments import main as compare_experiments_main
from skyeye.products.tx1.persistence import save_experiment
from skyeye.products.tx1.robustness import (
    compare_experiments,
    compute_regime_scores,
    compute_stability_score,
    detect_overfit_flags,
    render_comparison_report,
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


def _make_saved_experiment(
    tmp_path,
    name,
    ic_values,
    *,
    val_ic_values=None,
    gross_returns=None,
    net_returns=None,
    spreads=None,
    val_spreads=None,
    turnovers=None,
    drawdowns=None,
    regime_patterns=None,
):
    n = len(ic_values)
    gross_returns = gross_returns or [0.0004] * n
    net_returns = net_returns or gross_returns
    turnovers = turnovers or [0.02] * n
    drawdowns = drawdowns or [0.05] * n
    spreads = spreads or [0.01] * n
    val_spreads = val_spreads or spreads

    portfolio_returns = []
    for i in range(n):
        values = regime_patterns[i] if regime_patterns else [0.01, 0.008, -0.006, -0.004]
        dates = pd.bdate_range("2023-01-01", periods=len(values), freq="B") + pd.offsets.BDay(i * 5)
        portfolio_returns.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "portfolio_return": values,
                    "turnover": [turnovers[i]] * len(values),
                    "overlap": [max(0.0, 1.0 - turnovers[i])] * len(values),
                }
            )
        )

    folds = _make_fold_results(
        ic_values,
        spread_values=spreads,
        val_ic_values=val_ic_values,
        val_spread_values=val_spreads,
        portfolio_returns=portfolio_returns,
    )
    for i, fold in enumerate(folds):
        fold["portfolio_metrics"].update(
            {
                "mean_return": gross_returns[i],
                "net_mean_return": net_returns[i],
                "max_drawdown": drawdowns[i],
                "mean_turnover": turnovers[i],
                "mean_overlap": max(0.0, 1.0 - turnovers[i]),
                "cost_drag_annual": max(gross_returns[i] - net_returns[i], 0.0) * 252.0,
                "cost_erosion_ratio": (
                    1.0 - (net_returns[i] / gross_returns[i])
                    if abs(gross_returns[i]) > 1e-12
                    else 0.0
                ),
                "breakeven_cost_bps": (
                    gross_returns[i] / turnovers[i] * 10000.0
                    if turnovers[i] > 1e-12 and gross_returns[i] > 0
                    else 0.0
                ),
            }
        )
        start = pd.Timestamp("2021-01-01") + pd.DateOffset(months=6 * i)
        end = start + pd.DateOffset(months=5, days=27)
        fold["date_range"] = {
            "train_start": None,
            "train_end": (start - pd.DateOffset(days=30)).strftime("%Y-%m-%d 00:00:00"),
            "val_start": (start - pd.DateOffset(days=15)).strftime("%Y-%m-%d 00:00:00"),
            "val_end": (start - pd.DateOffset(days=1)).strftime("%Y-%m-%d 00:00:00"),
            "test_start": start.strftime("%Y-%m-%d 00:00:00"),
            "test_end": end.strftime("%Y-%m-%d 00:00:00"),
        }

    result = {
        "model_kind": "lgbm",
        "fold_results": folds,
        "aggregate_metrics": {},
    }
    output_dir = tmp_path / name
    save_experiment(
        result,
        str(output_dir),
        config={"robustness": {"enabled": True, "stability_metric": "rank_ic_mean"}},
        experiment_name=name,
    )
    return output_dir


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


class TestExperimentComparison:
    def test_pair_flags_few_window_uplift(self, tmp_path):
        baseline_dir = _make_saved_experiment(
            tmp_path,
            "baseline",
            ic_values=[0.05] * 5,
            gross_returns=[0.00042] * 5,
            net_returns=[0.00040] * 5,
        )
        candidate_dir = _make_saved_experiment(
            tmp_path,
            "candidate_few_windows",
            ic_values=[0.07, 0.048, 0.047, 0.046, 0.045],
            gross_returns=[0.00095, 0.00042, 0.00041, 0.00039, 0.00038],
            net_returns=[0.00090, 0.00040, 0.00039, 0.00037, 0.00036],
        )

        result = compare_experiments([str(baseline_dir), str(candidate_dir)])
        comparison = result["comparisons"][0]
        report = render_comparison_report(result)

        assert comparison["fold_level"]["few_window_flag"]
        assert comparison["decision"]["status"] == "HOLD"
        assert comparison["flags"]["few_window_uplift"]
        assert "收益来自少数窗口: YES" in report

    def test_pair_rejects_when_costs_erase_edge(self, tmp_path):
        baseline_dir = _make_saved_experiment(
            tmp_path,
            "baseline",
            ic_values=[0.05] * 5,
            gross_returns=[0.00042] * 5,
            net_returns=[0.00040] * 5,
        )
        candidate_dir = _make_saved_experiment(
            tmp_path,
            "candidate_cost_fade",
            ic_values=[0.06] * 5,
            gross_returns=[0.00060] * 5,
            net_returns=[0.00035] * 5,
            turnovers=[0.08] * 5,
        )

        result = compare_experiments([str(baseline_dir), str(candidate_dir)])
        comparison = result["comparisons"][0]

        assert comparison["cost_level"]["fades_after_cost"]
        assert not comparison["cost_level"]["survives_cost"]
        assert comparison["decision"]["status"] == "REJECT"

    def test_compare_experiments_supports_real_artifact_lines(self):
        result = compare_experiments(
            ["baseline_tree"],
            baseline_ref="baseline_lgbm",
            artifacts_root="skyeye/artifacts/experiments/tx1",
        )
        comparison = result["comparisons"][0]

        assert result["baseline"]["label"] == "baseline_lgbm"
        assert comparison["candidate"]["label"] == "baseline_tree"
        assert comparison["fold_level"]["aligned_folds"] == 14
        assert "decision" in comparison

    def test_cli_writes_report_and_json(self, tmp_path, capsys):
        baseline_dir = _make_saved_experiment(
            tmp_path,
            "baseline",
            ic_values=[0.05] * 5,
            gross_returns=[0.00042] * 5,
            net_returns=[0.00040] * 5,
        )
        candidate_dir = _make_saved_experiment(
            tmp_path,
            "candidate_merge",
            ic_values=[0.055] * 5,
            val_ic_values=[0.055] * 5,
            gross_returns=[0.00050] * 5,
            net_returns=[0.00046] * 5,
        )
        report_path = tmp_path / "report.txt"
        json_path = tmp_path / "report.json"

        rc = compare_experiments_main(
            [
                str(candidate_dir),
                "--baseline",
                str(baseline_dir),
                "--output",
                str(report_path),
                "--json-output",
                str(json_path),
            ]
        )

        stdout = capsys.readouterr().out
        report = report_path.read_text(encoding="utf-8")
        payload = json_path.read_text(encoding="utf-8")

        assert rc == 0
        assert "Fold-level" in stdout
        assert "Regime-level" in report
        assert "Stability-level" in report
        assert "Overfit-level" in report
        assert "candidate_merge" in payload
