# -*- coding: utf-8 -*-

import pandas as pd
import pytest


class _RiskModel:
    def __init__(self, covariance: pd.DataFrame):
        self._covariance = covariance

    def get_covariance_matrix(self) -> pd.DataFrame:
        return self._covariance.copy()


def test_effective_breadth_collapses_for_highly_correlated_etf_cluster():
    from skyeye.products.ax1.effective_breadth import effective_breadth_from_covariance

    covariance = pd.DataFrame(
        [
            [0.04, 0.0396, 0.0396],
            [0.0396, 0.04, 0.0396],
            [0.0396, 0.0396, 0.04],
        ],
        index=["ETF_A", "ETF_B", "ETF_C"],
        columns=["ETF_A", "ETF_B", "ETF_C"],
    )

    summary = effective_breadth_from_covariance(covariance)

    assert summary["nominal_count"] == 3
    assert summary["effective_breadth"] == pytest.approx(1.01, abs=0.05)
    assert summary["effective_breadth"] < 1.5
    assert summary["breadth_ratio"] < 0.50


def test_effective_breadth_matches_asset_count_for_identity_correlation():
    from skyeye.products.ax1.effective_breadth import effective_breadth_from_covariance

    covariance = pd.DataFrame(
        [[0.04, 0.0, 0.0, 0.0], [0.0, 0.09, 0.0, 0.0], [0.0, 0.0, 0.16, 0.0], [0.0, 0.0, 0.0, 0.25]],
        index=["ETF_A", "ETF_B", "ETF_C", "ETF_D"],
        columns=["ETF_A", "ETF_B", "ETF_C", "ETF_D"],
    )

    summary = effective_breadth_from_covariance(covariance)

    assert summary["nominal_count"] == 4
    assert summary["effective_breadth"] == pytest.approx(4.0)
    assert summary["breadth_ratio"] == pytest.approx(1.0)


def test_replay_summary_carries_latest_and_average_effective_breadth():
    from skyeye.products.ax1.effective_breadth import summarize_effective_breadth_by_date

    risk_by_date = {
        pd.Timestamp("2024-01-02"): _RiskModel(
            pd.DataFrame(
                [[0.04, 0.0], [0.0, 0.09]],
                index=["ETF_A", "ETF_B"],
                columns=["ETF_A", "ETF_B"],
            )
        ),
        pd.Timestamp("2024-01-03"): _RiskModel(
            pd.DataFrame(
                [[0.04, 0.0396, 0.0396], [0.0396, 0.04, 0.0396], [0.0396, 0.0396, 0.04]],
                index=["ETF_A", "ETF_B", "ETF_C"],
                columns=["ETF_A", "ETF_B", "ETF_C"],
            )
        ),
    }

    summary = summarize_effective_breadth_by_date(risk_by_date)

    assert summary["schema_version"] == 1
    assert summary["date_count"] == 2
    assert summary["latest"]["as_of_date"] == "2024-01-03"
    assert summary["latest"]["nominal_count"] == 3
    assert summary["latest"]["effective_breadth"] < summary["latest"]["nominal_count"]
    assert summary["mean_effective_breadth"] < summary["mean_nominal_count"]
    assert summary["p5_effective_breadth"] <= summary["mean_effective_breadth"]
    assert summary["p5_breadth_ratio"] <= summary["mean_breadth_ratio"]


def test_promotion_gate_surfaces_low_effective_breadth_as_soft_risk():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        effective_breadth_summary={
            "schema_version": 1,
            "date_count": 1,
            "latest": {
                "as_of_date": "2024-01-03",
                "nominal_count": 34,
                "effective_breadth": 11.0,
                "breadth_ratio": 0.32,
            },
            "mean_effective_breadth": 11.0,
            "mean_nominal_count": 34.0,
            "mean_breadth_ratio": 0.32,
            "p5_effective_breadth": 11.0,
            "p5_breadth_ratio": 0.32,
            "warning_count": 1,
            "warnings": ["low_effective_breadth_ratio"],
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["passed"] is True
    assert summary["metrics"]["effective_breadth"] == pytest.approx(11.0)
    assert summary["metrics"]["effective_breadth_ratio"] == pytest.approx(0.32)
    assert summary["checks"]["min_effective_breadth_ratio"]["passed"] is False
    assert summary["checks"]["min_effective_breadth_ratio"]["hard_gate"] is False
    assert "min_effective_breadth_ratio" not in summary["failed_checks"]


def test_promotion_gate_prefers_p5_breadth_over_latest_snapshot():
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    result = _gate_ready_result(
        effective_breadth_summary={
            "schema_version": 1,
            "date_count": 20,
            "latest": {
                "as_of_date": "2024-01-31",
                "nominal_count": 34,
                "effective_breadth": 14.0,
                "breadth_ratio": 14.0 / 34.0,
            },
            "mean_effective_breadth": 18.0,
            "mean_nominal_count": 34.0,
            "mean_breadth_ratio": 18.0 / 34.0,
            "p5_effective_breadth": 12.0,
            "p5_breadth_ratio": 12.0 / 34.0,
            "warning_count": 1,
            "warnings": ["low_effective_breadth_ratio"],
        }
    )

    summary = evaluate_promotion_gate(result)

    assert summary["metrics"]["effective_breadth"] == pytest.approx(12.0)
    assert summary["metrics"]["effective_breadth_ratio"] == pytest.approx(12.0 / 34.0)


def _gate_ready_result(**overrides):
    result = {
        "training_summary": {
            "fold_results": [{"fold_id": idx} for idx in range(6)],
            "aggregate_metrics": {
                "n_folds": 6,
                "prediction": {
                    "rank_ic_mean_mean": 0.035,
                    "top_bucket_spread_mean_mean": 0.004,
                },
            },
            "stability": {"stability_score": 55.0, "cv": 0.50},
            "positive_ratio": {"positive_ratio": 0.67},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
        },
        "calibration_bundle": {"summary": {"oos_rows": 120}},
        "evaluation": {
            "portfolio": {
                "net_mean_return": 0.001,
                "max_drawdown": 0.04,
                "opportunity_benchmark_available": True,
                "excess_net_mean_return": 0.001,
                "max_excess_drawdown": 0.03,
                "alpha_hit_rate": 0.60,
                "max_rolling_underperformance": 0.04,
                "active_day_ratio": 0.80,
                "mean_turnover": 0.10,
            }
        },
    }
    result.update(overrides)
    return result
