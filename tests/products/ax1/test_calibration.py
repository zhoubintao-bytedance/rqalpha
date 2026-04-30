# -*- coding: utf-8 -*-

import pandas as pd
import pytest

from skyeye.products.ax1.calibration import build_calibration_bundle, lookup_calibration_bucket


def test_build_calibration_bundle_uses_ax1_oos_predictions_and_label_priority():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 4 + ["2026-01-06"] * 4),
            "order_book_id": list("ABCDEFGH"),
            "expected_relative_net_return_10d": [0.10, 0.20, 0.80, 0.90, 0.15, 0.25, 0.70, 0.95],
            "label_return_10d": [0.50] * 8,
            "label_net_return_10d": [-0.03, 0.01, 0.02, 0.04, -0.01, 0.00, 0.03, 0.05],
            "label_volatility_10d": [0.22, 0.24, 0.30, 0.35, 0.20, 0.23, 0.28, 0.32],
        }
    )

    bundle = build_calibration_bundle(
        {"training_summary": {"fold_results": [{"predictions_df": frame}]}},
        bucket_count=2,
    )

    low_bucket = bundle["bucket_stats"][0]
    high_bucket = bundle["bucket_stats"][1]
    matched = lookup_calibration_bucket(bundle, 0.90)

    assert bundle["summary"]["oos_rows"] == 8
    assert bundle["summary"]["oos_dates"] == 2
    assert low_bucket["sample_count"] == 4
    assert low_bucket["win_rate"] == pytest.approx(0.25)
    assert high_bucket["sample_count"] == 4
    assert high_bucket["win_rate"] == pytest.approx(1.0)
    assert high_bucket["median_return"] == pytest.approx(0.035)
    assert high_bucket["volatility_quantiles"]["p50"] > 0
    assert "max_drawdown_quantiles" not in high_bucket
    assert matched["bucket_id"] == "b01"


def test_build_calibration_bundle_falls_back_to_return_label_and_empty_volatility_quantiles():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 3),
            "order_book_id": ["A", "B", "C"],
            "expected_relative_net_return_10d": [0.10, 0.20, 0.30],
            "label_return_5d": [-0.01, 0.02, 0.03],
        }
    )

    bundle = build_calibration_bundle({"fold_results": [{"predictions_df": frame}]}, bucket_count=3)

    assert bundle["return_label_column"] == "label_return_5d"
    assert bundle["volatility_label_column"] is None
    assert bundle["bucket_stats"][0]["volatility_quantiles"] == {}


def test_score_sanity_reference_contains_daily_score_distribution():
    rows = []
    for day_offset, base in enumerate([0.1, 0.2, 0.3], start=1):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_offset)
        for idx in range(10):
            rows.append(
                {
                    "date": date,
                    "order_book_id": "S{:02d}".format(idx),
                    "expected_relative_net_return_10d": base + idx * 0.05,
                    "label_net_return_10d": -0.02 + idx * 0.01,
                    "label_volatility_10d": 0.20 + idx * 0.01,
                }
            )
    bundle = build_calibration_bundle({"fold_results": [{"predictions_df": pd.DataFrame(rows)}]}, bucket_count=5)
    sanity = bundle["score_sanity_reference"]

    assert sanity["score_std_p05"] > 0
    assert sanity["score_top_spread_p05"] > 0
    assert sanity["score_min_p50"] < sanity["score_max_p50"]
    assert sanity["n_days"] == 3


def test_build_calibration_bundle_drops_rows_with_missing_return_labels():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 4),
            "order_book_id": ["A", "B", "C", "D"],
            "expected_relative_net_return_10d": [0.10, 0.20, 0.80, 0.90],
            "label_net_return_10d": [-0.01, None, 0.03, 0.04],
        }
    )

    bundle = build_calibration_bundle({"fold_results": [{"predictions_df": frame}]}, bucket_count=2)

    assert bundle["summary"]["oos_rows"] == 3
    assert sum(bucket["sample_count"] for bucket in bundle["bucket_stats"]) == 3
    assert all(bucket["win_rate"] >= 0.0 for bucket in bundle["bucket_stats"])


def test_confidence_calibrator_can_use_explicit_tradable_net_success_outcome():
    from skyeye.products.ax1.confidence import fit_confidence_calibrator

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 8),
            "order_book_id": [f"ETF_{idx}" for idx in range(8)],
            "confidence_raw": [0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95],
        }
    )
    labels = pd.DataFrame(
        {
            "date": predictions["date"],
            "order_book_id": predictions["order_book_id"],
            "label_relative_net_return_20d": [0.01] * 8,
            "tradable_net_success": [-1, -1, -1, -1, 1, 1, -1, -1],
        }
    )

    calibrator, summary = fit_confidence_calibrator(
        predictions,
        labels,
        outcome_column="tradable_net_success",
        bucket_count=2,
        min_samples=4,
    )

    assert summary["status"] == "calibrated"
    assert summary["outcome_column"] == "tradable_net_success"
    assert calibrator["outcome_column"] == "tradable_net_success"
    assert calibrator["buckets"][0]["hit_rate"] == pytest.approx(0.0)
    assert calibrator["buckets"][1]["hit_rate"] == pytest.approx(0.5)
