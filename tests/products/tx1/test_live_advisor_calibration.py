import pandas as pd

from skyeye.products.tx1.live_advisor.calibration import (
    build_calibration_bundle,
    lookup_calibration_bucket,
)


def test_build_calibration_bundle_uses_oos_predictions_only():
    """验证校准统计只基于 fold 的 OOS predictions_df 构建。"""
    fold_one = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05"] * 4),
            "order_book_id": ["A", "B", "C", "D"],
            "prediction": [0.10, 0.20, 0.80, 0.90],
            "label_return_raw": [-0.03, 0.01, 0.02, 0.04],
            "label_volatility_horizon": [0.22, 0.24, 0.30, 0.35],
            "label_max_drawdown_horizon": [0.08, 0.06, 0.05, 0.04],
        }
    )
    fold_two = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-06"] * 4),
            "order_book_id": ["E", "F", "G", "H"],
            "prediction": [0.15, 0.25, 0.70, 0.95],
            "label_return_raw": [-0.01, 0.00, 0.03, 0.05],
            "label_volatility_horizon": [0.20, 0.23, 0.28, 0.32],
            "label_max_drawdown_horizon": [0.09, 0.07, 0.05, 0.03],
        }
    )

    calibration_bundle = build_calibration_bundle(
        {
            "fold_results": [
                {"predictions_df": fold_one},
                {"predictions_df": fold_two},
            ]
        },
        bucket_count=2,
    )

    bucket_low = calibration_bundle["bucket_stats"][0]
    bucket_high = calibration_bundle["bucket_stats"][1]
    matched = lookup_calibration_bucket(calibration_bundle, 0.90)

    assert calibration_bundle["summary"]["oos_rows"] == 8
    assert calibration_bundle["summary"]["oos_dates"] == 2
    assert bucket_low["sample_count"] == 4
    assert bucket_low["win_rate"] == 0.25
    assert bucket_high["sample_count"] == 4
    assert bucket_high["win_rate"] == 1.0
    assert bucket_high["median_return"] == 0.035
    assert matched["bucket_id"] == bucket_high["bucket_id"]


def test_build_calibration_bundle_collects_score_sanity_reference():
    """验证校准包会附带运行时 stop-serve 所需的分布参考值。"""
    rows = []
    for day_offset, base in enumerate([0.1, 0.2, 0.3], start=1):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=day_offset)
        for idx in range(10):
            rows.append(
                {
                    "date": date,
                    "order_book_id": "S{:02d}".format(idx),
                    "prediction": base + idx * 0.05,
                    "label_return_raw": -0.02 + idx * 0.01,
                    "label_volatility_horizon": 0.20 + idx * 0.01,
                    "label_max_drawdown_horizon": 0.10 - idx * 0.005,
                }
            )
    frame = pd.DataFrame(rows)

    calibration_bundle = build_calibration_bundle(
        {"fold_results": [{"predictions_df": frame}]},
        bucket_count=5,
    )
    sanity = calibration_bundle["score_sanity_reference"]

    assert sanity["prediction_std_p01"] > 0
    assert sanity["top_spread_p05"] > 0
    assert sanity["n_days"] == 3
