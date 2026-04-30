"""阶段 5：LGBM 多目标 predictor 测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.models.lgbm_multi_target import LGBMMultiTargetPredictor


def _make_training_frame(n_dates: int = 25, n_assets: int = 12, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    order_book_ids = [f"A{idx:04d}" for idx in range(n_assets)]

    rows = []
    labels_rows = []
    for date in dates:
        for order_book_id in order_book_ids:
            momentum = float(rng.normal(0.0, 0.02))
            volatility = float(abs(rng.normal(0.02, 0.01)))
            dollar_volume = float(abs(rng.normal(1e7, 3e6)))
            # 合成 label：让头能学到一点信号。
            label_return_5d = 0.3 * momentum - 0.2 * volatility + rng.normal(0.0, 0.005)
            label_return_10d = 0.5 * momentum - 0.3 * volatility + rng.normal(0.0, 0.006)
            label_return_20d = 0.9 * momentum - 0.4 * volatility + rng.normal(0.0, 0.008)
            label_volatility_10d = volatility * float(abs(rng.normal(1.0, 0.1))) + 0.005
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "momentum_2d": momentum,
                    "volatility_3d": volatility,
                    "dollar_volume": dollar_volume,
                }
            )
            labels_rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "label_return_5d": label_return_5d,
                    "label_return_10d": label_return_10d,
                    "label_return_20d": label_return_20d,
                    "label_net_return_5d": label_return_5d - 0.002,
                    "label_net_return_10d": label_return_10d - 0.002,
                    "label_net_return_20d": label_return_20d - 0.002,
                    "label_volatility_10d": label_volatility_10d,
                }
            )
    labels = pd.DataFrame(labels_rows)
    for horizon in (5, 10, 20):
        column = f"label_net_return_{horizon}d"
        labels[f"label_relative_net_return_{horizon}d"] = (
            labels[column] - labels.groupby("date")[column].transform("mean")
        )
    return pd.DataFrame(rows), labels


def _fast_params() -> dict:
    return {
        "n_estimators": 25,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "min_child_samples": 5,
        "early_stopping_rounds": 5,
    }


def test_fit_predict_produces_relative_return_columns():
    features, labels = _make_training_frame()
    predictor = LGBMMultiTargetPredictor(
        feature_columns=("momentum_2d", "volatility_3d", "dollar_volume"),
        params=_fast_params(),
    )
    predictor.fit(features, labels)
    predictions = predictor.predict(features)

    expected_columns = {
        "expected_relative_net_return_5d",
        "expected_relative_net_return_10d",
        "expected_relative_net_return_20d",
        "risk_forecast",
        "confidence_raw",
        "confidence",
        "liquidity_score",
        "cost_forecast",
    }
    assert expected_columns.issubset(predictions.columns)
    assert predictions[["date", "order_book_id"]].equals(features[["date", "order_book_id"]])
    assert predictions["confidence_raw"].equals(predictions["confidence"])


def test_predict_values_within_reasonable_ranges():
    features, labels = _make_training_frame()
    predictor = LGBMMultiTargetPredictor(
        feature_columns=("momentum_2d", "volatility_3d", "dollar_volume"),
        params=_fast_params(),
    )
    predictor.fit(features, labels)
    predictions = predictor.predict(features)

    assert predictions["risk_forecast"].ge(0.0).all()
    for column in ["confidence", "liquidity_score"]:
        assert predictions[column].between(0.0, 1.0).all()
    assert predictions["cost_forecast"].ge(0.0).all()


def test_fit_without_labels_raises():
    features, _ = _make_training_frame()
    predictor = LGBMMultiTargetPredictor(
        feature_columns=("momentum_2d", "volatility_3d", "dollar_volume"),
        params=_fast_params(),
    )
    with pytest.raises(ValueError, match="labels"):
        predictor.fit(features, None)


def test_val_set_triggers_early_stopping():
    features, labels = _make_training_frame(n_dates=30, seed=1)
    split_date = features["date"].unique()[20]
    train_mask = features["date"] < split_date
    val_mask = ~train_mask

    params = _fast_params()
    params["n_estimators"] = 500
    params["early_stopping_rounds"] = 3

    predictor = LGBMMultiTargetPredictor(
        feature_columns=("momentum_2d", "volatility_3d", "dollar_volume"),
        params=params,
    )
    predictor.fit(
        features[train_mask],
        labels[train_mask],
        val_features=features[val_mask],
        val_labels=labels[val_mask],
    )

    # Booster 的 best_iteration / current_iteration 应小于 n_estimators，说明触发了早停。
    booster = predictor._multi_head.models_["relative_net_return_10d"]["model"]._model
    assert booster.current_iteration() < 500


def test_relative_net_return_20d_is_a_trained_head_not_linear_extrapolation():
    features, labels = _make_training_frame()
    predictor = LGBMMultiTargetPredictor(
        feature_columns=("momentum_2d", "volatility_3d", "dollar_volume"),
        params=_fast_params(),
    )
    predictor.fit(features, labels)
    predictions = predictor.predict(features)

    assert "relative_net_return_20d" in predictor._head_configs
    assert predictor._head_configs["relative_net_return_20d"]["target_column"] == "label_relative_net_return_20d"
    assert not np.allclose(
        predictions["expected_relative_net_return_20d"].to_numpy(dtype=float),
        predictions["expected_relative_net_return_10d"].to_numpy(dtype=float) * 2.0,
    )


def test_feature_columns_autodetect_from_numeric():
    features, labels = _make_training_frame()
    # feature_columns=None → 自动从 numeric 列里挑（过滤掉 label_ / expected_ / 派生列名）
    predictor = LGBMMultiTargetPredictor(feature_columns=None, params=_fast_params())
    predictor.fit(features, labels)

    assert set(predictor._trained_feature_columns) == {"momentum_2d", "volatility_3d", "dollar_volume"}
    predictions = predictor.predict(features)
    assert "expected_relative_net_return_10d" in predictions.columns


def test_missing_feature_column_in_predict_raises():
    features, labels = _make_training_frame()
    predictor = LGBMMultiTargetPredictor(
        feature_columns=("momentum_2d", "volatility_3d", "dollar_volume"),
        params=_fast_params(),
    )
    predictor.fit(features, labels)
    broken = features.drop(columns=["volatility_3d"])
    with pytest.raises(ValueError, match="features missing columns"):
        predictor.predict(broken)


def test_confidence_calibrator_uses_validation_labels_and_preserves_raw_signal():
    from skyeye.products.ax1.confidence import apply_confidence_calibration, fit_confidence_calibrator

    validation_predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-02-01", periods=8, freq="D"),
            "order_book_id": ["A"] * 8,
            "confidence_raw": [0.05, 0.10, 0.20, 0.30, 0.70, 0.80, 0.90, 0.95],
        }
    )
    validation_labels = pd.DataFrame(
        {
            "date": validation_predictions["date"],
            "order_book_id": validation_predictions["order_book_id"],
            "label_relative_net_return_20d": [-0.02, -0.01, 0.01, -0.03, 0.02, 0.03, 0.01, 0.04],
        }
    )
    test_predictions = validation_predictions.copy()

    calibrator, summary = fit_confidence_calibrator(
        validation_predictions,
        validation_labels,
        label_column="label_relative_net_return_20d",
        bucket_count=2,
        min_samples=4,
    )
    calibrated = apply_confidence_calibration(test_predictions, calibrator)

    assert summary["status"] == "calibrated"
    assert summary["sample_count"] == 8
    assert len(summary["buckets"]) == 2
    assert "confidence_raw" in calibrated.columns
    assert calibrated["confidence"].between(0.0, 1.0).all()
    assert calibrated.loc[0, "confidence"] < calibrated.loc[7, "confidence"]
