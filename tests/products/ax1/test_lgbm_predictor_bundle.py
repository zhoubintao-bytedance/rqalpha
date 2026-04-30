import json

import numpy as np
import pandas as pd

from skyeye.products.ax1.models.bundle import dump_predictor_bundle, load_predictor_bundle
from skyeye.products.ax1.models.lgbm_multi_target import LGBMMultiTargetPredictor


def _make_training_frame(n_dates: int = 24, n_assets: int = 10, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="D")
    rows = []
    labels = []
    for date in dates:
        for asset_idx in range(n_assets):
            momentum = float(rng.normal(0.0, 0.03))
            volatility = float(abs(rng.normal(0.02, 0.01)))
            dollar_volume = float(abs(rng.normal(1e7, 2e6)))
            rows.append(
                {
                    "date": date,
                    "order_book_id": f"A{asset_idx:04d}",
                    "momentum_5d": momentum,
                    "volatility_5d": volatility,
                    "dollar_volume": dollar_volume,
                }
            )
            labels.append(
                {
                    "date": date,
                    "order_book_id": f"A{asset_idx:04d}",
                    "label_return_5d": 0.4 * momentum - 0.1 * volatility + rng.normal(0.0, 0.004),
                    "label_return_10d": 0.7 * momentum - 0.2 * volatility + rng.normal(0.0, 0.005),
                    "label_return_20d": 1.0 * momentum - 0.3 * volatility + rng.normal(0.0, 0.006),
                    "label_net_return_5d": 0.4 * momentum - 0.1 * volatility + rng.normal(0.0, 0.004) - 0.002,
                    "label_net_return_10d": 0.7 * momentum - 0.2 * volatility + rng.normal(0.0, 0.005) - 0.002,
                    "label_net_return_20d": 1.0 * momentum - 0.3 * volatility + rng.normal(0.0, 0.006) - 0.002,
                    "label_volatility_10d": volatility + 0.003,
                }
            )
    labels_df = pd.DataFrame(labels)
    for horizon in (5, 10, 20):
        column = f"label_net_return_{horizon}d"
        labels_df[f"label_relative_net_return_{horizon}d"] = (
            labels_df[column] - labels_df.groupby("date")[column].transform("mean")
        )
    return pd.DataFrame(rows), labels_df


def _fast_params():
    return {
        "n_estimators": 12,
        "num_leaves": 7,
        "learning_rate": 0.1,
        "min_child_samples": 3,
        "early_stopping_rounds": 3,
        "num_threads": 1,
        "verbose": -1,
    }


def _assert_predictions_equal(left: pd.DataFrame, right: pd.DataFrame):
    assert left[["date", "order_book_id"]].equals(right[["date", "order_book_id"]])
    numeric_columns = [
        "expected_relative_net_return_5d",
        "expected_relative_net_return_10d",
        "expected_relative_net_return_20d",
        "risk_forecast",
        "confidence",
        "liquidity_score",
        "cost_forecast",
    ]
    for column in numeric_columns:
        np.testing.assert_allclose(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float))


def test_lgbm_predictor_bundle_round_trips_predictions():
    features, labels = _make_training_frame()
    feature_columns = ["momentum_5d", "volatility_5d", "dollar_volume"]
    predictor = LGBMMultiTargetPredictor(feature_columns=feature_columns, params=_fast_params())
    predictor.fit(features, labels)

    bundle = dump_predictor_bundle(
        predictor,
        preprocessor_bundle={"kind": "feature_preprocessor", "sector_optional": True},
        feature_columns=feature_columns,
    )
    loaded = load_predictor_bundle(bundle)

    assert bundle["model_bundle"]["model_kind"] == "multi_head"
    assert bundle["model_bundle"]["state"]["base_model_kind"] == "lgbm"
    assert bundle["rule_config"]["training_horizons"] == [5, 10, 20]
    assert loaded._trained_feature_columns == tuple(feature_columns)
    _assert_predictions_equal(predictor.predict(features), loaded.predict(features))


def test_lgbm_predictor_bundle_json_round_trip_preserves_rules_and_preprocessor_bundle():
    features, labels = _make_training_frame(seed=11)
    feature_columns = ["momentum_5d", "volatility_5d", "dollar_volume"]
    predictor = LGBMMultiTargetPredictor(
        horizons=(5, 10, 20, 30),
        feature_columns=feature_columns,
        liquidity_column="dollar_volume",
        params=_fast_params(),
    )
    predictor.fit(features, labels)
    preprocessor_bundle = {
        "kind": "feature_preprocessor",
        "sector_optional": False,
        "feature_columns": feature_columns,
    }

    raw_bundle = dump_predictor_bundle(
        predictor,
        preprocessor_bundle=preprocessor_bundle,
        feature_columns=feature_columns,
    )
    loaded_bundle = json.loads(json.dumps(raw_bundle))
    loaded = load_predictor_bundle(loaded_bundle)
    predictions = loaded.predict(features)

    assert loaded_bundle["preprocessor_bundle"] == preprocessor_bundle
    assert loaded.horizons == (5, 10, 20, 30)
    assert loaded.liquidity_column == "dollar_volume"
    assert "expected_relative_net_return_30d" in predictions.columns
    _assert_predictions_equal(predictor.predict(features), predictions)
