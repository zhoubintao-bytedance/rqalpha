import json

import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.feature_diagnostics import (
    analyze_feature_diagnostics,
    analyze_fold_feature_diagnostics,
)
from skyeye.products.ax1.models.lgbm_multi_target import LGBMMultiTargetPredictor


def _diagnostic_panel() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    assets = ["A", "B", "C", "D", "E"]
    base_scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    rows: list[dict] = []
    labels: list[dict] = []

    for date_idx, date in enumerate(dates):
        label_values = base_scores * 0.01 + date_idx * 0.0001
        duplicate_noise = np.array([0.0, 0.03, -0.02, 0.02, -0.03]) * (1.0 + date_idx * 0.01)
        for asset_idx, asset in enumerate(assets):
            score = base_scores[asset_idx]
            rows.append(
                {
                    "date": date,
                    "order_book_id": asset,
                    "positive_factor": score,
                    "negative_factor": -score,
                    "low_variance_factor": 0.001 + date_idx * 0.000001,
                    "high_missing_factor": score if asset_idx == 4 else np.nan,
                    "positive_factor_duplicate": score + duplicate_noise[asset_idx],
                }
            )
            labels.append(
                {
                    "date": date,
                    "order_book_id": asset,
                    "label_net_return_5d": label_values[asset_idx],
                }
            )

    feature_columns = [
        "positive_factor",
        "negative_factor",
        "low_variance_factor",
        "high_missing_factor",
        "positive_factor_duplicate",
    ]
    return pd.DataFrame(rows), pd.DataFrame(labels), feature_columns


def test_feature_diagnostics_classifies_factor_quality_and_conflicts():
    features, labels, feature_columns = _diagnostic_panel()

    report = analyze_feature_diagnostics(
        features=features,
        labels=labels,
        feature_columns=feature_columns,
        fold_id=0,
        top_k=2,
        conflict_abs_corr_threshold=0.85,
    )

    json.dumps(report)
    diagnostics = report["feature_diagnostics"]
    review = report["feature_review_summary"]["features"]
    conflicts = report["feature_conflicts"]

    assert diagnostics["positive_factor"]["coverage"] == pytest.approx(1.0)
    assert diagnostics["positive_factor"]["rank_ic_mean"] == pytest.approx(1.0)
    assert diagnostics["positive_factor"]["top_bucket_spread_mean"] > 0.0
    assert diagnostics["positive_factor"]["top_k_hit_rate"] == pytest.approx(1.0)
    assert diagnostics["positive_factor"]["group_monotonicity"] == pytest.approx(1.0)
    assert diagnostics["positive_factor"]["fold_summary"]["mean"] == pytest.approx(1.0)
    assert diagnostics["positive_factor"]["fold_summary"]["std"] == pytest.approx(0.0)
    assert diagnostics["positive_factor"]["fold_summary"]["positive_ratio"] == pytest.approx(1.0)

    assert review["positive_factor"]["decision"] == "keep"
    assert review["negative_factor"]["decision"] == "drop_candidate"
    assert "negative_rank_ic" in review["negative_factor"]["reasons"]
    assert review["low_variance_factor"]["decision"] == "drop_candidate"
    assert "low_cross_sectional_variance" in review["low_variance_factor"]["reasons"]
    assert review["high_missing_factor"]["decision"] == "review"
    assert "low_coverage" in review["high_missing_factor"]["reasons"]
    assert review["positive_factor_duplicate"]["decision"] == "drop_candidate"
    assert "redundant_weaker_than_group_representative" in review["positive_factor_duplicate"]["reasons"]

    positive_groups = conflicts["high_corr_groups"]
    assert any(
        group["representative"] == "positive_factor"
        and "positive_factor_duplicate" in group["features"]
        for group in positive_groups
    )
    inverse_groups = conflicts["inverse_corr_groups"]
    assert any(
        {"positive_factor", "negative_factor"}.issubset(set(group["features"]))
        for group in inverse_groups
    )


def test_fold_feature_diagnostics_accepts_runner_style_fold_results():
    features, labels, feature_columns = _diagnostic_panel()
    fold_results = [
        {
            "fold_id": 0,
            "features_df": features.iloc[:20],
            "labels": labels.iloc[:20],
        },
        {
            "fold_id": 1,
            "predictions_df": features.iloc[20:],
            "labels": labels.iloc[20:],
        },
    ]

    report = analyze_fold_feature_diagnostics(
        fold_results=fold_results,
        feature_columns=feature_columns,
        label_column="label_net_return_5d",
        top_k=2,
    )

    assert report["feature_diagnostics"]["positive_factor"]["fold_summary"]["mean"] == pytest.approx(1.0)
    assert report["feature_diagnostics"]["negative_factor"]["fold_summary"]["worst"] == pytest.approx(-1.0)
    assert report["feature_review_summary"]["features"]["positive_factor"]["decision"] == "keep"


def test_feature_diagnostics_emits_runner_aggregation_schema():
    features, labels, feature_columns = _diagnostic_panel()

    report = analyze_feature_diagnostics(
        features=features,
        labels=labels,
        feature_columns=feature_columns,
        top_k=2,
    )

    assert set(report) >= {"feature_diagnostics", "feature_conflicts", "feature_review_summary"}
    summary = report["feature_review_summary"]
    assert summary["schema_version"] == 1
    assert summary["feature_count"] == len(feature_columns)
    assert summary["decision_counts"]["keep"] == 1
    assert summary["decision_counts"]["review"] == 1
    assert summary["decision_counts"]["drop_candidate"] == 3
    assert summary["warning_count"] == 4
    assert summary["features"]["positive_factor"]["decision"] == "keep"
    assert {warning["feature"] for warning in summary["warnings"]} == {
        "negative_factor",
        "low_variance_factor",
        "high_missing_factor",
        "positive_factor_duplicate",
    }


def test_feature_diagnostics_emits_factor_scorecards():
    features, labels, feature_columns = _diagnostic_panel()

    report = analyze_feature_diagnostics(
        features=features,
        labels=labels,
        feature_columns=feature_columns,
        top_k=2,
    )

    scorecards = report["factor_scorecards"]
    assert set(scorecards) == set(feature_columns)
    assert scorecards["positive_factor"]["final_factor_score"] > scorecards["negative_factor"]["final_factor_score"]
    assert scorecards["positive_factor"]["signal_score"] > 0.0
    assert scorecards["positive_factor_duplicate"]["redundancy_penalty"] > 0.0
    assert scorecards["high_missing_factor"]["data_quality_penalty"] > 0.0
    assert scorecards["low_variance_factor"]["final_factor_score"] < scorecards["positive_factor"]["final_factor_score"]


def test_lgbm_multi_target_feature_importance_is_json_friendly_per_head_and_aggregate():
    features, labels, _ = _diagnostic_panel()
    train_features = features.rename(
        columns={
            "positive_factor": "momentum_2d",
            "negative_factor": "volatility_3d",
            "positive_factor_duplicate": "dollar_volume",
        }
    )
    train_labels = labels.rename(columns={"label_net_return_5d": "label_net_return_5d"})
    train_labels["label_net_return_10d"] = train_labels["label_net_return_5d"] * 1.3
    train_labels["label_net_return_20d"] = train_labels["label_net_return_5d"] * 1.8
    for horizon in (5, 10, 20):
        column = f"label_net_return_{horizon}d"
        train_labels[f"label_relative_net_return_{horizon}d"] = (
            train_labels[column] - train_labels.groupby("date")[column].transform("mean")
        )
    train_labels["label_volatility_10d"] = train_labels["label_net_return_5d"].abs() + 0.01
    feature_columns_for_model = ["momentum_2d", "volatility_3d", "dollar_volume"]
    predictor = LGBMMultiTargetPredictor(
        feature_columns=feature_columns_for_model,
        params={
            "n_estimators": 10,
            "num_leaves": 5,
            "learning_rate": 0.2,
            "min_child_samples": 2,
            "early_stopping_rounds": 0,
            "num_threads": 1,
            "verbose": -1,
        },
    )
    predictor.fit(train_features, train_labels)

    importance = predictor.feature_importance()

    json.dumps(importance)
    assert importance["schema_version"] == 1
    assert importance["feature_columns"] == feature_columns_for_model
    assert set(importance["heads"]) == {
        "relative_net_return_5d",
        "relative_net_return_10d",
        "relative_net_return_20d",
        "risk",
    }
    assert set(importance["aggregate"]) == {"gain", "split"}
    assert {item["feature"] for item in importance["aggregate"]["gain"]} == set(feature_columns_for_model)
    assert all("gain" in head and "split" in head for head in importance["heads"].values())
    assert all(
        set(item) == {"feature", "importance", "normalized_importance", "rank"}
        for item in importance["aggregate"]["gain"]
    )
