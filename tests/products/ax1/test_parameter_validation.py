from skyeye.products.ax1.config import normalize_config


def test_parameter_validation_aligns_label_horizon_with_score_column():
    import pandas as pd

    from skyeye.products.ax1.parameter_validation import build_parameter_validation_summary

    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "order_book_id": ["A", "B", "A", "B"],
            "expected_relative_net_return_10d": [0.20, 0.10, 0.05, 0.15],
        }
    )
    labels = pd.DataFrame(
        {
            "date": predictions["date"],
            "order_book_id": predictions["order_book_id"],
            "label_net_return_10d": [0.02, 0.01, 0.00, 0.03],
            "label_net_return_20d": [-0.10, 0.10, 0.10, -0.10],
        }
    )

    summary = build_parameter_validation_summary(normalize_config({}), predictions, labels)

    assert summary["score_column"] == "expected_relative_net_return_10d"
    assert summary["label_column"] == "label_net_return_10d"
    assert summary["score_horizon"] == 10
    assert summary["label_horizon"] == 10
    assert summary["label_kind"] == "net_return"


def test_lgbm_param_policy_reports_candidates_and_deviation():
    from skyeye.products.ax1.parameter_validation import build_lgbm_param_policy_summary

    config = normalize_config({})
    summary = build_lgbm_param_policy_summary(config)

    assert summary["schema_version"] == 1
    assert summary["status"] == "within_policy"
    assert summary["policies"]["min_child_samples"]["candidates"] == [30, 50, 80]
    assert summary["policies"]["learning_rate"]["candidates"] == [0.03, 0.05, 0.08]
    assert summary["policies"]["reg_lambda"]["candidates"] == [0.2, 0.5, 1.0, 2.0]
    assert summary["current"]["min_child_samples"] == 80
    assert summary["current"]["learning_rate"] == 0.03
    assert summary["current"]["reg_lambda"] == 2.0

    deviated = normalize_config({"model": {"params": {"learning_rate": 0.20}}})
    deviated_summary = build_lgbm_param_policy_summary(deviated)

    assert deviated_summary["status"] == "deviates_from_policy"
    assert "lgbm_param_outside_warning_range" in {item["reason_code"] for item in deviated_summary["warnings"]}
