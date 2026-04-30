from skyeye.products.ax1.config import normalize_config


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
