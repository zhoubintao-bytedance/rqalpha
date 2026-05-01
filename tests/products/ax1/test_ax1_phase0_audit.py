def test_phase0_feature_audit_marks_hard_exclude_and_cluster_representatives():
    from skyeye.products.ax1.autoresearch.phase0 import build_phase0_feature_audit

    training_summary = {
        "feature_diagnostics_meta": {"label_column": "label_relative_net_return_10d"},
        "feature_diagnostics": {
            "good_factor": {
                "coverage": 1.0,
                "mean_cross_sectional_std": 0.20,
                "rank_ic_mean": 0.08,
                "top_bucket_spread_mean": 0.01,
                "fold_summary": {"positive_ratio": 1.0},
                "folds": [
                    {"rank_ic_mean": 0.06},
                    {"rank_ic_mean": 0.08},
                    {"rank_ic_mean": 0.10},
                ],
            },
            "constant_factor": {
                "coverage": 1.0,
                "mean_cross_sectional_std": 0.0,
                "rank_ic_mean": 0.03,
                "top_bucket_spread_mean": 0.002,
                "fold_summary": {"positive_ratio": 1.0},
                "folds": [{"rank_ic_mean": 0.03}, {"rank_ic_mean": 0.02}],
            },
            "bad_factor": {
                "coverage": 1.0,
                "mean_cross_sectional_std": 0.15,
                "rank_ic_mean": -0.07,
                "top_bucket_spread_mean": -0.01,
                "fold_summary": {"positive_ratio": 0.0},
                "folds": [
                    {"rank_ic_mean": -0.06},
                    {"rank_ic_mean": -0.08},
                    {"rank_ic_mean": -0.07},
                ],
            },
            "duplicate_factor": {
                "coverage": 1.0,
                "mean_cross_sectional_std": 0.18,
                "rank_ic_mean": 0.04,
                "top_bucket_spread_mean": 0.006,
                "fold_summary": {"positive_ratio": 0.67},
                "folds": [{"rank_ic_mean": 0.03}, {"rank_ic_mean": 0.05}, {"rank_ic_mean": 0.04}],
            },
        },
        "feature_conflicts": {
            "high_corr_groups": [
                {"representative": "good_factor", "features": ["good_factor", "duplicate_factor"]}
            ]
        },
    }

    audit = build_phase0_feature_audit(training_summary)
    features = audit["features"]

    assert audit["label_column"] == "label_relative_net_return_10d"
    assert features["good_factor"]["decision"] == "keep_candidate"
    assert features["good_factor"]["ic_t_stat"] > 0.0
    assert features["constant_factor"]["decision"] == "hard_exclude"
    assert "low_cross_sectional_variance" in features["constant_factor"]["reasons"]
    assert features["bad_factor"]["decision"] == "hard_exclude"
    assert "stable_negative_ic" in features["bad_factor"]["reasons"]
    assert features["duplicate_factor"]["decision"] == "hard_exclude"
    assert features["duplicate_factor"]["cluster_representative"] == "good_factor"
    assert audit["decision_counts"]["hard_exclude"] == 3
