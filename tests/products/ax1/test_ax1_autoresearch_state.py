def test_ax1_autoresearch_state_records_lineage_and_leaderboard(tmp_path):
    from skyeye.products.ax1.autoresearch.state import AX1AutoresearchStateStore

    store = AX1AutoresearchStateStore(tmp_path)
    store.initialize(
        run_tag="ax1-test",
        baseline_candidate_id="baseline",
        baseline_summary={"metrics": {"net_mean_return": 0.001}},
    )
    store.append_candidate(
        candidate_id="candidate_1",
        parent_candidate_id="baseline",
        status="discard",
        reason_code="data_audit_failed",
        stage_reached="audit",
        metrics={"net_mean_return": 0.0},
    )

    state = store.load()

    assert state["run_tag"] == "ax1-test"
    assert state["baseline_candidate_id"] == "baseline"
    assert state["leaderboard"][0]["candidate_id"] == "candidate_1"
    assert state["leaderboard"][0]["parent_candidate_id"] == "baseline"
    assert state["leaderboard"][0]["reason_code"] == "data_audit_failed"


def test_ax1_judge_requires_audit_and_promotes_only_material_improvement():
    from skyeye.products.ax1.autoresearch.judge import judge_feature_set_candidate

    baseline = {
        "metrics": {
            "net_mean_return": 0.001,
            "max_drawdown": 0.05,
            "mean_turnover": 0.10,
            "top_bucket_spread_mean": 0.002,
            "stability_score": 45.0,
        }
    }
    failed_audit = judge_feature_set_candidate(
        {"data_audit": {"passed": False, "hard_blocks": [{"reason_code": "latest_snapshot_not_point_in_time"}]}},
        baseline_summary=baseline,
        best_summary=baseline,
    )
    assert failed_audit["status"] == "discard"
    assert failed_audit["reason_code"] == "data_audit_failed"

    champion = judge_feature_set_candidate(
        {
            "data_audit": {"passed": True},
            "metrics": {
                "net_mean_return": 0.0016,
                "max_drawdown": 0.045,
                "mean_turnover": 0.10,
                "top_bucket_spread_mean": 0.003,
                "stability_score": 48.0,
            },
            "feature_review_summary": {"warning_count": 0},
        },
        baseline_summary=baseline,
        best_summary=baseline,
    )
    assert champion["status"] == "champion"
    assert champion["reason_code"] == "full_improved"


def test_ax1_judge_rejects_tiny_or_statistically_weak_improvement():
    from skyeye.products.ax1.autoresearch.judge import judge_feature_set_candidate

    baseline = {
        "metrics": {
            "net_mean_return": 0.0010,
            "max_drawdown": 0.05,
            "mean_turnover": 0.10,
            "top_bucket_spread_mean": 0.0020,
            "stability_score": 45.0,
        }
    }

    tiny_delta = judge_feature_set_candidate(
        {
            "data_audit": {"passed": True},
            "metrics": {
                "net_mean_return": 0.00105,
                "max_drawdown": 0.05,
                "mean_turnover": 0.10,
                "top_bucket_spread_mean": 0.0021,
                "stability_score": 45.0,
            },
            "feature_review_summary": {"warning_count": 0},
        },
        baseline_summary=baseline,
        best_summary=baseline,
    )
    assert tiny_delta["status"] == "keep"
    assert tiny_delta["reason_code"] == "insufficient_material_improvement"

    weak_ci = judge_feature_set_candidate(
        {
            "data_audit": {"passed": True},
            "metrics": {
                "net_mean_return": 0.0020,
                "max_drawdown": 0.045,
                "mean_turnover": 0.10,
                "top_bucket_spread_mean": 0.0030,
                "stability_score": 47.0,
            },
            "feature_review_summary": {"warning_count": 0},
            "robustness_summary": {"bootstrap_ci": {"ci_low": -0.0001, "ci_crosses_zero": True}},
        },
        baseline_summary=baseline,
        best_summary=baseline,
    )
    assert weak_ci["status"] == "discard"
    assert "bootstrap_ci" in weak_ci["failed_guards"]
