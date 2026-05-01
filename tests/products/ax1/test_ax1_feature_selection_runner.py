def test_feature_selection_candidate_plan_uses_phase0_filters_and_scope_ablations():
    from skyeye.products.ax1.feature_select import build_feature_selection_candidate_specs

    phase0 = {
        "features": {
            "good": {"decision": "keep_candidate", "reasons": []},
            "watch": {"decision": "watch", "reasons": ["weak_statistical_evidence"]},
            "constant": {"decision": "hard_exclude", "reasons": ["low_cross_sectional_variance"]},
            "bad": {"decision": "hard_exclude", "reasons": ["stable_negative_ic"]},
            "duplicate": {"decision": "hard_exclude", "reasons": ["redundant_non_representative"]},
        }
    }

    specs = build_feature_selection_candidate_specs(
        baseline_features=["good", "watch", "constant", "bad", "duplicate"],
        phase0_audit=phase0,
    )
    by_id = {spec.candidate_id: spec for spec in specs}

    assert by_id["phase0_no_hard_exclude"].feature_allowlist == ("good", "watch")
    assert by_id["phase0_keep_only"].feature_allowlist == ("good",)
    assert by_id["drop_low_variance"].feature_allowlist == ("good", "watch", "bad", "duplicate")
    assert by_id["drop_stable_negative_ic"].feature_allowlist == ("good", "watch", "constant", "duplicate")
    assert by_id["drop_redundant"].feature_allowlist == ("good", "watch", "constant", "bad")
    assert by_id["scope_common"].include_scopes == ("common",)


def test_feature_selection_leaderboard_summary_extracts_gate_and_stability_metrics():
    from skyeye.products.ax1.feature_select import FeatureSelectionCandidateSpec, summarize_experiment_for_leaderboard

    summary = summarize_experiment_for_leaderboard(
        FeatureSelectionCandidateSpec(
            candidate_id="c1",
            rationale="test",
            feature_allowlist=("a", "b"),
        ),
        {
            "status": "ok",
            "gate_summary": {"passed": False, "failed_checks": ["min_stability_score"]},
            "training_summary": {
                "feature_review_summary": {"warning_count": 1},
                "positive_ratio": {"positive_ratio": 0.5},
                "stability": {"cv": 2.0, "stability_score": 10.0},
            },
            "evaluation": {
                "signal": {
                    "rank_ic_mean": 0.03,
                    "rank_ic_significance": {"p_value": 0.04},
                    "top_bucket_spread_mean": 0.002,
                },
                "portfolio": {
                    "net_mean_return": 0.01,
                    "max_drawdown": 0.08,
                    "max_excess_drawdown": 0.10,
                    "max_rolling_underperformance": 0.12,
                    "mean_turnover": 0.02,
                },
            },
        },
    )

    assert summary["candidate_id"] == "c1"
    assert summary["feature_count"] == 2
    assert summary["rank_ic_p_value"] == 0.04
    assert summary["positive_ratio"] == 0.5
    assert summary["failed_checks"] == ["min_stability_score"]


def test_feature_selection_resume_skips_candidates_already_in_leaderboard():
    from skyeye.products.ax1.feature_select import FeatureSelectionCandidateSpec, _remaining_specs

    specs = [
        FeatureSelectionCandidateSpec(candidate_id="scope_common", rationale="a"),
        FeatureSelectionCandidateSpec(candidate_id="scope_full", rationale="b"),
        FeatureSelectionCandidateSpec(candidate_id="phase0_keep_only", rationale="c"),
    ]
    leaderboard = [
        {"candidate_id": "baseline", "status": "ok"},
        {"candidate_id": "scope_common", "status": "ok"},
    ]

    remaining = _remaining_specs(specs, leaderboard)

    assert [spec.candidate_id for spec in remaining] == ["scope_full", "phase0_keep_only"]


def test_feature_selection_candidate_spec_round_trips_from_json_payload():
    from skyeye.products.ax1.feature_select import FeatureSelectionCandidateSpec

    spec = FeatureSelectionCandidateSpec.from_dict(
        {
            "candidate_id": "c1",
            "phase": "phase0_filter",
            "rationale": "drop hard exclusions",
            "include_scopes": ["common", "regime"],
            "feature_allowlist": ["a", "b"],
            "feature_blocklist": ["c"],
        }
    )

    assert spec.candidate_id == "c1"
    assert spec.phase == "phase0_filter"
    assert spec.include_scopes == ("common", "regime")
    assert spec.feature_allowlist == ("a", "b")
    assert spec.feature_blocklist == ("c",)
