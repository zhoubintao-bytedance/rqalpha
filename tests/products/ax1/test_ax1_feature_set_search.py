def test_feature_set_search_expands_champion_with_scorecard_and_conflict_guard():
    from skyeye.products.ax1.autoresearch.search import generate_feature_set_candidates

    scorecards = {
        "factor_a": {"final_factor_score": 0.90},
        "factor_b": {"final_factor_score": 0.85},
        "factor_c": {"final_factor_score": 0.70},
        "factor_bad": {"final_factor_score": -0.20},
    }
    conflicts = {
        "high_corr_groups": [
            {"representative": "factor_a", "features": ["factor_a", "factor_b"]},
        ]
    }
    data_audit = {
        "factor_a": {"passed": True},
        "factor_b": {"passed": True},
        "factor_c": {"passed": True},
        "factor_bad": {"passed": True},
        "factor_blocked": {"passed": False},
    }

    candidates = generate_feature_set_candidates(
        base_features=["base_factor"],
        scorecards=scorecards,
        conflicts=conflicts,
        data_audit_by_feature=data_audit,
        max_additions=2,
        beam_width=3,
    )

    assert candidates
    assert candidates[0].base_features == ("base_factor",)
    assert candidates[0].added_features[0] == "factor_a"
    assert all("factor_bad" not in candidate.added_features for candidate in candidates)
    assert all("factor_blocked" not in candidate.added_features for candidate in candidates)
    assert all(
        not {"factor_a", "factor_b"}.issubset(set(candidate.added_features))
        for candidate in candidates
    )
    assert all(candidate.parent_candidate_id for candidate in candidates)


def test_feature_set_search_respects_candidate_budget_and_score_floor():
    from skyeye.products.ax1.autoresearch.search import generate_feature_set_candidates

    candidates = generate_feature_set_candidates(
        base_features=["base_factor"],
        scorecards={
            "factor_a": {"final_factor_score": 0.90},
            "factor_b": {"final_factor_score": 0.70},
            "factor_c": {"final_factor_score": 0.10},
        },
        max_additions=2,
        beam_width=10,
        max_candidates=2,
        early_stop_score_floor=0.50,
    )

    assert len(candidates) == 2
    assert all("factor_c" not in candidate.added_features for candidate in candidates)


def test_feature_set_candidate_roundtrip_is_json_friendly():
    from skyeye.products.ax1.autoresearch.feature_set import FeatureSetCandidate

    candidate = FeatureSetCandidate(
        candidate_id="c1",
        base_features=("base",),
        added_features=("factor_a",),
        removed_features=(),
        rationale="positive OOS score",
        parent_candidate_id="champion",
        search_phase="single_add",
    )

    payload = candidate.to_dict()
    restored = FeatureSetCandidate.from_dict(payload)

    assert payload["candidate_id"] == "c1"
    assert restored == candidate
