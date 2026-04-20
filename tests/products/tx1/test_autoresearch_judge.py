from skyeye.products.tx1.autoresearch.judge import judge_candidate


def test_judge_candidate_rejects_candidate_that_breaks_stability_guard():
    baseline = {
        "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.010},
        "portfolio": {"net_mean_return": 0.0015, "max_drawdown": 0.09, "mean_turnover": 0.18},
        "robustness": {
            "stability": {"stability_score": 58.0, "cv": 0.55},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {"metric_consistency": {"positive_ratio": 0.78}},
        },
    }
    candidate = {
        "prediction": {"rank_ic_mean": 0.08, "top_bucket_spread_mean": 0.014},
        "portfolio": {"net_mean_return": 0.0030, "max_drawdown": 0.16, "mean_turnover": 0.19},
        "robustness": {
            "stability": {"stability_score": 61.0, "cv": 0.52},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {"metric_consistency": {"positive_ratio": 0.80}},
        },
    }

    decision = judge_candidate(candidate, baseline_summary=baseline, stage="full")

    assert decision["status"] == "discard"
    assert decision["reason_code"] == "guardrail_failed"
    assert "max_drawdown" in decision["failed_guards"]


def test_judge_candidate_marks_champion_when_full_eval_is_more_robust_and_profitable():
    baseline = {
        "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.010},
        "portfolio": {"net_mean_return": 0.0015, "max_drawdown": 0.09, "mean_turnover": 0.18},
        "robustness": {
            "stability": {"stability_score": 58.0, "cv": 0.55},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {"metric_consistency": {"positive_ratio": 0.78}},
        },
    }
    candidate = {
        "prediction": {"rank_ic_mean": 0.06, "top_bucket_spread_mean": 0.012},
        "portfolio": {"net_mean_return": 0.0021, "max_drawdown": 0.08, "mean_turnover": 0.16},
        "robustness": {
            "stability": {"stability_score": 66.0, "cv": 0.48},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {"metric_consistency": {"positive_ratio": 0.86}},
        },
    }

    decision = judge_candidate(candidate, baseline_summary=baseline, stage="full")

    assert decision["status"] == "champion"
    assert decision["reason_code"] == "full_improved"
    assert decision["score_delta"]["net_mean_return"] > 0


def test_judge_candidate_never_promotes_smoke_result_to_champion():
    baseline = {
        "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.010},
        "portfolio": {"net_mean_return": 0.0015, "max_drawdown": 0.09, "mean_turnover": 0.18},
        "robustness": {
            "stability": {"stability_score": 58.0, "cv": 0.55},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {"metric_consistency": {"positive_ratio": 0.78}},
        },
    }
    candidate = {
        "prediction": {"rank_ic_mean": 0.06, "top_bucket_spread_mean": 0.012},
        "portfolio": {"net_mean_return": 0.0021, "max_drawdown": 0.08, "mean_turnover": 0.16},
        "robustness": {
            "stability": {"stability_score": 66.0, "cv": 0.48},
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {"metric_consistency": {"positive_ratio": 0.86}},
        },
    }

    decision = judge_candidate(candidate, baseline_summary=baseline, stage="smoke")

    assert decision["status"] == "keep"
    assert decision["reason_code"] == "smoke_pass"

