from skyeye.products.tx1.autoresearch import search


def _summary(net_mean_return, max_drawdown, stability_score, cv, positive_ratio, experiment_path):
    return {
        "prediction": {
            "rank_ic_mean": 0.05,
            "top_bucket_spread_mean": 0.01,
        },
        "portfolio": {
            "net_mean_return": net_mean_return,
            "max_drawdown": max_drawdown,
            "mean_turnover": 0.02,
        },
        "robustness": {
            "stability": {
                "stability_score": stability_score,
                "cv": cv,
            },
            "overfit_flags": {
                "flag_ic_decay": False,
                "flag_spread_decay": False,
                "flag_val_dominant": False,
            },
            "regime_scores": {
                "metric_consistency": {
                    "positive_ratio": positive_ratio,
                }
            },
        },
        "experiment_path": experiment_path,
    }


def test_run_catalog_search_writes_leaderboard(monkeypatch, tmp_path):
    baseline_candidate = {
        "id": "baseline_5f_default",
        "description": "baseline",
        "features": ["mom_40d"],
    }
    candidates = [
        {
            "id": "liq_keep",
            "description": "liquidity keep",
            "features": ["mom_40d", "volume_ratio_20d"],
        },
        {
            "id": "bad_candidate",
            "description": "bad candidate",
            "features": ["mom_40d", "beta_60d"],
        },
    ]

    def _fake_build_candidate_config(candidate, **kwargs):
        return {
            "experiment_name": candidate["id"],
            "features": list(candidate["features"]),
        }

    def _fake_run_config_trial(*, run_root, experiment_index, raw_df, config, stage=None, max_folds=None):
        candidate_id = config["experiment_name"]
        if candidate_id == "baseline_5f_default":
            return _summary(0.00043, 0.045, 14.3, 1.04, 0.93, str(tmp_path / "baseline"))
        if candidate_id == "liq_keep" and stage == "smoke":
            return _summary(0.00046, 0.043, 15.0, 0.98, 0.90, str(tmp_path / "liq_smoke"))
        if candidate_id == "liq_keep" and stage == "full":
            return _summary(0.00051, 0.041, 15.8, 0.95, 0.90, str(tmp_path / "liq_full"))
        return _summary(0.00044, 0.042, 8.0, 1.30, 0.70, str(tmp_path / "bad"))

    monkeypatch.setattr(search, "build_research_raw_df", lambda **kwargs: {"raw_df": "ok"})
    monkeypatch.setattr(search, "build_baseline_candidate", lambda: baseline_candidate)
    monkeypatch.setattr(search, "build_candidate_catalog", lambda catalog_name: candidates)
    monkeypatch.setattr(search, "build_candidate_config", _fake_build_candidate_config)
    monkeypatch.setattr(search, "run_config_trial", _fake_run_config_trial)

    result = search.run_catalog_search(
        run_tag="demo",
        runs_root=tmp_path / "runs",
        catalog_name="risk_reward_v1",
        max_experiments=0,
    )

    assert result["status"] == "completed"
    assert result["champion"]["candidate_id"] == "liq_keep"
    assert [item["candidate_id"] for item in result["leaderboard"]] == ["liq_keep", "bad_candidate"]
    assert (tmp_path / "runs" / "demo" / "catalog_results.json").exists()
    assert (tmp_path / "runs" / "demo" / "catalog_leaderboard.tsv").exists()
