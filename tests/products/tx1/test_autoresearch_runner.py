from skyeye.products.tx1.autoresearch import runner


def test_runner_builds_dedicated_experiment_path(tmp_path):
    run_root = tmp_path / "tx1_run"

    path = runner.build_experiment_root(run_root, experiment_index=7)

    assert path == run_root / "experiments" / "exp_0007"


def test_runner_loads_summary_from_experiment_result():
    result = {
        "output_dir": "/tmp/demo_exp",
        "aggregate_metrics": {
            "prediction": {
                "rank_ic_mean": 0.051,
                "top_bucket_spread_mean": 0.011,
            },
            "portfolio": {
                "net_mean_return": 0.0022,
                "max_drawdown": 0.081,
                "mean_turnover": 0.155,
            },
            "robustness": {
                "stability": {
                    "stability_score": 63.0,
                    "cv": 0.49,
                },
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {
                    "metric_consistency": {
                        "positive_ratio": 0.84,
                    }
                },
            },
        },
    }

    summary = runner.load_summary_from_experiment_result(result)

    assert summary["prediction"]["rank_ic_mean"] == 0.051
    assert summary["portfolio"]["net_mean_return"] == 0.0022
    assert summary["robustness"]["stability"]["stability_score"] == 63.0
    assert summary["experiment_path"] == "/tmp/demo_exp"


def test_runner_loads_summary_from_experiment_path(monkeypatch, tmp_path):
    captured = {}

    def _fake_load_experiment(path):
        captured["path"] = path
        return {
            "output_dir": path,
            "aggregate_metrics": {
                "prediction": {"rank_ic_mean": 0.05, "top_bucket_spread_mean": 0.01},
                "portfolio": {"net_mean_return": 0.002, "max_drawdown": 0.08, "mean_turnover": 0.16},
                "robustness": {
                    "stability": {"stability_score": 60.0, "cv": 0.5},
                    "overfit_flags": {
                        "flag_ic_decay": False,
                        "flag_spread_decay": False,
                        "flag_val_dominant": False,
                    },
                    "regime_scores": {"metric_consistency": {"positive_ratio": 0.8}},
                },
            },
        }

    monkeypatch.setattr(runner, "load_experiment", _fake_load_experiment)

    summary = runner.load_summary_from_experiment_path(tmp_path / "demo_exp")

    assert captured["path"] == str(tmp_path / "demo_exp")
    assert summary["portfolio"]["max_drawdown"] == 0.08


def test_runner_runs_feature_trial_and_returns_summary(monkeypatch, tmp_path, make_raw_panel):
    captured = {}

    def _fake_run_feature_experiments(
        *,
        raw_df,
        output_dir,
        variant_names,
        model_kind,
        label_transform,
        horizon_days,
        max_folds,
    ):
        captured["shape"] = tuple(raw_df.shape)
        captured["output_dir"] = output_dir
        captured["variant_names"] = list(variant_names)
        captured["model_kind"] = model_kind
        captured["label_transform"] = label_transform
        captured["horizon_days"] = horizon_days
        captured["max_folds"] = max_folds
        return {
            "model_kind": model_kind,
            "label_transform": label_transform,
            "variants": [
                {
                    "name": "baseline_5f",
                    "fold_results": [
                        {
                            "fold_index": 1,
                            "prediction_metrics": {
                                "rank_ic_mean": 0.05,
                                "top_bucket_spread_mean": 0.01,
                            },
                            "portfolio_metrics": {
                                "net_mean_return": 0.002,
                                "mean_return": 0.002,
                                "max_drawdown": 0.08,
                                "mean_turnover": 0.16,
                            },
                            "date_range": {
                                "test_start": "2024-01-01",
                                "test_end": "2024-06-30",
                            },
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr(runner, "run_feature_experiments", _fake_run_feature_experiments)

    summary = runner.run_feature_trial(
        run_root=tmp_path / "tx1_run",
        experiment_index=1,
        raw_df=make_raw_panel(periods=260, extended=True),
        variant_name="baseline_5f",
        model_kind="linear",
        label_transform="rank",
        horizon_days=20,
        max_folds=2,
    )

    assert captured["variant_names"] == ["baseline_5f"]
    assert captured["model_kind"] == "linear"
    assert captured["max_folds"] == 2
    assert summary["prediction"]["rank_ic_mean"] == 0.05
    assert summary["experiment_path"].endswith("exp_0001")


def test_runner_builds_research_raw_df_from_universe_and_date_window(monkeypatch):
    captured = {}

    def _fake_get_liquid_universe(universe_size, data_end=None):
        captured["universe_size"] = universe_size
        captured["data_end"] = data_end
        return ["000001.XSHE", "000002.XSHE"]

    def _fake_build_raw_df(universe, start_date=None, end_date=None):
        captured["universe"] = list(universe)
        captured["start_date"] = start_date
        captured["end_date"] = end_date
        return {"raw_df": "ok"}

    monkeypatch.setattr(runner, "_get_liquid_universe", _fake_get_liquid_universe)
    monkeypatch.setattr(runner, "build_raw_df", _fake_build_raw_df)

    raw_df = runner.build_research_raw_df(
        universe_size=123,
        start_date="2020-01-01",
        end_date="2024-12-31",
    )

    assert raw_df == {"raw_df": "ok"}
    assert captured == {
        "universe_size": 123,
        "data_end": "2024-12-31",
        "universe": ["000001.XSHE", "000002.XSHE"],
        "start_date": "2020-01-01",
        "end_date": "2024-12-31",
    }
