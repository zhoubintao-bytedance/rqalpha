from skyeye.products.tx1.main import main


def test_main_runs_end_to_end(make_raw_panel):
    raw_df = make_raw_panel(periods=2200)

    result = main({"model": {"kind": "linear"}}, raw_df=raw_df)

    assert result["model_kind"] == "linear"
    assert result["fold_results"]
    assert "aggregate_metrics" in result
    assert "prediction" in result["aggregate_metrics"]
    assert "portfolio" in result["aggregate_metrics"]
    assert "validation_metrics" in result["fold_results"][0]
    assert result["fold_results"][0]["row_counts"]["val"] > 0


def test_main_runs_multi_output_research_mode(make_raw_panel, tmp_path):
    raw_df = make_raw_panel(periods=2200)

    result = main(
        {
            "model": {"kind": "linear"},
            "multi_output": {
                "enabled": True,
                "volatility": {"enabled": True, "transform": "log1p"},
                "max_drawdown": {"enabled": True, "transform": "robust"},
                "prediction": {
                    "combine_auxiliary": True,
                    "volatility_weight": 0.2,
                    "max_drawdown_weight": 0.1,
                },
                "reliability_score": {"enabled": True},
            },
        },
        raw_df=raw_df,
        output_dir=str(tmp_path / "tx1_multi_output"),
    )

    first_fold = result["fold_results"][0]
    predictions_df = first_fold["predictions_df"]

    assert result["model_heads"] == ["return", "volatility", "max_drawdown"]
    assert {"head_metrics", "head_validation_metrics", "prediction_blend", "selection"}.issubset(
        result["aggregate_metrics"].keys()
    )
    assert {
        "prediction",
        "prediction_ret",
        "prediction_vol",
        "prediction_mdd",
        "reliability_score",
        "target_return",
        "target_volatility",
        "target_max_drawdown",
    }.issubset(predictions_df.columns)
    assert predictions_df["prediction_vol"].notna().all()
    assert predictions_df["prediction_mdd"].notna().all()
    assert predictions_df["reliability_score"].between(0.0, 1.0).all()
    assert result["output_dir"].endswith("tx1_multi_output")
