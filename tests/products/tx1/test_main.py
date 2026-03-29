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
