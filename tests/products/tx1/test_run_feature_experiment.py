import pandas as pd

from skyeye.products.tx1.evaluator import BASELINE_4F_COLUMNS, BASELINE_5F_COLUMNS, FEATURE_COLUMNS
from skyeye.products.tx1.run_feature_experiment import (
    GOOD_COMPANY_UNIVERSE_FILTER,
    _apply_universe_filter,
    build_variants,
    run_feature_experiments,
)


def test_run_feature_experiment_smoke(tmp_path, make_raw_panel):
    raw_df = make_raw_panel(periods=1400, extended=True)

    result = run_feature_experiments(
        raw_df=raw_df,
        output_dir=tmp_path,
        variant_names=["baseline_5f", "baseline_5f_roe"],
        model_kind="linear",
        max_folds=1,
    )

    variant_names = [variant["name"] for variant in result["variants"]]
    assert variant_names == ["baseline_5f", "baseline_5f_roe"]
    assert result["variants"][0]["n_folds"] == 1
    assert result["variants"][0]["features"]
    assert (tmp_path / "feature_experiment_results.json").exists()
    assert (tmp_path / "feature_experiment_report.txt").exists()
    assert (tmp_path / "variant_metrics.csv").exists()

    # 报告里必须显式暴露 spread 与稳定性 delta，便于判断是否值得继续推进。
    report = (tmp_path / "feature_experiment_report.txt").read_text()
    assert "top_bucket_spread_mean" in report
    assert "fold_net_return_std" in report

    # CSV 也要导出对应 delta 字段，便于后续程序化比较。
    metrics = pd.read_csv(tmp_path / "variant_metrics.csv")
    assert {"delta_top_bucket_spread_mean", "delta_fold_net_return_std"}.issubset(metrics.columns)


def test_build_variants_contains_new_variants():
    variants = build_variants()
    names = [v["name"] for v in variants]
    assert "baseline_5f" in names
    assert "baseline_5f_roe" in names
    assert "baseline_4f_fundamental_filter" in names
    assert "baseline_5f_fundamental_filter" in names
    assert "elite_ohlcv_3f" in names
    assert "fundamental_5f" in names
    assert "elite_combined_8f" in names
    assert "elite_combined_8f_lt" in names


def test_baseline_5f_variant_uses_expected_features():
    variants = build_variants()
    baseline_5f = next(v for v in variants if v["name"] == "baseline_5f")
    assert baseline_5f["features"] == list(BASELINE_5F_COLUMNS)


def test_baseline_5f_roe_variant_uses_expected_features():
    variants = build_variants()
    baseline_5f_roe = next(v for v in variants if v["name"] == "baseline_5f_roe")
    assert baseline_5f_roe["features"] == list(BASELINE_5F_COLUMNS) + ["return_on_equity_ttm"]


def test_default_feature_columns_match_baseline_5f():
    assert FEATURE_COLUMNS == list(BASELINE_5F_COLUMNS)


def test_baseline_4f_variant_uses_legacy_feature_set():
    variants = build_variants()
    baseline_4f = next(v for v in variants if v["name"] == "baseline_4f")
    assert baseline_4f["features"] == list(BASELINE_4F_COLUMNS)


def test_filtered_variants_use_good_company_filter():
    variants = build_variants()
    filtered_names = {"baseline_4f_fundamental_filter", "baseline_5f_fundamental_filter"}

    filtered_variants = [v for v in variants if v["name"] in filtered_names]

    assert len(filtered_variants) == 2
    for variant in filtered_variants:
        assert variant["universe_filter"] == GOOD_COMPANY_UNIVERSE_FILTER


def test_elite_combined_lt_has_portfolio_config():
    variants = build_variants()
    lt = next(v for v in variants if v["name"] == "elite_combined_8f_lt")
    assert lt["portfolio_config"]["rebalance_interval"] == 40
    assert lt["portfolio_config"]["holding_bonus"] == 1.0


def test_run_elite_variants_smoke(tmp_path, make_raw_panel):
    raw_df = make_raw_panel(periods=1400, extended=True)

    result = run_feature_experiments(
        raw_df=raw_df,
        output_dir=tmp_path,
        variant_names=["elite_ohlcv_3f", "fundamental_5f", "elite_combined_8f"],
        model_kind="linear",
        max_folds=1,
    )

    for variant in result["variants"]:
        assert variant["n_folds"] == 1, f"{variant['name']} has 0 folds"
        assert variant["features"], f"{variant['name']} has no features"


def test_apply_universe_filter_keeps_only_above_median_rows_per_date():
    df = pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
            ],
            "order_book_id": ["a", "b", "c", "a", "b", "c"],
            "ep_ratio_ttm": [1.0, 2.0, 3.0, 3.0, 1.0, 2.0],
            "return_on_equity_ttm": [1.0, 3.0, 2.0, 3.0, 1.0, 2.0],
        }
    )

    filtered = _apply_universe_filter(df, GOOD_COMPANY_UNIVERSE_FILTER)

    assert list(filtered["order_book_id"]) == ["b", "c", "a", "c"]


def test_run_filtered_variants_smoke(tmp_path, make_raw_panel):
    raw_df = make_raw_panel(periods=1400, extended=True)

    result = run_feature_experiments(
        raw_df=raw_df,
        output_dir=tmp_path,
        variant_names=["baseline_4f", "baseline_5f", "baseline_4f_fundamental_filter"],
        model_kind="linear",
        max_folds=1,
    )

    by_name = {variant["name"]: variant for variant in result["variants"]}

    assert by_name["baseline_4f"]["n_folds"] == 1
    assert by_name["baseline_5f"]["n_folds"] == 1
    assert by_name["baseline_5f"]["features"] == list(BASELINE_5F_COLUMNS)

    filtered_variant = by_name["baseline_4f_fundamental_filter"]
    assert filtered_variant["n_folds"] == 1
    assert filtered_variant["universe_filter"] == GOOD_COMPANY_UNIVERSE_FILTER
    assert filtered_variant["row_count"] < by_name["baseline_4f"]["row_count"]

    report = (tmp_path / "feature_experiment_report.txt").read_text()
    assert "universe_filter={'ep_ratio_ttm': 'above_median', 'return_on_equity_ttm': 'above_median'}" in report
