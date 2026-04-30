import pandas as pd
import pytest

from skyeye.products.ax1.config import normalize_config


def _raw_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"]),
            "order_book_id": ["510300.XSHG", "510300.XSHG", "510500.XSHG", "510500.XSHG"],
            "close": [10.0, 10.2, 20.0, 20.4],
            "adjusted_close": [10.0, 10.2, 20.0, 20.4],
            "volume": [1_000_000, 1_100_000, 2_000_000, 2_100_000],
        }
    )


def test_training_blocks_unadjusted_or_unknown_price_contract():
    from skyeye.products.ax1.data_quality import build_raw_data_quality_report, enforce_data_quality

    config = normalize_config({})
    raw_df = _raw_panel().drop(columns=["adjusted_close"])

    report = build_raw_data_quality_report(raw_df, config)

    assert report["passed"] is False
    assert "missing_price_adjustment_contract" in {item["reason_code"] for item in report["hard_blocks"]}
    with pytest.raises(ValueError, match="missing_price_adjustment_contract"):
        enforce_data_quality(report, context="raw_data_quality")


def test_run_experiment_blocks_missing_price_adjustment_contract(tmp_path):
    from skyeye.products.ax1.run_experiment import run_experiment

    raw_df = _raw_panel().drop(columns=["adjusted_close"])

    with pytest.raises(ValueError, match="raw_data_quality.*missing_price_adjustment_contract"):
        run_experiment(
            profile_path=None,
            output_dir=tmp_path / "experiments",
            raw_df=raw_df,
            experiment_name="ax1_missing_adjustment_contract_test",
        )


def test_raw_data_quality_blocks_duplicate_non_positive_and_missing_prices():
    from skyeye.products.ax1.data_quality import build_raw_data_quality_report

    config = normalize_config({})
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "order_book_id": ["510300.XSHG", "510300.XSHG", "510300.XSHG"],
            "close": [10.0, -1.0, None],
            "adjusted_close": [10.0, -1.0, None],
            "volume": [1000, 1000, 1000],
        }
    )

    report = build_raw_data_quality_report(raw_df, config)

    assert report["passed"] is False
    reason_codes = {item["reason_code"] for item in report["hard_blocks"]}
    assert {"duplicate_date_order_book_id", "non_positive_price", "missing_label_price"} <= reason_codes
    assert report["row_count"] == 3
    assert report["data_version"]["row_count"] == 3
    assert report["data_version"]["data_hash"]


def test_data_version_hash_changes_when_feature_columns_change():
    from skyeye.products.ax1.data_quality import build_data_version

    raw_df = _raw_panel()

    version_a = build_data_version(raw_df, feature_columns=["feature_momentum_5d", "feature_liquidity_score"])
    version_b = build_data_version(raw_df, feature_columns=["feature_momentum_5d", "feature_cost_forecast"])

    assert version_a["feature_column_count"] == 2
    assert version_a["feature_columns"] == ["feature_momentum_5d", "feature_liquidity_score"]
    assert version_a["feature_schema_hash"] != version_b["feature_schema_hash"]
    assert version_a["data_hash"] != version_b["data_hash"]


def test_feature_quality_reports_missingness_before_lgbm_fillna():
    from skyeye.products.ax1.data_quality import build_feature_matrix_quality_report

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "order_book_id": ["A", "A", "A"],
            "feature_ok": [1.0, 2.0, 3.0],
            "feature_sparse": [1.0, None, None],
            "feature_all_missing": [None, None, None],
        }
    )

    report = build_feature_matrix_quality_report(
        frame,
        ["feature_ok", "feature_sparse", "feature_all_missing"],
        config={"data": {"quality": {"feature_missing_warning_ratio": 0.50}}},
    )

    assert report["passed"] is False
    assert "all_empty_feature_column" in {item["reason_code"] for item in report["hard_blocks"]}
    assert "high_feature_missingness" in {item["reason_code"] for item in report["warnings"]}
    assert report["features"]["feature_sparse"]["missing_ratio"] == pytest.approx(2.0 / 3.0)


def test_explicit_adjusted_close_status_is_recorded_as_price_contract():
    from skyeye.products.ax1.data_quality import build_raw_data_quality_report

    config = normalize_config(
        {
            "data": {
                "price_adjustment": {
                    "adjusted_price_column": None,
                    "adjustment_factor_column": None,
                    "allow_declared_adjusted_close": True,
                }
            }
        }
    )
    raw_df = _raw_panel().drop(columns=["adjusted_close"])
    raw_df["price_adjustment_status"] = "adjusted"

    report = build_raw_data_quality_report(raw_df, config)

    assert report["passed"] is True
    assert report["price_adjustment"]["method"] == "declared_adjusted_price_column"
    assert report["price_adjustment"]["price_column"] == "close"
