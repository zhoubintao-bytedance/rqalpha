import numpy as np
import pandas as pd
import pytest

from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import BASELINE_FEATURE_COLUMNS, FUNDAMENTAL_FEATURE_COLUMNS


def test_dataset_builder_builds_sorted_feature_frame(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert list(result.columns[:4]) == ["date", "order_book_id", "close", "benchmark_close"]
    assert set(BASELINE_FEATURE_COLUMNS).issubset(result.columns)
    assert {
        "mom_20d",
        "mom_60d",
        "excess_mom_20d",
        "ma_gap_20d",
        "price_position_60d",
        "volume_ratio_20d",
        "beta_60d",
        "max_drawdown_20d",
    }.issubset(result.columns)
    assert result[["date", "order_book_id"]].equals(
        result[["date", "order_book_id"]].sort_values(["date", "order_book_id"]).reset_index(drop=True)
    )


def test_dataset_builder_rejects_empty_input():
    with pytest.raises(ValueError):
        DatasetBuilder().build(pd.DataFrame(columns=["date", "order_book_id", "close", "volume", "benchmark_close"]))


def test_dataset_builder_amihud_uses_total_turnover(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert "amihud_20d" in result.columns
    assert result["amihud_20d"].notna().all()


def test_dataset_builder_without_turnover_skips_amihud_and_keeps_fallback_liquidity_features(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=False)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert "amihud_20d" not in result.columns
    assert {"turnover_ratio_20d", "dollar_volume_20d_log", "vol_adj_turnover_20d"}.issubset(result.columns)
    assert result["turnover_ratio_20d"].notna().all()


def test_dataset_builder_feature_formulae_basic_correctness(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)
    asset = raw_df["order_book_id"].iloc[0]
    asset_raw = raw_df[raw_df["order_book_id"] == asset].sort_values("date").reset_index(drop=True)
    asset_result = result[result["order_book_id"] == asset].sort_values("date").reset_index(drop=True)
    target_date = asset_result["date"].iloc[-1]
    raw_idx = asset_raw.index[asset_raw["date"] == target_date][0]
    row = asset_result.iloc[-1]

    closes = asset_raw["close"]
    benchmark = asset_raw["benchmark_close"]
    volumes = asset_raw["volume"]

    expected_mom_20d = closes.pct_change(20, fill_method=None).iloc[raw_idx]
    expected_excess_mom_20d = expected_mom_20d - benchmark.pct_change(20, fill_method=None).iloc[raw_idx]
    expected_ma_gap_20d = closes.iloc[raw_idx] / closes.rolling(20, min_periods=20).mean().iloc[raw_idx] - 1.0
    rolling_low_20d = closes.rolling(20, min_periods=20).min().iloc[raw_idx]
    rolling_high_20d = closes.rolling(20, min_periods=20).max().iloc[raw_idx]
    expected_price_position_20d = (closes.iloc[raw_idx] - rolling_low_20d) / (rolling_high_20d - rolling_low_20d)
    expected_volume_ratio_20d = volumes.iloc[raw_idx] / volumes.rolling(20, min_periods=20).mean().iloc[raw_idx]

    assert row["mom_20d"] == pytest.approx(expected_mom_20d)
    assert row["excess_mom_20d"] == pytest.approx(expected_excess_mom_20d)
    assert row["ma_gap_20d"] == pytest.approx(expected_ma_gap_20d)
    assert row["price_position_20d"] == pytest.approx(expected_price_position_20d)
    assert row["volume_ratio_20d"] == pytest.approx(expected_volume_ratio_20d)


def test_dataset_builder_sector_passthrough(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert "sector" in result.columns
    assert set(result["sector"].unique()) == {"Financials", "RealEstate", "Industrials"}


def test_dataset_builder_price_position_handles_flat_window():
    dates = pd.bdate_range("2020-01-01", periods=80)
    rows = []
    for asset_idx, asset in enumerate(["a", "b"]):
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "order_book_id": asset,
                    "close": 10.0 + asset_idx,
                    "volume": 1000.0 + asset_idx,
                    "benchmark_close": 100.0,
                }
            )
    raw_df = pd.DataFrame(rows)

    result = DatasetBuilder().build(raw_df)

    assert np.isfinite(result["price_position_20d"]).all()
    assert np.isfinite(result["price_position_60d"]).all()


def test_dataset_builder_fundamental_passthrough(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)

    for col in FUNDAMENTAL_FEATURE_COLUMNS:
        assert col in result.columns, f"fundamental column {col} missing from output"
        assert result[col].notna().any(), f"fundamental column {col} is all NaN"
