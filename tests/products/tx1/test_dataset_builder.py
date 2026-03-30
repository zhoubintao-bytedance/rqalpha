import pandas as pd
import pytest

from skyeye.products.tx1.dataset_builder import DatasetBuilder


def test_dataset_builder_builds_sorted_feature_frame(make_raw_panel):
    raw_df = make_raw_panel(periods=120)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert list(result.columns[:4]) == ["date", "order_book_id", "close", "benchmark_close"]
    assert {"mom_40d", "volatility_20d", "reversal_5d"}.issubset(result.columns)
    # Removed features must not be present
    for removed in ("mom_20d", "mom_60d", "excess_mom_20d", "regime_support",
                    "volume_ratio_20d", "overnight_gap_std_40d", "close_position_20d"):
        assert removed not in result.columns
    assert result[["date", "order_book_id"]].equals(result[["date", "order_book_id"]].sort_values(["date", "order_book_id"]).reset_index(drop=True))


def test_dataset_builder_rejects_empty_input():
    with pytest.raises(ValueError):
        DatasetBuilder().build(pd.DataFrame(columns=["date", "order_book_id", "close", "volume", "benchmark_close"]))


def test_dataset_builder_amihud_uses_total_turnover(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert "amihud_20d" in result.columns
    assert result["amihud_20d"].notna().all()


def test_dataset_builder_without_turnover_skips_amihud(make_raw_panel):
    raw_df = make_raw_panel(periods=120)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert "amihud_20d" not in result.columns


def test_dataset_builder_sector_passthrough(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert "sector" in result.columns
    assert set(result["sector"].unique()) == {"Financials", "RealEstate", "Industrials"}
