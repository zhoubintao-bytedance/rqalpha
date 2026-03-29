import pandas as pd
import pytest

from skyeye.products.tx1.dataset_builder import DatasetBuilder


def test_dataset_builder_builds_sorted_feature_frame(make_raw_panel):
    raw_df = make_raw_panel(periods=120)

    result = DatasetBuilder(input_window=60).build(raw_df)

    assert not result.empty
    assert list(result.columns[:4]) == ["date", "order_book_id", "close", "benchmark_close"]
    assert {"mom_20d", "mom_40d", "mom_60d", "excess_mom_20d", "volatility_20d", "volume_ratio_20d", "benchmark_mom_20d"}.issubset(result.columns)
    assert result[["date", "order_book_id"]].equals(result[["date", "order_book_id"]].sort_values(["date", "order_book_id"]).reset_index(drop=True))
    assert result["date"].min() >= pd.Timestamp("2018-03-23")


def test_dataset_builder_rejects_empty_input():
    with pytest.raises(ValueError):
        DatasetBuilder().build(pd.DataFrame(columns=["date", "order_book_id", "close", "volume", "benchmark_close"]))
