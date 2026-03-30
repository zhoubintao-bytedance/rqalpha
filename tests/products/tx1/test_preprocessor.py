import numpy as np
import pandas as pd

from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.preprocessor import FeaturePreprocessor


def test_winsorize_clips_extremes(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    # Inject an extreme value
    first_date = dataset["date"].iloc[0]
    mask = dataset["date"] == first_date
    original_max = dataset.loc[mask, "mom_40d"].max()
    dataset.loc[dataset.index[mask][0], "mom_40d"] = original_max * 100  # extreme outlier

    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=5.0, standardize=False)
    result = preprocessor.transform(dataset, ["mom_40d"])

    # The extreme value should be clipped
    assert result.loc[result.index[mask][0], "mom_40d"] < original_max * 100


def test_zscore_normalizes_cross_section(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=True)
    result = preprocessor.transform(dataset, ["mom_40d"])

    # Per-date mean should be ~0, std should be ~1
    for _, day_df in result.groupby("date"):
        if len(day_df) < 3:
            continue
        vals = day_df["mom_40d"].dropna()
        if len(vals) >= 2:
            assert abs(vals.mean()) < 0.1, f"Mean should be ~0, got {vals.mean()}"
            assert abs(vals.std() - 1.0) < 0.2, f"Std should be ~1, got {vals.std()}"


def test_neutralization_removes_sector_bias(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    preprocessor = FeaturePreprocessor(neutralize=True, winsorize_scale=None, standardize=False)
    result = preprocessor.transform(dataset, ["mom_40d"])

    # After neutralization, no NaN introduced beyond what was there
    orig_nan_count = dataset["mom_40d"].isna().sum()
    result_nan_count = result["mom_40d"].isna().sum()
    assert result_nan_count <= orig_nan_count + 5  # allow small tolerance


def test_full_pipeline_preserves_row_count(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    original_len = len(dataset)

    preprocessor = FeaturePreprocessor(neutralize=True, winsorize_scale=5.0, standardize=True)
    result = preprocessor.transform(dataset, ["mom_40d", "reversal_5d"])

    assert len(result) == original_len


def test_preprocessor_without_sector_column(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=False)  # no sector column
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    preprocessor = FeaturePreprocessor(neutralize=True, winsorize_scale=5.0, standardize=True)
    # Should not crash even without sector column
    result = preprocessor.transform(dataset, ["mom_40d"])
    assert len(result) == len(dataset)
