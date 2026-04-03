import numpy as np
import pandas as pd

from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.preprocessor import FeaturePreprocessor


def _make_cross_section_frame():
    date = pd.Timestamp("2024-01-31")
    rows = []
    sectors = ["A", "A", "A", "A", "B", "B", "B", "B"]
    closes = np.array([10.0, 14.0, 20.0, 24.0, 12.0, 16.0, 22.0, 26.0])
    size_signal = np.log(closes)
    sector_bias = np.array([1.5 if sector == "A" else -1.5 for sector in sectors])
    feature = 2.0 * size_signal + sector_bias
    for idx, (sector, close, value) in enumerate(zip(sectors, closes, feature)):
        rows.append(
            {
                "date": date,
                "order_book_id": f"asset_{idx}",
                "close": close,
                "sector": sector,
                "mom_40d": value,
            }
        )
    return pd.DataFrame(rows)


def test_winsorize_clips_extremes(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    first_date = dataset["date"].iloc[0]
    mask = dataset["date"] == first_date
    original_max = dataset.loc[mask, "mom_40d"].max()
    dataset.loc[dataset.index[mask][0], "mom_40d"] = original_max * 100

    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=5.0, standardize=False)
    result = preprocessor.transform(dataset, ["mom_40d"])

    assert result.loc[result.index[mask][0], "mom_40d"] < original_max * 100


def test_zscore_normalizes_cross_section(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=True)
    result = preprocessor.transform(dataset, ["mom_40d"])

    for _, day_df in result.groupby("date"):
        vals = day_df["mom_40d"].dropna()
        if len(vals) >= 2:
            assert abs(vals.mean()) < 0.1
            assert abs(vals.std() - 1.0) < 0.2


def test_neutralization_removes_size_and_sector_bias():
    dataset = _make_cross_section_frame()
    preprocessor = FeaturePreprocessor(neutralize=True, winsorize_scale=None, standardize=False)

    result = preprocessor.transform(dataset, ["mom_40d"])

    sector_means = result.groupby("sector")["mom_40d"].mean()
    assert np.abs(result["mom_40d"]).max() < 1e-10
    assert abs(sector_means["A"] - sector_means["B"]) < 1e-10


def test_full_pipeline_preserves_row_count(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=True)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    original_len = len(dataset)

    preprocessor = FeaturePreprocessor(neutralize=True, winsorize_scale=5.0, standardize=True)
    result = preprocessor.transform(dataset, ["mom_40d", "reversal_5d", "turnover_ratio_20d"])

    assert len(result) == original_len


def test_preprocessor_without_sector_column_uses_size_only(make_raw_panel):
    raw_df = make_raw_panel(periods=160, extended=False)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    preprocessor = FeaturePreprocessor(neutralize=True, winsorize_scale=5.0, standardize=True)
    result = preprocessor.transform(dataset, ["mom_40d"])

    assert len(result) == len(dataset)
    assert result["mom_40d"].notna().sum() > 0


def test_standardize_zero_variance_maps_to_zero():
    frame = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-02-01")] * 4,
            "order_book_id": ["a", "b", "c", "d"],
            "close": [10.0, 11.0, 12.0, 13.0],
            "mom_40d": [1.0, 1.0, 1.0, 1.0],
        }
    )

    preprocessor = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=True)
    result = preprocessor.transform(frame, ["mom_40d"])

    assert (result["mom_40d"] == 0.0).all()
