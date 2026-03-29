from dateutil.relativedelta import relativedelta

from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.splitter import WalkForwardSplitter


def test_splitter_builds_non_overlapping_walk_forward_folds(make_raw_panel):
    raw_df = make_raw_panel(periods=2200)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    labeled = LabelBuilder(horizon=20, transform="raw").build(dataset)

    splitter = WalkForwardSplitter(train_years=3, val_months=6, test_months=6, embargo_days=20)
    folds = splitter.split(labeled)

    assert folds
    first = folds[0]
    train_dates = sorted(first["train_df"]["date"].unique())
    val_dates = sorted(first["val_df"]["date"].unique())
    test_dates = sorted(first["test_df"]["date"].unique())
    all_dates = sorted(labeled["date"].unique())

    train_end_idx = all_dates.index(train_dates[-1])
    val_start_idx = all_dates.index(val_dates[0])
    val_end_idx = all_dates.index(val_dates[-1])
    test_start_idx = all_dates.index(test_dates[0])

    assert train_dates[-1] < val_dates[0] < test_dates[0]
    assert val_start_idx - train_end_idx > 20
    assert test_start_idx - val_end_idx > 20

    expected_train_end = train_dates[0] + relativedelta(years=3) - relativedelta(days=1)
    assert train_dates[-1] <= expected_train_end

    if len(folds) > 1:
        second = folds[1]
        second_train_dates = sorted(second["train_df"]["date"].unique())
        second_expected_train_end = second_train_dates[0] + relativedelta(years=3) - relativedelta(days=1)
        assert second_train_dates[-1] <= second_expected_train_end
        assert second_train_dates[0] > train_dates[0]


def test_splitter_rejects_too_short_dataset(make_raw_panel):
    raw_df = make_raw_panel(periods=200)
    dataset = DatasetBuilder(input_window=60).build(raw_df)
    labeled = LabelBuilder(horizon=20, transform="raw").build(dataset)

    folds = WalkForwardSplitter(train_years=3, val_months=6, test_months=6, embargo_days=20).split(labeled)

    assert folds == []
