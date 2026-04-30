import pandas as pd
import pytest

from skyeye.products.ax1.training import SingleSplitSplitter, WalkForwardSplitter


def make_labeled_panel(n_days: int = 200, n_assets: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for asset_index in range(n_assets):
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "order_book_id": f"A{asset_index:03d}.XSHE",
                    "close": 10.0 + asset_index + (date - dates[0]).days * 0.01,
                    "label_return_5d": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_split_produces_single_fold_with_correct_date_ranges():
    labeled = make_labeled_panel(n_days=200)

    folds = SingleSplitSplitter(
        train_end="2023-04-01",
        val_months=1,
        test_months=1,
        embargo_days=3,
    ).split(labeled)

    assert len(folds) == 1
    fold = folds[0]
    assert fold["fold_id"] == 0
    assert fold["train_end"] == pd.Timestamp("2023-04-01")
    # val_start 应在 train_end 之后，至少隔 embargo_days 个交易日
    assert fold["val_start"] > fold["train_end"]
    assert fold["test_start"] > fold["val_end"]


def test_fold_schema_matches_tx1_walkforward_keys():
    labeled = make_labeled_panel(n_days=200)

    folds = SingleSplitSplitter(
        train_end="2023-04-01",
        val_months=1,
        test_months=1,
        embargo_days=2,
    ).split(labeled)

    fold = folds[0]
    expected_keys = {
        "fold_id",
        "train_df",
        "val_df",
        "test_df",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    }
    assert set(fold.keys()) == expected_keys


def test_split_returns_empty_when_train_end_beyond_data():
    labeled = make_labeled_panel(n_days=30)

    folds = SingleSplitSplitter(
        train_end="2099-12-31",
        val_months=1,
        test_months=1,
    ).split(labeled)

    assert folds == []


def test_split_returns_empty_when_data_insufficient():
    labeled = make_labeled_panel(n_days=10)

    folds = SingleSplitSplitter(
        train_end="2023-01-05",
        val_months=6,
        test_months=6,
        embargo_days=20,
    ).split(labeled)

    assert folds == []


def test_split_respects_embargo_days():
    """embargo_days 越大，val_start 越晚。"""
    labeled = make_labeled_panel(n_days=200)

    fold_small_embargo = SingleSplitSplitter(
        train_end="2023-04-01",
        val_months=1,
        test_months=1,
        embargo_days=1,
    ).split(labeled)[0]

    fold_large_embargo = SingleSplitSplitter(
        train_end="2023-04-01",
        val_months=1,
        test_months=1,
        embargo_days=10,
    ).split(labeled)[0]

    assert fold_large_embargo["val_start"] > fold_small_embargo["val_start"]


def test_split_dfs_are_disjoint_across_train_val_test():
    labeled = make_labeled_panel(n_days=200)

    fold = SingleSplitSplitter(
        train_end="2023-04-01",
        val_months=1,
        test_months=1,
        embargo_days=3,
    ).split(labeled)[0]

    train_max = fold["train_df"]["date"].max()
    val_min = fold["val_df"]["date"].min()
    val_max = fold["val_df"]["date"].max()
    test_min = fold["test_df"]["date"].min()

    assert train_max < val_min
    assert val_max < test_min


def test_split_empty_frame_returns_empty():
    empty = pd.DataFrame({"date": [], "order_book_id": [], "close": []})

    folds = SingleSplitSplitter(train_end="2023-04-01").split(empty)

    assert folds == []


def test_invalid_months_raises():
    with pytest.raises(ValueError):
        SingleSplitSplitter(train_end="2023-04-01", val_months=0)
    with pytest.raises(ValueError):
        SingleSplitSplitter(train_end="2023-04-01", test_months=-1)


def test_invalid_embargo_days_raises():
    with pytest.raises(ValueError):
        SingleSplitSplitter(train_end="2023-04-01", embargo_days=-5)


# ---------------------------------------------------------------------------
# WalkForwardSplitter tests
# ---------------------------------------------------------------------------


def test_walk_forward_produces_multiple_folds_with_rolling_train_end():
    labeled = make_labeled_panel(n_days=540)

    splitter = WalkForwardSplitter(
        train_end="2023-06-01",
        val_months=1,
        test_months=1,
        embargo_days=2,
        n_folds=3,
        step_months=1,
    )
    folds = splitter.split(labeled)

    assert len(folds) == 3
    assert [fold["fold_id"] for fold in folds] == [0, 1, 2]
    # train_end 应按 step_months 滚动往后
    assert folds[0]["train_end"] < folds[1]["train_end"] < folds[2]["train_end"]


def test_walk_forward_fold_schema_matches_single_split():
    labeled = make_labeled_panel(n_days=400)

    folds = WalkForwardSplitter(
        train_end="2023-05-01",
        val_months=1,
        test_months=1,
        embargo_days=2,
        n_folds=2,
        step_months=1,
    ).split(labeled)

    expected_keys = {
        "fold_id",
        "train_df",
        "val_df",
        "test_df",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    }
    for fold in folds:
        assert set(fold.keys()) == expected_keys


def test_walk_forward_folds_are_disjoint_within_each_fold():
    labeled = make_labeled_panel(n_days=400)

    folds = WalkForwardSplitter(
        train_end="2023-05-01",
        val_months=1,
        test_months=1,
        embargo_days=3,
        n_folds=2,
        step_months=1,
    ).split(labeled)

    for fold in folds:
        train_max = fold["train_df"]["date"].max()
        val_min = fold["val_df"]["date"].min()
        val_max = fold["val_df"]["date"].max()
        test_min = fold["test_df"]["date"].min()
        assert train_max < val_min
        assert val_max < test_min


def test_walk_forward_stops_when_data_exhausted():
    labeled = make_labeled_panel(n_days=200)

    folds = WalkForwardSplitter(
        train_end="2023-04-01",
        val_months=1,
        test_months=1,
        embargo_days=2,
        n_folds=10,
        step_months=1,
    ).split(labeled)

    # 即使请求 10 fold，数据不足时只返回能完整切出的 fold
    assert 0 < len(folds) < 10


def test_walk_forward_auto_train_end_keeps_last_fold_inside_complete_labels():
    labeled = make_labeled_panel(n_days=260)
    labeled["label_return_20d"] = 0.02
    last_complete_date = pd.Timestamp(labeled["date"].max()) - pd.Timedelta(days=20)
    labeled.loc[labeled["date"] > last_complete_date, "label_return_20d"] = pd.NA

    folds = WalkForwardSplitter(
        train_end="auto",
        val_months=1,
        test_months=1,
        embargo_days=3,
        n_folds=3,
        step_months=1,
    ).split(labeled)

    assert len(folds) == 3
    assert all(fold["train_end"] < fold["val_start"] < fold["test_start"] for fold in folds)
    assert max(fold["test_end"] for fold in folds) <= last_complete_date
    assert max(fold["test_end"] for fold in folds) >= last_complete_date - pd.Timedelta(days=35)


def test_walk_forward_invalid_params_raise():
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_end="2023-04-01", n_folds=0)
    with pytest.raises(ValueError):
        WalkForwardSplitter(train_end="2023-04-01", step_months=0)


def test_walk_forward_default_embargo_matches_ax1_profile_default():
    splitter = WalkForwardSplitter(train_end="2023-04-01")

    assert splitter.embargo_days == 20


def test_walk_forward_empty_when_data_insufficient():
    labeled = make_labeled_panel(n_days=10)

    folds = WalkForwardSplitter(
        train_end="2023-01-05",
        val_months=6,
        test_months=6,
        embargo_days=20,
        n_folds=3,
        step_months=1,
    ).split(labeled)

    assert folds == []
