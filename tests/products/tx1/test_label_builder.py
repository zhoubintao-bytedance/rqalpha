from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.label_builder import LabelBuilder


def test_label_builder_adds_forward_labels_without_nans(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    labeled = LabelBuilder(horizon=20, transform="raw").build(dataset)

    assert not labeled.empty
    assert {
        "label_return_raw",
        "label_volatility_horizon",
        "label_max_drawdown_horizon",
        "target_label",
        "target_return",
        "target_volatility",
        "target_max_drawdown",
    }.issubset(labeled.columns)
    assert labeled["target_label"].notna().all()
    assert labeled["target_return"].notna().all()
    assert labeled["target_volatility"].notna().all()
    assert labeled["target_max_drawdown"].notna().all()
    assert labeled["label_return_raw"].notna().all()
    assert labeled["label_volatility_horizon"].notna().all()
    assert labeled["label_max_drawdown_horizon"].notna().all()
    assert (labeled["target_label"] == labeled["target_return"]).all()


def test_label_builder_supports_rank_transform(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    labeled = LabelBuilder(horizon=20, transform="rank").build(dataset)

    assert labeled["target_label"].between(0.0, 1.0).all()


def test_label_builder_quantile_differs_from_rank(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    rank_labeled = LabelBuilder(horizon=20, transform="rank").build(dataset)
    quant_labeled = LabelBuilder(horizon=20, transform="quantile").build(dataset)

    merged = rank_labeled[["date", "order_book_id", "target_label"]].merge(
        quant_labeled[["date", "order_book_id", "target_label"]],
        on=["date", "order_book_id"],
        suffixes=("_rank", "_quantile"),
    )
    assert not merged.empty
    assert (merged["target_label_rank"] != merged["target_label_quantile"]).any()


def test_label_builder_winsorize_clips_extremes(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    raw_labeled = LabelBuilder(horizon=20, transform="raw").build(dataset)
    win_labeled = LabelBuilder(horizon=20, transform="raw", winsorize=(0.01, 0.99)).build(dataset)

    # target_label should differ (clipped extremes)
    merged = raw_labeled[["date", "order_book_id", "target_label"]].merge(
        win_labeled[["date", "order_book_id", "target_label"]],
        on=["date", "order_book_id"],
        suffixes=("_raw", "_win"),
    )
    assert not merged.empty
    # Winsorized range should be <= raw range
    assert merged["target_label_win"].max() <= merged["target_label_raw"].max()
    assert merged["target_label_win"].min() >= merged["target_label_raw"].min()


def test_label_builder_winsorize_preserves_label_return_raw(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    raw_labeled = LabelBuilder(horizon=20, transform="raw").build(dataset)
    win_labeled = LabelBuilder(horizon=20, transform="raw", winsorize=(0.01, 0.99)).build(dataset)

    # label_return_raw must be identical (winsorize only affects target_label)
    merged = raw_labeled[["date", "order_book_id", "label_return_raw"]].merge(
        win_labeled[["date", "order_book_id", "label_return_raw"]],
        on=["date", "order_book_id"],
        suffixes=("_raw", "_win"),
    )
    assert (merged["label_return_raw_raw"] == merged["label_return_raw_win"]).all()


def test_label_builder_horizon_10(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    labeled_20 = LabelBuilder(horizon=20, transform="raw").build(dataset)
    labeled_10 = LabelBuilder(horizon=10, transform="raw").build(dataset)

    # Horizon 10 should produce more labeled rows (less data lost at tail)
    assert len(labeled_10) >= len(labeled_20)
    assert labeled_10["target_label"].notna().all()
    assert labeled_10["label_return_raw"].notna().all()


def test_label_builder_supports_auxiliary_target_transforms(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    labeled = LabelBuilder(
        horizon=20,
        transform="rank",
        target_config={
            "volatility": {"transform": "log1p"},
            "max_drawdown": {"transform": "robust"},
        },
    ).build(dataset)

    assert labeled["target_volatility"].ge(0.0).all()
    assert labeled["target_max_drawdown"].notna().all()
    assert not labeled["target_volatility"].equals(labeled["label_volatility_horizon"])
    assert not labeled["target_max_drawdown"].equals(labeled["label_max_drawdown_horizon"])
