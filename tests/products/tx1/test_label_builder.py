from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.label_builder import LabelBuilder


def test_label_builder_adds_forward_labels_without_nans(make_raw_panel):
    raw_df = make_raw_panel(periods=160)
    dataset = DatasetBuilder(input_window=60).build(raw_df)

    labeled = LabelBuilder(horizon=20, transform="raw").build(dataset)

    assert not labeled.empty
    assert {"label_return_raw", "label_volatility_horizon", "label_max_drawdown_horizon", "target_label"}.issubset(labeled.columns)
    assert labeled["target_label"].notna().all()
    assert labeled["label_return_raw"].notna().all()
    assert labeled["label_volatility_horizon"].notna().all()
    assert labeled["label_max_drawdown_horizon"].notna().all()


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
