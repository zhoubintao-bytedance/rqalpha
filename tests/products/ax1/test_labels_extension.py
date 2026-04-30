import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.labels import MultiHorizonLabelBuilder


def make_raw_panel(n_assets: int = 5, n_days: int = 30, seed: int = 0) -> pd.DataFrame:
    """构造一个简单的 close 面板，用于 label 扩展测试。"""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    asset_ids = [f"A{i:03d}.XSHE" for i in range(n_assets)]
    rows = []
    for asset_index, order_book_id in enumerate(asset_ids):
        price = 10.0 + asset_index
        for date in dates:
            step = float(rng.normal(0.0, 0.5))
            price = max(1.0, price + step)
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "close": price,
                }
            )
    return pd.DataFrame(rows)


def test_builder_emits_only_absolute_return_labels_without_relative_contract():
    raw_df = make_raw_panel()

    labeled = MultiHorizonLabelBuilder(horizons=(1, 3)).build(raw_df)

    label_cols = {column for column in labeled.columns if column.startswith("label_")}
    assert label_cols == {"label_return_1d", "label_return_3d"}
    # 默认不做 demean / winsorize / volatility
    assert "label_volatility_3d" not in labeled.columns


def test_return_labels_use_next_session_entry_price_for_after_close_signal():
    raw_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "order_book_id": ["A"] * 5,
            "close": [10.0, 11.0, 13.0, 16.0, 20.0],
        }
    )

    labeled = MultiHorizonLabelBuilder(horizons=(1, 2), entry_lag_days=1).build(raw_df)
    first = labeled.sort_values("date").iloc[0]

    assert first["label_return_1d"] == pytest.approx(13.0 / 11.0 - 1.0)
    assert first["label_return_2d"] == pytest.approx(16.0 / 11.0 - 1.0)
    assert "label_entry_date" in labeled.columns
    assert "label_exit_date_1d" in labeled.columns
    assert first["label_entry_date"] == pd.Timestamp("2024-01-02")
    assert first["label_exit_date_1d"] == pd.Timestamp("2024-01-03")


def test_return_labels_use_adjusted_price_column():
    raw_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "order_book_id": ["A"] * 4,
            # Raw close has a split-like jump from 100 to 50.
            "close": [100.0, 50.0, 55.0, 60.0],
            "adjusted_close": [50.0, 50.0, 55.0, 60.0],
        }
    )

    labeled = MultiHorizonLabelBuilder(horizons=(1,), adjusted_price_column="adjusted_close").build(raw_df)
    first = labeled.sort_values("date").iloc[0]

    assert first["label_return_1d"] == pytest.approx(0.0)


def test_return_labels_can_use_adjustment_factor_column():
    raw_df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "order_book_id": ["A"] * 4,
            "close": [100.0, 50.0, 55.0, 60.0],
            "adjust_factor": [0.5, 1.0, 1.0, 1.0],
        }
    )

    labeled = MultiHorizonLabelBuilder(horizons=(1,), adjustment_factor_column="adjust_factor").build(raw_df)
    first = labeled.sort_values("date").iloc[0]

    assert first["label_return_1d"] == pytest.approx(0.0)


def test_cost_aware_labels_deduct_stock_and_etf_costs_separately():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "order_book_id": ["000001.XSHE", "000001.XSHE", "000001.XSHE", "510300.XSHG", "510300.XSHG", "510300.XSHG"],
            "asset_type": ["stock", "stock", "stock", "etf", "etf", "etf"],
            "close": [10.0, 10.5, 11.0, 10.0, 10.5, 11.0],
        }
    )

    labeled = MultiHorizonLabelBuilder(
        horizons=(1, 2),
        cost_config={
            "enabled": True,
            "stock": {
                "commission_rate": 0.001,
                "stamp_tax_rate": 0.001,
                "slippage_bps": 10.0,
                "impact_bps": 5.0,
            },
            "etf": {
                "commission_rate": 0.0003,
                "stamp_tax_rate": 0.0,
                "slippage_bps": 2.0,
                "impact_bps": 1.0,
            },
        },
    ).build(raw_df)

    assert {"label_net_return_1d", "label_net_return_2d"}.issubset(labeled.columns)
    stock_first = labeled[labeled["order_book_id"] == "000001.XSHE"].sort_values("date").iloc[0]
    etf_first = labeled[labeled["order_book_id"] == "510300.XSHG"].sort_values("date").iloc[0]
    assert stock_first["label_return_2d"] == pytest.approx(etf_first["label_return_2d"])
    assert stock_first["label_net_return_2d"] < etf_first["label_net_return_2d"]
    assert stock_first["label_net_return_2d"] == pytest.approx(0.10 - 0.0055)
    assert etf_first["label_net_return_2d"] == pytest.approx(0.10 - 0.0011)


def test_relative_net_labels_demean_peer_group_after_costs():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"]
            ),
            "order_book_id": ["STOCK", "STOCK", "STOCK", "ETF", "ETF", "ETF"],
            "asset_type": ["stock", "stock", "stock", "etf", "etf", "etf"],
            "universe_layer": ["industry", "industry", "industry", "industry", "industry", "industry"],
            "close": [10.0, 10.5, 11.0, 10.0, 10.5, 11.0],
        }
    )

    labeled = MultiHorizonLabelBuilder(
        horizons=(2,),
        cost_config={
            "enabled": True,
            "stock": {
                "commission_rate": 0.001,
                "stamp_tax_rate": 0.001,
                "slippage_bps": 10.0,
                "impact_bps": 5.0,
            },
            "etf": {
                "commission_rate": 0.0003,
                "stamp_tax_rate": 0.0,
                "slippage_bps": 2.0,
                "impact_bps": 1.0,
            },
        },
        relative_return_enabled=True,
        relative_group_columns=("universe_layer",),
        relative_min_group_count=2,
    ).build(raw_df)

    first_day = labeled[labeled["date"] == pd.Timestamp("2024-01-01")].sort_values("order_book_id")
    assert {"label_net_return_2d", "label_relative_net_return_2d"}.issubset(labeled.columns)
    assert first_day["label_net_return_2d"].tolist() == pytest.approx([0.10 - 0.0011, 0.10 - 0.0055])
    assert first_day["label_relative_net_return_2d"].mean() == pytest.approx(0.0, abs=1e-12)
    assert first_day.set_index("order_book_id").loc["ETF", "label_relative_net_return_2d"] > 0
    assert first_day.set_index("order_book_id").loc["STOCK", "label_relative_net_return_2d"] < 0


def test_relative_net_labels_fallback_to_date_cross_section_when_layer_too_small():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "order_book_id": ["CORE", "CORE", "INDUSTRY", "INDUSTRY"],
            "asset_type": ["etf", "etf", "etf", "etf"],
            "universe_layer": ["core", "core", "industry", "industry"],
            "close": [10.0, 11.0, 10.0, 12.0],
        }
    )

    labeled = MultiHorizonLabelBuilder(
        horizons=(1,),
        cost_config={"enabled": True, "etf": {}},
        relative_return_enabled=True,
        relative_group_columns=("universe_layer",),
        relative_min_group_count=2,
        relative_fallback="date",
    ).build(raw_df)

    first_day = labeled[labeled["date"] == pd.Timestamp("2024-01-01")].set_index("order_book_id")
    assert first_day.loc["CORE", "label_relative_net_return_1d"] == pytest.approx(-0.05)
    assert first_day.loc["INDUSTRY", "label_relative_net_return_1d"] == pytest.approx(0.05)


def test_winsorize_does_not_mutate_absolute_realized_returns():
    # 绝对收益用于组合回放和 PnL，不能被训练标签的 winsorize 改写。
    base = make_raw_panel(n_assets=10, n_days=5)
    raw_df = base.copy()
    mask = (raw_df["order_book_id"] == "A009.XSHE") & (raw_df["date"] == raw_df["date"].min())
    raw_df.loc[mask, "close"] = 1000.0

    unwinsorized = MultiHorizonLabelBuilder(horizons=(1,)).build(raw_df)
    winsorized = MultiHorizonLabelBuilder(
        horizons=(1,),
        winsorize_quantiles=(0.05, 0.95),
    ).build(raw_df)

    day0 = raw_df["date"].min()
    raw_values = unwinsorized[unwinsorized["date"] == day0]["label_return_1d"].to_numpy(dtype=float)
    winsorized_values = winsorized[winsorized["date"] == day0]["label_return_1d"].to_numpy(dtype=float)

    np.testing.assert_allclose(winsorized_values, raw_values)


def test_volatility_label_positive_and_finite():
    raw_df = make_raw_panel(n_assets=3, n_days=30)

    builder = MultiHorizonLabelBuilder(
        horizons=(1,),
        volatility_horizons=(5,),
    )
    labeled = builder.build(raw_df)

    assert "label_volatility_5d" in labeled.columns
    vol = labeled["label_volatility_5d"].dropna()
    assert len(vol) > 0
    assert (vol >= 0).all()
    assert np.isfinite(vol).all()
    # 末尾 horizon 个观测点应为 NaN（窗口外推不足）
    for _, asset_df in labeled.groupby("order_book_id"):
        asset_df = asset_df.sort_values("date")
        assert np.isnan(asset_df["label_volatility_5d"].iloc[-1])


def test_volatility_label_annualization_factor():
    """forward vol 应按 sqrt(trading_days_per_year) 年化。"""
    raw_df = make_raw_panel(n_assets=3, n_days=20, seed=1)

    default_builder = MultiHorizonLabelBuilder(
        horizons=(1,),
        volatility_horizons=(5,),
        trading_days_per_year=252,
    )
    custom_builder = MultiHorizonLabelBuilder(
        horizons=(1,),
        volatility_horizons=(5,),
        trading_days_per_year=63,  # quarterly
    )
    default_labeled = default_builder.build(raw_df).dropna(subset=["label_volatility_5d"])
    custom_labeled = custom_builder.build(raw_df).dropna(subset=["label_volatility_5d"])

    ratio = default_labeled["label_volatility_5d"].to_numpy() / custom_labeled["label_volatility_5d"].to_numpy()
    expected_ratio = np.sqrt(252.0 / 63.0)
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-9)


def test_invalid_winsorize_quantiles_raises():
    with pytest.raises(ValueError):
        MultiHorizonLabelBuilder(horizons=(1,), winsorize_quantiles=(0.9, 0.1))
    with pytest.raises(ValueError):
        MultiHorizonLabelBuilder(horizons=(1,), winsorize_quantiles=(-0.1, 0.5))
    with pytest.raises(ValueError):
        MultiHorizonLabelBuilder(horizons=(1,), winsorize_quantiles=(0.5,))


def test_invalid_volatility_horizons_raises():
    with pytest.raises(ValueError):
        MultiHorizonLabelBuilder(horizons=(1,), volatility_horizons=(0,))
    with pytest.raises(ValueError):
        MultiHorizonLabelBuilder(horizons=(1,), volatility_horizons=(-3,))


def test_relative_labels_remain_finite_with_winsorization():
    """relative label 使用 winsorized net return 再按 peer group 去均值。"""
    raw_df = make_raw_panel(n_assets=8, n_days=6)
    raw_df["asset_type"] = "etf"
    raw_df["universe_layer"] = "industry"

    builder = MultiHorizonLabelBuilder(
        horizons=(1,),
        cost_config={"enabled": True, "etf": {}},
        relative_return_enabled=True,
        relative_group_columns=("universe_layer",),
        winsorize_quantiles=(0.1, 0.9),
    )
    labeled = builder.build(raw_df)

    non_nan = labeled.dropna(subset=["label_relative_net_return_1d"])
    cross_section_means = non_nan.groupby("date")["label_relative_net_return_1d"].mean()
    assert np.isfinite(non_nan["label_relative_net_return_1d"]).all()
    for mean in cross_section_means:
        assert mean == pytest.approx(0.0, abs=1e-12)
