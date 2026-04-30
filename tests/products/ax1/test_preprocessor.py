import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.preprocessing import FeaturePreprocessor


def make_factor_panel(with_sector: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    sectors = ["tech", "finance", "health", "tech", "finance"]
    rows = []
    for asset_index in range(5):
        base_price = 10.0 + asset_index * 3.0
        for step, date in enumerate(dates):
            row = {
                "date": date,
                "order_book_id": f"A{asset_index:03d}.XSHE",
                "close": base_price + step + float(rng.normal(0, 0.1)),
                "factor_a": float(rng.normal(0, 1)),
                "factor_b": float(rng.normal(0, 2)),
            }
            if with_sector:
                row["sector"] = sectors[asset_index]
            rows.append(row)
    return pd.DataFrame(rows)


def test_required_columns_skips_sector_when_optional():
    pre = FeaturePreprocessor(sector_optional=True)

    required = pre.required_columns(["factor_a", "factor_b"])

    assert "sector" not in required
    assert {"date", "order_book_id", "close", "factor_a", "factor_b"}.issubset(set(required))


def test_required_columns_includes_sector_when_not_optional():
    pre = FeaturePreprocessor(sector_optional=False)

    required = pre.required_columns(["factor_a"])

    assert "sector" in required


def test_required_columns_skips_close_when_no_neutralize():
    pre = FeaturePreprocessor(neutralize=False)

    required = pre.required_columns(["factor_a"])

    assert "close" not in required
    assert "sector" not in required


def test_transform_produces_zero_mean_unit_variance_per_date():
    df = make_factor_panel(with_sector=False)
    # 关闭中性化，只看 standardize 效果
    pre = FeaturePreprocessor(neutralize=False, winsorize_scale=None, standardize=True)

    out = pre.transform(df, feature_columns=["factor_a", "factor_b"])

    for _, day in out.groupby("date"):
        for col in ["factor_a", "factor_b"]:
            values = day[col].to_numpy(dtype=float)
            assert values.mean() == pytest.approx(0.0, abs=1e-9)
            # std ddof=1 for pandas default; 5 obs → std can be 0 only if all equal
            assert values.std(ddof=1) > 0


def test_transform_without_sector_column_uses_ln_close_only():
    df = make_factor_panel(with_sector=False)
    pre = FeaturePreprocessor(neutralize=True, sector_optional=True)

    out = pre.transform(df, feature_columns=["factor_a"])

    # 跑通即通过（不抛异常），且输出列存在
    assert "factor_a" in out.columns
    assert len(out) == len(df)


def test_transform_with_sector_column_runs():
    df = make_factor_panel(with_sector=True)
    pre = FeaturePreprocessor(neutralize=True, sector_optional=True)

    out = pre.transform(df, feature_columns=["factor_a"])

    assert "factor_a" in out.columns
    assert len(out) == len(df)


def test_transform_preserves_row_order_and_ids():
    df = make_factor_panel(with_sector=False)
    pre = FeaturePreprocessor(neutralize=False)

    out = pre.transform(df, feature_columns=["factor_a"])

    assert (out["date"].values == df["date"].values).all()
    assert (out["order_book_id"].values == df["order_book_id"].values).all()


def test_transform_skips_missing_feature_column():
    df = make_factor_panel(with_sector=False)
    pre = FeaturePreprocessor(neutralize=False)

    # 不存在的列应被跳过而不是抛错
    out = pre.transform(df, feature_columns=["factor_a", "does_not_exist"])

    assert "factor_a" in out.columns
    assert "does_not_exist" not in out.columns


def test_winsorize_caps_extreme_feature_values():
    df = make_factor_panel(with_sector=False).copy()
    # 把某一天的 factor_a 做成有极端 outlier
    first_day = df["date"].min()
    mask = df["date"] == first_day
    df.loc[mask, "factor_a"] = [-10.0, -1.0, 0.0, 1.0, 100.0]
    pre = FeaturePreprocessor(neutralize=False, winsorize_scale=3.0, standardize=False)

    out = pre.transform(df, feature_columns=["factor_a"])

    day0_out = out[out["date"] == first_day]["factor_a"].to_numpy()
    # 100.0 被 MAD clip 后应远小于 100
    assert day0_out.max() < 100.0
