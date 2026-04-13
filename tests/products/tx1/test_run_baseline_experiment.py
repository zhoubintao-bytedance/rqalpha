import pandas as pd
import pytest
from rqdatac.share.errors import QuotaExceeded

from skyeye.data.facade import DataFacade
from skyeye.products.tx1 import run_baseline_experiment as baseline_runner


def test_get_liquid_universe_applies_market_cap_floor_before_liquidity_ranking(monkeypatch):
    instruments = pd.DataFrame(
        {
            "order_book_id": ["A", "B", "C"],
            "circulating_market_cap": [10.0, 100.0, 90.0],
        }
    )
    dates = pd.bdate_range("2020-01-01", periods=520)
    rows = []
    volume_map = {"A": 300.0, "B": 200.0, "C": 100.0}
    for order_book_id, volume in volume_map.items():
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "volume": volume,
                }
            )
    bars = pd.DataFrame(rows)

    class FakeFacade:
        def get_daily_bars(self, order_book_ids, *args, **kwargs):
            if isinstance(order_book_ids, str):
                return pd.DataFrame(columns=["date", "order_book_id", "volume"])
            return bars[bars["order_book_id"].isin(order_book_ids)].copy()

    monkeypatch.setattr(
        baseline_runner,
        "_load_stock_instruments",
        lambda active_only=True: instruments.copy(),
    )
    monkeypatch.setattr(baseline_runner, "DATA_FACADE", FakeFacade())

    assert baseline_runner.get_liquid_universe(2) == ["A", "B"]
    assert baseline_runner.get_liquid_universe(2, market_cap_floor_quantile=0.5) == ["B", "C"]


def test_build_experiment_config_enables_multi_output_and_portfolio_overrides():
    config = baseline_runner.build_experiment_config(
        "lgbm",
        experiment_name="tx1_combo_guarded_b25_h45",
        multi_output_enabled=True,
        volatility_weight=0.15,
        max_drawdown_weight=0.20,
        enable_reliability_score=True,
        holding_bonus=0.2,
        rebalance_interval=15,
    )

    assert config["experiment_name"] == "tx1_combo_guarded_b25_h45"
    assert config["portfolio"]["holding_bonus"] == 0.2
    assert config["portfolio"]["rebalance_interval"] == 15
    assert config["multi_output"]["enabled"] is True
    assert config["multi_output"]["volatility"]["enabled"] is True
    assert config["multi_output"]["max_drawdown"]["enabled"] is True
    assert config["multi_output"]["prediction"]["combine_auxiliary"] is True
    assert config["multi_output"]["prediction"]["volatility_weight"] == 0.15
    assert config["multi_output"]["prediction"]["max_drawdown_weight"] == 0.20
    assert config["multi_output"]["reliability_score"]["enabled"] is True


def test_data_facade_bundle_path_prefers_env_override(monkeypatch):
    monkeypatch.setenv("RQALPHA_BUNDLE_PATH", "/tmp/custom-bundle")
    facade = DataFacade()

    assert facade._bundle_path() == "/tmp/custom-bundle"


def test_resolve_data_end_uses_latest_available_benchmark_date(monkeypatch):
    dates = pd.bdate_range("2026-03-25", periods=5)
    bars = pd.DataFrame(
        {
            "date": dates,
            "order_book_id": ["000300.XSHG"] * len(dates),
            "close": [1, 2, 3, 4, 5],
        }
    )

    class FakeFacade:
        def get_daily_bars(self, order_book_ids, *args, **kwargs):
            return bars.copy()

    monkeypatch.setattr(baseline_runner, "DATA_FACADE", FakeFacade())

    assert baseline_runner.resolve_data_end() == pd.Timestamp("2026-03-31")


def test_resolve_data_end_falls_back_to_default_when_benchmark_missing(monkeypatch):
    class FakeFacade:
        def get_daily_bars(self, order_book_ids, *args, **kwargs):
            return pd.DataFrame(columns=["date", "order_book_id", "close"])

    monkeypatch.setattr(baseline_runner, "DATA_FACADE", FakeFacade())

    assert baseline_runner.resolve_data_end() == pd.Timestamp(baseline_runner.DEFAULT_DATA_END)


def test_build_live_raw_df_passes_trade_date_into_universe_selection(monkeypatch):
    """live snapshot 指定历史日期时，选股池也必须以该日期为截止日。"""
    captured = {}

    def fake_get_liquid_universe(n, market_cap_floor_quantile=None, market_cap_column=None, data_end=None):
        captured["data_end"] = data_end
        return ["A", "B"]

    def fake_build_raw_df(universe, *, start_date=None, end_date=None):
        captured["universe"] = list(universe)
        captured["end_date"] = end_date
        return pd.DataFrame({"order_book_id": universe, "date": [pd.Timestamp(end_date)] * len(universe)})

    monkeypatch.setattr(baseline_runner, "get_liquid_universe", fake_get_liquid_universe)
    monkeypatch.setattr(baseline_runner, "build_raw_df", fake_build_raw_df)

    baseline_runner.build_live_raw_df(trade_date="2026-01-15")

    assert captured["data_end"] == pd.Timestamp("2026-01-15")
    assert captured["end_date"] == pd.Timestamp("2026-01-15")
    assert captured["universe"] == ["A", "B"]


def test_build_live_raw_df_runtime_fast_uses_runtime_universe_snapshot(monkeypatch):
    """runtime fast path 应优先使用缓存候选池，而不是研究侧全市场重算。"""
    captured = {}

    def fake_resolve_runtime_liquid_universe(
        *,
        trade_date,
        universe_size,
        cache_root=None,
        bundle_path=None,
        min_history_days=500,
    ):
        captured["trade_date"] = trade_date
        captured["universe_size"] = universe_size
        captured["cache_root"] = cache_root
        return {
            "order_book_ids": ["C", "D"],
            "data_end_date": "2026-01-14",
        }

    def fake_build_raw_df(universe, *, start_date=None, end_date=None):
        captured["universe"] = list(universe)
        captured["end_date"] = end_date
        return pd.DataFrame({"order_book_id": universe, "date": [pd.Timestamp(end_date)] * len(universe)})

    monkeypatch.setattr(
        baseline_runner,
        "resolve_runtime_liquid_universe",
        fake_resolve_runtime_liquid_universe,
    )
    monkeypatch.setattr(baseline_runner, "build_raw_df", fake_build_raw_df)

    baseline_runner.build_live_raw_df(
        trade_date="2026-01-15",
        universe_source="runtime_fast",
    )

    assert captured["trade_date"] == pd.Timestamp("2026-01-15")
    assert captured["universe_size"] == baseline_runner.UNIVERSE_SIZE
    assert captured["cache_root"] is None
    assert captured["end_date"] == pd.Timestamp("2026-01-15")
    assert captured["universe"] == ["C", "D"]


def test_build_live_raw_df_passes_required_features_into_raw_loader(monkeypatch):
    """验证 live raw panel 会把 package 的 required_features 透传给原始数据装载。"""
    captured = {}

    def fake_get_liquid_universe(n, market_cap_floor_quantile=None, market_cap_column=None, data_end=None):
        return ["A", "B"]

    def fake_build_raw_df(
        universe,
        *,
        start_date=None,
        end_date=None,
        required_features=None,
        data_dependency_summary=None,
        collect_source_diagnostics=False,
    ):
        captured["required_features"] = list(required_features or [])
        captured["collect_source_diagnostics"] = collect_source_diagnostics
        return pd.DataFrame({"order_book_id": universe, "date": [pd.Timestamp(end_date)] * len(universe)})

    monkeypatch.setattr(baseline_runner, "get_liquid_universe", fake_get_liquid_universe)
    monkeypatch.setattr(baseline_runner, "build_raw_df", fake_build_raw_df)

    baseline_runner.build_live_raw_df(
        trade_date="2026-01-15",
        required_features=["mom_40d", "volatility_20d"],
    )

    assert captured["required_features"] == ["mom_40d", "volatility_20d"]
    assert captured["collect_source_diagnostics"] is True


def test_build_raw_df_skips_optional_sources_when_package_features_do_not_need_them(monkeypatch):
    """验证 baseline_5f 这类 package 不会再无条件拉 northbound/fundamental。"""
    dates = pd.bdate_range("2026-01-01", periods=3)
    benchmark = pd.DataFrame({"date": dates, "benchmark_close": [1.0, 1.1, 1.2]})
    stocks = pd.DataFrame(
        {
            "date": list(dates),
            "order_book_id": ["A"] * len(dates),
            "close": [10.0, 10.5, 10.8],
            "volume": [1000.0, 1100.0, 1200.0],
            "total_turnover": [1.0, 1.1, 1.2],
        }
    )

    monkeypatch.setattr(baseline_runner, "_load_benchmark_close", lambda end_date=None: benchmark.copy())
    monkeypatch.setattr(baseline_runner, "_load_all_stocks", lambda universe, start_date, end_date: stocks.copy())
    monkeypatch.setattr(baseline_runner, "_load_sector_map", lambda: {"A": "Bank"})
    monkeypatch.setattr(
        baseline_runner,
        "_load_northbound_flow",
        lambda data_end: (_ for _ in ()).throw(AssertionError("northbound should be skipped")),
    )
    monkeypatch.setattr(
        baseline_runner,
        "_load_fundamental_factors",
        lambda universe, start_date, end_date: (_ for _ in ()).throw(AssertionError("fundamental should be skipped")),
    )

    raw_df = baseline_runner.build_raw_df(
        ["A"],
        start_date="2026-01-01",
        end_date="2026-01-05",
        required_features=["mom_40d", "volatility_20d", "reversal_5d", "amihud_20d", "turnover_stability_20d"],
        collect_source_diagnostics=True,
    )

    source_summary = raw_df.attrs["data_source_summary"]

    assert source_summary["northbound_flow"]["status"] == "skipped_feature_not_required"
    assert source_summary["fundamental_factors"]["status"] == "skipped_feature_not_required"


def test_data_facade_get_daily_bars_raises_quota_exceeded_without_bundle_fallback(monkeypatch, tmp_path):
    """验证本地无法覆盖的日线缺口遇到 QuotaExceeded 时，会显式报错。"""

    class FakeProvider:
        """模拟在线 provider 的流量额度耗尽。"""

        def get_price(self, *args, **kwargs):
            """日线查询直接抛出配额异常。"""
            raise QuotaExceeded()

    facade = DataFacade()
    facade.provider = FakeProvider()
    monkeypatch.setattr(facade, "_bundle_path", lambda: str(tmp_path))

    with pytest.raises(QuotaExceeded):
        facade.get_daily_bars("000300.XSHG", "2026-04-01", "2026-04-10", fields=["close"], adjust_type="none")


def test_data_facade_all_instruments_raises_quota_exceeded_without_bundle_fallback(monkeypatch, tmp_path):
    """验证合约列表查询遇到 QuotaExceeded 时，不会静默改读本地 instruments.pk。"""

    class FakeProvider:
        """模拟在线 provider 的合约查询额度耗尽。"""

        def get_instruments(self, *args, **kwargs):
            """合约查询直接抛出配额异常。"""
            raise QuotaExceeded()

    facade = DataFacade()
    facade.provider = FakeProvider()
    monkeypatch.setattr(facade, "_bundle_path", lambda: str(tmp_path))

    with pytest.raises(QuotaExceeded):
        facade.all_instruments(type="CS")


def test_data_facade_get_factor_raises_quota_exceeded(monkeypatch):
    """验证因子查询遇到 QuotaExceeded 时会显式报错。"""

    class FakeProvider:
        """模拟在线 provider 的因子查询额度耗尽。"""

        def get_factors(self, *args, **kwargs):
            """因子查询直接抛出配额异常。"""
            raise QuotaExceeded()

    facade = DataFacade()
    facade.provider = FakeProvider()

    with pytest.raises(QuotaExceeded):
        facade.get_factor(["000001.XSHE"], ["ep_ratio_ttm"], "2026-03-01", "2026-03-31")
