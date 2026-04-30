import pandas as pd
import pytest


def test_data_source_registry_marks_implemented_sources():
    from skyeye.products.ax1.data_sources import build_default_data_source_registry

    registry = build_default_data_source_registry()
    capabilities = {capability.name: capability for capability in registry.capabilities()}

    # Implemented sources
    assert capabilities["price_volume.daily_ohlcv"].status == "implemented"
    assert capabilities["price_volume.daily_ohlcv"].point_in_time is True
    assert capabilities["fundamental.valuation_quality"].status == "implemented"
    assert capabilities["flow.capital_flow"].status == "implemented"

    # Not implemented sources
    assert capabilities["macro.bond_yield"].status == "implemented"
    assert capabilities["macro.northbound_aggregate"].status == "implemented"
    assert capabilities["macro.pmi"].status == "implemented"
    assert capabilities["technical.indicators"].status == "implemented"


def test_price_volume_source_returns_declared_columns():
    from skyeye.products.ax1.data_sources.price_volume import PriceVolumeDataSource

    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "order_book_id": ["510300.XSHG"],
            "open": [9.9],
            "high": [10.2],
            "low": [9.8],
            "close": [10.0],
            "adjusted_close": [10.0],
            "volume": [1_000_000],
        }
    )

    loaded = PriceVolumeDataSource(raw_df).load_panel()

    assert list(loaded.columns) == ["date", "order_book_id", "open", "high", "low", "close", "adjusted_close", "volume"]
    assert loaded.loc[0, "order_book_id"] == "510300.XSHG"


class TestFundamentalDataSource:
    """Tests for FundamentalDataSource."""

    def test_capabilities_declare_point_in_time(self):
        """Test that fundamental data source declares point-in-time capability."""
        from skyeye.products.ax1.data_sources.fundamental import FundamentalDataSource

        source = FundamentalDataSource()
        capabilities = source.capabilities()

        assert len(capabilities) == 1
        assert capabilities[0].source_family == "fundamental"
        assert capabilities[0].point_in_time is True
        assert capabilities[0].observable_lag_days == 1
        assert capabilities[0].status == "implemented"

    def test_merge_and_normalize_handles_empty_inputs(self):
        """Test that _merge_and_normalize handles empty inputs gracefully."""
        from skyeye.products.ax1.data_sources.fundamental import FundamentalDataSource

        source = FundamentalDataSource()

        # Both None
        result = source._merge_and_normalize(None, None)
        assert result.empty
        assert "feature_pe_ttm" in result.columns
        assert "feature_pb_ratio" in result.columns
        assert "feature_roe_ttm" in result.columns

    def test_merge_and_normalize_merges_dataframes(self):
        """Test that _merge_and_normalize correctly merges DataFrames."""
        from skyeye.products.ax1.data_sources.fundamental import FundamentalDataSource

        source = FundamentalDataSource()

        factor_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "order_book_id": ["000001.XSHE", "000001.XSHE"],
                "feature_pe_ttm": [10.0, 11.0],
                "feature_pb_ratio": [1.5, 1.6],
            }
        )

        roe_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "order_book_id": ["000001.XSHE", "000001.XSHE"],
                "feature_roe_ttm": [0.15, 0.16],
            }
        )

        result = source._merge_and_normalize(factor_df, roe_df)

        assert len(result) == 2
        assert "feature_pe_ttm" in result.columns
        assert "feature_pb_ratio" in result.columns
        assert "feature_roe_ttm" in result.columns
        assert result["feature_pe_ttm"].tolist() == [10.0, 11.0]

    def test_merge_and_normalize_clips_extreme_values(self):
        """Test that _merge_and_normalize clips extreme values."""
        from skyeye.products.ax1.data_sources.fundamental import FundamentalDataSource

        source = FundamentalDataSource()

        factor_df = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-01")],
                "order_book_id": ["000001.XSHE"],
                "feature_pe_ttm": [50000.0],  # Extreme high
                "feature_pb_ratio": [-5000.0],  # Extreme low
            }
        )

        result = source._merge_and_normalize(factor_df, None)

        assert result["feature_pe_ttm"].iloc[0] == 10000.0  # Clipped to upper bound
        assert result["feature_pb_ratio"].iloc[0] == -1000.0  # Clipped to lower bound


class TestFlowDataSource:
    """Tests for FlowDataSource."""

    def test_capabilities_declare_point_in_time(self):
        """Test that flow data source declares point-in-time capability."""
        from skyeye.products.ax1.data_sources.flow import FlowDataSource

        source = FlowDataSource()
        capabilities = source.capabilities()

        assert len(capabilities) == 1
        assert capabilities[0].source_family == "flow"
        assert capabilities[0].point_in_time is True
        assert capabilities[0].observable_lag_days == 1
        assert capabilities[0].status == "implemented"

    def test_is_stock_identifies_stocks_correctly(self):
        """Test _is_stock static method."""
        from skyeye.products.ax1.data_sources.flow import FlowDataSource

        source = FlowDataSource()

        # Stocks (6-digit codes not starting with 51, 159, 58)
        assert source._is_stock("000001.XSHE") is True
        assert source._is_stock("600000.XSHG") is True
        assert source._is_stock("601318.XSHG") is True

        # ETFs (codes starting with 51, 159, 58)
        assert source._is_stock("510300.XSHG") is False  # 51x ETF
        assert source._is_stock("159915.XSHE") is False  # 159x ETF
        assert source._is_stock("588000.XSHG") is False  # 58x ETF
        assert source._is_stock("512880.XSHG") is False  # 51x ETF

        # Edge cases
        assert source._is_stock("") is False
        assert source._is_stock(None) is False

    def test_load_panel_returns_empty_dataframe_on_no_data(self):
        """Test that load_panel returns empty DataFrame when no data is available."""
        from skyeye.products.ax1.data_sources.flow import FlowDataSource

        source = FlowDataSource()

        # Use invalid date range to trigger empty response
        result = source.load_panel(
            order_book_ids=["000001.XSHE"],
            start_date=pd.Timestamp("1990-01-01"),
            end_date=pd.Timestamp("1990-01-02"),
        )

        # Should return empty DataFrame with correct columns
        assert isinstance(result, pd.DataFrame)
        assert "date" in result.columns
        assert "order_book_id" in result.columns

    def test_load_northbound_data_exposes_institutional_holding_ratio(self):
        from skyeye.products.ax1.data_sources.flow import FlowDataSource

        class FakeDataFacade:
            def get_stock_connect(self, order_book_ids, start_date, end_date, fields=None):
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                        "order_book_id": ["000001.XSHE", "000001.XSHE"],
                        "shares_holding": [100.0, 110.0],
                        "holding_ratio": [0.05, 0.055],
                    }
                )

        source = FlowDataSource()
        result = source._load_northbound_data(
            order_book_ids=["000001.XSHE"],
            start_date=pd.Timestamp("2024-01-02"),
            end_date=pd.Timestamp("2024-01-03"),
            data_facade=FakeDataFacade(),
        )

        assert result is not None
        assert "feature_northbound_net_flow" in result.columns
        assert "feature_institutional_holding_ratio" in result.columns
        assert result["feature_institutional_holding_ratio"].tolist() == [0.05, 0.055]


class TestMacroDataSource:
    def test_load_panel_merges_and_forward_fills_macro_features(self):
        from skyeye.products.ax1.data_sources.macro import MacroDataSource

        class FakeDataFacade:
            def get_bond_yield(self, start_date, end_date, tenor="10Y"):
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                        "bond_yield_10Y": [2.5, 2.6, 2.7],
                    }
                )

            def get_northbound_flow(self, start_date, end_date):
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
                        "northbound_net_flow": [10.0, 11.0, 12.0],
                    }
                )

            def get_macro_pmi(self, start_date, end_date):
                return pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-01", "2024-01-31"]),
                        "pmi": [49.5, 50.2],
                    }
                )

        source = MacroDataSource()
        result = source.load_panel(
            order_book_ids=["000001.XSHE"],
            start_date=pd.Timestamp("2024-01-02"),
            end_date=pd.Timestamp("2024-01-04"),
            data_facade=FakeDataFacade(),
        )

        assert list(result.columns) == [
            "date",
            "feature_bond_yield_10y",
            "feature_northbound_aggregate_flow",
            "feature_macro_pmi",
        ]
        assert result["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-03", "2024-01-04"]
        assert result["feature_bond_yield_10y"].tolist() == pytest.approx([0.025, 0.026, 0.027])
        assert result["feature_macro_pmi"].tolist() == pytest.approx([49.5, 49.5, 49.5])


class TestTechnicalIndicatorDataSource:
    def test_load_panel_builds_rsi_and_macd_from_close_prices(self):
        from skyeye.products.ax1.data_sources.technical import TechnicalIndicatorDataSource

        dates = pd.date_range("2024-01-02", periods=40, freq="D")

        class FakeDataFacade:
            def get_daily_bars(self, order_book_ids, start_date, end_date, fields=None, adjust_type="pre"):
                rows = []
                for idx, date in enumerate(dates):
                    rows.append(
                        {
                            "date": date,
                            "order_book_id": "000001.XSHE",
                            "close": 10.0 + idx,
                        }
                    )
                return pd.DataFrame(rows)

        source = TechnicalIndicatorDataSource()
        result = source.load_panel(
            order_book_ids=["000001.XSHE"],
            start_date=dates.min(),
            end_date=dates.max(),
            data_facade=FakeDataFacade(),
        )

        assert list(result.columns) == ["date", "order_book_id", "feature_rsi_14d", "feature_macd"]
        assert len(result) == len(dates)
        assert result["feature_rsi_14d"].dropna().between(0.0, 100.0).all()
        assert result["feature_macd"].notna().all()


class TestDataSourceIntegration:
    """Integration tests for data sources with feature engineering."""

    def test_fundamental_source_integrates_with_catalog(self):
        """Test that fundamental features are properly registered in catalog."""
        from skyeye.products.ax1.features.catalog import build_default_feature_catalog

        catalog = build_default_feature_catalog()

        # Check that fundamental features are registered
        assert "feature_pe_ttm" in catalog
        assert "feature_pb_ratio" in catalog
        assert "feature_roe_ttm" in catalog

        # Check status
        pe_def = catalog.get("feature_pe_ttm")
        assert pe_def.status == "implemented"
        assert pe_def.observable_lag_days == 1

    def test_flow_source_integrates_with_catalog(self):
        """Test that flow features are properly registered in catalog."""
        from skyeye.products.ax1.features.catalog import build_default_feature_catalog

        catalog = build_default_feature_catalog()

        # Check that flow features are registered
        assert "feature_margin_financing_balance" in catalog
        assert "feature_northbound_net_flow" in catalog

        # Check status
        margin_def = catalog.get("feature_margin_financing_balance")
        assert margin_def.status == "implemented"
        assert margin_def.observable_lag_days == 1

        northbound_def = catalog.get("feature_northbound_net_flow")
        assert northbound_def.status == "implemented"
        assert northbound_def.observable_lag_days == 1
