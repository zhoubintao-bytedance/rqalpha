"""Tests for AX1 universe liquidity and AUM filtering."""

import pandas as pd
import pytest

from skyeye.products.ax1.universe import DynamicUniverseBuilder


@pytest.fixture
def base_frame():
    """Create a basic ETF panel for testing."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        # High liquidity ETF (passes filters)
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,  # 4亿成交额
            "asset_type": "etf",
            "aum": 100_000_000,  # 1亿规模
        })
        # Low liquidity ETF (should be filtered)
        rows.append({
            "date": date,
            "order_book_id": "512345.XSHG",
            "close": 1.0,
            "volume": 5_000_000,  # 500万成交额
            "asset_type": "etf",
            "aum": 20_000_000,  # 2000万规模
        })
        # Very low liquidity ETF (should be filtered)
        rows.append({
            "date": date,
            "order_book_id": "159999.XSHE",
            "close": 0.5,
            "volume": 1_000_000,  # 50万成交额
            "asset_type": "etf",
            "aum": 5_000_000,  # 500万规模
        })
    return pd.DataFrame(rows)


def test_filter_by_aum_filters_low_aum_etfs(base_frame):
    """ETFs with AUM below threshold should be filtered out."""
    builder = DynamicUniverseBuilder()
    config = {
        "layers": {
            "core": {"asset_type": "etf", "include": ["510300.XSHG", "512345.XSHG", "159999.XSHE"]},
        },
        "min_aum": 50_000_000,  # 5000万
    }

    result = builder.build(base_frame, as_of_date="2026-01-30", config=config)

    # Only 510300.XSHG passes AUM filter (1亿 >= 5000万)
    assert "510300.XSHG" in result
    assert "512345.XSHG" not in result  # 2000万 < 5000万
    assert "159999.XSHE" not in result  # 500万 < 5000万


def test_filter_by_liquidity_filters_low_volume_etfs(base_frame):
    """ETFs with avg daily dollar volume below threshold should be filtered."""
    builder = DynamicUniverseBuilder()
    config = {
        "layers": {
            "core": {"asset_type": "etf", "include": ["510300.XSHG", "512345.XSHG", "159999.XSHE"]},
        },
        "min_daily_dollar_volume": 10_000_000,  # 1000万
        "liquidity_lookback_days": 20,
    }

    result = builder.build(base_frame, as_of_date="2026-01-30", config=config)

    # 510300.XSHG: 4.0 * 1亿 = 4亿 >= 1000万 ✓
    # 512345.XSHG: 1.0 * 500万 = 500万 < 1000万 ✗
    # 159999.XSHE: 0.5 * 100万 = 50万 < 1000万 ✗
    assert "510300.XSHG" in result
    assert "512345.XSHG" not in result
    assert "159999.XSHE" not in result


def test_combined_aum_and_liquidity_filters(base_frame):
    """Both filters should be applied together."""
    builder = DynamicUniverseBuilder()
    config = {
        "layers": {
            "core": {"asset_type": "etf", "include": ["510300.XSHG", "512345.XSHG", "159999.XSHE"]},
        },
        "min_aum": 50_000_000,
        "min_daily_dollar_volume": 10_000_000,
        "liquidity_lookback_days": 20,
    }

    result = builder.build(base_frame, as_of_date="2026-01-30", config=config)

    # Only 510300 passes both filters
    assert result == ["510300.XSHG"]


def test_no_filters_when_config_disabled(base_frame):
    """When filters are not configured, all ETFs should pass."""
    builder = DynamicUniverseBuilder()
    config = {
        "layers": {
            "core": {"asset_type": "etf", "include": ["510300.XSHG", "512345.XSHG", "159999.XSHE"]},
        },
    }

    result = builder.build(base_frame, as_of_date="2026-01-30", config=config)

    # All ETFs should be present
    assert "510300.XSHG" in result
    assert "512345.XSHG" in result
    assert "159999.XSHE" in result


def test_aum_filter_respects_point_in_time():
    """AUM filter should use the latest AUM value as of cutoff date."""
    builder = DynamicUniverseBuilder()

    # Create frame with AUM changing over time
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        # ETF grows from small to large
        aum = 10_000_000 if date.day < 15 else 60_000_000
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
            "aum": aum,
        })
    frame = pd.DataFrame(rows)

    config = {
        "layers": {"core": {"asset_type": "etf", "include": ["510300.XSHG"]}},
        "min_aum": 50_000_000,
    }

    # As of early date (AUM = 10M), should be filtered
    result_early = builder.build(frame, as_of_date="2026-01-10", config=config)
    assert result_early == []

    # As of later date (AUM = 60M), should pass
    result_late = builder.build(frame, as_of_date="2026-01-20", config=config)
    assert "510300.XSHG" in result_late


def test_liquidity_filter_calculates_average_over_lookback():
    """Liquidity filter should calculate avg over lookback period, not single day."""
    builder = DynamicUniverseBuilder()

    # Create frame with fluctuating volume
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for i, date in enumerate(dates):
        # Average volume over 20 days should be ~25M
        # But last few days have high volume spike
        volume = 20_000_000 if i < 25 else 100_000_000
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 1.0,
            "volume": volume,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    config = {
        "layers": {"core": {"asset_type": "etf", "include": ["510300.XSHG"]}},
        "min_daily_dollar_volume": 15_000_000,  # 1500万
        "liquidity_lookback_days": 20,
    }

    # As of Jan 30, last 20 days avg is > 15M, should pass
    result = builder.build(frame, as_of_date="2026-01-30", config=config)
    assert "510300.XSHG" in result


def test_build_with_metadata_includes_audit_info(base_frame):
    """build_with_metadata should include audit information for liquidity filters."""
    builder = DynamicUniverseBuilder()
    config = {
        "layers": {
            "core": {"asset_type": "etf", "include": ["510300.XSHG"]},
        },
        "min_aum": 50_000_000,
        "min_daily_dollar_volume": 10_000_000,
    }

    metadata = builder.build_with_metadata(base_frame, as_of_date="2026-01-30", config=config)

    assert "pit_audit" in metadata.attrs
    audit = metadata.attrs["pit_audit"]
    assert audit["source_status"]["aum"] == "frame_point_in_time"
    assert audit["source_status"]["liquidity"] == "frame_point_in_time"


def test_filter_handles_missing_aum_column():
    """When AUM column is missing, filter should pass through unchanged."""
    builder = DynamicUniverseBuilder()

    # Frame without AUM column
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    config = {
        "layers": {"core": {"asset_type": "etf", "include": ["510300.XSHG"]}},
        "min_aum": 50_000_000,
    }

    # Should pass through unchanged (no data to filter)
    result = builder.build(frame, as_of_date="2026-01-30", config=config)
    assert "510300.XSHG" in result


def test_filter_handles_missing_volume_column():
    """When volume column is missing, liquidity filter should pass through unchanged."""
    builder = DynamicUniverseBuilder()

    # Frame without volume column
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "asset_type": "etf",
            "aum": 100_000_000,
        })
    frame = pd.DataFrame(rows)

    config = {
        "layers": {"core": {"asset_type": "etf", "include": ["510300.XSHG"]}},
        "min_daily_dollar_volume": 10_000_000,
    }

    # Should pass through unchanged (no data to filter)
    result = builder.build(frame, as_of_date="2026-01-30", config=config)
    assert "510300.XSHG" in result


def test_filter_aum_from_provider():
    """AUM filter should fetch data from provider when not in frame."""
    builder = DynamicUniverseBuilder()

    # Frame without AUM
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    # Mock provider with AUM data
    class MockProvider:
        def get_fund_info(self, order_book_ids, date):
            return pd.DataFrame({
                "order_book_id": order_book_ids,
                "aum": [100_000_000 if "510300" in oid else 10_000_000 for oid in order_book_ids],
            })

    config = {
        "layers": {"core": {"asset_type": "etf", "include": ["510300.XSHG"]}},
        "min_aum": 50_000_000,
    }

    result = builder.build(
        frame,
        as_of_date="2026-01-30",
        config=config,
        data_provider=MockProvider(),
    )

    assert "510300.XSHG" in result


def test_pit_universe_validation_filters_delisted_etfs():
    """PIT universe validation should exclude ETFs that didn't exist on as_of_date.

    This prevents survivorship bias where backtests only include currently trading ETFs.
    """
    builder = DynamicUniverseBuilder()

    # Create frame with an ETF that looks valid in raw data
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",  # Active ETF
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
        })
        rows.append({
            "date": date,
            "order_book_id": "159999.XSHE",  # This ETF delisted before as_of_date
            "close": 1.0,
            "volume": 50_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    # Mock provider: only 510300 exists on 2026-01-30, 159999 is delisted
    class MockProvider:
        def all_instruments(self, type, date):
            # Only return active ETF (simulating delisted ETF not appearing)
            return pd.DataFrame({
                "order_book_id": ["510300.XSHG"],
                "symbol": ["300ETF"],
            })

    config = {
        "layers": {"core": {"asset_type": "etf"}},
        "validate_pit_universe": True,  # Enable PIT validation
    }

    result = builder.build(
        frame,
        as_of_date="2026-01-30",
        config=config,
        data_provider=MockProvider(),
    )

    # Only active ETF should pass, delisted ETF should be filtered
    assert "510300.XSHG" in result
    assert "159999.XSHE" not in result


def test_pit_universe_validation_can_be_disabled():
    """PIT universe validation can be disabled for cases where raw_df is trusted."""
    builder = DynamicUniverseBuilder()

    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "159999.XSHE",
            "close": 1.0,
            "volume": 50_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    # Mock provider returns empty (ETF delisted)
    class MockProvider:
        def all_instruments(self, type, date):
            return pd.DataFrame(columns=["order_book_id", "symbol"])

    config = {
        "layers": {"core": {"asset_type": "etf"}},
        "validate_pit_universe": False,  # Disable PIT validation
    }

    result = builder.build(
        frame,
        as_of_date="2026-01-30",
        config=config,
        data_provider=MockProvider(),
    )

    # When disabled, ETF passes even if not in provider
    assert "159999.XSHE" in result


def test_pit_universe_audit_warns_when_no_provider():
    """Audit should warn when PIT validation is enabled but no provider given (research purpose)."""
    builder = DynamicUniverseBuilder()

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    config = {
        "layers": {"core": {"asset_type": "etf"}},
        "validate_pit_universe": True,
    }

    # Default purpose is "research", should produce warning
    metadata = builder.build_with_metadata(
        frame,
        as_of_date="2026-01-10",
        config=config,
        data_provider=None,  # No provider!
        purpose="research",
    )

    audit = metadata.attrs["pit_audit"]
    assert audit["source_status"]["pit_universe"] == "validation_skipped_no_provider"
    assert any("survivorship bias" in w["message"] for w in audit["warnings"])


def test_pit_universe_audit_blocks_when_no_provider_for_promotable():
    """Audit should hard block for promotable purpose when no provider given."""
    builder = DynamicUniverseBuilder()

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    config = {
        "layers": {"core": {"asset_type": "etf"}},
        "validate_pit_universe": True,
    }

    # For promotable training, should produce hard block
    metadata = builder.build_with_metadata(
        frame,
        as_of_date="2026-01-10",
        config=config,
        data_provider=None,  # No provider!
        purpose="promotable_training",
    )

    audit = metadata.attrs["pit_audit"]
    assert audit["source_status"]["pit_universe"] == "validation_skipped_no_provider"
    assert any("survivorship bias" in w["message"] for w in audit["hard_blocks"])
    assert not audit["passed"]


def test_pit_universe_audit_shows_validated_status():
    """Audit should show validated status when PIT check succeeds."""
    builder = DynamicUniverseBuilder()

    dates = pd.date_range("2026-01-01", periods=10, freq="D")
    rows = []
    for date in dates:
        rows.append({
            "date": date,
            "order_book_id": "510300.XSHG",
            "close": 4.0,
            "volume": 100_000_000,
            "asset_type": "etf",
        })
    frame = pd.DataFrame(rows)

    class MockProvider:
        def all_instruments(self, type, date):
            return pd.DataFrame({
                "order_book_id": ["510300.XSHG"],
                "symbol": ["300ETF"],
            })

    config = {
        "layers": {"core": {"asset_type": "etf"}},
        "validate_pit_universe": True,
    }

    metadata = builder.build_with_metadata(
        frame,
        as_of_date="2026-01-10",
        config=config,
        data_provider=MockProvider(),
    )

    audit = metadata.attrs["pit_audit"]
    assert audit["source_status"]["pit_universe"] == "validated_via_provider"
