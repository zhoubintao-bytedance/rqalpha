import numpy as np
import pandas as pd
import pytest


ASSETS = ("000001.XSHE", "000002.XSHE", "000004.XSHE")

SECTORS = {
    "000001.XSHE": "Financials",
    "000002.XSHE": "RealEstate",
    "000004.XSHE": "Industrials",
}


@pytest.fixture
def make_raw_panel():
    def _make_raw_panel(periods=200, assets=ASSETS, extended=False):
        dates = pd.bdate_range("2018-01-01", periods=periods)
        benchmark_close = 100.0 + np.linspace(0.0, 20.0, periods) + 2.0 * np.sin(np.arange(periods) / 25.0)
        rows = []
        for asset_idx, asset in enumerate(assets):
            base = 20.0 + asset_idx * 5.0
            trend = np.linspace(0.0, 15.0 + asset_idx * 2.0, periods)
            seasonal = 0.8 * np.sin(np.arange(periods) / (7.0 + asset_idx))
            asset_bias = asset_idx * 0.3 * np.cos(np.arange(periods) / 17.0)
            close = base + trend + seasonal + asset_bias
            volume = 1_000_000 + asset_idx * 200_000 + 100_000 * np.cos(np.arange(periods) / 9.0)
            for i, date in enumerate(dates):
                row = {
                    "date": date,
                    "order_book_id": asset,
                    "close": float(close[i]),
                    "volume": float(max(volume[i], 10_000.0)),
                    "benchmark_close": float(benchmark_close[i]),
                }
                if extended:
                    # Generate realistic OHLC data
                    c = close[i]
                    daily_range = abs(c * 0.02 * (1 + 0.5 * np.sin(i / 5.0)))
                    row["high"] = float(c + daily_range * 0.6)
                    row["low"] = float(c - daily_range * 0.4)
                    row["open"] = float(c + daily_range * 0.1 * np.sin(i / 3.0))
                    row["prev_close"] = float(close[max(0, i - 1)])
                    row["total_turnover"] = float(max(volume[i], 10_000.0) * c)
                    row["sector"] = SECTORS.get(asset, "Unknown")
                    # Simulated northbound net flow (market-level, same for all assets)
                    row["north_net_flow"] = float(5.0 * np.sin(i / 10.0) + np.random.default_rng(i).normal(0, 2))
                rows.append(row)
        return pd.DataFrame(rows)

    return _make_raw_panel
