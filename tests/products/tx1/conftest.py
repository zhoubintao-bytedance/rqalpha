import numpy as np
import pandas as pd
import pytest


ASSETS = ("000001.XSHE", "000002.XSHE", "000004.XSHE")


@pytest.fixture
def make_raw_panel():
    def _make_raw_panel(periods=200, assets=ASSETS):
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
                rows.append(
                    {
                        "date": date,
                        "order_book_id": asset,
                        "close": float(close[i]),
                        "volume": float(max(volume[i], 10_000.0)),
                        "benchmark_close": float(benchmark_close[i]),
                    }
                )
        return pd.DataFrame(rows)

    return _make_raw_panel
