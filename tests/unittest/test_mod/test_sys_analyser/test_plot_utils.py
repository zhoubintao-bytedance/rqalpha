import pandas as pd

from rqalpha.mod.rqalpha_mod_sys_analyser.plot.utils import max_dd, max_ddd


def test_max_drawdown_helpers_accept_series_without_futurewarning():
    series = pd.Series([1.0, 1.1, 0.9, 0.8], index=pd.date_range("2024-01-01", periods=4))

    dd = max_dd(series, series.index)
    ddd = max_ddd(series, series.index)

    assert dd.start == 1
    assert dd.end == 3
    assert ddd.start == 1
    assert ddd.end == 3
