from __future__ import annotations

import math

import numpy as np
import pandas as pd


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return pd.Series(numerator, dtype=float) / pd.Series(denominator, dtype=float).replace(0.0, np.nan)


def normalize_benchmark_bars(bars: pd.DataFrame | None) -> pd.DataFrame:
    if bars is None or getattr(bars, "empty", True):
        return pd.DataFrame()
    out = bars.copy()
    out.columns = [str(col).lower() for col in out.columns]
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index.name = "date"
    return out


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.Series(series, dtype=float).ewm(span=int(span), adjust=False).mean()


def wilder_smooth(values: pd.Series | np.ndarray, window: int) -> pd.Series:
    x = pd.Series(values, dtype=float)
    return x.ewm(alpha=1.0 / float(window), adjust=False, min_periods=int(window)).mean()


def percentile_rank(window_values: pd.Series, current: float) -> float:
    valid = pd.Series(window_values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return math.nan
    return float((valid <= float(current)).mean())


def build_factor_value(history: pd.Series, window: int) -> dict[str, float] | None:
    valid = pd.Series(history, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        return None
    current = float(valid.iloc[-1])
    lookback = valid.iloc[-int(window) :]
    min_required = min(int(window), 20)
    percentile = math.nan if len(lookback) < min_required else percentile_rank(lookback, current)
    return {"value": current, "percentile": percentile}


def add_factor_from_series(target: dict[str, dict[str, float]], name: str, history: pd.Series, window: int) -> None:
    payload = build_factor_value(history, window=window)
    if payload is not None:
        target[name] = payload
