from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorLayerConfig:
    percentile_window: int = 252
    rsi_period: int = 14
    kdj_n: int = 9
    kdj_m1: int = 3
    kdj_m2: int = 3
    cci_period: int = 20
    mom_period: int = 10
    roc_period: int = 10
    bias_period: int = 6
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_periods: tuple[int, ...] = (5, 10, 20, 60)
    ema_periods: tuple[int, ...] = (12, 26)
    boll_period: int = 20
    boll_std: float = 2.0
    dc_period: int = 20
    atr_period: int = 14
    obv_ma_period: int = 20
    mfi_period: int = 14
    rsrs_n: int = 18  # 与 market_regime_layer 保持一致
    rsrs_m: int = 600  # RSRS zscore 计算窗口
