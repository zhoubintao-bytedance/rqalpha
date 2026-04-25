# -*- coding: utf-8 -*-
"""市场状态分类层（Market Regime Layer）

目标：输出稳定、可降级的市场分类信息，供各策略/研究模块复用。

输出（对外最小格式）：
    {"regime": "bull_co_move"|... , "strength": 0~1}

实现说明：
- 将“方向/趋势强度”和“结构/轮动强度”分离计算，再合成 6 类 regime。
- 对数据缺失/样本不足进行降级，保证能返回结果（并将 strength 降为 0）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import math

import numpy as np
import pandas as pd


RegimeLabel = Literal[
    "bull_co_move",
    "bull_rotation",
    "range_co_move",
    "range_rotation",
    "bear_co_move",
    "bear_rotation",
]


@dataclass(frozen=True)
class MarketRegime:
    regime: RegimeLabel
    strength: float
    # 诊断信息：不属于对外最小输出，但对调参/验证很有用。
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"regime": self.regime, "strength": float(self.strength)}


@dataclass(frozen=True)
class MarketRegimeConfig:
    # -----------------------------
    # 数据窗口（交易日）
    # -----------------------------
    adx_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    ma_short: int = 5
    ma_mid: int = 20
    ma_long: int = 60
    return_window: int = 20
    price_position_window: int = 60
    boll_window: int = 20
    boll_k: float = 2.0
    boll_percentile_window: int = 120
    atr_window: int = 14
    atr_ma_window: int = 60
    vol_trend_window: int = 20
    hurst_window: int = 200
    hurst_max_lag: int = 20
    rsrs_n: int = 18
    rsrs_m: int = 600

    # 结构指标用的窗口
    industry_return_window: int = 5
    dispersion_percentile_window: int = 120

    # -----------------------------
    # 权重（缺失时会自动 renormalize）
    # -----------------------------
    trendiness_weights: dict[str, float] = field(
        default_factory=lambda: {
            "adx_score": 0.30,
            "hurst_score": 0.25,
            "boll_width_score": 0.20,
            "atr_ratio_score": 0.15,
            "volume_trend_score": 0.10,
        }
    )
    direction_weights: dict[str, float] = field(
        default_factory=lambda: {
            "ma_arrangement": 0.30,
            "macd_histogram": 0.25,
            "return_20d_sign": 0.20,
            "price_position": 0.15,
            "rsrs_score": 0.10,
        }
    )
    rotation_weights: dict[str, float] = field(
        default_factory=lambda: {
            "dispersion_percentile": 0.35,
            "skewness_abs": 0.20,
            "ad_ratio_neutrality": 0.25,
            "concentration_score": 0.20,
        }
    )

    # -----------------------------
    # 阈值
    # -----------------------------
    trendiness_range_threshold: float = 0.35
    direction_bull_threshold: float = 0.15
    direction_bear_threshold: float = -0.15
    rotation_threshold: float = 0.40

    # ADX 映射（20~35 映射到 0~1）
    adx_low: float = 20.0
    adx_high: float = 35.0

    # ATR 比值映射（0.7~1.5 映射到 0~1）
    atr_ratio_low: float = 0.7
    atr_ratio_high: float = 1.5

    # 偏度绝对值映射（abs(skew)/2 -> 0~1）
    skew_abs_scale: float = 2.0

    # 行业集中度映射：concentration 由 0.25~0.50 映射到 0~1
    concentration_low: float = 0.25
    concentration_high: float = 0.50

    # AD ratio “接近 1” 的映射：ratio=2 或 0.5 -> 0
    ad_ratio_boundary: float = 2.0


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _weighted_avg(values: dict[str, float | None], weights: dict[str, float]) -> float | None:
    total_w = 0.0
    total_v = 0.0
    for k, w in weights.items():
        v = values.get(k)
        if v is None:
            continue
        wv = float(w)
        if wv <= 0:
            continue
        total_w += wv
        total_v += wv * float(v)
    if total_w <= 0:
        return None
    return float(total_v / total_w)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False).mean()


def _percentile_rank(window_values: np.ndarray, x: float) -> float | None:
    arr = np.asarray(window_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float(np.mean(arr <= float(x)))


def _hurst_exponent(prices: pd.Series, window: int, max_lag: int) -> float | None:
    s = pd.Series(prices).astype(float).dropna()
    if len(s) < int(window):
        return None
    x = s.iloc[-int(window) :].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max_lag + 2:
        return None
    lags = np.arange(2, int(max_lag) + 1)
    tau = []
    for lag in lags:
        diff = x[lag:] - x[:-lag]
        sd = float(np.std(diff))
        if not math.isfinite(sd) or sd <= 0:
            return None
        tau.append(sd)
    try:
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
    except Exception:
        return None
    h = float(poly[0] * 2.0)
    if not math.isfinite(h):
        return None
    # 常见区间约束（避免数值爆炸）
    return float(max(0.0, min(1.0, h)))


def _wilder_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Wilder's smoothing (RMA) used by ADX/ATR."""
    n = int(window)
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    if x.size == 0 or n <= 0:
        return out
    if x.size < n:
        return out
    # 初值用简单均值
    out[n - 1] = float(np.nanmean(x[:n]))
    alpha = 1.0 / float(n)
    for i in range(n, x.size):
        prev = out[i - 1]
        cur = x[i]
        if not math.isfinite(prev):
            prev = float(np.nanmean(x[max(0, i - n + 1) : i + 1]))
        if not math.isfinite(cur):
            out[i] = prev
        else:
            out[i] = (1.0 - alpha) * prev + alpha * cur
    return out


def _compute_adx(high: pd.Series | None, low: pd.Series | None, close: pd.Series, window: int) -> float | None:
    """ADX(窗口)；若缺 high/low，则用 close 的简化 DMI 近似。"""
    c = pd.Series(close).astype(float)
    if len(c) < int(window) + 2:
        return None
    if high is None or low is None:
        # 简化近似：用 close 的变化模拟方向动量；TR 用 |diff(close)|
        diff = c.diff().to_numpy(dtype=float)
        tr = np.abs(diff)
        plus_dm = np.where(diff > 0, diff, 0.0)
        minus_dm = np.where(diff < 0, -diff, 0.0)
    else:
        h = pd.Series(high).astype(float)
        l = pd.Series(low).astype(float)
        if len(h) != len(c) or len(l) != len(c):
            return None
        prev_close = c.shift(1)
        prev_high = h.shift(1)
        prev_low = l.shift(1)
        tr = np.maximum.reduce(
            [
                (h - l).to_numpy(dtype=float),
                (h - prev_close).abs().to_numpy(dtype=float),
                (l - prev_close).abs().to_numpy(dtype=float),
            ]
        )
        up_move = (h - prev_high).to_numpy(dtype=float)
        down_move = (prev_low - l).to_numpy(dtype=float)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = _wilder_smooth(tr, window)
    plus_di = 100.0 * _wilder_smooth(plus_dm, window) / atr
    minus_di = 100.0 * _wilder_smooth(minus_dm, window) / atr
    denom = plus_di + minus_di
    dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    adx = _wilder_smooth(dx, window)
    last = float(adx[-1])
    if not math.isfinite(last):
        return None
    return last


def _compute_atr_ratio(high: pd.Series | None, low: pd.Series | None, close: pd.Series, atr_window: int, ma_window: int) -> float | None:
    c = pd.Series(close).astype(float)
    if len(c) < int(atr_window) + int(ma_window) + 2:
        return None
    if high is None or low is None:
        tr = c.diff().abs().to_numpy(dtype=float)
    else:
        h = pd.Series(high).astype(float)
        l = pd.Series(low).astype(float)
        prev_close = c.shift(1)
        tr = np.maximum.reduce(
            [
                (h - l).to_numpy(dtype=float),
                (h - prev_close).abs().to_numpy(dtype=float),
                (l - prev_close).abs().to_numpy(dtype=float),
            ]
        )
    atr = pd.Series(_wilder_smooth(tr, atr_window), index=c.index)
    atr_ma = atr.rolling(int(ma_window)).mean()
    v = _safe_float(atr.iloc[-1])
    m = _safe_float(atr_ma.iloc[-1])
    if v is None or m is None or m <= 0:
        return None
    return float(v / m)


def _compute_macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> pd.Series:
    c = pd.Series(close).astype(float)
    macd = _ema(c, fast) - _ema(c, slow)
    sig = _ema(macd, signal)
    hist = macd - sig
    return hist


def _compute_rsrs_zscore(high: pd.Series, low: pd.Series, n: int = 18, m: int = 600) -> float | None:
    """RSRS 标准分：对 slope 做 rolling zscore。

    计算方式（简化实现，性能足够日频）：
    - 对每个时点，用最近 n 日 high ~ a + b * low 回归，得到 slope b。
    - 用最近 m 个 slope 计算 zscore。
    """
    h = pd.Series(high).astype(float).dropna()
    l = pd.Series(low).astype(float).dropna()
    if len(h) != len(l):
        return None
    if len(h) < int(n) + int(m):
        return None
    hh = h.to_numpy(dtype=float)
    ll = l.to_numpy(dtype=float)
    slopes = []
    start = len(hh) - int(m)
    for i in range(start, len(hh)):
        seg_h = hh[i - int(n) + 1 : i + 1]
        seg_l = ll[i - int(n) + 1 : i + 1]
        if seg_h.size != int(n) or seg_l.size != int(n):
            continue
        if not (np.isfinite(seg_h).all() and np.isfinite(seg_l).all()):
            continue
        x = seg_l
        y = seg_h
        vx = float(np.var(x))
        if vx <= 0:
            continue
        cov = float(np.cov(x, y, bias=True)[0, 1])
        b = cov / vx
        if not math.isfinite(b):
            continue
        slopes.append(float(b))
    if len(slopes) < int(m) * 0.8:
        return None
    s = np.asarray(slopes, dtype=float)
    mu = float(np.mean(s))
    sd = float(np.std(s, ddof=0))
    if not math.isfinite(sd) or sd <= 0:
        return None
    z = (float(s[-1]) - mu) / sd
    if not math.isfinite(z):
        return None
    # 不做硬截断（保留信息），下游映射成 direction 时只取符号。
    return float(z)


def _compute_direction_label_and_scores(
    bars: pd.DataFrame,
    cfg: MarketRegimeConfig,
) -> tuple[str, float | None, float | None, dict[str, Any]]:
    """返回 (direction_label, trendiness, direction_score, diagnostics)."""
    diag: dict[str, Any] = {}
    if bars is None or bars.empty:
        return ("range", None, None, {"reason": "empty_bars"})

    bars = bars.copy()
    if not isinstance(bars.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index()

    close = bars.get("close")
    if close is None:
        return ("range", None, None, {"reason": "missing_close"})
    high = bars.get("high")
    low = bars.get("low")
    volume = bars.get("volume")

    close = pd.Series(close).astype(float)
    high_s = pd.Series(high).astype(float) if high is not None else None
    low_s = pd.Series(low).astype(float) if low is not None else None
    vol_s = pd.Series(volume).astype(float) if volume is not None else None

    required = max(
        cfg.ma_long,
        cfg.return_window,
        cfg.price_position_window,
        cfg.boll_percentile_window,
        cfg.hurst_window,
        cfg.atr_window + cfg.atr_ma_window,
        cfg.adx_window,
        cfg.macd_slow + cfg.macd_signal,
    ) + 5
    if len(close) < int(required):
        return (
            "range",
            None,
            None,
            {"reason": "insufficient_history", "need": int(required), "have": int(len(close))},
        )

    # -----------------------------
    # trendiness 0~1
    # -----------------------------
    adx = _compute_adx(high_s, low_s, close, cfg.adx_window)
    adx_score = None
    if adx is not None:
        adx_score = _clamp01((float(adx) - cfg.adx_low) / (cfg.adx_high - cfg.adx_low))
    hurst = _hurst_exponent(close, cfg.hurst_window, cfg.hurst_max_lag)
    hurst_score = None
    if hurst is not None:
        hurst_score = _clamp01(abs(float(hurst) - 0.5) * 2.0)
    # Boll width percentile
    ma = close.rolling(cfg.boll_window).mean()
    sd = close.rolling(cfg.boll_window).std(ddof=0)
    mid = ma
    width = (2.0 * cfg.boll_k * sd) / mid
    width_last = _safe_float(width.iloc[-1])
    boll_width_score = None
    if width_last is not None:
        width_hist = width.iloc[-cfg.boll_percentile_window :].to_numpy(dtype=float)
        boll_width_score = _percentile_rank(width_hist, width_last)
    atr_ratio = _compute_atr_ratio(high_s, low_s, close, cfg.atr_window, cfg.atr_ma_window)
    atr_ratio_score = None
    if atr_ratio is not None:
        atr_ratio_score = _clamp01(
            (float(atr_ratio) - cfg.atr_ratio_low) / (cfg.atr_ratio_high - cfg.atr_ratio_low)
        )
    volume_trend_score = None
    if vol_s is not None and vol_s.notna().sum() >= cfg.vol_trend_window + 2:
        rets = close.pct_change().iloc[-cfg.vol_trend_window :]
        vchg = vol_s.pct_change().iloc[-cfg.vol_trend_window :].replace([np.inf, -np.inf], np.nan)
        if rets.notna().sum() >= cfg.vol_trend_window * 0.8 and vchg.notna().sum() >= cfg.vol_trend_window * 0.8:
            corr = float(rets.corr(vchg))
            if math.isfinite(corr):
                volume_trend_score = _clamp01(max(corr, 0.0))

    trendiness_components = {
        "adx_score": adx_score,
        "hurst_score": hurst_score,
        "boll_width_score": boll_width_score,
        "atr_ratio_score": atr_ratio_score,
        "volume_trend_score": volume_trend_score,
    }
    trendiness = _weighted_avg(trendiness_components, cfg.trendiness_weights)

    # -----------------------------
    # direction -1~+1
    # -----------------------------
    ma_s = close.rolling(cfg.ma_short).mean().iloc[-1]
    ma_m = close.rolling(cfg.ma_mid).mean().iloc[-1]
    ma_l = close.rolling(cfg.ma_long).mean().iloc[-1]
    ma_arrangement = None
    if all(map(math.isfinite, [float(ma_s), float(ma_m), float(ma_l)])):
        if ma_s > ma_m > ma_l:
            ma_arrangement = 1.0
        elif ma_s < ma_m < ma_l:
            ma_arrangement = -1.0
        else:
            ma_arrangement = 0.0
    macd_hist = _compute_macd_hist(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    macd_histogram = None
    hist_tail = macd_hist.iloc[-3:]
    if hist_tail.notna().sum() >= 3:
        if (hist_tail > 0).all():
            macd_histogram = 1.0
        elif (hist_tail < 0).all():
            macd_histogram = -1.0
        else:
            macd_histogram = 0.0
    ret_20 = close.pct_change(cfg.return_window).iloc[-1]
    return_20d_sign = None
    if math.isfinite(float(ret_20)):
        if ret_20 > 0:
            return_20d_sign = 1.0
        elif ret_20 < 0:
            return_20d_sign = -1.0
        else:
            return_20d_sign = 0.0
    hi = close.rolling(cfg.price_position_window).max().iloc[-1]
    lo = close.rolling(cfg.price_position_window).min().iloc[-1]
    price_position = None
    if math.isfinite(float(hi)) and math.isfinite(float(lo)) and hi > lo:
        pos = (float(close.iloc[-1]) - float(lo)) / (float(hi) - float(lo))
        price_position = float(_clamp01(pos) * 2.0 - 1.0)
    rsrs = None
    if high_s is not None and low_s is not None:
        rsrs = _compute_rsrs_zscore(high_s, low_s, n=cfg.rsrs_n, m=cfg.rsrs_m)
    rsrs_score = None
    if rsrs is not None:
        rsrs_score = 1.0 if rsrs > 0 else (-1.0 if rsrs < 0 else 0.0)

    direction_components = {
        "ma_arrangement": ma_arrangement,
        "macd_histogram": macd_histogram,
        "return_20d_sign": return_20d_sign,
        "price_position": price_position,
        "rsrs_score": rsrs_score,
    }
    direction = _weighted_avg(direction_components, cfg.direction_weights)

    diag.update(
        {
            "trendiness_components": trendiness_components,
            "direction_components": direction_components,
            "adx": adx,
            "hurst": hurst,
            "atr_ratio": atr_ratio,
            "direction": direction,
            "trendiness": trendiness,
        }
    )

    # 判定
    if trendiness is None or direction is None:
        return ("range", trendiness, direction, {**diag, "reason": "missing_scores"})
    if float(trendiness) < cfg.trendiness_range_threshold:
        label = "range"
    elif float(direction) > cfg.direction_bull_threshold:
        label = "bull"
    elif float(direction) < cfg.direction_bear_threshold:
        label = "bear"
    else:
        label = "range"
    return (label, float(trendiness), float(direction), diag)


def _map_ad_ratio_to_neutrality(ratio: float, boundary: float) -> float | None:
    r = _safe_float(ratio)
    if r is None or r <= 0:
        return None
    b = float(boundary)
    if b <= 1:
        return None
    # 用 log 尺度，ratio=1 -> 1；ratio=b 或 1/b -> 0
    dist = abs(math.log(r))
    max_dist = abs(math.log(b))
    if max_dist <= 0:
        return None
    return _clamp01(1.0 - dist / max_dist)


def _compute_rotation_score(
    industry_close: pd.DataFrame | None,
    cfg: MarketRegimeConfig,
    ad_ratio: float | None = None,
) -> tuple[str, float | None, dict[str, Any]]:
    """返回 (structure_label, rotation_score, diagnostics).

    industry_close: index=date, columns=industry_id（或反过来亦可，函数内会整理）。
    ad_ratio: 可选，涨跌家数比（adv/dec）。
    """
    diag: dict[str, Any] = {}

    dispersion_percentile = None
    skewness_abs = None
    concentration_score = None
    ad_ratio_neutrality = None

    if industry_close is not None and not industry_close.empty:
        df = industry_close.copy()
        # 允许传入：MultiIndex bars（order_book_id/date）或已 pivot 的 close 表。
        if isinstance(df.index, pd.MultiIndex):
            if "order_book_id" in df.index.names and "date" in df.index.names:
                # 期望 df 含 close 列
                if "close" in df.columns:
                    df = df["close"].unstack("order_book_id")
                else:
                    # 尝试 unstack 最后一层
                    df = df.unstack("order_book_id")
        if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 计算行业收益（横截面）
        rets = df.pct_change(cfg.industry_return_window)
        latest = rets.iloc[-1]
        latest = latest.replace([np.inf, -np.inf], np.nan).dropna()
        if len(latest) >= 5:
            dispersion = float(latest.std(ddof=0))
            # 用过去 dispersion 的分位数（历史窗口基于 rolling std）
            rolling_disp = rets.std(axis=1, ddof=0).iloc[-cfg.dispersion_percentile_window :]
            rolling_disp = rolling_disp.replace([np.inf, -np.inf], np.nan).dropna()
            if not rolling_disp.empty and math.isfinite(dispersion):
                dispersion_percentile = _percentile_rank(rolling_disp.to_numpy(dtype=float), dispersion)
            skew = float(latest.skew())
            if math.isfinite(skew):
                skewness_abs = _clamp01(abs(skew) / float(cfg.skew_abs_scale))

            # 行业集中度：top3 行业绝对收益占比
            abs_ret = latest.abs()
            denom = float(abs_ret.sum())
            if denom > 0 and math.isfinite(denom):
                top3 = float(abs_ret.sort_values(ascending=False).head(3).sum())
                conc = top3 / denom
                concentration_score = _clamp01(
                    (conc - cfg.concentration_low) / (cfg.concentration_high - cfg.concentration_low)
                )
                diag["industry_concentration"] = conc

            diag["industry_dispersion"] = dispersion
            diag["industry_skewness"] = skew
            diag["industry_count"] = int(len(latest))

    if ad_ratio is not None:
        ad_ratio_neutrality = _map_ad_ratio_to_neutrality(ad_ratio, cfg.ad_ratio_boundary)

    rotation_components = {
        "dispersion_percentile": dispersion_percentile,
        "skewness_abs": skewness_abs,
        "ad_ratio_neutrality": ad_ratio_neutrality,
        "concentration_score": concentration_score,
    }
    rotation_score = _weighted_avg(rotation_components, cfg.rotation_weights)
    diag.update({"rotation_components": rotation_components, "rotation_score": rotation_score})

    if rotation_score is None:
        # 两类数据都不可用：按方案降级
        return ("co_move", None, {**diag, "reason": "missing_structure_inputs"})
    structure_label = "rotation" if float(rotation_score) >= cfg.rotation_threshold else "co_move"
    return (structure_label, float(rotation_score), diag)


def compute_market_regime(
    benchmark_bars: pd.DataFrame,
    industry_close: pd.DataFrame | None = None,
    *,
    cfg: MarketRegimeConfig | None = None,
    ad_ratio: float | None = None,
) -> MarketRegime:
    """主入口：计算 6 类市场分类标签 + 0~1 强度。

    参数：
    - benchmark_bars: 至少包含 close；推荐包含 high/low/volume。
    - industry_close: 行业/板块指数 close 的宽表（date x industry）。可为空。
    - ad_ratio: 可选，涨跌家数比（adv/dec）。若上游能提供，会增强结构判别。
    """
    cfg = cfg or MarketRegimeConfig()

    direction_label, trendiness, direction_score, dir_diag = _compute_direction_label_and_scores(
        benchmark_bars, cfg
    )
    structure_label, rotation_score, str_diag = _compute_rotation_score(industry_close, cfg, ad_ratio=ad_ratio)

    # 合成
    regime: RegimeLabel = f"{direction_label}_{structure_label}"  # type: ignore[assignment]

    # strength：方向强度（趋势 vs 震荡）与结构强度取 min，保守表示“最弱短板”。
    direction_strength = None
    if trendiness is not None:
        direction_strength = float(trendiness) if direction_label != "range" else float(1.0 - trendiness)
    structure_strength = None
    if rotation_score is not None:
        structure_strength = float(rotation_score) if structure_label == "rotation" else float(1.0 - rotation_score)

    if direction_strength is None or structure_strength is None:
        strength = 0.0
    else:
        strength = float(min(direction_strength, structure_strength))

    diagnostics = {
        "direction_label": direction_label,
        "structure_label": structure_label,
        "trendiness": trendiness,
        "direction": direction_score,
        "rotation_score": rotation_score,
        "direction_strength": direction_strength,
        "structure_strength": structure_strength,
        "strength": strength,
        "direction_diagnostics": dir_diag,
        "structure_diagnostics": str_diag,
    }
    return MarketRegime(regime=regime, strength=float(_clamp01(strength)), diagnostics=diagnostics)


def normalize_single_instrument_bars(df: pd.DataFrame, order_book_id: str | None = None) -> pd.DataFrame:
    """将 DataFacade.get_daily_bars 的输出规范化成 index=date 的单标的 bars。

    复制了 `skyeye/evaluation/rolling_score/engine.py:_normalize_single_instrument_bars` 的核心逻辑，
    避免从 evaluation 包引用实现细节。
    """
    if df is None or df.empty:
        return df
    out = df
    if isinstance(out.index, pd.MultiIndex):
        if order_book_id is not None and "order_book_id" in out.index.names:
            try:
                out = out.xs(order_book_id, level="order_book_id")
            except KeyError:
                pass
        if isinstance(out.index, pd.MultiIndex):
            if "date" in out.index.names:
                levels_to_drop = [lvl for lvl in out.index.names if lvl != "date"]
                if levels_to_drop:
                    out = out.reset_index(level=levels_to_drop, drop=True)
            else:
                out = out.reset_index(level=list(range(out.index.nlevels - 1)), drop=True)
    if "order_book_id" in out.columns:
        out = out.drop(columns=["order_book_id"])
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out.index.name = "date"
    return out


def pivot_close_wide(bars: pd.DataFrame) -> pd.DataFrame:
    """将多标的 bars（order_book_id/date MultiIndex）转成 close 宽表（date x order_book_id）。"""
    if bars is None or bars.empty:
        return pd.DataFrame()
    df = bars
    if isinstance(df.index, pd.MultiIndex) and "order_book_id" in df.index.names and "date" in df.index.names:
        if "close" not in df.columns:
            raise ValueError("bars missing close column")
        wide = df["close"].unstack("order_book_id")
        wide.index = pd.to_datetime(wide.index)
        return wide.sort_index()
    # DataFacade.get_daily_bars(list[ids]) 的常见返回：index=date，列包含 order_book_id/close
    if "order_book_id" in df.columns and "close" in df.columns:
        out = df.copy()
        if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            out.index = pd.to_datetime(out.index)
        wide = out.pivot_table(index=out.index, columns="order_book_id", values="close", aggfunc="last")
        wide.index.name = "date"
        return wide.sort_index()
    # 已经是宽表（date x industry）
    if not isinstance(df.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def compute_market_regime_from_data_facade(
    *,
    end_date: str | pd.Timestamp,
    benchmark_id: str = "000300.XSHG",
    industry_index_ids: list[str] | None = None,
    cfg: MarketRegimeConfig | None = None,
    lookback_calendar_days: int = 1200,
    fields: list[str] | None = None,
    adjust_type: str = "none",
    ad_ratio: float | None = None,
) -> MarketRegime:
    """便捷入口：用 DataFacade 拉数并计算 MarketRegime。

    说明：
    - 默认只用 benchmark 的 OHLCV + 可选 industry indices close。
    - 行业指数列表建议由上游配置注入；缺省不取行业数据，会触发结构侧降级。
    """
    # 局部导入，避免在无 rqdatac 环境下导入失败。
    from skyeye.data.facade import DataFacade

    # 可选：自动发现一组“申万一级风格”的行业指数（约 30+）。
    # 若发现失败，则保持为空，触发结构侧降级。
    if industry_index_ids is None:
        try:
            industry_index_ids = discover_shenwan_l1_like_indices()
        except Exception:
            industry_index_ids = None

    cfg = cfg or MarketRegimeConfig()
    end_ts = pd.to_datetime(end_date)
    start_ts = end_ts - pd.Timedelta(days=int(lookback_calendar_days))
    start_date = start_ts.strftime("%Y-%m-%d")
    end_date_str = end_ts.strftime("%Y-%m-%d")

    data = DataFacade()
    bench_fields = fields or ["open", "high", "low", "close", "volume"]
    bench_raw = data.get_daily_bars(
        benchmark_id,
        start_date,
        end_date_str,
        fields=bench_fields,
        adjust_type=adjust_type,
    )
    if bench_raw is None or getattr(bench_raw, "empty", True):
        return MarketRegime(regime="range_co_move", strength=0.0, diagnostics={"reason": "benchmark_bars_empty"})
    bench = normalize_single_instrument_bars(bench_raw, benchmark_id)

    industry_close = None
    if industry_index_ids:
        ind_raw = data.get_daily_bars(
            industry_index_ids,
            start_date,
            end_date_str,
            fields=["close"],
            adjust_type="none",
        )
        if ind_raw is not None and not ind_raw.empty:
            industry_close = pivot_close_wide(ind_raw)

    return compute_market_regime(bench, industry_close=industry_close, cfg=cfg, ad_ratio=ad_ratio)


def discover_shenwan_l1_like_indices() -> list[str]:
    """从 rqdatac INDX 库里发现一组“申万一级风格”的行业指数代码。

    经验规则（不绑定具体版本）：
    - 代码形态：`801??0.INDX`（一级/大类常见编码）
    - 名称前缀：`申银万国指数-`
    - 排除合成类：后缀包含“申万...”的宽基/风格集合（避免把综合指数揉进去）

    返回：排序后的 order_book_id 列表。
    """
    import re

    # 走 DataFacade -> Provider 的初始化逻辑，避免重复 rqdatac.init() 的噪音告警。
    from skyeye.data.facade import DataFacade

    df = DataFacade().all_instruments(type="INDX")
    if df is None or df.empty:
        return []
    pat = re.compile(r"^801\d{2}0\.INDX$")
    sub = df[df["order_book_id"].astype(str).str.match(pat)].copy()
    prefix = "申银万国指数-"
    sub = sub[sub["symbol"].astype(str).str.startswith(prefix)]
    tail = sub["symbol"].astype(str).str[len(prefix) :]
    sub = sub[~tail.str.contains("申万", na=False)]
    ids = sorted(sub["order_book_id"].astype(str).unique().tolist())
    return ids
