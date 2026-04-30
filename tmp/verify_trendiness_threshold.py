# -*- coding: utf-8 -*-
"""临时脚本：验证 AX1 trendiness_range_threshold=0.35 是否合理

任务背景：
  issue 指出 trendiness_range_threshold=0.35 过低，导致"震荡市"几乎从不被触发。
  A 股实际 60-70% 时间处于震荡，但系统约 50% 概率判定趋势。

本脚本：
  1. 用真实 A 股数据（沪深300）滚动计算 trendiness 的历史分布
  2. 统计不同阈值下"震荡市"占比，找到与 A 股实际震荡比例(60-70%)匹配的阈值
  3. 输出 trendiness 的分位数表和方向分类占比对比

运行后可删除。
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from skyeye.market_regime_layer import (
    MarketRegimeConfig,
    compute_market_regime,
    required_market_regime_history_days,
    _compute_direction_label_and_scores,
)


def main():
    # ---- 1. 拉取沪深300真实数据 ----
    from skyeye.data.facade import DataFacade

    data = DataFacade()
    bars = data.get_daily_bars(
        "000300.XSHG",
        "2016-01-01",
        "2026-04-25",
        fields=["open", "high", "low", "close", "volume"],
    )
    if bars is None or bars.empty:
        print("ERROR: 无法获取沪深300数据，跳过")
        return

    # 规范化
    from skyeye.market_regime_layer import normalize_single_instrument_bars

    bench = normalize_single_instrument_bars(bars, "000300.XSHG")
    print(f"沪深300 数据行数: {len(bench)}, 日期范围: {bench.index[0]} ~ {bench.index[-1]}")

    cfg = MarketRegimeConfig()
    required_days = required_market_regime_history_days(cfg)
    print(f"所需最少历史天数: {required_days}")

    # ---- 2. 滚动计算每日 trendiness ----
    lookback = 650
    trendiness_list = []
    direction_list = []
    date_list = []

    for i in range(required_days + lookback, len(bench) + 1):
        sub = bench.iloc[i - lookback : i]
        label, trendiness, direction, diag = _compute_direction_label_and_scores(sub, cfg)
        if trendiness is not None:
            trendiness_list.append(trendiness)
            direction_list.append(direction if direction is not None else 0.0)
            date_list.append(sub.index[-1])

    trendiness_s = pd.Series(trendiness_list, index=date_list, name="trendiness")
    direction_s = pd.Series(direction_list, index=date_list, name="direction")
    print(f"\n滚动计算完成，有效天数: {len(trendiness_s)}")

    # ---- 3. trendiness 分布统计 ----
    print("\n========== trendiness 分布统计 ==========")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    for p in percentiles:
        print(f"  P{p:02d} = {trendiness_s.quantile(p / 100):.4f}")
    print(f"  均值 = {trendiness_s.mean():.4f}")
    print(f"  标准差 = {trendiness_s.std():.4f}")

    # ---- 4. 不同阈值下"震荡"占比 ----
    # A 股实际经验：60-70% 时间震荡
    print("\n========== 不同 trendiness_range_threshold 下的方向占比 ==========")
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    print(f"{'阈值':>6s} | {'震荡%':>6s} | {'趋势%':>6s} | {'其中牛%':>6s} | {'其中熊%':>6s}")
    print("-" * 50)

    for th in thresholds:
        is_range = trendiness_s < th
        is_trend = ~is_range
        range_pct = is_range.mean() * 100
        trend_pct = is_trend.mean() * 100

        # 在趋势日中，看 direction 是 bull 还是 bear
        trend_dir = direction_s[is_trend]
        bull_pct = (trend_dir > cfg.direction_bull_threshold).mean() * 100 if len(trend_dir) > 0 else 0
        bear_pct = (trend_dir < cfg.direction_bear_threshold).mean() * 100 if len(trend_dir) > 0 else 0
        print(f"  {th:.2f} | {range_pct:6.1f} | {trend_pct:6.1f} | {bull_pct:6.1f} | {bear_pct:6.1f}")

    # ---- 5. 与 A 股 ADX 经验值交叉验证 ----
    print("\n========== ADX 子指标分布 ==========")
    adx_list = []
    for i in range(required_days + lookback, len(bench) + 1):
        sub = bench.iloc[i - lookback : i]
        label, trendiness, direction, diag = _compute_direction_label_and_scores(sub, cfg)
        adx_val = diag.get("adx")
        if adx_val is not None:
            adx_list.append(adx_val)
    adx_s = pd.Series(adx_list)
    print(f"  ADX 样本数: {len(adx_s)}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  ADX P{p:02d} = {adx_s.quantile(p / 100):.2f}")
    print(f"  ADX 均值 = {adx_s.mean():.2f}")

    # adx_score 的分布（映射后）
    adx_score_s = ((adx_s - cfg.adx_low) / (cfg.adx_high - cfg.adx_low)).clip(0, 1)
    print(f"\n  adx_score 均值 = {adx_score_s.mean():.4f}")
    for p in [10, 25, 50, 75, 90]:
        print(f"  adx_score P{p:02d} = {adx_score_s.quantile(p / 100):.4f}")

    # ---- 6. 分年份统计（检查稳定性）----
    print("\n========== 分年份震荡占比（threshold=0.35 vs 0.50 vs 0.55）==========")
    trendiness_df = pd.DataFrame({"trendiness": trendiness_s, "year": trendiness_s.index.year})
    for th in [0.35, 0.50, 0.55]:
        range_by_year = trendiness_df.groupby("year")["trendiness"].apply(lambda x: (x < th).mean() * 100)
        print(f"\n  threshold={th:.2f}:")
        for yr, pct in range_by_year.items():
            print(f"    {yr}: 震荡占比 {pct:.1f}%")
        print(f"    整体: 震荡占比 {(trendiness_s < th).mean() * 100:.1f}%")

    # ---- 7. 结论 ----
    print("\n========== 结论 ==========")
    print("A 股实际震荡市占比约 60-70%。")
    print('找到使"震荡%最接近 60-70% 区间中值(65%)的阈值:')
    target = 0.65
    best_th = None
    best_diff = 999
    for th in thresholds:
        diff = abs((trendiness_s < th).mean() - target)
        if diff < best_diff:
            best_diff = diff
            best_th = th
    range_pct_at_best = (trendiness_s < best_th).mean() * 100
    print(f"  最优阈值 ≈ {best_th:.2f}，对应震荡占比 ≈ {range_pct_at_best:.1f}%")


if __name__ == "__main__":
    main()
