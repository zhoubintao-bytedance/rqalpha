"""验证交易员关于top_k_hit_rate的建议。

核心问题：截面去均值后的标签，top_k_hit_rate是否确实结构性无意义？
"""
import numpy as np
import pandas as pd

# 模拟截面去均值后的标签
np.random.seed(42)
n_days = 100
n_assets = 50

# 生成原始收益（模拟真实股票收益分布，有偏度）
raw_returns = np.random.normal(0.001, 0.02, (n_days, n_assets))  # 有正偏

# 截面去均值
demeaned_returns = raw_returns - raw_returns.mean(axis=1, keepdims=True)

# 验证截面均值为0
daily_means = demeaned_returns.mean(axis=1)
print(f"✓ 截面均值: {daily_means.mean():.2e} (理论应为0)")

# 统计每天有多少比例为正
positive_ratio_per_day = (demeaned_returns > 0).mean(axis=1)
print(f"\n✓ 每天标签>0的比例:")
print(f"  - 均值: {positive_ratio_per_day.mean():.1%}")
print(f"  - 标准差: {positive_ratio_per_day.std():.1%}")
print(f"  - 范围: [{positive_ratio_per_day.min():.1%}, {positive_ratio_per_day.max():.1%}]")

# 模拟三种信号质量
def simulate_hit_rate(returns, signal_quality='random'):
    """模拟不同信号质量下的hit_rate"""
    hit_rates = []

    for day in range(returns.shape[0]):
        day_returns = returns[day]

        if signal_quality == 'random':
            # 随机信号：随机选择top组
            top_indices = np.random.choice(n_assets, size=n_assets//5, replace=False)
        elif signal_quality == 'good':
            # 好信号：优先选择实际收益高的股票（70%准确率）
            sorted_indices = np.argsort(day_returns)[::-1]  # 从高到低
            # 给实际高收益股票更高的分数
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(n_assets)
            noise = np.random.random(n_assets)
            mixed_score = 0.7 * ranks + 0.3 * noise * n_assets
            top_indices = np.argsort(mixed_score)[-n_assets//5:]
        elif signal_quality == 'perfect':
            # 完美信号：选择实际收益最高的top组
            top_indices = np.argsort(day_returns)[-n_assets//5:]

        top_returns = day_returns[top_indices]
        hit_rate = (top_returns > 0).mean()
        hit_rates.append(hit_rate)

    return np.mean(hit_rates)

print("\n" + "="*60)
print("不同信号质量下的hit_rate:")
print("="*60)

random_hit_rate = simulate_hit_rate(demeaned_returns, 'random')
good_hit_rate = simulate_hit_rate(demeaned_returns, 'good')
perfect_hit_rate = simulate_hit_rate(demeaned_returns, 'perfect')

print(f"1. 随机信号:   hit_rate = {random_hit_rate:.1%}")
print(f"2. 好信号:     hit_rate = {good_hit_rate:.1%}")
print(f"3. 完美信号:   hit_rate = {perfect_hit_rate:.1%}")

print("\n" + "="*60)
print("验证建议的替代方案:")
print("="*60)

# 验证建议的替代方案
def compare_mean_returns(returns, signal_quality='random'):
    """比较top组均值 vs 全截面均值"""
    top_means = []
    all_means = []

    for day in range(returns.shape[0]):
        day_returns = returns[day]

        if signal_quality == 'random':
            top_indices = np.random.choice(n_assets, size=n_assets//5, replace=False)
        elif signal_quality == 'good':
            sorted_indices = np.argsort(day_returns)[::-1]
            ranks = np.empty_like(sorted_indices)
            ranks[sorted_indices] = np.arange(n_assets)
            noise = np.random.random(n_assets)
            mixed_score = 0.7 * ranks + 0.3 * noise * n_assets
            top_indices = np.argsort(mixed_score)[-n_assets//5:]
        elif signal_quality == 'perfect':
            top_indices = np.argsort(day_returns)[-n_assets//5:]

        top_means.append(day_returns[top_indices].mean())
        all_means.append(day_returns.mean())

    return np.mean(top_means), np.mean(all_means)

# 对于截面去均值的数据，all_means应该≈0
random_top, random_all = compare_mean_returns(demeaned_returns, 'random')
good_top, good_all = compare_mean_returns(demeaned_returns, 'good')
perfect_top, perfect_all = compare_mean_returns(demeaned_returns, 'perfect')

print(f"\n随机信号:")
print(f"  top组均值: {random_top:.4f}")
print(f"  全截面均值: {random_all:.4f}")
print(f"  差值: {random_top - random_all:.4f}")

print(f"\n好信号:")
print(f"  top组均值: {good_top:.4f}")
print(f"  全截面均值: {good_all:.4f}")
print(f"  差值: {good_top - good_all:.4f}")

print(f"\n完美信号:")
print(f"  top组均值: {perfect_top:.4f}")
print(f"  全截面均值: {perfect_all:.4f}")
print(f"  差值: {perfect_top - perfect_all:.4f}")

print("\n" + "="*60)
print("结论:")
print("="*60)
print("1. 截面去均值后，确实约50%标签为正（接近50%，但不绝对）")
print("2. 对于随机信号，hit_rate确实≈50%")
print("3. 对于有效信号，hit_rate仍可能显著>50%")
print("4. 建议的替代方案 'top.mean > all.mean' 对于截面去均值数据无意义（all≈0）")
print("5. 更好的方案：直接看 top.mean 的绝对值或统计显著性")
