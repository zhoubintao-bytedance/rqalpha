"""验证交易员建议 - 正确理解截面去均值"""
import numpy as np
import pandas as pd

np.random.seed(42)

print("="*70)
print("核心问题：截面去均值后，top_k_hit_rate是否还有意义？")
print("="*70)

# 生成更真实的股票收益分布（有正偏）
n_days = 100
n_assets = 50

# 模拟原始收益（正偏分布）
raw_returns = np.random.lognormal(mean=0.0005, sigma=0.02, size=(n_days, n_assets))
raw_returns = raw_returns - 1  # 转换为收益率

# 截面去均值
demeaned = raw_returns - raw_returns.mean(axis=1, keepdims=True)

print(f"\n截面去均值统计:")
print(f"  - 截面均值: {demeaned.mean(axis=1).mean():.2e} (理论=0)")
print(f"  - 每天>0的比例: {np.mean(demeaned > 0, axis=1).mean():.1%}")
print(f"  - 每天>0的比例范围: [{np.mean(demeaned > 0, axis=1).min():.1%}, {np.mean(demeaned > 0, axis=1).max():.1%}]")

# 验证hit_rate在不同信号质量下的表现
bucket_size = n_assets // 5

def compute_metrics(returns, signal_quality):
    """计算hit_rate和top均值"""
    hit_rates = []
    top_means = []

    for day in range(len(returns)):
        day_returns = returns[day]

        # 根据信号质量生成排名
        if signal_quality == 'random':
            signal_rank = np.random.permutation(n_assets)
        elif signal_quality == 'poor':  # 故意选错的
            # 优先选择实际收益低的
            signal_rank = np.argsort(day_returns)  # 从低到高
        elif signal_quality == 'good':
            # 70%准确率
            true_rank = np.argsort(day_returns)[::-1]
            ranks = np.empty_like(true_rank)
            ranks[true_rank] = np.arange(n_assets)
            noise = np.random.random(n_assets) * n_assets
            mixed = 0.7 * ranks + 0.3 * noise
            signal_rank = np.argsort(mixed)[::-1]  # 从高到低
        elif signal_quality == 'perfect':
            signal_rank = np.argsort(day_returns)[::-1]  # 从高到低

        # 选择top组
        top_idx = signal_rank[:bucket_size]
        top_returns = day_returns[top_idx]

        # 计算指标
        hit_rate = (top_returns > 0).mean()
        top_mean = top_returns.mean()

        hit_rates.append(hit_rate)
        top_means.append(top_mean)

    return np.mean(hit_rates), np.mean(top_means)

print("\n" + "="*70)
print("不同信号质量下的表现:")
print("="*70)

for quality in ['poor', 'random', 'good', 'perfect']:
    hr, tm = compute_metrics(demeaned, quality)
    print(f"\n{quality:8s}信号:")
    print(f"  hit_rate: {hr:.1%}")
    print(f"  top均值:  {tm:.4f}")

print("\n" + "="*70)
print("关键发现:")
print("="*70)
print("1. 截面去均值后，标签分布不是严格对称的（受收益分布形状影响）")
print("2. 随机信号的hit_rate ≈ 50% 附近")
print("3. 有效信号的hit_rate可以显著>50%")
print("4. 但对于'poor'信号，hit_rate可能<50%")
print("\n结论:")
print("✓ hit_rate对截面去均值的标签仍有区分度")
print("✓ 但需要结合其他指标（top均值、IC、spread）")
print("✗ 'top.mean > all.mean' 对截面去均值数据无意义（all≈0）")
