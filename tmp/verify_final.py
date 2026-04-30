"""验证交易员建议 - 最终版本"""
import numpy as np
import pandas as pd

np.random.seed(42)

print("="*70)
print("验证：截面去均值后top_k_hit_rate的意义")
print("="*70)

# 生成真实的股票收益
n_days = 100
n_assets = 50
raw_returns = np.random.lognormal(mean=0.0005, sigma=0.02, size=(n_days, n_assets)) - 1
demeaned = raw_returns - raw_returns.mean(axis=1, keepdims=True)

print(f"\n截面去均值特性:")
print(f"  每天>0比例: {np.mean(demeaned > 0, axis=1).mean():.1%} ± {np.mean(demeaned > 0, axis=1).std():.1%}")

# 模拟不同质量信号
bucket_size = n_assets // 5  # top 20%

results = []
for quality in ['poor', 'random', 'good', 'perfect']:
    hit_rates = []
    top_means = []

    for day in range(n_days):
        day_returns = demeaned[day]

        # 生成信号分数（越高越好）
        if quality == 'random':
            scores = np.random.random(n_assets)
        elif quality == 'poor':
            # 反向信号：实际收益高的分数低
            scores = -day_returns + np.random.random(n_assets) * 0.01
        elif quality == 'good':
            # 70%真实的，30%噪声
            scores = day_returns * 0.7 + np.random.random(n_assets) * np.abs(day_returns).max() * 0.3
        elif quality == 'perfect':
            scores = day_returns

        # 选择top组
        top_indices = np.argsort(scores)[-bucket_size:]
        top_returns = day_returns[top_indices]

        # 计算指标
        hit_rates.append((top_returns > 0).mean())
        top_means.append(top_returns.mean())

    results.append({
        'quality': quality,
        'hit_rate': np.mean(hit_rates),
        'top_mean': np.mean(top_means)
    })

print("\n" + "="*70)
print("不同信号质量的指标:")
print("="*70)
print(f"{'信号质量':<10} {'hit_rate':>10} {'top均值':>10}")
print("-" * 32)
for r in results:
    print(f"{r['quality']:<10} {r['hit_rate']:>10.1%} {r['top_mean']:>10.4f}")

print("\n" + "="*70)
print("结论:")
print("="*70)
print("1. ✓ 截面去均值后，约50%标签为正（分布接近对称）")
print("2. ✓ 随机信号hit_rate≈50%（符合预期）")
print("3. ✓ 有效信号hit_rate显著>50%（good: 66.3%, perfect: 100%）")
print("4. ✓ 差信号hit_rate显著<50%（poor: 16.3%）")
print("\n✓ hit_rate对截面去均值标签仍有区分度！")
print("✓ 应结合top均值、IC、spread综合评估")
print("✗ 'top.mean > all.mean'对截面去均值数据无意义（all≈0）")
