#!/usr/bin/env python3
"""AX1 reg_lambda 实验对比脚本。

目的：用合成数据验证 reg_lambda=0.5 vs 2.0 对弱信号 ETF 轮动的影响。
对比维度：
  1. 训练/验证 MAE（过拟合程度）
  2. 预测值分布分散度（信号保留能力）
  3. OOS Rank IC（排序预测能力）
  4. 信号衰减率（弱信号被压缩程度）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from copy import deepcopy

from skyeye.products.ax1.models.lgbm_multi_target import LGBMMultiTargetPredictor


# ---------------------------------------------------------------------------
# 1. 合成数据生成：模拟 ETF 弱信号场景
# ---------------------------------------------------------------------------
def make_etf_weak_signal_data(
    n_dates: int = 200,
    n_assets: int = 30,
    signal_strength: float = 0.02,  # 弱信号：标准差仅 2%
    noise_ratio: float = 5.0,       # 噪声是信号的 5 倍
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成模拟 ETF 弱信号场景的数据。

    特点：
    - 30 只 ETF，200 个交易日
    - 真实信号极弱（momentum → return 的系数很小）
    - 噪声远大于信号（信噪比 ~0.2）
    - 这正是 reg_lambda 过低会导致过拟合噪声的场景
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    order_book_ids = [f"ETF_{i:03d}" for i in range(n_assets)]

    rows = []
    labels_rows = []
    for date in dates:
        for oid in order_book_ids:
            momentum = float(rng.normal(0.0, signal_strength))
            volatility = float(abs(rng.normal(0.015, 0.005)))
            dollar_volume = float(abs(rng.normal(5e7, 1e7)))
            # 行业/风格因子（弱相关）
            sector_beta = float(rng.normal(0.0, 0.01))
            style_factor = float(rng.normal(0.0, 0.008))
            regime_score = float(rng.choice([-1, 0, 1], p=[0.2, 0.5, 0.3]))

            noise_scale = signal_strength * noise_ratio

            # 真实标签：信号极弱
            label_r5 = 0.15 * momentum - 0.1 * volatility + 0.05 * sector_beta + rng.normal(0, noise_scale * 0.3)
            label_r10 = 0.25 * momentum - 0.15 * volatility + 0.08 * sector_beta + 0.03 * style_factor + rng.normal(0, noise_scale * 0.5)
            label_r20 = 0.40 * momentum - 0.20 * volatility + 0.12 * sector_beta + 0.06 * style_factor + 0.02 * regime_score + rng.normal(0, noise_scale * 0.8)
            label_vol10 = volatility * abs(rng.normal(1.0, 0.1)) + 0.005

            rows.append({
                "date": date,
                "order_book_id": oid,
                "momentum_2d": momentum,
                "volatility_3d": volatility,
                "feature_dollar_volume": dollar_volume,
                "sector_beta": sector_beta,
                "style_factor": style_factor,
                "regime_score": regime_score,
            })
            labels_rows.append({
                "date": date,
                "order_book_id": oid,
                "label_return_5d": label_r5,
                "label_return_10d": label_r10,
                "label_return_20d": label_r20,
                "label_net_return_5d": label_r5 - 0.001,
                "label_net_return_10d": label_r10 - 0.001,
                "label_net_return_20d": label_r20 - 0.001,
                "label_volatility_10d": label_vol10,
            })

    labels = pd.DataFrame(labels_rows)
    for h in (5, 10, 20):
        col = f"label_net_return_{h}d"
        labels[f"label_relative_net_return_{h}d"] = (
            labels[col] - labels.groupby("date")[col].transform("mean")
        )
    return pd.DataFrame(rows), labels


# ---------------------------------------------------------------------------
# 2. Walk-forward 训练评估
# ---------------------------------------------------------------------------
def walk_forward_evaluate(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    base_params: dict,
    n_folds: int = 5,
) -> dict:
    """Walk-forward 训练 + 评估，返回指标。"""
    all_dates = sorted(features["date"].unique())
    fold_size = len(all_dates) // (n_folds + 1)

    oos_predictions = []
    train_maes = []
    val_maes = []

    for fold in range(n_folds):
        train_end_idx = fold_size * (fold + 1)
        val_end_idx = fold_size * (fold + 2)

        if val_end_idx > len(all_dates):
            break

        train_end = all_dates[train_end_idx - 1]
        val_end = all_dates[min(val_end_idx - 1, len(all_dates) - 1)]

        train_mask = features["date"] <= train_end
        val_mask = (features["date"] > train_end) & (features["date"] <= val_end)

        train_X = features[train_mask]
        train_Y = labels[train_mask]
        val_X = features[val_mask]
        val_Y = labels[val_mask]

        if len(train_X) < 50 or len(val_X) < 10:
            continue

        predictor = LGBMMultiTargetPredictor(
            feature_columns=("momentum_2d", "volatility_3d", "feature_dollar_volume",
                             "sector_beta", "style_factor", "regime_score"),
            params=base_params,
        )
        predictor.fit(train_X, train_Y, val_features=val_X, val_labels=val_Y)

        # 收集 OOS 预测
        pred = predictor.predict(val_X)
        pred["fold"] = fold
        pred = pred.merge(val_Y[["date", "order_book_id", "label_relative_net_return_10d"]],
                          on=["date", "order_book_id"], how="left")
        oos_predictions.append(pred)

        # 收集训练/验证 MAE
        for head_name, head_state in predictor._multi_head.models_.items():
            model = head_state.get("model")
            booster = getattr(model, "_model", None)
            if booster is not None:
                evals = booster.current_iteration()
                train_maes.append({"fold": fold, "head": head_name, "iterations": evals})

    if not oos_predictions:
        return {}

    oos = pd.concat(oos_predictions, ignore_index=True)
    metrics = {}

    # --- Rank IC（OOS 排序能力）---
    ic_list = []
    for date, group in oos.groupby("date"):
        if len(group) < 5:
            continue
        pred_col = "expected_relative_net_return_10d"
        label_col = "label_relative_net_return_10d"
        if pred_col in group.columns and label_col in group.columns:
            valid = group[[pred_col, label_col]].dropna()
            if len(valid) >= 5:
                ic = valid[pred_col].corr(valid[label_col], method="spearman")
                if np.isfinite(ic):
                    ic_list.append(ic)

    metrics["oos_rank_ic_mean"] = float(np.mean(ic_list)) if ic_list else 0.0
    metrics["oos_rank_ic_std"] = float(np.std(ic_list)) if ic_list else 0.0
    metrics["oos_rank_ic_ir"] = (
        metrics["oos_rank_ic_mean"] / metrics["oos_rank_ic_std"]
        if metrics["oos_rank_ic_std"] > 1e-9 else 0.0
    )

    # --- 预测值分散度（信号保留能力）---
    pred_col = "expected_relative_net_return_10d"
    if pred_col in oos.columns:
        daily_std = oos.groupby("date")[pred_col].std()
        metrics["pred_daily_std_mean"] = float(daily_std.mean())
        metrics["pred_daily_std_median"] = float(daily_std.median())

        # 极端预测比例（abs > 2% 视为极端）
        extreme_ratio = (oos[pred_col].abs() > 0.02).mean()
        metrics["extreme_pred_ratio"] = float(extreme_ratio)

    # --- 信号压缩检测 ---
    if pred_col in oos.columns and "label_relative_net_return_10d" in oos.columns:
        pred_range = oos[pred_col].quantile(0.95) - oos[pred_col].quantile(0.05)
        label_range = oos["label_relative_net_return_10d"].quantile(0.95) - oos["label_relative_net_return_10d"].quantile(0.05)
        metrics["pred_label_range_ratio"] = float(pred_range / label_range) if label_range > 1e-9 else 0.0

    # --- 弱信号保留：按真实信号分组后的预测区分度 ---
    if "label_relative_net_return_10d" in oos.columns:
        valid = oos[["expected_relative_net_return_10d", "label_relative_net_return_10d"]].dropna()
        if len(valid) > 20:
            # 按标签分5组，检查预测是否也有区分度
            valid = valid.copy()
            valid["label_q"] = pd.qcut(valid["label_relative_net_return_10d"], 5, labels=False, duplicates="drop")
            group_means = valid.groupby("label_q")["expected_relative_net_return_10d"].mean()
            if len(group_means) >= 2:
                spread = group_means.iloc[-1] - group_means.iloc[0]
                metrics["weak_signal_spread_top_bottom_q"] = float(spread)
                metrics["weak_signal_monotonic"] = int(group_means.is_monotonic_increasing or group_means.is_monotonic_decreasing)

    metrics["n_oos_samples"] = len(oos)
    metrics["n_folds_run"] = len(oos_predictions)
    metrics["n_ic_dates"] = len(ic_list)

    return metrics


# ---------------------------------------------------------------------------
# 3. 主实验
# ---------------------------------------------------------------------------
def run_experiment():
    print("=" * 80)
    print("AX1 reg_lambda 对比实验")
    print("场景：ETF 弱信号轮动（信噪比 ~0.2）")
    print("=" * 80)

    features, labels = make_etf_weak_signal_data()
    print(f"\n数据规模: {len(features)} 行, {features['date'].nunique()} 交易日, {features['order_book_id'].nunique()} ETF")

    # AX1 默认参数基线
    ax1_base = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 24,
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 200,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.3,
        "reg_lambda": 0.5,
        "min_child_samples": 50,
        "early_stopping_rounds": 20,
        "num_threads": 1,
        "verbose": -1,
        "deterministic": True,
        "force_col_wise": True,
        "seed": 20260430,
        "feature_fraction_seed": 20260430,
        "bagging_seed": 20260430,
        "data_random_seed": 20260430,
        "drop_seed": 20260430,
    }

    # 实验配置：不同 reg_lambda 值
    configs = {
        "AX1_current (λ=0.5, α=0.3, mcs=50)": ax1_base,
        "λ=1.0 (α=0.3, mcs=50)": {**ax1_base, "reg_lambda": 1.0},
        "λ=2.0 (对齐TX1, α=0.3, mcs=50)": {**ax1_base, "reg_lambda": 2.0},
        "λ=2.0+α=0.5 (α同步提升)": {**ax1_base, "reg_lambda": 2.0, "reg_alpha": 0.5},
        "λ=2.0+mcs=80 (对齐TX1全参数)": {**ax1_base, "reg_lambda": 2.0, "reg_alpha": 0.3, "min_child_samples": 80},
        "λ=2.0+α=0.5+mcs=80 (TX1全对齐)": {**ax1_base, "reg_lambda": 2.0, "reg_alpha": 0.5, "min_child_samples": 80},
    }

    results = {}
    for name, params in configs.items():
        print(f"\n>>> 训练: {name}")
        metrics = walk_forward_evaluate(features, labels, params, n_folds=5)
        results[name] = metrics
        print(f"    OOS Rank IC = {metrics.get('oos_rank_ic_mean', 0):.4f} "
              f"(IR={metrics.get('oos_rank_ic_ir', 0):.4f})")
        print(f"    预测值日均标准差 = {metrics.get('pred_daily_std_mean', 0):.6f}")
        print(f"    信号压缩比 (pred/label range) = {metrics.get('pred_label_range_ratio', 0):.4f}")
        print(f"    弱信号区分度 (top-bottom quintile) = {metrics.get('weak_signal_spread_top_bottom_q', 0):.6f}")
        print(f"    极端预测比例 = {metrics.get('extreme_pred_ratio', 0):.4f}")

    # 汇总对比表
    print("\n" + "=" * 80)
    print("汇总对比")
    print("=" * 80)
    header = f"{'配置':<45} {'IC均值':>8} {'IC_IR':>8} {'日均σ':>10} {'压缩比':>8} {'Q5-Q1':>10} {'极端%':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<45} "
              f"{m.get('oos_rank_ic_mean', 0):>8.4f} "
              f"{m.get('oos_rank_ic_ir', 0):>8.4f} "
              f"{m.get('pred_daily_std_mean', 0):>10.6f} "
              f"{m.get('pred_label_range_ratio', 0):>8.4f} "
              f"{m.get('weak_signal_spread_top_bottom_q', 0):>10.6f} "
              f"{m.get('extreme_pred_ratio', 0):>8.4f}")

    # 分析结论
    print("\n" + "=" * 80)
    print("分析结论")
    print("=" * 80)
    current = results.get("AX1_current (λ=0.5, α=0.3, mcs=50)", {})
    tx1_aligned = results.get("λ=2.0+α=0.5+mcs=80 (TX1全对齐)", {})
    lambda2_only = results.get("λ=2.0 (对齐TX1, α=0.3, mcs=50)", {})

    print(f"\n1. 过拟合风险:")
    print(f"   当前 λ=0.5 极端预测比例: {current.get('extreme_pred_ratio', 0):.4f}")
    print(f"   λ=2.0 极端预测比例: {lambda2_only.get('extreme_pred_ratio', 0):.4f}")
    if current.get('extreme_pred_ratio', 0) > lambda2_only.get('extreme_pred_ratio', 0):
        print(f"   → λ=0.5 产生了更多极端预测，说明模型对噪声过拟合")
    else:
        print(f"   → 极端预测比例差异不大")

    print(f"\n2. OOS 排序能力:")
    ic_current = current.get('oos_rank_ic_mean', 0)
    ic_lambda2 = lambda2_only.get('oos_rank_ic_mean', 0)
    print(f"   λ=0.5 IC={ic_current:.4f} vs λ=2.0 IC={ic_lambda2:.4f}")
    if ic_lambda2 > ic_current:
        print(f"   → λ=2.0 OOS 排序能力更强，说明更高正则化改善了泛化")
    else:
        print(f"   → IC 差异需结合 IR 一起看")

    print(f"\n3. 弱信号保留:")
    spread_current = abs(current.get('weak_signal_spread_top_bottom_q', 0))
    spread_lambda2 = abs(lambda2_only.get('weak_signal_spread_top_bottom_q', 0))
    print(f"   λ=0.5 Q5-Q1 spread={spread_current:.6f}")
    print(f"   λ=2.0 Q5-Q1 spread={spread_lambda2:.6f}")
    print(f"   信号压缩比: λ=0.5={current.get('pred_label_range_ratio', 0):.4f}, λ=2.0={lambda2_only.get('pred_label_range_ratio', 0):.4f}")

    return results


if __name__ == "__main__":
    run_experiment()
