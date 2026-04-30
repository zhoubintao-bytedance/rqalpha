#!/usr/bin/env python3
"""AX1 reg_lambda 参数网格实验 v3 —— 优化网格规模。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

def make_etf_data(n_dates=400, n_assets=30, snr=0.15, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    order_book_ids = [f"ETF_{i:03d}" for i in range(n_assets)]
    etf_sector = {oid: rng.integers(0, 5) for oid in order_book_ids}
    rows, labels_rows = [], []
    for date in dates:
        sector_noise = {s: float(rng.normal(0, 0.01)) for s in range(5)}
        for oid in order_book_ids:
            momentum = float(rng.normal(0.0, 0.025))
            volatility = float(abs(rng.normal(0.018, 0.006)))
            dollar_volume = float(abs(rng.normal(5e7, 1.5e7)))
            sector = etf_sector[oid]
            sector_exposure = float(rng.normal(0, 0.005))
            style_exposure = float(rng.normal(0, 0.003))
            signal_10d = 0.3 * momentum + 0.4 * sector_exposure + 0.2 * style_exposure
            collinear_noise = sector_noise[sector]
            pure_noise = rng.normal(0, abs(signal_10d) / snr) if abs(signal_10d) > 1e-9 else rng.normal(0, 0.01)
            label_r10 = signal_10d + collinear_noise + pure_noise
            label_r5 = 0.5 * signal_10d + 0.5 * collinear_noise + rng.normal(0, abs(signal_10d * 0.5) / snr + 0.003)
            label_r20 = 1.5 * signal_10d + 0.8 * collinear_noise + rng.normal(0, abs(signal_10d * 1.5) / snr + 0.005)
            label_vol10 = volatility * abs(rng.normal(1.0, 0.15)) + 0.008
            rows.append({"date": date, "order_book_id": oid, "momentum_2d": momentum,
                         "volatility_3d": volatility, "feature_dollar_volume": dollar_volume,
                         "sector_exposure": sector_exposure, "style_exposure": style_exposure})
            labels_rows.append({"date": date, "order_book_id": oid,
                                "label_return_5d": label_r5, "label_return_10d": label_r10,
                                "label_return_20d": label_r20,
                                "label_net_return_5d": label_r5 - 0.001,
                                "label_net_return_10d": label_r10 - 0.001,
                                "label_net_return_20d": label_r20 - 0.001,
                                "label_volatility_10d": label_vol10})
    labels = pd.DataFrame(labels_rows)
    for h in (5, 10, 20):
        col = f"label_net_return_{h}d"
        labels[f"label_relative_net_return_{h}d"] = (
            labels[col] - labels.groupby("date")[col].transform("mean"))
    return pd.DataFrame(rows), labels


def evaluate_config(features, labels, params, n_folds=5):
    feat_cols = ["momentum_2d", "volatility_3d", "feature_dollar_volume",
                 "sector_exposure", "style_exposure"]
    all_dates = sorted(features["date"].unique())
    fold_size = len(all_dates) // (n_folds + 1)
    oos_ics, train_ics = [], []
    for fold in range(n_folds):
        train_end_idx = fold_size * (fold + 1)
        val_end_idx = fold_size * (fold + 2)
        if val_end_idx > len(all_dates):
            break
        train_end = all_dates[train_end_idx - 1]
        val_end = all_dates[min(val_end_idx - 1, len(all_dates) - 1)]
        train_mask = features["date"] <= train_end
        val_mask = (features["date"] > train_end) & (features["date"] <= val_end)
        train_X = features.loc[train_mask, feat_cols].fillna(0)
        train_Y = labels.loc[train_mask]
        val_X = features.loc[val_mask, feat_cols].fillna(0)
        val_Y = labels.loc[val_mask]
        if len(train_X) < 50 or len(val_X) < 10:
            continue
        target_col = "label_relative_net_return_10d"
        train_data = lgb.Dataset(train_X, label=train_Y[target_col].values)
        val_data = lgb.Dataset(val_X, label=val_Y[target_col].values, reference=train_data)
        callbacks = [lgb.early_stopping(params.get("early_stopping_rounds", 20), verbose=False)]
        model = lgb.train(params, train_data, num_boost_round=params.get("n_estimators", 200),
                          valid_sets=[val_data], callbacks=callbacks)
        train_pred = model.predict(train_X)
        train_ic = pd.Series(train_pred).corr(pd.Series(train_Y[target_col].values), method="spearman")
        train_ics.append(train_ic)
        val_pred = model.predict(val_X)
        val_df = val_Y[["date", "order_book_id", target_col]].copy()
        val_df["pred"] = val_pred
        daily_ics = []
        for date, group in val_df.groupby("date"):
            if len(group) < 5:
                continue
            ic = group["pred"].corr(group[target_col], method="spearman")
            if np.isfinite(ic):
                daily_ics.append(ic)
        oos_ics.append(float(np.mean(daily_ics)) if daily_ics else 0.0)
    if not oos_ics:
        return {}
    train_ic_avg = float(np.mean(train_ics))
    oos_ic_avg = float(np.mean(oos_ics))
    return {"train_ic": train_ic_avg, "oos_ic": oos_ic_avg,
            "ic_gap": train_ic_avg - oos_ic_avg,
            "ic_gap_ratio": (train_ic_avg - oos_ic_avg) / train_ic_avg if abs(train_ic_avg) > 1e-9 else 0.0,
            "oos_ic_ir": np.mean(oos_ics) / np.std(oos_ics) if np.std(oos_ics) > 1e-9 else 0.0}


def main():
    # 网格：5 λ × 5 α × 4 mcs = 100 组
    reg_lambdas = [0.3, 0.5, 1.0, 2.0, 3.0]
    reg_alphas = [0.2, 0.3, 0.5, 0.8, 1.0]
    mcs_list = [30, 50, 65, 80]

    n_seeds = 8
    snr_list = [(0.15, "中SNR"), (0.25, "高SNR")]

    base_params = {
        "objective": "regression", "metric": "mae", "boosting_type": "gbdt",
        "num_leaves": 24, "max_depth": 5, "learning_rate": 0.03, "n_estimators": 200,
        "subsample": 0.75, "subsample_freq": 1, "colsample_bytree": 0.75,
        "early_stopping_rounds": 20, "num_threads": 1, "verbose": -1,
        "deterministic": True, "force_col_wise": True,
    }

    all_results = []
    total = len(reg_lambdas) * len(reg_alphas) * len(mcs_list) * n_seeds * len(snr_list)
    done = 0

    for snr_val, snr_label in snr_list:
        for seed in range(n_seeds):
            features, labels = make_etf_data(n_dates=400, n_assets=30, snr=snr_val, seed=seed * 100 + 7)
            for lam in reg_lambdas:
                for alp in reg_alphas:
                    for mcs in mcs_list:
                        done += 1
                        if done % 200 == 0:
                            print(f"  进度: {done}/{total} ({done/total:.0%})")
                        params = {**base_params, "reg_lambda": lam, "reg_alpha": alp,
                                  "min_child_samples": mcs, "seed": seed * 100 + 7,
                                  "feature_fraction_seed": seed * 100 + 7,
                                  "bagging_seed": seed * 100 + 7,
                                  "data_random_seed": seed * 100 + 7,
                                  "drop_seed": seed * 100 + 7}
                        r = evaluate_config(features, labels, params, n_folds=5)
                        if r:
                            all_results.append({"snr": snr_label, "seed": seed,
                                                "reg_lambda": lam, "reg_alpha": alp,
                                                "min_child_samples": mcs, **r})

    # ============ 中SNR 详细分析 ============
    df_mid = pd.DataFrame([r for r in all_results if r["snr"] == "中SNR"])
    summary_mid = df_mid.groupby(["reg_lambda", "reg_alpha", "min_child_samples"]).agg(
        oos_ic=("oos_ic", "mean"), ic_gap=("ic_gap", "mean"),
        ic_gap_ratio=("ic_gap_ratio", "mean"), oos_ir=("oos_ir", "mean"),
        train_ic=("train_ic", "mean")).reset_index()

    print(f"\n{'='*100}")
    print("中SNR 场景 —— 全网格结果")
    print(f"{'='*100}")

    # Top 20 by OOS IR
    top_ir = summary_mid.nlargest(20, "oos_ir")
    print(f"\nTop 20 by OOS IR:")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'TrainIC':>9} {'OOS_IC':>9} {'IC_gap':>9} {'Gap%':>8} {'OOS_IR':>8}")
    print("-" * 65)
    for _, row in top_ir.iterrows():
        print(f"{row['reg_lambda']:>5.1f} {row['reg_alpha']:>5.1f} {row['min_child_samples']:>5.0f} "
              f"{row['train_ic']:>9.4f} {row['oos_ic']:>9.4f} {row['ic_gap']:>9.4f} "
              f"{row['ic_gap_ratio']:>8.1%} {row['oos_ir']:>8.4f}")

    # Top 20 by OOS IC
    top_ic = summary_mid.nlargest(20, "oos_ic")
    print(f"\nTop 20 by OOS IC:")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'TrainIC':>9} {'OOS_IC':>9} {'IC_gap':>9} {'Gap%':>8} {'OOS_IR':>8}")
    print("-" * 65)
    for _, row in top_ic.iterrows():
        print(f"{row['reg_lambda']:>5.1f} {row['reg_alpha']:>5.1f} {row['min_child_samples']:>5.0f} "
              f"{row['train_ic']:>9.4f} {row['oos_ic']:>9.4f} {row['ic_gap']:>9.4f} "
              f"{row['ic_gap_ratio']:>8.1%} {row['oos_ir']:>8.4f}")

    # Pareto 综合
    summary_mid["oos_ic_rank"] = summary_mid["oos_ic"].rank(ascending=False)
    summary_mid["ir_rank"] = summary_mid["oos_ir"].rank(ascending=False)
    summary_mid["gap_rank"] = summary_mid["ic_gap"].rank(ascending=True)
    summary_mid["composite"] = (summary_mid["oos_ic_rank"] + summary_mid["ir_rank"] + summary_mid["gap_rank"]) / 3

    pareto = summary_mid.nsmallest(20, "composite")
    print(f"\nPareto Top 20 (IC_rank + IR_rank + Gap_rank 综合):")
    print(f"{'λ':>5} {'α':>5} {'mcs':>5} {'TrainIC':>9} {'OOS_IC':>9} {'IC_gap':>9} {'Gap%':>8} {'OOS_IR':>8} {'IC_rk':>6} {'IR_rk':>6} {'GapRk':>6} {'Cmp':>6}")
    print("-" * 100)
    for _, row in pareto.iterrows():
        print(f"{row['reg_lambda']:>5.1f} {row['reg_alpha']:>5.1f} {row['min_child_samples']:>5.0f} "
              f"{row['train_ic']:>9.4f} {row['oos_ic']:>9.4f} {row['ic_gap']:>9.4f} "
              f"{row['ic_gap_ratio']:>8.1%} {row['oos_ir']:>8.4f} "
              f"{row['oos_ic_rank']:>6.0f} {row['ir_rank']:>6.0f} {row['gap_rank']:>6.0f} {row['composite']:>6.1f}")

    # mcs=50 热力图
    for metric, label in [("oos_ir", "OOS IR"), ("ic_gap_ratio", "IC_gap%"), ("oos_ic", "OOS IC")]:
        pivot = summary_mid[summary_mid["min_child_samples"] == 50].pivot_table(
            index="reg_lambda", columns="reg_alpha", values=metric)
        print(f"\n{label} 热力图 (mcs=50):")
        print(f"λ\\α", end="")
        for col in pivot.columns:
            print(f"  {col:.1f}  ", end="")
        print()
        for idx, row in pivot.iterrows():
            print(f"{idx:>3.1f}", end="")
            for val in row:
                if pd.notna(val):
                    if metric == "ic_gap_ratio":
                        print(f"  {val:>4.0%} ", end="")
                    else:
                        print(f"  {val:>5.3f}", end="")
                else:
                    print(f"   N/A ", end="")
            print()

    # ============ 配对 t 检验 ============
    current = df_mid[(df_mid["reg_lambda"] == 0.5) & (df_mid["reg_alpha"] == 0.3) & (df_mid["min_child_samples"] == 50)]
    current_oos = current.set_index("seed")["oos_ic"]
    current_gap = current.set_index("seed")["ic_gap"]

    print(f"\n{'='*100}")
    print("配对 t 检验 (vs AX1 当前: λ=0.5, α=0.3, mcs=50)")
    print(f"{'='*100}")

    comparisons = []
    for lam in reg_lambdas:
        for alp in reg_alphas:
            for mcs in mcs_list:
                if lam == 0.5 and alp == 0.3 and mcs == 50:
                    continue
                comparisons.append((f"λ={lam},α={alp},mcs={mcs}", lam, alp, mcs))

    results_t = []
    for label, lam, alp, mcs in comparisons:
        cand = df_mid[(df_mid["reg_lambda"] == lam) & (df_mid["reg_alpha"] == alp) & (df_mid["min_child_samples"] == mcs)]
        if len(cand) < 3:
            continue
        cand_oos = cand.set_index("seed")["oos_ic"]
        cand_gap = cand.set_index("seed")["ic_gap"]
        common = current_oos.index.intersection(cand_oos.index)
        if len(common) < 3:
            continue
        t_ic, p_ic = stats.ttest_rel(cand_oos[common].values, current_oos[common].values)
        t_gap, p_gap = stats.ttest_rel(cand_gap[common].values, current_gap[common].values)
        delta_ic = cand_oos[common].mean() - current_oos[common].mean()
        delta_gap = cand_gap[common].mean() - current_gap[common].mean()
        results_t.append({"label": label, "lam": lam, "alp": alp, "mcs": mcs,
                          "delta_ic": delta_ic, "delta_gap": delta_gap,
                          "p_ic": p_ic, "p_gap": p_gap})

    df_t = pd.DataFrame(results_t)

    # 显著改善组 (p < 0.1 且 OOS IC 更高)
    improved = df_t[(df_t["delta_ic"] > 0) & (df_t["p_ic"] < 0.1)].sort_values("p_ic")
    print(f"\n显著改善 OOS IC 的配置 (p < 0.1, IC 更高):")
    print(f"{'配置':<30} {'ΔOOS_IC':>9} {'ΔIC_gap':>9} {'p(IC)':>8} {'p(gap)':>8}")
    print("-" * 70)
    for _, row in improved.head(20).iterrows():
        sig = "***" if row["p_ic"] < 0.01 else "**" if row["p_ic"] < 0.05 else "*"
        print(f"{row['label']:<30} {row['delta_ic']:>+9.4f} {row['delta_gap']:>+9.4f} {row['p_ic']:>8.4f}{sig:<3} {row['p_gap']:>8.4f}")

    # 显著减小 IC_gap 的组
    gap_improved = df_t[(df_t["delta_gap"] < 0) & (df_t["p_gap"] < 0.1)].sort_values("p_gap")
    print(f"\n显著减小 IC_gap 的配置 (p < 0.1, gap 更小):")
    print(f"{'配置':<30} {'ΔOOS_IC':>9} {'ΔIC_gap':>9} {'p(IC)':>8} {'p(gap)':>8}")
    print("-" * 70)
    for _, row in gap_improved.head(20).iterrows():
        sig = "***" if row["p_gap"] < 0.01 else "**" if row["p_gap"] < 0.05 else "*"
        print(f"{row['label']:<30} {row['delta_ic']:>+9.4f} {row['delta_gap']:>+9.4f} {row['p_ic']:>8.4f} {row['p_gap']:>8.4f}{sig:<3}")

    # ============ 高SNR 热力图 ============
    df_high = pd.DataFrame([r for r in all_results if r["snr"] == "高SNR"])
    summary_high = df_high.groupby(["reg_lambda", "reg_alpha", "min_child_samples"]).agg(
        oos_ic=("oos_ic", "mean"), ic_gap=("ic_gap", "mean"),
        ic_gap_ratio=("ic_gap_ratio", "mean"), oos_ir=("oos_ir", "mean")).reset_index()

    print(f"\n{'='*100}")
    print("高SNR 场景 —— OOS IR 热力图 (mcs=50)")
    print(f"{'='*100}")
    pivot = summary_high[summary_high["min_child_samples"] == 50].pivot_table(
        index="reg_lambda", columns="reg_alpha", values="oos_ir")
    print(f"λ\\α", end="")
    for col in pivot.columns:
        print(f"  {col:.1f}  ", end="")
    print()
    for idx, row in pivot.iterrows():
        print(f"{idx:>3.1f}", end="")
        for val in row:
            if pd.notna(val):
                print(f"  {val:>5.3f}", end="")
            else:
                print(f"   N/A ", end="")
        print()

    # 保存
    pd.DataFrame(all_results).to_csv("/home/tiger/rqalpha/tmp/reg_lambda_grid_results.csv", index=False)
    print(f"\n完整结果已保存至 tmp/reg_lambda_grid_results.csv ({len(all_results)} 行)")


if __name__ == "__main__":
    main()
