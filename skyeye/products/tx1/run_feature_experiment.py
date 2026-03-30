# -*- coding: utf-8 -*-
"""
TX1 Feature Engineering Experiment

Usage:
    python -m skyeye.products.tx1.run_feature_experiment
    python -m skyeye.products.tx1.run_feature_experiment --universe-size 100
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from skyeye.products.tx1.baseline_models import create_model
from skyeye.products.tx1.cost_layer import CostConfig
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import (
    FEATURE_COLUMNS,
    build_portfolio_returns,
    evaluate_portfolios,
    evaluate_predictions,
)
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.preprocessor import FeaturePreprocessor
from skyeye.products.tx1.run_baseline_experiment import build_raw_df, _get_liquid_universe
from skyeye.products.tx1.splitter import WalkForwardSplitter


MODEL_CONFIG = {
    "kind": "lgbm",
    "params": {
        "num_leaves": 24,
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 200,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "min_child_samples": 80,
        "early_stopping_rounds": 20,
    },
}

COST_CONFIG = CostConfig(
    commission_rate=0.0008,
    stamp_tax_rate=0.0005,
    slippage_bps=5.0,
)


VARIANTS = [
    {
        "name": "V1_4feat_rank",
        "features": FEATURE_COLUMNS,
        "transform": "rank",
        "preprocess": False,
        "purpose": "Best 4-feature set with rank labels",
    },
    {
        "name": "V2_4feat_raw",
        "features": FEATURE_COLUMNS,
        "transform": "raw",
        "preprocess": False,
        "purpose": "Best 4-feature set with raw labels",
    },
]


def run_variant(dataset_df, variant):
    """Run a single variant through walk-forward pipeline."""
    name = variant["name"]
    features = variant["features"]
    transform = variant["transform"]
    use_preprocess = variant["preprocess"]

    # Filter to available features
    available_features = [f for f in features if f in dataset_df.columns]
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"  Warning: missing features {missing}, using {len(available_features)} available")
        features = available_features

    print(f"\n{'='*60}")
    print(f"Variant: {name} ({variant['purpose']})")
    print(f"  features={len(features)}, transform={transform}, preprocess={use_preprocess}")
    print(f"{'='*60}")

    label_builder = LabelBuilder(horizon=20, transform=transform)
    labeled = label_builder.build(dataset_df)

    splitter = WalkForwardSplitter(train_years=3, val_months=6, test_months=6, embargo_days=20)
    folds = splitter.split(labeled)
    print(f"  Folds: {len(folds)}")

    preprocessor = FeaturePreprocessor() if use_preprocess else None
    portfolio_builder = PortfolioProxy(buy_top_k=20, hold_top_k=50, rebalance_interval=20, holding_bonus=0.5)
    fold_results = []
    feature_importances = []

    for idx, fold in enumerate(folds, start=1):
        train_df = fold["train_df"].copy()
        val_df = fold["val_df"].copy()
        test_df = fold["test_df"].copy()

        # Preprocessing inside fold
        if preprocessor is not None:
            feat_cols = [c for c in features if c in train_df.columns]
            train_df = preprocessor.transform(train_df, feat_cols)
            val_df = preprocessor.transform(val_df, feat_cols)
            test_df = preprocessor.transform(test_df, feat_cols)

        model = create_model(MODEL_CONFIG["kind"], params=MODEL_CONFIG.get("params"))
        fit_kwargs = {}
        if hasattr(model, "fit") and "val_X" in model.fit.__code__.co_varnames:
            fit_kwargs["val_X"] = val_df[features]
            fit_kwargs["val_y"] = val_df["target_label"]
        model.fit(train_df[features], train_df["target_label"], **fit_kwargs)

        test_df["prediction"] = model.predict(test_df[features])
        prediction_metrics = evaluate_predictions(test_df, top_k=20)

        weights_df = portfolio_builder.build(test_df[["date", "order_book_id", "prediction"]])
        portfolio_returns = build_portfolio_returns(test_df, weights_df, horizon_days=20)
        portfolio_metrics = evaluate_portfolios(portfolio_returns, cost_config=COST_CONFIG)

        fold_results.append({
            "fold_index": idx,
            "prediction_metrics": prediction_metrics,
            "portfolio_metrics": portfolio_metrics,
        })

        # Extract feature importance from LightGBM
        if hasattr(model, "_model") and model._model is not None:
            importance = model._model.feature_importance(importance_type="gain")
            fi = dict(zip(features, importance.tolist()))
            feature_importances.append(fi)

        sys.stdout.write(f"\r  Fold {idx}/{len(folds)} (IC={prediction_metrics['rank_ic_mean']:.4f})")
        sys.stdout.flush()

    print()

    n = len(fold_results)
    if n == 0:
        return {"name": name, "n_folds": 0}

    agg_pred = {k: sum(f["prediction_metrics"][k] for f in fold_results) / n
                for k in fold_results[0]["prediction_metrics"]}
    agg_port = {k: sum(f["portfolio_metrics"][k] for f in fold_results) / n
                for k in fold_results[0]["portfolio_metrics"]}

    # Aggregate feature importance
    avg_importance = {}
    if feature_importances:
        all_feats = set()
        for fi in feature_importances:
            all_feats.update(fi.keys())
        for feat in all_feats:
            vals = [fi.get(feat, 0.0) for fi in feature_importances]
            avg_importance[feat] = float(np.mean(vals))

    return {
        "name": name,
        "n_folds": n,
        "features": features,
        "prediction": agg_pred,
        "portfolio": agg_port,
        "fold_results": fold_results,
        "feature_importance": avg_importance,
    }


def compute_single_factor_ic(dataset_df, features, horizon=20):
    """Compute single-factor Rank IC for each feature."""
    label_builder = LabelBuilder(horizon=horizon, transform="raw")
    labeled = label_builder.build(dataset_df)

    results = {}
    for feat in features:
        if feat not in labeled.columns:
            continue
        ics = []
        for _, day_df in labeled.groupby("date"):
            if len(day_df) < 10:
                continue
            pred_rank = day_df[feat].rank(method="average")
            label_rank = day_df["label_return_raw"].rank(method="average")
            ic = float(pred_rank.corr(label_rank, method="pearson"))
            if np.isfinite(ic):
                ics.append(ic)
        if ics:
            results[feat] = {
                "ic_mean": float(np.mean(ics)),
                "ic_std": float(np.std(ics)),
                "ic_ir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
            }
    return results


def compute_correlation_matrix(dataset_df, features):
    """Compute average cross-sectional correlation matrix."""
    available = [f for f in features if f in dataset_df.columns]
    corr_sum = None
    count = 0
    for _, day_df in dataset_df.groupby("date"):
        if len(day_df) < 20:
            continue
        c = day_df[available].corr(method="spearman")
        if corr_sum is None:
            corr_sum = c.values
        else:
            corr_sum += c.values
        count += 1
    if count == 0:
        return pd.DataFrame()
    avg_corr = pd.DataFrame(corr_sum / count, index=available, columns=available)
    return avg_corr


def format_results(results, single_factor_ics, corr_matrix, output_dir):
    """Format and save all results."""
    lines = []
    lines.append("\n=== TX1 Feature Engineering Experiment Results ===\n")

    header = f"{'Variant':<22} | {'#Feat':>5} | {'Rank IC':>8} | {'IC IR':>8} | {'Spread':>8} | {'Hit%':>6} | {'NetRet':>9} | {'MaxDD':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        if r["n_folds"] == 0:
            lines.append(f"{r['name']:<22} | {'N/A':>5}")
            continue
        p = r["prediction"]
        pt = r["portfolio"]
        lines.append(
            f"{r['name']:<22} | {len(r['features']):>5} | {p['rank_ic_mean']:>8.4f} | "
            f"{p['rank_ic_ir']:>8.4f} | {p['top_bucket_spread_mean']:>8.4f} | "
            f"{p['top_k_hit_rate']:>5.1%} | {pt.get('net_mean_return', pt['mean_return']):>9.6f} | "
            f"{pt['max_drawdown']:>5.1%}"
        )

    # Single factor IC table
    lines.append("\n--- Single Factor Rank IC ---")
    lines.append(f"{'Feature':<25} | {'IC Mean':>8} | {'IC Std':>8} | {'IC IR':>8}")
    lines.append("-" * 58)
    for feat, stats in sorted(single_factor_ics.items(), key=lambda x: -abs(x[1]["ic_mean"])):
        lines.append(
            f"{feat:<25} | {stats['ic_mean']:>8.4f} | {stats['ic_std']:>8.4f} | {stats['ic_ir']:>8.4f}"
        )

    # Feature importance for best variant
    best = max((r for r in results if r["n_folds"] > 0), key=lambda r: r["prediction"]["rank_ic_ir"])
    if best.get("feature_importance"):
        lines.append(f"\n--- Feature Importance ({best['name']}) ---")
        sorted_fi = sorted(best["feature_importance"].items(), key=lambda x: -x[1])
        for feat, imp in sorted_fi:
            lines.append(f"  {feat:<25}: {imp:.1f}")

    # Correlation matrix
    lines.append("\n--- Cross-Sectional Correlation Matrix (Spearman) ---")
    if not corr_matrix.empty:
        lines.append(corr_matrix.round(2).to_string())

    report = "\n".join(lines)
    print(report)

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for r in results:
        entry = {"name": r["name"], "n_folds": r["n_folds"]}
        if r["n_folds"] > 0:
            entry["prediction"] = r["prediction"]
            entry["portfolio"] = r["portfolio"]
            entry["features"] = r["features"]
            entry["feature_importance"] = r.get("feature_importance", {})
        summary.append(entry)

    with open(output_dir / "feature_experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(output_dir / "feature_experiment_report.txt", "w") as f:
        f.write(report)
    with open(output_dir / "single_factor_ic.json", "w") as f:
        json.dump(single_factor_ics, f, indent=2)
    if not corr_matrix.empty:
        corr_matrix.to_csv(output_dir / "correlation_matrix.csv")

    print(f"\nResults saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="TX1 Feature Engineering Experiment")
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument("--output-dir", default="skyeye/artifacts/experiments/tx1_feature_eng_new")
    args = parser.parse_args()

    # Load data once
    print("Loading universe and raw data...")
    universe = _get_liquid_universe(args.universe_size)
    raw_df = build_raw_df(universe)

    # Build dataset once (with all features)
    print("Building dataset (features)...")
    dataset_builder = DatasetBuilder(input_window=60)
    dataset_df = dataset_builder.build(raw_df)
    print(f"Dataset shape: {dataset_df.shape}")
    print(f"Columns: {list(dataset_df.columns)}")

    # Diagnostics: single factor IC and correlation matrix
    print("\nComputing single factor IC...")
    all_features = [f for f in FEATURE_COLUMNS if f in dataset_df.columns]
    single_factor_ics = compute_single_factor_ic(dataset_df, all_features)

    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(dataset_df, all_features)

    # Run all variants
    results = []
    for variant in VARIANTS:
        result = run_variant(dataset_df, variant)
        results.append(result)

    # Format and save
    format_results(results, single_factor_ics, corr_matrix, args.output_dir)


if __name__ == "__main__":
    main()
