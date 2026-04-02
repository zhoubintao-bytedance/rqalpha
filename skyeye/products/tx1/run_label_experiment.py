# -*- coding: utf-8 -*-
"""
TX1 Label Optimization Experiment

Compares label construction variants to find the configuration that
maximizes Rank IC. Loads data once, builds features once, then runs
each variant through the full walk-forward pipeline.

Usage:
    python -m skyeye.products.tx1.run_label_experiment
    python -m skyeye.products.tx1.run_label_experiment --universe-size 100
"""

import argparse
import json
import sys
from pathlib import Path

from skyeye.products.tx1.baseline_models import create_model
from skyeye.products.tx1.cost_layer import CostConfig
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import FEATURE_COLUMNS, build_portfolio_returns, evaluate_portfolios, evaluate_predictions
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.run_baseline_experiment import build_raw_df, _get_liquid_universe
from skyeye.products.tx1.splitter import WalkForwardSplitter


VARIANTS = [
    {"name": "A_baseline",  "horizon": 20, "transform": "raw",  "winsorize": None},
    {"name": "B_winsorize", "horizon": 20, "transform": "raw",  "winsorize": (0.01, 0.99)},
    {"name": "C_rank",      "horizon": 20, "transform": "rank", "winsorize": None},
    {"name": "D_horizon10", "horizon": 10, "transform": "raw",  "winsorize": None},
]

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
        "early_stopping_rounds": 0,
    },
}

COST_CONFIG = CostConfig(
    commission_rate=0.0008,
    stamp_tax_rate=0.0005,
    slippage_bps=5.0,
)


def run_variant(dataset_df, variant):
    """Run a single label variant through the full walk-forward pipeline."""
    name = variant["name"]
    horizon = variant["horizon"]
    transform = variant["transform"]
    winsorize = variant["winsorize"]

    print(f"\n{'='*60}")
    print(f"Running variant: {name}")
    print(f"  horizon={horizon}, transform={transform}, winsorize={winsorize}")
    print(f"{'='*60}")

    label_builder = LabelBuilder(horizon=horizon, transform=transform, winsorize=winsorize)
    labeled = label_builder.build(dataset_df)
    print(f"  Labeled rows: {len(labeled)}")

    # Embargo should be at least as long as the horizon
    embargo_days = max(20, horizon)
    splitter = WalkForwardSplitter(
        train_years=3, val_months=6, test_months=6, embargo_days=embargo_days,
    )
    folds = splitter.split(labeled)
    print(f"  Folds: {len(folds)}")

    portfolio_builder = PortfolioProxy(buy_top_k=20, hold_top_k=30)
    fold_results = []

    for idx, fold in enumerate(folds, start=1):
        train_df = fold["train_df"]
        val_df = fold["val_df"].copy()
        test_df = fold["test_df"].copy()

        model = create_model(MODEL_CONFIG["kind"], params=MODEL_CONFIG.get("params"))
        fit_kwargs = {}
        if hasattr(model, "fit") and "val_X" in model.fit.__code__.co_varnames:
            fit_kwargs["val_X"] = val_df[FEATURE_COLUMNS]
            fit_kwargs["val_y"] = val_df["target_label"]
        model.fit(train_df[FEATURE_COLUMNS], train_df["target_label"], **fit_kwargs)

        test_df["prediction"] = model.predict(test_df[FEATURE_COLUMNS])
        prediction_metrics = evaluate_predictions(test_df, top_k=20)

        weights_df = portfolio_builder.build(test_df[["date", "order_book_id", "prediction"]])
        portfolio_returns = build_portfolio_returns(test_df, weights_df, horizon_days=horizon)
        portfolio_metrics = evaluate_portfolios(portfolio_returns, cost_config=COST_CONFIG)

        fold_results.append({
            "fold_index": idx,
            "prediction_metrics": prediction_metrics,
            "portfolio_metrics": portfolio_metrics,
        })

        sys.stdout.write(f"\r  Fold {idx}/{len(folds)} done (IC={prediction_metrics['rank_ic_mean']:.4f})")
        sys.stdout.flush()

    print()

    # Aggregate across folds
    n = len(fold_results)
    if n == 0:
        return {"name": name, "n_folds": 0}

    agg_pred = {}
    for key in fold_results[0]["prediction_metrics"]:
        agg_pred[key] = sum(f["prediction_metrics"][key] for f in fold_results) / n
    agg_port = {}
    for key in fold_results[0]["portfolio_metrics"]:
        agg_port[key] = sum(f["portfolio_metrics"][key] for f in fold_results) / n

    return {
        "name": name,
        "config": variant,
        "n_folds": n,
        "prediction": agg_pred,
        "portfolio": agg_port,
        "fold_results": fold_results,
    }


def format_comparison(results):
    """Generate a markdown comparison table."""
    lines = []
    lines.append("\n=== TX1 Label Optimization Experiment Results ===\n")

    header = f"{'Variant':<16} | {'Rank IC':>8} | {'IC IR':>8} | {'Spread':>8} | {'Hit Rate':>8} | {'Net Ret':>8} | {'MaxDD':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    best_ic = -999
    best_name = ""

    for r in results:
        if r["n_folds"] == 0:
            lines.append(f"{r['name']:<16} | {'(no folds)':>8}")
            continue
        p = r["prediction"]
        pt = r["portfolio"]
        ic = p["rank_ic_mean"]
        if ic > best_ic:
            best_ic = ic
            best_name = r["name"]
        lines.append(
            f"{r['name']:<16} | {ic:>8.4f} | {p['rank_ic_ir']:>8.4f} | "
            f"{p['top_bucket_spread_mean']:>8.4f} | {p['top_k_hit_rate']:>7.1%} | "
            f"{pt.get('net_mean_return', pt['mean_return']):>8.5f} | {pt['max_drawdown']:>7.1%}"
        )

    lines.append("")
    lines.append(f"Winner: {best_name} (Rank IC = {best_ic:.4f})")
    lines.append("")

    # Per-fold detail for winner
    winner = next((r for r in results if r["name"] == best_name), None)
    if winner and winner["n_folds"] > 0:
        lines.append(f"--- {best_name} per-fold Rank IC ---")
        for f in winner["fold_results"]:
            lines.append(f"  Fold {f['fold_index']:2d}: IC = {f['prediction_metrics']['rank_ic_mean']:.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="TX1 Label Optimization Experiment")
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument("--output-dir", default="skyeye/artifacts/experiments/tx1_label_opt")
    args = parser.parse_args()

    # Load data once
    print("Loading universe and raw data...")
    universe = _get_liquid_universe(args.universe_size)
    raw_df = build_raw_df(universe)

    # Build features once
    print("Building dataset (features)...")
    dataset_builder = DatasetBuilder(input_window=60)
    dataset_df = dataset_builder.build(raw_df)
    print(f"Dataset shape: {dataset_df.shape}")

    # Run all variants
    results = []
    for variant in VARIANTS:
        result = run_variant(dataset_df, variant)
        results.append(result)

    # Print comparison
    report = format_comparison(results)
    print(report)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for r in results:
        entry = {"name": r["name"], "n_folds": r["n_folds"]}
        if r["n_folds"] > 0:
            entry["prediction"] = r["prediction"]
            entry["portfolio"] = r["portfolio"]
            entry["config"] = r["config"]
        summary.append(entry)
    with open(output_dir / "label_experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(output_dir / "label_experiment_report.txt", "w") as f:
        f.write(report)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
