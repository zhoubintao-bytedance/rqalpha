# -*- coding: utf-8 -*-
"""
TX1 Feature Engineering Experiment

Usage:
    python -m skyeye.products.tx1.run_feature_experiment
    python -m skyeye.products.tx1.run_feature_experiment --variant baseline_5f combined_v2 --max-folds 1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyeye.products.tx1.baseline_models import create_model
from skyeye.products.tx1.cost_layer import CostConfig
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.evaluator import (
    BASELINE_4F_COLUMNS,
    BASELINE_FEATURE_COLUMNS,
    BASELINE_5F_COLUMNS,
    CANDIDATE_FEATURE_COLUMNS,
    ELITE_OHLCV_COLUMNS,
    FEATURE_LIBRARY,
    FUNDAMENTAL_FEATURE_COLUMNS,
    LIQUIDITY_FEATURE_COLUMNS,
    MOMENTUM_FEATURE_COLUMNS,
    RISK_FEATURE_COLUMNS,
    TREND_FEATURE_COLUMNS,
    build_portfolio_returns,
    evaluate_portfolios,
    evaluate_predictions,
    get_available_feature_columns,
)
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.preprocessor import FeaturePreprocessor
from skyeye.products.tx1.run_baseline_experiment import build_raw_df, _get_liquid_universe
from skyeye.products.tx1.splitter import WalkForwardSplitter


MODEL_CONFIG = {
    "linear": {},
    "tree": {},
    "lgbm": {
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

DEFAULT_PREPROCESS_CONFIG = {
    "neutralize": True,
    "winsorize_scale": 5.0,
    "standardize": True,
}

GOOD_COMPANY_UNIVERSE_FILTER = {
    "ep_ratio_ttm": "above_median",
    "return_on_equity_ttm": "above_median",
}


def _dedupe(features):
    result = []
    for feature in features:
        if feature not in result:
            result.append(feature)
    return result


def build_variants():
    combined_v1_features = _dedupe(
        list(BASELINE_FEATURE_COLUMNS)
        + [
            "excess_mom_20d",
            "excess_mom_60d",
            "ma_crossover_10_40d",
            "price_position_60d",
            "volume_trend_5_20d",
            "turnover_stability_20d",
            "downside_volatility_20d",
            "beta_60d",
        ]
    )
    combined_v2_features = _dedupe(
        list(BASELINE_FEATURE_COLUMNS)
        + [
            "mom_20d",
            "excess_mom_60d",
            "ma_gap_20d",
            "distance_to_high_60d",
            "turnover_ratio_20d",
            "vol_adj_turnover_20d",
            "downside_volatility_20d",
            "max_drawdown_20d",
        ]
    )
    elite_combined_features = _dedupe(
        list(ELITE_OHLCV_COLUMNS) + list(FUNDAMENTAL_FEATURE_COLUMNS)
    )
    return [
        {
            "name": "baseline_5f",
            "purpose": "Current default TX1 baseline: 4-factor core plus turnover stability.",
            "features": list(BASELINE_5F_COLUMNS),
            "preprocess": None,
        },
        {
            "name": "baseline_5f_roe",
            "purpose": "Minimal fundamental additive check: baseline_5f plus return_on_equity_ttm.",
            "features": _dedupe(list(BASELINE_5F_COLUMNS) + ["return_on_equity_ttm"]),
            "preprocess": None,
        },
        {
            "name": "baseline_4f",
            "purpose": "Legacy 4-factor baseline kept only as a historical reference line.",
            "features": list(BASELINE_4F_COLUMNS),
            "preprocess": None,
        },
        {
            "name": "baseline_4f_fundamental_filter",
            "purpose": "Legacy 4-factor timing model inside a value-and-quality universe filtered by EP and ROE.",
            "features": list(BASELINE_4F_COLUMNS),
            "preprocess": None,
            "universe_filter": dict(GOOD_COMPANY_UNIVERSE_FILTER),
        },
        {
            "name": "baseline_5f_fundamental_filter",
            "purpose": "Current 5-factor timing model inside a value-and-quality universe filtered by EP and ROE.",
            "features": list(BASELINE_5F_COLUMNS),
            "preprocess": None,
            "universe_filter": dict(GOOD_COMPANY_UNIVERSE_FILTER),
        },
        {
            "name": "baseline_4f_preproc",
            "purpose": "Legacy 4-factor baseline plus cross-sectional winsorize/neutralize/zscore preprocessing.",
            "features": list(BASELINE_4F_COLUMNS),
            "preprocess": dict(DEFAULT_PREPROCESS_CONFIG),
        },
        {
            "name": "momentum_plus",
            "purpose": "Baseline plus multi-horizon and benchmark-relative momentum features.",
            "features": _dedupe(list(BASELINE_FEATURE_COLUMNS) + list(MOMENTUM_FEATURE_COLUMNS)),
            "preprocess": None,
        },
        {
            "name": "trend_plus",
            "purpose": "Baseline plus moving-average structure and range-position trend features.",
            "features": _dedupe(list(BASELINE_FEATURE_COLUMNS) + list(TREND_FEATURE_COLUMNS)),
            "preprocess": None,
        },
        {
            "name": "liquidity_plus",
            "purpose": "Baseline plus volume, turnover, and volatility-adjusted liquidity features.",
            "features": _dedupe(list(BASELINE_FEATURE_COLUMNS) + list(LIQUIDITY_FEATURE_COLUMNS)),
            "preprocess": None,
        },
        {
            "name": "risk_plus",
            "purpose": "Baseline plus downside-risk, beta, skew, and drawdown proxies.",
            "features": _dedupe(list(BASELINE_FEATURE_COLUMNS) + list(RISK_FEATURE_COLUMNS)),
            "preprocess": None,
        },
        {
            "name": "combined_v1",
            "purpose": "Cross-group mix that keeps benchmark-relative momentum and turnover stability.",
            "features": combined_v1_features,
            "preprocess": None,
        },
        {
            "name": "combined_v2",
            "purpose": "Compact cross-group mix with preprocessing enabled to test fold stability.",
            "features": combined_v2_features,
            "preprocess": dict(DEFAULT_PREPROCESS_CONFIG),
        },
        {
            "name": "elite_ohlcv_3f",
            "purpose": "Best representative from each OHLCV cluster: momentum, volatility, liquidity stability.",
            "features": list(ELITE_OHLCV_COLUMNS),
            "preprocess": None,
        },
        {
            "name": "fundamental_5f",
            "purpose": "Pure fundamental factors: valuation, profitability, growth, cash flow.",
            "features": list(FUNDAMENTAL_FEATURE_COLUMNS),
            "preprocess": None,
        },
        {
            "name": "elite_combined_8f",
            "purpose": "Elite OHLCV (3) + fundamental (5) for maximum orthogonal signal.",
            "features": elite_combined_features,
            "preprocess": None,
        },
        {
            "name": "elite_combined_8f_lt",
            "purpose": "Elite combined with lower turnover: rebalance_interval=40, holding_bonus=1.0.",
            "features": elite_combined_features,
            "preprocess": None,
            "portfolio_config": {
                "rebalance_interval": 40,
                "holding_bonus": 1.0,
            },
        },
    ]


def _make_preprocessor(preprocess_config):
    if not preprocess_config:
        return None
    return FeaturePreprocessor(
        neutralize=preprocess_config.get("neutralize", True),
        winsorize_scale=preprocess_config.get("winsorize_scale", 5.0),
        standardize=preprocess_config.get("standardize", True),
        min_obs=preprocess_config.get("min_obs", 5),
    )


def _aggregate_metric_dicts(items, key):
    if not items:
        return {}
    metric_keys = items[0][key].keys()
    return {
        metric_key: float(np.mean([item[key][metric_key] for item in items]))
        for metric_key in metric_keys
    }


def _extract_feature_importance(model, features):
    if not hasattr(model, "_model") or model._model is None:
        return {}
    importance = model._model.feature_importance(importance_type="gain")
    return {feature: float(value) for feature, value in zip(features, importance.tolist())}


def _apply_universe_filter(df, universe_filter):
    """Apply per-date cross-sectional universe filter.

    universe_filter is a dict like:
        {"ep_ratio_ttm": "above_median", "return_on_equity_ttm": "above_median"}

    For each date, keeps only rows where ALL filter conditions are met.
    "above_median" means the value >= cross-sectional median for that date.
    """
    if not universe_filter:
        return df
    filtered_parts = []
    for date, day_df in df.groupby("date"):
        mask = pd.Series(True, index=day_df.index)
        for col, condition in universe_filter.items():
            if col not in day_df.columns or day_df[col].isna().all():
                continue
            if condition == "above_median":
                median_val = day_df[col].median()
                mask = mask & (day_df[col] >= median_val)
        filtered_parts.append(day_df.loc[mask])
    if not filtered_parts:
        return df.iloc[:0]
    return pd.concat(filtered_parts, ignore_index=True)


def run_variant(
    labeled_df,
    variant,
    splitter,
    model_kind="lgbm",
    horizon_days=20,
    top_k=20,
    max_folds=None,
):
    requested_features = list(variant["features"])
    available_features = get_available_feature_columns(labeled_df.columns, requested_features)
    missing_features = [feature for feature in requested_features if feature not in available_features]

    print("\n" + "=" * 72)
    print(f"Variant: {variant['name']}")
    print(f"Purpose: {variant['purpose']}")
    print(
        "Features: requested={} available={} preprocess={}".format(
            len(requested_features),
            len(available_features),
            bool(variant.get("preprocess")),
        )
    )
    if missing_features:
        print(f"Missing features skipped: {missing_features}")
    print("=" * 72)

    if not available_features:
        return {
            "name": variant["name"],
            "purpose": variant["purpose"],
            "requested_features": requested_features,
            "features": [],
            "missing_features": missing_features,
            "preprocess": variant.get("preprocess"),
            "n_folds": 0,
        }

    variant_df = labeled_df.dropna(subset=available_features).copy()
    universe_filter = variant.get("universe_filter")
    if universe_filter:
        pre_filter_count = len(variant_df)
        variant_df = _apply_universe_filter(variant_df, universe_filter)
        retained_ratio = float(len(variant_df) / pre_filter_count) if pre_filter_count else 0.0
        print(f"Universe filter applied: {pre_filter_count} -> {len(variant_df)} rows ({retained_ratio:.1%} retained)")
    variant_df = variant_df.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    folds = splitter.split(variant_df)
    if max_folds is not None:
        folds = folds[:max_folds]
    print(f"Rows after feature availability filter: {len(variant_df)} / {len(labeled_df)}")
    print(f"Folds: {len(folds)}")

    if not folds:
        return {
            "name": variant["name"],
            "purpose": variant["purpose"],
            "requested_features": requested_features,
            "features": available_features,
            "missing_features": missing_features,
            "preprocess": variant.get("preprocess"),
            "n_folds": 0,
            "row_count": len(variant_df),
            "row_ratio": float(len(variant_df) / len(labeled_df)) if len(labeled_df) else 0.0,
        }

    preprocessor = _make_preprocessor(variant.get("preprocess"))
    portfolio_config = variant.get("portfolio_config") or {}
    portfolio_builder = PortfolioProxy(
        buy_top_k=portfolio_config.get("buy_top_k", 25),
        hold_top_k=portfolio_config.get("hold_top_k", 45),
        rebalance_interval=portfolio_config.get("rebalance_interval", 20),
        holding_bonus=portfolio_config.get("holding_bonus", 0.5),
    )
    feature_importances = []
    fold_results = []

    for idx, fold in enumerate(folds, start=1):
        train_df = fold["train_df"].copy()
        val_df = fold["val_df"].copy()
        test_df = fold["test_df"].copy()

        if preprocessor is not None:
            train_df = preprocessor.transform(train_df, available_features)
            val_df = preprocessor.transform(val_df, available_features)
            test_df = preprocessor.transform(test_df, available_features)

        model = create_model(model_kind, params=MODEL_CONFIG.get(model_kind))
        fit_kwargs = {}
        if hasattr(model, "fit") and "val_X" in model.fit.__code__.co_varnames:
            fit_kwargs["val_X"] = val_df[available_features]
            fit_kwargs["val_y"] = val_df["target_label"]
        model.fit(train_df[available_features], train_df["target_label"], **fit_kwargs)

        test_df["prediction"] = model.predict(test_df[available_features])
        prediction_metrics = evaluate_predictions(test_df, top_k=top_k)

        weights_df = portfolio_builder.build(test_df[["date", "order_book_id", "prediction"]])
        portfolio_returns = build_portfolio_returns(test_df, weights_df, horizon_days=horizon_days)
        portfolio_metrics = evaluate_portfolios(portfolio_returns, cost_config=COST_CONFIG)

        fold_results.append(
            {
                "fold_index": idx,
                "date_range": {
                    "train_end": fold.get("train_end"),
                    "val_start": fold.get("val_start"),
                    "val_end": fold.get("val_end"),
                    "test_start": fold.get("test_start"),
                    "test_end": fold.get("test_end"),
                },
                "row_counts": {
                    "train": int(len(train_df)),
                    "val": int(len(val_df)),
                    "test": int(len(test_df)),
                },
                "prediction_metrics": prediction_metrics,
                "portfolio_metrics": portfolio_metrics,
            }
        )
        importance = _extract_feature_importance(model, available_features)
        if importance:
            feature_importances.append(importance)
        print(
            "  Fold {}/{} rank_ic={:.4f} net_ret={:.6f}".format(
                idx,
                len(folds),
                prediction_metrics["rank_ic_mean"],
                portfolio_metrics.get("net_mean_return", portfolio_metrics["mean_return"]),
            )
        )

    prediction = _aggregate_metric_dicts(fold_results, "prediction_metrics")
    portfolio = _aggregate_metric_dicts(fold_results, "portfolio_metrics")
    fold_rank_ics = [item["prediction_metrics"]["rank_ic_mean"] for item in fold_results]
    fold_net_returns = [
        item["portfolio_metrics"].get("net_mean_return", item["portfolio_metrics"]["mean_return"])
        for item in fold_results
    ]
    aggregated_importance = {}
    if feature_importances:
        for feature in available_features:
            aggregated_importance[feature] = float(np.mean([fi.get(feature, 0.0) for fi in feature_importances]))

    return {
        "name": variant["name"],
        "purpose": variant["purpose"],
        "requested_features": requested_features,
        "features": available_features,
        "missing_features": missing_features,
        "preprocess": variant.get("preprocess"),
        "portfolio_config": variant.get("portfolio_config"),
        "universe_filter": variant.get("universe_filter"),
        "n_folds": len(fold_results),
        "row_count": len(variant_df),
        "row_ratio": float(len(variant_df) / len(labeled_df)) if len(labeled_df) else 0.0,
        "date_range": {
            "start": variant_df["date"].min(),
            "end": variant_df["date"].max(),
        },
        "prediction": prediction,
        "portfolio": portfolio,
        "stability": {
            "fold_rank_ic_std": float(np.std(fold_rank_ics)) if fold_rank_ics else 0.0,
            "fold_rank_ic_min": float(np.min(fold_rank_ics)) if fold_rank_ics else 0.0,
            "fold_rank_ic_max": float(np.max(fold_rank_ics)) if fold_rank_ics else 0.0,
            "fold_net_return_std": float(np.std(fold_net_returns)) if fold_net_returns else 0.0,
            "fold_net_return_min": float(np.min(fold_net_returns)) if fold_net_returns else 0.0,
            "fold_net_return_max": float(np.max(fold_net_returns)) if fold_net_returns else 0.0,
        },
        "fold_results": fold_results,
        "feature_importance": aggregated_importance,
    }


def compute_single_factor_ic(labeled_df, features):
    """Compute single-factor Rank IC for each feature."""
    results = {}
    for feature in features:
        if feature not in labeled_df.columns:
            continue
        factor_frame = labeled_df.dropna(subset=[feature, "label_return_raw"])
        ics = []
        for _, day_df in factor_frame.groupby("date"):
            if len(day_df) < 10:
                continue
            pred_rank = day_df[feature].rank(method="average")
            label_rank = day_df["label_return_raw"].rank(method="average")
            ic = float(pred_rank.corr(label_rank, method="pearson"))
            if np.isfinite(ic):
                ics.append(ic)
        if ics:
            results[feature] = {
                "ic_mean": float(np.mean(ics)),
                "ic_std": float(np.std(ics)),
                "ic_ir": float(np.mean(ics) / np.std(ics)) if np.std(ics) > 0 else 0.0,
            }
    return results


def compute_correlation_matrix(dataset_df, features):
    """Compute average cross-sectional correlation matrix."""
    available = get_available_feature_columns(dataset_df.columns, features)
    if not available:
        return pd.DataFrame()

    corr_sum = None
    count = 0
    for _, day_df in dataset_df.groupby("date"):
        current = day_df[available].dropna(axis=1, how="all")
        if current.shape[0] < 20 or current.shape[1] < 2:
            continue
        corr = current.corr(method="spearman")
        if corr_sum is None:
            corr_sum = corr
        else:
            aligned_index = corr_sum.index.union(corr.index)
            corr_sum = corr_sum.reindex(index=aligned_index, columns=aligned_index, fill_value=0.0)
            corr = corr.reindex(index=aligned_index, columns=aligned_index, fill_value=0.0)
            corr_sum = corr_sum + corr
        count += 1

    if count == 0 or corr_sum is None:
        return pd.DataFrame()
    return corr_sum / float(count)


def _variant_metric_row(result):
    prediction = result.get("prediction", {})
    portfolio = result.get("portfolio", {})
    stability = result.get("stability", {})
    delta = result.get("delta_vs_baseline", {})
    # 导出 spread / 稳定性 delta，便于后续脚本直接筛选“值得继续推进”的候选。
    return {
        "variant": result["name"],
        "n_features": len(result.get("features", [])),
        "n_folds": result.get("n_folds", 0),
        "rank_ic_mean": prediction.get("rank_ic_mean"),
        "rank_ic_ir": prediction.get("rank_ic_ir"),
        "top_bucket_spread_mean": prediction.get("top_bucket_spread_mean"),
        "top_k_hit_rate": prediction.get("top_k_hit_rate"),
        "net_mean_return": portfolio.get("net_mean_return", portfolio.get("mean_return")),
        "max_drawdown": portfolio.get("max_drawdown"),
        "fold_rank_ic_std": stability.get("fold_rank_ic_std"),
        "fold_net_return_std": stability.get("fold_net_return_std"),
        "delta_rank_ic_mean": delta.get("prediction", {}).get("rank_ic_mean"),
        "delta_top_bucket_spread_mean": delta.get("prediction", {}).get("top_bucket_spread_mean"),
        "delta_net_mean_return": delta.get("portfolio", {}).get("net_mean_return"),
        "delta_fold_rank_ic_std": delta.get("stability", {}).get("fold_rank_ic_std"),
        "delta_fold_net_return_std": delta.get("stability", {}).get("fold_net_return_std"),
        "features": ",".join(result.get("features", [])),
    }


def _compute_baseline_deltas(results, baseline_name):
    baseline = next((item for item in results if item["name"] == baseline_name and item.get("n_folds", 0) > 0), None)
    if baseline is None:
        return

    baseline_net = baseline["portfolio"].get("net_mean_return", baseline["portfolio"].get("mean_return", 0.0))
    for result in results:
        if result.get("n_folds", 0) == 0:
            result["delta_vs_baseline"] = {}
            continue
        result_net = result["portfolio"].get("net_mean_return", result["portfolio"].get("mean_return", 0.0))
        result["delta_vs_baseline"] = {
            "prediction": {
                "rank_ic_mean": float(result["prediction"]["rank_ic_mean"] - baseline["prediction"]["rank_ic_mean"]),
                "rank_ic_ir": float(result["prediction"]["rank_ic_ir"] - baseline["prediction"]["rank_ic_ir"]),
                "top_bucket_spread_mean": float(
                    result["prediction"]["top_bucket_spread_mean"] - baseline["prediction"]["top_bucket_spread_mean"]
                ),
                "top_k_hit_rate": float(result["prediction"]["top_k_hit_rate"] - baseline["prediction"]["top_k_hit_rate"]),
            },
            "portfolio": {
                "net_mean_return": float(result_net - baseline_net),
                "max_drawdown": float(result["portfolio"]["max_drawdown"] - baseline["portfolio"]["max_drawdown"]),
                "mean_turnover": float(result["portfolio"]["mean_turnover"] - baseline["portfolio"]["mean_turnover"]),
            },
            "stability": {
                "fold_rank_ic_std": float(
                    result["stability"]["fold_rank_ic_std"] - baseline["stability"]["fold_rank_ic_std"]
                ),
                "fold_net_return_std": float(
                    result["stability"]["fold_net_return_std"] - baseline["stability"]["fold_net_return_std"]
                ),
            },
        }


def format_results(payload, output_dir):
    """Format and save experiment outputs."""
    results = payload["variants"]
    lines = []
    lines.append("=== TX1 Feature Engineering Experiment Results ===")
    lines.append("")
    lines.append(f"Model: {payload['model_kind']}")
    lines.append(f"Label transform: {payload['label_transform']}")
    lines.append(f"Dataset shape: {tuple(payload['dataset_shape'])}")
    lines.append(f"Available feature count: {len(payload['available_features'])}")
    lines.append(f"Baseline variant: {payload['baseline_variant']}")
    lines.append("")

    header = (
        f"{'Variant':<20} | {'#Feat':>5} | {'Folds':>5} | {'RankIC':>8} | {'ICstd':>8} | "
        f"{'Spread':>8} | {'NetRet':>9} | {'MaxDD':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for result in results:
        if result.get("n_folds", 0) == 0:
            lines.append(f"{result['name']:<20} | {'N/A':>5} | {'0':>5}")
            continue
        prediction = result["prediction"]
        portfolio = result["portfolio"]
        stability = result["stability"]
        lines.append(
            f"{result['name']:<20} | {len(result['features']):>5} | {result['n_folds']:>5} | "
            f"{prediction['rank_ic_mean']:>8.4f} | {stability['fold_rank_ic_std']:>8.4f} | "
            f"{prediction['top_bucket_spread_mean']:>8.4f} | "
            f"{portfolio.get('net_mean_return', portfolio['mean_return']):>9.6f} | "
            f"{portfolio['max_drawdown']:>6.1%}"
        )

    lines.append("")
    lines.append("--- Variant Details ---")
    for result in results:
        lines.append("")
        lines.append(f"{result['name']}: {result['purpose']}")
        lines.append(f"  features={result.get('features', [])}")
        lines.append(f"  missing_features={result.get('missing_features', [])}")
        lines.append(f"  preprocess={result.get('preprocess') or 'off'}")
        if result.get("portfolio_config"):
            lines.append(f"  portfolio_config={result['portfolio_config']}")
        if result.get("universe_filter"):
            lines.append(f"  universe_filter={result['universe_filter']}")
        if result.get("n_folds", 0) > 0:
            delta = result.get("delta_vs_baseline", {})
            lines.append(
                # 显式写出 spread 与稳定性 delta，避免只看 aggregate 收益后误判候选质量。
                "  delta_vs_baseline: rank_ic_mean={:+.4f}, top_bucket_spread_mean={:+.4f}, net_mean_return={:+.6f}, fold_rank_ic_std={:+.4f}, fold_net_return_std={:+.6f}".format(
                    delta.get("prediction", {}).get("rank_ic_mean", 0.0),
                    delta.get("prediction", {}).get("top_bucket_spread_mean", 0.0),
                    delta.get("portfolio", {}).get("net_mean_return", 0.0),
                    delta.get("stability", {}).get("fold_rank_ic_std", 0.0),
                    delta.get("stability", {}).get("fold_net_return_std", 0.0),
                )
            )

    lines.append("")
    lines.append("--- Single Factor Rank IC ---")
    lines.append(f"{'Feature':<25} | {'IC Mean':>8} | {'IC Std':>8} | {'IC IR':>8}")
    lines.append("-" * 58)
    for feature, stats in sorted(
        payload["single_factor_ic"].items(),
        key=lambda item: (-abs(item[1]["ic_mean"]), item[0]),
    ):
        lines.append(
            f"{feature:<25} | {stats['ic_mean']:>8.4f} | {stats['ic_std']:>8.4f} | {stats['ic_ir']:>8.4f}"
        )

    best_variant = max(
        (result for result in results if result.get("n_folds", 0) > 0),
        key=lambda item: item["prediction"]["rank_ic_mean"],
        default=None,
    )
    if best_variant is not None and best_variant.get("feature_importance"):
        lines.append("")
        lines.append(f"--- Feature Importance ({best_variant['name']}) ---")
        for feature, importance in sorted(best_variant["feature_importance"].items(), key=lambda item: -item[1]):
            lines.append(f"  {feature:<25}: {importance:.1f}")

    lines.append("")
    lines.append("--- Feature Descriptions ---")
    for feature in payload["available_features"]:
        lines.append(f"  {feature}: {FEATURE_LIBRARY.get(feature, 'n/a')}")

    lines.append("")
    lines.append("--- Cross-Sectional Correlation Matrix (Spearman) ---")
    if payload["correlation_matrix"].empty:
        lines.append("  Not enough cross-sectional coverage to compute.")
    else:
        lines.append(payload["correlation_matrix"].round(2).to_string())

    report = "\n".join(lines)
    print(report)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serializable_payload = dict(payload)
    serializable_payload["correlation_matrix"] = payload["correlation_matrix"].to_dict() if not payload["correlation_matrix"].empty else {}

    with open(output_dir / "feature_experiment_results.json", "w") as handle:
        json.dump(serializable_payload, handle, indent=2, default=str)
    with open(output_dir / "feature_experiment_report.txt", "w") as handle:
        handle.write(report)
    with open(output_dir / "single_factor_ic.json", "w") as handle:
        json.dump(payload["single_factor_ic"], handle, indent=2)
    if not payload["correlation_matrix"].empty:
        payload["correlation_matrix"].to_csv(output_dir / "correlation_matrix.csv")

    metrics_df = pd.DataFrame([_variant_metric_row(result) for result in results])
    metrics_df.to_csv(output_dir / "variant_metrics.csv", index=False)

    print(f"\nResults saved to {output_dir}")


def run_feature_experiments(
    raw_df,
    output_dir,
    variant_names=None,
    model_kind="lgbm",
    label_transform="rank",
    horizon_days=20,
    max_folds=None,
):
    dataset_builder = DatasetBuilder(input_window=60)
    dataset_df = dataset_builder.build(raw_df)
    print(f"Dataset shape: {dataset_df.shape}")

    label_builder = LabelBuilder(horizon=horizon_days, transform=label_transform)
    labeled_df = label_builder.build(dataset_df)
    print(f"Labeled shape: {labeled_df.shape}")

    available_features = get_available_feature_columns(dataset_df.columns, CANDIDATE_FEATURE_COLUMNS)
    single_factor_ics = compute_single_factor_ic(labeled_df, available_features)
    corr_matrix = compute_correlation_matrix(dataset_df, available_features)

    variants = build_variants()
    if variant_names:
        name_set = set(variant_names)
        variants = [variant for variant in variants if variant["name"] in name_set]
        missing_variants = sorted(name_set - {variant["name"] for variant in variants})
        if missing_variants:
            raise ValueError(f"unknown variant(s): {missing_variants}")

    splitter = WalkForwardSplitter(train_years=3, val_months=6, test_months=6, embargo_days=20)
    results = []
    for variant in variants:
        results.append(
            run_variant(
                labeled_df=labeled_df,
                variant=variant,
                splitter=splitter,
                model_kind=model_kind,
                horizon_days=horizon_days,
                top_k=20,
                max_folds=max_folds,
            )
        )

    baseline_name = "baseline_5f"
    _compute_baseline_deltas(results, baseline_name)

    payload = {
        "model_kind": model_kind,
        "label_transform": label_transform,
        "dataset_shape": list(dataset_df.shape),
        "available_features": available_features,
        "baseline_variant": baseline_name,
        "single_factor_ic": single_factor_ics,
        "correlation_matrix": corr_matrix,
        "variants": results,
    }
    format_results(payload, output_dir)
    return payload


def main(argv=None, raw_df=None):
    parser = argparse.ArgumentParser(description="TX1 Feature Engineering Experiment")
    parser.add_argument("--universe-size", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        default="skyeye/artifacts/experiments/tx1_feature_eng_session1",
    )
    parser.add_argument("--variant", nargs="*", default=None)
    parser.add_argument("--model-kind", choices=["linear", "tree", "lgbm"], default="lgbm")
    parser.add_argument("--label-transform", choices=["raw", "rank", "quantile"], default="rank")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--horizon-days", type=int, default=20)
    args = parser.parse_args(argv)

    if raw_df is None:
        print("Loading universe and raw data...")
        universe = _get_liquid_universe(args.universe_size)
        raw_df = build_raw_df(universe)

    return run_feature_experiments(
        raw_df=raw_df,
        output_dir=args.output_dir,
        variant_names=args.variant,
        model_kind=args.model_kind,
        label_transform=args.label_transform,
        horizon_days=args.horizon_days,
        max_folds=args.max_folds,
    )


if __name__ == "__main__":
    main()
