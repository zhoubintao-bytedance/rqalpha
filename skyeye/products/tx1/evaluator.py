# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "mom_40d",
    "volatility_20d",
    "reversal_5d",
    "amihud_20d",
]


def evaluate_predictions(prediction_df, top_k=20):
    if prediction_df is None or len(prediction_df) == 0:
        raise ValueError("prediction_df must not be empty")
    grouped = prediction_df.groupby("date", sort=True)
    rank_ics = []
    spreads = []
    hit_rates = []
    for _, day_df in grouped:
        if len(day_df) < 2:
            continue
        pred_rank = day_df["prediction"].rank(method="average")
        label_rank = day_df["label_return_raw"].rank(method="average")
        rank_ic = float(pred_rank.corr(label_rank, method="pearson"))
        if np.isfinite(rank_ic):
            rank_ics.append(rank_ic)
        ranked = day_df.sort_values("prediction", ascending=False)
        top = ranked.head(min(top_k, len(ranked)))
        bottom = ranked.tail(min(top_k, len(ranked)))
        spreads.append(float(top["label_return_raw"].mean() - bottom["label_return_raw"].mean()))
        hit_rates.append(float((top["label_return_raw"] > 0).mean()))
    rank_ic_mean = float(np.mean(rank_ics)) if rank_ics else 0.0
    rank_ic_ir = float(rank_ic_mean / np.std(rank_ics)) if len(rank_ics) > 1 and np.std(rank_ics) > 0 else 0.0
    return {
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_ir": rank_ic_ir,
        "top_bucket_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
        "top_k_hit_rate": float(np.mean(hit_rates)) if hit_rates else 0.0,
    }


def build_portfolio_returns(test_df, weights_df, horizon_days=1):
    """Build a daily proxy return series from forward-horizon labels.

    `label_return_raw` is a forward excess return over `horizon_days`. This
    function keeps that raw horizon return for auditability and derives a daily
    proxy return by linear scaling, which is then used by portfolio/cost
    evaluation so all downstream metrics share the same daily unit.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if weights_df is None or len(weights_df) == 0:
        return pd.DataFrame(columns=["date", "portfolio_return_horizon_raw", "portfolio_return", "turnover", "overlap"])
    merged = weights_df.merge(
        test_df[["date", "order_book_id", "label_return_raw"]],
        on=["date", "order_book_id"],
        how="left",
    )
    if merged["label_return_raw"].isna().any():
        raise ValueError("weights_df contains assets without matching label_return_raw")
    rows = []
    prev_weights = {}
    prev_assets = set()
    for date, day_df in merged.groupby("date", sort=True):
        current_weights = {row["order_book_id"]: float(row["weight"]) for _, row in day_df.iterrows()}
        assets = set(current_weights)
        turnover = 0.5 * sum(abs(current_weights.get(a, 0.0) - prev_weights.get(a, 0.0)) for a in assets | set(prev_weights))
        overlap = float(len(assets & prev_assets) / len(assets | prev_assets)) if (assets or prev_assets) else 1.0
        portfolio_return_horizon_raw = float((day_df["weight"] * day_df["label_return_raw"]).sum())
        rows.append({
            "date": pd.Timestamp(date),
            "portfolio_return_horizon_raw": portfolio_return_horizon_raw,
            "portfolio_return": portfolio_return_horizon_raw / float(horizon_days),
            "turnover": turnover,
            "overlap": overlap,
        })
        prev_weights = current_weights
        prev_assets = assets
    return pd.DataFrame(rows)


def evaluate_portfolios(portfolio_returns_df, cost_config=None):
    if portfolio_returns_df is None or len(portfolio_returns_df) == 0:
        raise ValueError("portfolio_returns_df must not be empty")
    equity = (1.0 + portfolio_returns_df["portfolio_return"].fillna(0.0)).cumprod()
    running_max = equity.cummax()
    drawdown = 1.0 - equity / running_max
    volatility = float(portfolio_returns_df["portfolio_return"].std(ddof=0) * np.sqrt(252.0)) if len(portfolio_returns_df) > 1 else 0.0
    mean_return = float(portfolio_returns_df["portfolio_return"].mean())
    ir = float(mean_return / portfolio_returns_df["portfolio_return"].std(ddof=0)) if len(portfolio_returns_df) > 1 and portfolio_returns_df["portfolio_return"].std(ddof=0) > 0 else 0.0
    result = {
        "mean_return": mean_return,
        "max_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "volatility": volatility,
        "information_ratio_proxy": ir,
        "mean_turnover": float(portfolio_returns_df["turnover"].mean()),
        "mean_overlap": float(portfolio_returns_df["overlap"].mean()),
    }

    if cost_config is not None:
        from skyeye.products.tx1.cost_layer import compute_cost_metrics
        cost_metrics = compute_cost_metrics(portfolio_returns_df, cost_config)
        result["net_mean_return"] = cost_metrics["net_mean_return"]
        result["cost_drag_annual"] = cost_metrics["cost_drag_annual"]
        result["cost_erosion_ratio"] = cost_metrics["cost_erosion_ratio"]
        result["breakeven_cost_bps"] = cost_metrics["breakeven_cost_bps"]

    return result
