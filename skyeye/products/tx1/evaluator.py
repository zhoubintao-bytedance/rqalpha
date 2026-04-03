# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


BASELINE_4F_COLUMNS = [
    "mom_40d",
    "volatility_20d",
    "reversal_5d",
    "amihud_20d",
]

MOMENTUM_FEATURE_COLUMNS = [
    "mom_20d",
    "mom_60d",
    "excess_mom_20d",
    "excess_mom_60d",
]

TREND_FEATURE_COLUMNS = [
    "ma_gap_20d",
    "ma_gap_60d",
    "ma_crossover_10_40d",
    "price_position_20d",
    "price_position_60d",
    "distance_to_high_60d",
]

LIQUIDITY_FEATURE_COLUMNS = [
    "volume_ratio_20d",
    "volume_trend_5_20d",
    "turnover_ratio_20d",
    "turnover_stability_20d",
    "dollar_volume_20d_log",
    "vol_adj_turnover_20d",
]

RISK_FEATURE_COLUMNS = [
    "downside_volatility_20d",
    "return_skew_20d",
    "beta_60d",
    "max_drawdown_20d",
]

ELITE_OHLCV_COLUMNS = [
    "mom_40d",
    "volatility_20d",
    "turnover_stability_20d",
]

BASELINE_5F_COLUMNS = [
    "mom_40d",
    "volatility_20d",
    "reversal_5d",
    "amihud_20d",
    "turnover_stability_20d",
]

# Default TX1 baseline: promote the validated 5-factor set to the shared alias
# used by the training/evaluation pipeline. Keep the 4-factor set as an
# explicit legacy comparison baseline for feature experiments.
BASELINE_FEATURE_COLUMNS = list(BASELINE_5F_COLUMNS)
FEATURE_COLUMNS = list(BASELINE_FEATURE_COLUMNS)

FUNDAMENTAL_FEATURE_COLUMNS = [
    "ep_ratio_ttm",
    "return_on_equity_ttm",
    "operating_revenue_growth_ratio_ttm",
    "net_profit_growth_ratio_ttm",
    "pcf_ratio_ttm",
]

FEATURE_GROUPS = {
    "baseline_4f": list(BASELINE_4F_COLUMNS),
    "baseline": list(BASELINE_FEATURE_COLUMNS),
    "momentum": list(MOMENTUM_FEATURE_COLUMNS),
    "trend": list(TREND_FEATURE_COLUMNS),
    "liquidity": list(LIQUIDITY_FEATURE_COLUMNS),
    "risk": list(RISK_FEATURE_COLUMNS),
    "baseline_5f": list(BASELINE_5F_COLUMNS),
    "elite_ohlcv": list(ELITE_OHLCV_COLUMNS),
    "fundamental": list(FUNDAMENTAL_FEATURE_COLUMNS),
}

FEATURE_LIBRARY = {
    "mom_40d": "40-day price momentum; the current baseline trend signal.",
    "volatility_20d": "20-day annualized realized volatility from daily returns.",
    "reversal_5d": "Negative 5-day return; short-term mean-reversion proxy.",
    "amihud_20d": "20-day average Amihud illiquidity using absolute return over turnover.",
    "mom_20d": "20-day price momentum for a shorter relative-strength horizon.",
    "mom_60d": "60-day price momentum to capture slower medium-term trend persistence.",
    "excess_mom_20d": "20-day asset return minus benchmark return; benchmark-relative momentum.",
    "excess_mom_60d": "60-day asset return minus benchmark return; slower benchmark-relative momentum.",
    "ma_gap_20d": "Distance between price and the 20-day moving average.",
    "ma_gap_60d": "Distance between price and the 60-day moving average.",
    "ma_crossover_10_40d": "10-day versus 40-day moving-average spread; compact trend-structure proxy.",
    "price_position_20d": "Position of price within the trailing 20-day high-low range.",
    "price_position_60d": "Position of price within the trailing 60-day high-low range.",
    "distance_to_high_60d": "Distance from the trailing 60-day high; current drawdown-from-peak proxy.",
    "volume_ratio_20d": "Current volume divided by trailing 20-day average volume.",
    "volume_trend_5_20d": "5-day average volume relative to the 20-day average volume.",
    "turnover_ratio_20d": "Current turnover proxy divided by trailing 20-day average turnover proxy.",
    "turnover_stability_20d": "20-day turnover mean divided by turnover volatility; crowding/liquidity stability proxy.",
    "dollar_volume_20d_log": "Log of 20-day average turnover proxy; robust scale-stabilized trading activity.",
    "vol_adj_turnover_20d": "20-day average turnover proxy scaled by trailing volatility; volatility-adjusted liquidity.",
    "downside_volatility_20d": "20-day downside volatility using only negative-return magnitude.",
    "return_skew_20d": "20-day return skewness; tail-asymmetry proxy.",
    "beta_60d": "60-day rolling beta to the benchmark from daily returns.",
    "max_drawdown_20d": "Worst drawdown observed over the trailing 20-day drawdown path.",
    "ep_ratio_ttm": "Earnings-to-price ratio (TTM); inverse of P/E, higher = cheaper valuation.",
    "return_on_equity_ttm": "Return on equity (TTM); profitability quality measure.",
    "operating_revenue_growth_ratio_ttm": "Operating revenue YoY growth (TTM); top-line growth signal.",
    "net_profit_growth_ratio_ttm": "Net profit YoY growth (TTM); bottom-line growth signal.",
    "pcf_ratio_ttm": "Price-to-cash-flow ratio (TTM); cash-flow-based valuation.",
}

CANDIDATE_FEATURE_COLUMNS = []
for _feature_group in FEATURE_GROUPS.values():
    for _feature_name in _feature_group:
        if _feature_name not in CANDIDATE_FEATURE_COLUMNS:
            CANDIDATE_FEATURE_COLUMNS.append(_feature_name)


def collect_feature_columns(*group_names):
    features = []
    for group_name in group_names:
        for feature_name in FEATURE_GROUPS.get(group_name, []):
            if feature_name not in features:
                features.append(feature_name)
    return features


def get_available_feature_columns(columns, requested_features):
    available = set(columns)
    return [feature for feature in requested_features if feature in available]


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
