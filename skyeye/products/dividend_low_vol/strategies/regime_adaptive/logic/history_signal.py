"""History-aware score helpers for the regime-adaptive strategy."""

from __future__ import annotations

import pandas as pd

from rqalpha.environment import Environment


def resolve_dividend_scorer(context):
    if context.dividend_scorer is not None:
        return context.dividend_scorer

    env = Environment.get_instance()
    mod = getattr(env, "mod_dict", {}).get("dividend_scorer")
    scorer = getattr(mod, "_scorer", None) if mod is not None else None
    context.dividend_scorer = scorer
    return scorer


def hazen_percentile_of_last(series):
    valid = pd.Series(series).dropna()
    if len(valid) == 0:
        return None
    current = valid.iloc[-1]
    ranks = valid.rank(method="average")
    rank = ranks.iloc[-1]
    return float((rank - 0.5) / float(len(valid)))


def bind_precomputed_score_history(context):
    if context.precomputed_score_df is not None:
        return context.precomputed_score_df

    scorer = resolve_dividend_scorer(context)
    if scorer is None or scorer.score_history_df is None:
        return None

    score_df = scorer.score_history_df.loc[:, ["total_score", "score_percentile"]].copy()
    signal_percentile = score_df["score_percentile"].copy()
    missing_dates = score_df.index[signal_percentile.isna() & score_df["total_score"].notna()]

    for date in missing_dates:
        trailing = score_df.loc[:date, "total_score"].dropna().tail(context.bootstrap_percentile_window)
        if len(trailing) < context.bootstrap_min_score_samples:
            continue
        signal_percentile.loc[date] = hazen_percentile_of_last(trailing)

    score_df["signal_percentile"] = signal_percentile
    context.precomputed_score_df = score_df
    return score_df


def collect_recent_scores(context, score_date, required_count):
    score_df = bind_precomputed_score_history(context)
    if score_df is None:
        return []

    asof = pd.Timestamp(score_date)
    history = score_df.loc[:asof, ["signal_percentile"]].dropna()
    if history.empty:
        return []

    scores = []
    for idx, row in history.tail(required_count).iterrows():
        scores.append(
            {
                "date": idx.strftime("%Y-%m-%d"),
                "score_percentile": float(row["signal_percentile"]),
            }
        )
    return scores


def mean_percentile(scores, window):
    values = []
    for score in scores[:window]:
        percentile = score.get("score_percentile")
        if percentile is None:
            continue
        values.append(float(percentile))
    if not values:
        return None
    return sum(values) / float(len(values))
