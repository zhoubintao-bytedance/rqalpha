"""
红利低波打分器历史感知策略

设计原则:
1. 主信号: 用最近分数分位的历史均值和趋势决定主仓位
2. 主仓位: 连续分段仓位，不做满仓/清仓
3. 卖出增强: 只有价格/溢价/RSI 同时偏热时才把仓位压得很低

运行前请先准备:
1. RQAlpha bundle（用于回测）
2. 红利低波打分器 SQLite 缓存（用于估值打分）
"""
import datetime

import pandas as pd

from rqalpha.apis import *
from rqalpha.environment import Environment

# 使用方法
# env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig uv run python strategy_scorer.py rqalpha/examples/dividend_low_vol_score_strategy_history_aware.py --mod dividend_scorer

# Used by strategy_scorer.py to choose default rolling-score windows.
STRATEGY_SCORER_START_DATE = "2020-02-01"


def init(context):
    context.etf = "512890.XSHG"
    context.max_target_percent = 0.90
    context.fast_window = 5
    context.slow_window = 15
    context.deadband = 0.08
    context.default_step_limit = 0.18
    context.initial_step_limit = 0.30
    context.bootstrap_percentile_window = 252
    context.bootstrap_min_score_samples = 20
    context.heat_cooldown_weeks = 2
    context.heat_cooldown_min_percentile = 0.80
    context.heat_cooldown_release_buffer = 0.10
    context.heat_cooldown_left = 0
    context.heat_cooldown_cap = None
    context.high_reentry_guard_percentile = 0.85
    context.high_reentry_confirm_weeks = 2
    context.high_reentry_confirm_trend = -0.01
    context.high_reentry_clear_count = 0
    context.last_rebalance_week = None
    context.precomputed_score_df = None
    context.dividend_scorer = None
    update_universe(context.etf)


def before_trading(context):
    pass


def _resolve_dividend_scorer(context):
    if context.dividend_scorer is not None:
        return context.dividend_scorer

    env = Environment.get_instance()
    mod = getattr(env, "mod_dict", {}).get("dividend_scorer")
    scorer = getattr(mod, "_scorer", None) if mod is not None else None
    context.dividend_scorer = scorer
    return scorer


def _hazen_percentile_of_last(series):
    valid = pd.Series(series).dropna()
    if len(valid) == 0:
        return None
    current = valid.iloc[-1]
    ranks = valid.rank(method="average")
    rank = ranks.iloc[-1]
    return float((rank - 0.5) / float(len(valid)))


def _bind_precomputed_score_history(context):
    if context.precomputed_score_df is not None:
        return context.precomputed_score_df

    scorer = _resolve_dividend_scorer(context)
    if scorer is None or scorer.score_history_df is None:
        return None

    score_df = scorer.score_history_df.loc[:, ["total_score", "score_percentile"]].copy()
    signal_percentile = score_df["score_percentile"].copy()
    missing_dates = score_df.index[signal_percentile.isna() & score_df["total_score"].notna()]

    for date in missing_dates:
        trailing = score_df.loc[:date, "total_score"].dropna().tail(context.bootstrap_percentile_window)
        if len(trailing) < context.bootstrap_min_score_samples:
            continue
        signal_percentile.loc[date] = _hazen_percentile_of_last(trailing)

    score_df["signal_percentile"] = signal_percentile
    context.precomputed_score_df = score_df
    return score_df


def _collect_recent_scores(context, score_date, required_count):
    score_df = _bind_precomputed_score_history(context)
    if score_df is None:
        return []

    asof = pd.Timestamp(score_date)
    history = score_df.loc[:asof, ["signal_percentile"]].dropna()
    if history.empty:
        return []

    scores = []
    for idx, row in history.tail(required_count).iterrows():
        scores.append({
            "date": idx.strftime("%Y-%m-%d"),
            "score_percentile": float(row["signal_percentile"]),
        })
    return scores


def _mean_percentile(scores, window):
    values = []
    for score in scores[:window]:
        percentile = score.get("score_percentile")
        if percentile is None:
            continue
        values.append(float(percentile))
    if not values:
        return None
    return sum(values) / float(len(values))


def _base_target_from_percentile(percentile):
    # Keep the original broad shape, but flatten the neutral zone so mid-range
    # percentiles do not swing target exposure too aggressively.
    anchors = (
        (0.00, 0.90),
        (0.10, 0.90),
        (0.20, 0.80),
        (0.40, 0.70),
        (0.52, 0.55),
        (0.68, 0.55),
        (0.80, 0.40),
        (0.90, 0.30),
        (1.00, 0.25),
    )
    p = min(max(float(percentile), 0.0), 1.0)
    for (left_p, left_target), (right_p, right_target) in zip(anchors, anchors[1:]):
        if p <= right_p:
            if right_p == left_p:
                return right_target
            ratio = (p - left_p) / float(right_p - left_p)
            return left_target + ratio * (right_target - left_target)
    return anchors[-1][1]


def _safe_feature_percentile(score, feature_name):
    features = score.get("features") or {}
    feature = features.get(feature_name) or {}
    percentile = feature.get("percentile")
    if percentile is None:
        return None
    return float(percentile)


def _safe_feature_raw(score, feature_name):
    features = score.get("features") or {}
    feature = features.get(feature_name) or {}
    raw = feature.get("raw")
    if raw is None:
        return None
    return float(raw)


def _compute_hot_signals(score):
    price_pct = _safe_feature_percentile(score, "price_percentile")
    premium_pct = max(
        value for value in (
            _safe_feature_percentile(score, "premium_rate"),
            _safe_feature_percentile(score, "premium_rate_ma20"),
        ) if value is not None
    ) if any(
        value is not None for value in (
            _safe_feature_percentile(score, "premium_rate"),
            _safe_feature_percentile(score, "premium_rate_ma20"),
        )
    ) else None
    rsi_raw = _safe_feature_raw(score, "rsi_20")
    rsi_pct = _safe_feature_percentile(score, "rsi_20")

    hot_price = price_pct is not None and price_pct >= 0.85
    hot_premium = premium_pct is not None and premium_pct >= 0.85
    hot_rsi = (rsi_raw is not None and rsi_raw >= 70.0) or (rsi_pct is not None and rsi_pct >= 0.90)

    return {
        "hot_price": hot_price,
        "hot_premium": hot_premium,
        "hot_rsi": hot_rsi,
        "hot_count": int(hot_price) + int(hot_premium) + int(hot_rsi),
    }


def _trigger_heat_cap(percentile_ref, hot_signals):
    hot_count = hot_signals["hot_count"]
    if percentile_ref >= 0.90 and hot_count >= 3:
        return 0.05
    if percentile_ref >= 0.80 and hot_count >= 2:
        return 0.10
    return None


def _apply_heat_override(context, base_target, percentile_ref, hot_signals):
    trigger_cap = _trigger_heat_cap(percentile_ref, hot_signals)
    if trigger_cap is not None:
        context.heat_cooldown_left = context.heat_cooldown_weeks
        context.heat_cooldown_cap = trigger_cap
        return {
            "target": min(base_target, trigger_cap),
            "mode": "triggered",
            "cap": trigger_cap,
            "cooldown_left": context.heat_cooldown_left,
        }

    if (
        context.heat_cooldown_left > 0
        and context.heat_cooldown_cap is not None
        and percentile_ref >= context.heat_cooldown_min_percentile
    ):
        release_cap = min(0.25, context.heat_cooldown_cap + context.heat_cooldown_release_buffer)
        context.heat_cooldown_left -= 1
        if context.heat_cooldown_left == 0:
            context.heat_cooldown_cap = None
        return {
            "target": min(base_target, release_cap),
            "mode": "cooldown",
            "cap": release_cap,
            "cooldown_left": context.heat_cooldown_left,
        }

    context.heat_cooldown_left = 0
    context.heat_cooldown_cap = None
    return {
        "target": base_target,
        "mode": "none",
        "cap": None,
        "cooldown_left": 0,
    }


def _apply_high_reentry_guard(
    context,
    current_target,
    desired_target,
    percentile_ref,
    trend,
    hot_signals,
    heat_override,
):
    if heat_override["mode"] == "triggered":
        context.high_reentry_clear_count = 0
        return {
            "target": desired_target,
            "mode": "triggered",
            "clear_count": context.high_reentry_clear_count,
        }

    if heat_override["mode"] == "cooldown":
        return {
            "target": desired_target,
            "mode": "cooldown",
            "clear_count": context.high_reentry_clear_count,
        }

    if desired_target <= current_target:
        if percentile_ref < context.high_reentry_guard_percentile:
            context.high_reentry_clear_count = 0
        return {
            "target": desired_target,
            "mode": "none",
            "clear_count": context.high_reentry_clear_count,
        }

    if percentile_ref < context.high_reentry_guard_percentile:
        context.high_reentry_clear_count = 0
        return {
            "target": desired_target,
            "mode": "none",
            "clear_count": context.high_reentry_clear_count,
        }

    confirmation_ready = (
        hot_signals["hot_count"] <= 1
        and trend <= context.high_reentry_confirm_trend
    )
    if confirmation_ready:
        context.high_reentry_clear_count += 1
    else:
        context.high_reentry_clear_count = 0

    if context.high_reentry_clear_count < context.high_reentry_confirm_weeks:
        return {
            "target": current_target,
            "mode": "waiting",
            "clear_count": context.high_reentry_clear_count,
        }

    return {
        "target": desired_target,
        "mode": "confirmed",
        "clear_count": context.high_reentry_clear_count,
    }


def _step_limit(percentile_ref, trend, current_target, desired_target, default_step_limit, initial_step_limit):
    if current_target < 0.01 and desired_target >= 0.40:
        return max(default_step_limit, min(initial_step_limit, desired_target))

    step = default_step_limit
    if percentile_ref <= 0.40:
        if trend > 0.03:
            return 0.18
        if trend < -0.03:
            return 0.08
    if percentile_ref >= 0.80:
        if trend > 0.03:
            return 0.18
        if trend < -0.03:
            return 0.08
    return step


def _confidence_adjusted_target(current_target, desired_target, confidence):
    if confidence == "low" and desired_target > current_target:
        return current_target
    return desired_target


def _confidence_step_multiplier(confidence):
    if confidence == "lowered":
        return 2.0 / 3.0
    return 1.0


def _bounded_target(current_target, desired_target, max_step):
    delta = desired_target - current_target
    if delta > max_step:
        return current_target + max_step
    if delta < -max_step:
        return current_target - max_step
    return desired_target


def _current_target_percent(context):
    position = get_position(context.etf)
    current_market_value = position.market_value if position is not None else 0.0
    portfolio_value = context.portfolio.total_value
    if portfolio_value <= 0:
        return 0.0
    return current_market_value / portfolio_value


def _trade_date(score):
    trade_date = score.get("trade_date") or score.get("date")
    if not trade_date:
        return None
    return datetime.datetime.strptime(trade_date, "%Y-%m-%d").date()


def _week_key(day):
    iso = day.isocalendar()
    return (iso[0], iso[1])


def handle_bar(context, bar_dict):
    current_score = get_dividend_score()
    if not current_score or current_score.get("error"):
        return

    signal_date = current_score.get("date")
    trade_day = _trade_date(current_score)
    if trade_day is None or signal_date is None:
        return

    week_key = _week_key(trade_day)
    if week_key == context.last_rebalance_week:
        return
    context.last_rebalance_week = week_key

    history = _collect_recent_scores(context, signal_date, context.slow_window)
    if len(history) < context.slow_window:
        logger.info(
            "skip rebalance on {}: score history {} < required {}".format(
                trade_day, len(history), context.slow_window
            )
        )
        return

    p_fast = _mean_percentile(history, context.fast_window)
    p_slow = _mean_percentile(history, context.slow_window)
    if p_fast is None or p_slow is None:
        return

    percentile_ref = 0.7 * p_fast + 0.3 * p_slow
    trend = p_fast - p_slow

    base_target = min(_base_target_from_percentile(percentile_ref), context.max_target_percent)
    hot_signals = _compute_hot_signals(current_score)
    heat_override = _apply_heat_override(context, base_target, percentile_ref, hot_signals)
    current_target = _current_target_percent(context)
    reentry_guard = _apply_high_reentry_guard(
        context=context,
        current_target=current_target,
        desired_target=heat_override["target"],
        percentile_ref=percentile_ref,
        trend=trend,
        hot_signals=hot_signals,
        heat_override=heat_override,
    )
    desired_target = reentry_guard["target"]
    desired_target = _confidence_adjusted_target(
        current_target=current_target,
        desired_target=desired_target,
        confidence=current_score.get("confidence"),
    )

    if abs(desired_target - current_target) < context.deadband:
        logger.info(
            "hold on {}: current={:.0%} desired={:.0%} p_ref={:.1%} trend={:+.1%} hot_count={} heat_mode={} heat_cap={} cooldown_left={} reentry_mode={} clear_count={} confidence={}".format(
                trade_day,
                current_target,
                desired_target,
                percentile_ref,
                trend,
                hot_signals["hot_count"],
                heat_override["mode"],
                "{:.0%}".format(heat_override["cap"]) if heat_override["cap"] is not None else "-",
                heat_override["cooldown_left"],
                reentry_guard["mode"],
                reentry_guard["clear_count"],
                current_score.get("confidence"),
            )
        )
        return

    step_limit = _step_limit(
        percentile_ref=percentile_ref,
        trend=trend,
        current_target=current_target,
        desired_target=desired_target,
        default_step_limit=context.default_step_limit,
        initial_step_limit=context.initial_step_limit,
    )
    step_limit *= _confidence_step_multiplier(current_score.get("confidence"))
    next_target = _bounded_target(current_target, desired_target, step_limit)

    logger.info(
        (
            "history-aware rebalance on {} signal_date={} current={:.0%} next={:.0%} desired={:.0%} "
            "p_fast={:.1%} p_slow={:.1%} p_ref={:.1%} trend={:+.1%} hot_count={} "
            "hot_price={} hot_premium={} hot_rsi={} heat_mode={} heat_cap={} cooldown_left={} "
            "reentry_mode={} clear_count={} confidence={} method={}"
        ).format(
            trade_day,
            current_score.get("date"),
            current_target,
            next_target,
            desired_target,
            p_fast,
            p_slow,
            percentile_ref,
            trend,
            hot_signals["hot_count"],
            hot_signals["hot_price"],
            hot_signals["hot_premium"],
            hot_signals["hot_rsi"],
            heat_override["mode"],
            "{:.0%}".format(heat_override["cap"]) if heat_override["cap"] is not None else "-",
            heat_override["cooldown_left"],
            reentry_guard["mode"],
            reentry_guard["clear_count"],
            current_score.get("confidence"),
            current_score.get("model_meta", {}).get("method"),
        )
    )
    order_target_percent(context.etf, next_target)


def after_trading(context):
    pass
