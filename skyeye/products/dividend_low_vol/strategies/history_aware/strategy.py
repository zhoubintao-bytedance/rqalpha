"""
红利低波历史感知策略

运行入口只保留 RQAlpha 适配层。
具体交易逻辑拆在同目录下的 `logic/`，参数 profile 拆在 `profiles/`。
"""
import datetime

from rqalpha.apis import *
from skyeye.products.dividend_low_vol.registry import get_strategy_spec
from skyeye.products.dividend_low_vol.strategies.history_aware.logic.confidence_guard import (
    confidence_adjusted_target,
    confidence_step_multiplier,
)
from skyeye.products.dividend_low_vol.strategies.history_aware.logic.heat_guard import (
    apply_heat_override,
    compute_hot_signals,
)
from skyeye.products.dividend_low_vol.strategies.history_aware.logic.history_signal import (
    collect_recent_scores,
    mean_percentile,
)
from skyeye.products.dividend_low_vol.strategies.history_aware.logic.position_curve import (
    base_target_from_percentile,
    bounded_target,
    step_limit,
)
from skyeye.products.dividend_low_vol.strategies.history_aware.logic.reentry_guard import (
    apply_high_reentry_guard,
)
from skyeye.products.dividend_low_vol.strategies.history_aware.params import configure_context


STRATEGY_ID = "dividend_low_vol.history_aware"
ROLLING_SCORE_START_DATE = "2020-02-01"


def init(context):
    context.strategy_spec = get_strategy_spec(STRATEGY_ID).raw
    configure_context(context)
    update_universe(context.etf)


def before_trading(context):
    pass


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

    history = collect_recent_scores(context, signal_date, context.slow_window)
    if len(history) < context.slow_window:
        logger.info(
            "skip rebalance on {}: score history {} < required {}".format(
                trade_day, len(history), context.slow_window
            )
        )
        return

    p_fast = mean_percentile(history, context.fast_window)
    p_slow = mean_percentile(history, context.slow_window)
    if p_fast is None or p_slow is None:
        return

    percentile_ref = 0.7 * p_fast + 0.3 * p_slow
    trend = p_fast - p_slow

    base_target = min(base_target_from_percentile(percentile_ref), context.max_target_percent)
    hot_signals = compute_hot_signals(current_score)
    heat_override = apply_heat_override(context, base_target, percentile_ref, hot_signals)
    current_target = _current_target_percent(context)
    reentry_guard = apply_high_reentry_guard(
        context=context,
        current_target=current_target,
        desired_target=heat_override["target"],
        percentile_ref=percentile_ref,
        trend=trend,
        hot_signals=hot_signals,
        heat_override=heat_override,
    )
    desired_target = reentry_guard["target"]
    desired_target = confidence_adjusted_target(
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

    max_step = step_limit(
        percentile_ref=percentile_ref,
        trend=trend,
        current_target=current_target,
        desired_target=desired_target,
        default_step_limit=context.default_step_limit,
        initial_step_limit=context.initial_step_limit,
    )
    max_step *= confidence_step_multiplier(current_score.get("confidence"))
    next_target = bounded_target(current_target, desired_target, max_step)

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
