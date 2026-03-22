"""
红利低波状态自适应策略

核心思路：
- 趋势强时保留较高底仓，避免单边上涨里长期空着现金
- 估值分位仍然决定战术层的加减仓
- 过热和低置信度只限制战术层，不轻易破坏趋势底仓
"""

from __future__ import annotations

import datetime

import numpy as np

from rqalpha.apis import *
from skyeye.products.dividend_low_vol.registry import get_strategy_spec
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.allocation import (
    bounded_target,
    combine_targets,
    step_limit,
    trend_support_target,
    valuation_target_from_percentile,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.confidence_guard import (
    apply_cheap_trend_cap,
    confidence_adjusted_target,
    confidence_step_multiplier,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.heat_guard import (
    apply_tactical_heat_ceiling,
    compute_hot_signals,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.history_signal import (
    collect_recent_scores,
    mean_percentile,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.market_regime import (
    compute_market_state,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.reentry_guard import (
    apply_high_reentry_guard,
    apply_high_heat_reentry_state,
    apply_low_percentile_add_guard,
    apply_low_percentile_recovery_state,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.params import configure_context


STRATEGY_ID = "dividend_low_vol.regime_adaptive"
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


def _load_close_history(context):
    close_history = history_bars(context.etf, context.trend_window, "1d", "close")
    if close_history is None:
        return None
    prices = np.asarray(close_history, dtype=float)
    prices = prices[np.isfinite(prices)]
    if len(prices) == 0:
        return None
    return prices


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

    close_history = _load_close_history(context)
    required_trend_bars = max(
        context.trend_short_window,
        context.trend_mid_window,
        context.trend_long_window,
        context.trend_breakout_window,
    ) + 1
    if close_history is None or len(close_history) < required_trend_bars:
        logger.info(
            "skip rebalance on {}: close history {} < required {}".format(
                trade_day,
                0 if close_history is None else len(close_history),
                required_trend_bars,
            )
        )
        return

    p_fast = mean_percentile(history, context.fast_window)
    p_slow = mean_percentile(history, context.slow_window)
    if p_fast is None or p_slow is None:
        return

    percentile_ref = context.fast_percentile_weight * p_fast + (1.0 - context.fast_percentile_weight) * p_slow
    valuation_trend = p_fast - p_slow
    market_state = compute_market_state(
        close_history=close_history,
        short_window=context.trend_short_window,
        mid_window=context.trend_mid_window,
        long_window=context.trend_long_window,
        breakout_window=context.trend_breakout_window,
        core_floor_min=context.core_floor_min,
        core_floor_max=context.core_floor_max,
    )
    hot_signals = compute_hot_signals(current_score)
    valuation_target = valuation_target_from_percentile(
        percentile=percentile_ref,
        max_target_percent=context.max_target_percent,
    )
    support_target = trend_support_target(
        percentile_ref=percentile_ref,
        trend_strength=market_state["trend_strength"],
        drawdown=market_state["drawdown"],
        hot_count=hot_signals["hot_count"],
        trend_support_threshold=context.trend_support_threshold,
        trend_support_min_percentile=context.trend_support_min_percentile,
        trend_support_max_percentile=context.trend_support_max_percentile,
        trend_support_shallow_drawdown=context.trend_support_shallow_drawdown,
        trend_support_max_hot_count=context.trend_support_max_hot_count,
        trend_support_min_target=context.trend_support_min_target,
        trend_support_max_target=context.trend_support_max_target,
        trend_support_extreme_percentile=context.trend_support_extreme_percentile,
        trend_support_extreme_max_target=context.trend_support_extreme_max_target,
    )
    combined_target = combine_targets(
        valuation_target=valuation_target,
        support_target=support_target["target"],
    )
    heat_result = apply_tactical_heat_ceiling(
        context=context,
        base_target=combined_target["target"],
        support_floor=combined_target["active_floor"],
        percentile_ref=percentile_ref,
        hot_signals=hot_signals,
        trend_strength=market_state["trend_strength"],
        drawdown=market_state["drawdown"],
        heat_trigger_percentile=context.heat_trigger_percentile,
        heat_extreme_percentile=context.heat_extreme_percentile,
        heat_cooldown_min_percentile=context.heat_cooldown_min_percentile,
        heat_cooldown_release_buffer=context.heat_cooldown_release_buffer,
        heat_constructive_trend_threshold=context.heat_constructive_trend_threshold,
        heat_strong_trend_threshold=context.heat_strong_trend_threshold,
        heat_shallow_drawdown=context.heat_shallow_drawdown,
        heat_cap_hot_weak=context.heat_cap_hot_weak,
        heat_cap_hot_constructive=context.heat_cap_hot_constructive,
        heat_cap_hot_strong=context.heat_cap_hot_strong,
        heat_cap_extreme_weak=context.heat_cap_extreme_weak,
        heat_cap_extreme_constructive=context.heat_cap_extreme_constructive,
        heat_cap_extreme_strong=context.heat_cap_extreme_strong,
    )
    current_target = _current_target_percent(context)
    high_reentry_result = apply_high_reentry_guard(
        context=context,
        current_target=current_target,
        desired_target=heat_result["target"],
        percentile_ref=percentile_ref,
        valuation_trend=valuation_trend,
        hot_count=hot_signals["hot_count"],
        heat_mode=heat_result["mode"],
        high_reentry_guard_percentile=context.high_reentry_guard_percentile,
        high_reentry_confirm_weeks=context.high_reentry_confirm_weeks,
        high_reentry_confirm_trend=context.high_reentry_confirm_trend,
    )
    desired_target = confidence_adjusted_target(
        current_target=current_target,
        desired_target=high_reentry_result["target"],
        confidence=current_score.get("confidence"),
        core_floor=combined_target["active_floor"],
    )
    high_heat_result = apply_high_heat_reentry_state(
        context=context,
        current_target=current_target,
        desired_target=desired_target,
        percentile_ref=percentile_ref,
        valuation_trend=valuation_trend,
        hot_count=hot_signals["hot_count"],
        heat_mode=heat_result["mode"],
        high_heat_reentry_arm_percentile=context.high_heat_reentry_arm_percentile,
        high_heat_reentry_reset_percentile=context.high_heat_reentry_reset_percentile,
        high_heat_reentry_watch_weeks=context.high_heat_reentry_watch_weeks,
        high_heat_reentry_stage1_confirm_weeks=context.high_heat_reentry_stage1_confirm_weeks,
        high_heat_reentry_stage2_confirm_weeks=context.high_heat_reentry_stage2_confirm_weeks,
        high_heat_reentry_stage1_max_hot_count=context.high_heat_reentry_stage1_max_hot_count,
        high_heat_reentry_stage2_max_hot_count=context.high_heat_reentry_stage2_max_hot_count,
        high_heat_reentry_stage1_max_valuation_trend=context.high_heat_reentry_stage1_max_valuation_trend,
        high_heat_reentry_stage2_max_valuation_trend=context.high_heat_reentry_stage2_max_valuation_trend,
        high_heat_reentry_stage1_target=context.high_heat_reentry_stage1_target,
        high_heat_reentry_stage2_target=context.high_heat_reentry_stage2_target,
        high_heat_reentry_hold_weeks=context.high_heat_reentry_hold_weeks,
    )
    desired_target = high_heat_result["target"]
    recovery_source_target = desired_target
    cheap_cap_result = apply_cheap_trend_cap(
        desired_target=desired_target,
        percentile_ref=percentile_ref,
        trend_strength=market_state["trend_strength"],
        cheap_entry_cap_percentile=context.cheap_entry_cap_percentile,
        cheap_entry_cap_trend_threshold=context.cheap_entry_cap_trend_threshold,
        cheap_entry_cap_target=context.cheap_entry_cap_target,
    )
    desired_target = cheap_cap_result["target"]
    low_entry_result = apply_low_percentile_add_guard(
        current_target=current_target,
        desired_target=desired_target,
        percentile_ref=percentile_ref,
        valuation_trend=valuation_trend,
        trend_strength=market_state["trend_strength"],
        low_entry_guard_percentile=context.low_entry_guard_percentile,
        low_entry_guard_current_target=context.low_entry_guard_current_target,
        low_entry_guard_max_valuation_trend=context.low_entry_guard_max_valuation_trend,
        low_entry_guard_max_trend_strength=context.low_entry_guard_max_trend_strength,
    )
    desired_target = low_entry_result["target"]
    low_recovery_result = apply_low_percentile_recovery_state(
        context=context,
        current_target=current_target,
        guarded_target=desired_target,
        recovery_source_target=recovery_source_target,
        percentile_ref=percentile_ref,
        valuation_trend=valuation_trend,
        trend_strength=market_state["trend_strength"],
        hot_count=hot_signals["hot_count"],
        low_recovery_arm_percentile=context.low_recovery_arm_percentile,
        low_recovery_reset_percentile=context.low_recovery_reset_percentile,
        low_recovery_min_current_target=context.low_recovery_min_current_target,
        low_recovery_confirm_weeks=context.low_recovery_confirm_weeks,
        low_recovery_confirm_valuation_trend=context.low_recovery_confirm_valuation_trend,
        low_recovery_confirm_trend_strength=context.low_recovery_confirm_trend_strength,
        low_recovery_stage1_target=context.low_recovery_stage1_target,
        low_recovery_stage2_target=context.low_recovery_stage2_target,
        low_recovery_stage2_confirm_weeks=context.low_recovery_stage2_confirm_weeks,
        low_recovery_stage2_confirm_valuation_trend=context.low_recovery_stage2_confirm_valuation_trend,
        low_recovery_stage2_confirm_trend_strength=context.low_recovery_stage2_confirm_trend_strength,
        low_recovery_hold_weeks=context.low_recovery_hold_weeks,
        low_recovery_fail_valuation_trend=context.low_recovery_fail_valuation_trend,
        low_recovery_max_hot_count=context.low_recovery_max_hot_count,
    )
    desired_target = low_recovery_result["target"]
    deadband_override = bool(
        low_recovery_result.get("deadband_override") or high_heat_result.get("deadband_override")
    )

    if not deadband_override and abs(desired_target - current_target) < context.deadband:
        logger.info(
            "hold on {}: current={:.0%} desired={:.0%} p_ref={:.1%} v_trend={:+.1%} regime={} trend_strength={:.0%} core_floor={:.0%} support_floor={:.0%} valuation_target={:.0%} hot_count={} heat_mode={} high_reentry_mode={} high_heat_mode={} cheap_cap_mode={} low_entry_mode={} low_recovery_mode={} cooldown_left={} clear_count={} high_heat_count={} high_heat_hold={} high_heat_watch={} high_heat_deadband={} recovery_count={} recovery_hold={} recovery_stage={} recovery_deadband={} confidence={}".format(
                trade_day,
                current_target,
                desired_target,
                percentile_ref,
                valuation_trend,
                market_state["mode"],
                market_state["trend_strength"],
                market_state["core_floor"],
                combined_target["active_floor"],
                valuation_target,
                hot_signals["hot_count"],
                heat_result["mode"],
                high_reentry_result["mode"],
                high_heat_result["mode"],
                cheap_cap_result["mode"],
                low_entry_result["mode"],
                low_recovery_result["mode"],
                heat_result["cooldown_left"],
                high_reentry_result["clear_count"],
                high_heat_result["confirm_count"],
                high_heat_result["hold_left"],
                high_heat_result["watch_left"],
                high_heat_result["deadband_override"],
                low_recovery_result["confirm_count"],
                low_recovery_result["hold_left"],
                low_recovery_result["stage"],
                low_recovery_result["deadband_override"],
                current_score.get("confidence"),
            )
        )
        return

    max_step = step_limit(
        current_target=current_target,
        desired_target=desired_target,
        trend_strength=market_state["trend_strength"],
        default_raise_step=context.default_raise_step,
        default_cut_step=context.default_cut_step,
        initial_step_limit=context.initial_step_limit,
        strong_trend_raise_step=context.strong_trend_raise_step,
        weak_trend_raise_step=context.weak_trend_raise_step,
        breakdown_cut_step=context.breakdown_cut_step,
        strong_trend_cut_step=context.strong_trend_cut_step,
    )
    max_step *= confidence_step_multiplier(current_score.get("confidence"))
    next_target = bounded_target(current_target, desired_target, max_step)

    logger.info(
        (
            "regime-adaptive rebalance on {} signal_date={} current={:.0%} next={:.0%} desired={:.0%} "
            "p_fast={:.1%} p_slow={:.1%} p_ref={:.1%} v_trend={:+.1%} regime={} trend_strength={:.0%} "
            "core_floor={:.0%} support_floor={:.0%} valuation_target={:.0%} support_mode={} combine_mode={} hot_count={} heat_mode={} high_reentry_mode={} high_heat_mode={} cheap_cap_mode={} low_entry_mode={} low_recovery_mode={} heat_cap={} cooldown_left={} clear_count={} high_heat_count={} high_heat_hold={} high_heat_watch={} high_heat_deadband={} recovery_count={} recovery_hold={} recovery_stage={} recovery_deadband={} "
            "ret20={:+.1%} ret60={:+.1%} drawdown={:+.1%} confidence={} method={}"
        ).format(
            trade_day,
            current_score.get("date"),
            current_target,
            next_target,
            desired_target,
            p_fast,
            p_slow,
            percentile_ref,
            valuation_trend,
            market_state["mode"],
            market_state["trend_strength"],
            market_state["core_floor"],
            combined_target["active_floor"],
            valuation_target,
            support_target["mode"],
            combined_target["mode"],
            hot_signals["hot_count"],
            heat_result["mode"],
            high_reentry_result["mode"],
            high_heat_result["mode"],
            cheap_cap_result["mode"],
            low_entry_result["mode"],
            low_recovery_result["mode"],
            "{:.0%}".format(heat_result["cap"]) if heat_result["cap"] is not None else "-",
            heat_result["cooldown_left"],
            high_reentry_result["clear_count"],
            high_heat_result["confirm_count"],
            high_heat_result["hold_left"],
            high_heat_result["watch_left"],
            high_heat_result["deadband_override"],
            low_recovery_result["confirm_count"],
            low_recovery_result["hold_left"],
            low_recovery_result["stage"],
            low_recovery_result["deadband_override"],
            market_state["ret_short"],
            market_state["ret_mid"],
            market_state["drawdown"],
            current_score.get("confidence"),
            current_score.get("model_meta", {}).get("method"),
        )
    )
    order_target_percent(context.etf, next_target)


def after_trading(context):
    pass
