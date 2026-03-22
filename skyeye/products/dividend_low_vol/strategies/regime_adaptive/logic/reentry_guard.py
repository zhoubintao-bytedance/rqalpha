"""Reentry guards for expensive and cheap add-back scenarios."""

from __future__ import annotations


def apply_high_reentry_guard(
    context,
    current_target,
    desired_target,
    percentile_ref,
    valuation_trend,
    hot_count,
    heat_mode,
    high_reentry_guard_percentile,
    high_reentry_confirm_weeks,
    high_reentry_confirm_trend,
):
    if heat_mode not in ("none", "cooldown"):
        context.high_reentry_clear_count = 0
        return {
            "target": float(desired_target),
            "mode": "triggered",
            "clear_count": context.high_reentry_clear_count,
        }

    if heat_mode == "cooldown":
        return {
            "target": float(desired_target),
            "mode": "cooldown",
            "clear_count": context.high_reentry_clear_count,
        }

    if float(desired_target) <= float(current_target):
        if float(percentile_ref) < float(high_reentry_guard_percentile):
            context.high_reentry_clear_count = 0
        return {
            "target": float(desired_target),
            "mode": "none",
            "clear_count": context.high_reentry_clear_count,
        }

    if float(percentile_ref) < float(high_reentry_guard_percentile):
        context.high_reentry_clear_count = 0
        return {
            "target": float(desired_target),
            "mode": "none",
            "clear_count": context.high_reentry_clear_count,
        }

    confirmation_ready = int(hot_count) <= 1 and float(valuation_trend) <= float(high_reentry_confirm_trend)
    if confirmation_ready:
        context.high_reentry_clear_count += 1
    else:
        context.high_reentry_clear_count = 0

    if context.high_reentry_clear_count < int(high_reentry_confirm_weeks):
        return {
            "target": float(current_target),
            "mode": "waiting",
            "clear_count": context.high_reentry_clear_count,
        }

    return {
        "target": float(desired_target),
        "mode": "confirmed",
        "clear_count": context.high_reentry_clear_count,
    }


def apply_low_percentile_add_guard(
    current_target,
    desired_target,
    percentile_ref,
    valuation_trend,
    trend_strength,
    low_entry_guard_percentile,
    low_entry_guard_current_target,
    low_entry_guard_max_valuation_trend,
    low_entry_guard_max_trend_strength,
):
    if float(desired_target) <= float(current_target):
        return {
            "target": float(desired_target),
            "mode": "none",
        }

    if (
        float(current_target) >= float(low_entry_guard_current_target)
        and float(percentile_ref) <= float(low_entry_guard_percentile)
        and float(valuation_trend) <= float(low_entry_guard_max_valuation_trend)
        and float(trend_strength) <= float(low_entry_guard_max_trend_strength)
    ):
        return {
            "target": float(current_target),
            "mode": "waiting",
        }

    return {
        "target": float(desired_target),
        "mode": "none",
    }


def apply_low_percentile_recovery_state(
    context,
    current_target,
    guarded_target,
    recovery_source_target,
    percentile_ref,
    valuation_trend,
    trend_strength,
    hot_count,
    low_recovery_arm_percentile,
    low_recovery_reset_percentile,
    low_recovery_min_current_target,
    low_recovery_confirm_weeks,
    low_recovery_confirm_valuation_trend,
    low_recovery_confirm_trend_strength,
    low_recovery_stage1_target,
    low_recovery_stage2_target,
    low_recovery_stage2_confirm_weeks,
    low_recovery_stage2_confirm_valuation_trend,
    low_recovery_stage2_confirm_trend_strength,
    low_recovery_hold_weeks,
    low_recovery_fail_valuation_trend,
    low_recovery_max_hot_count,
):
    current = float(current_target)
    guarded = float(guarded_target)
    source = float(recovery_source_target)
    percentile = float(percentile_ref)
    valuation = float(valuation_trend)
    strength = float(trend_strength)
    hot = int(hot_count)
    previous_stage = int(context.low_recovery_stage)

    suppressed_upside = source > guarded + 1e-9
    armed = (
        percentile <= float(low_recovery_arm_percentile)
        and current >= float(low_recovery_min_current_target)
        and suppressed_upside
    )
    confirmation_ready = (
        armed
        and hot <= int(low_recovery_max_hot_count)
        and valuation >= float(low_recovery_confirm_valuation_trend)
        and strength >= float(low_recovery_confirm_trend_strength)
    )
    stage2_ready = (
        confirmation_ready
        and valuation >= float(low_recovery_stage2_confirm_valuation_trend)
        and strength >= float(low_recovery_stage2_confirm_trend_strength)
    )

    if confirmation_ready:
        context.low_recovery_confirm_count += 1
    else:
        context.low_recovery_confirm_count = 0

    fail_recovery = (
        percentile >= float(low_recovery_reset_percentile)
        or not suppressed_upside
        or hot > int(low_recovery_max_hot_count)
        or valuation <= float(low_recovery_fail_valuation_trend)
    )
    if fail_recovery:
        context.low_recovery_stage = 0
        context.low_recovery_hold_left = 0
        if not armed:
            context.low_recovery_confirm_count = 0
        return {
            "target": guarded,
            "mode": "none" if not armed else "armed",
            "confirm_count": context.low_recovery_confirm_count,
            "hold_left": context.low_recovery_hold_left,
            "stage": context.low_recovery_stage,
            "deadband_override": False,
        }

    if context.low_recovery_confirm_count >= int(low_recovery_confirm_weeks):
        context.low_recovery_stage = max(int(context.low_recovery_stage), 1)
        context.low_recovery_hold_left = int(low_recovery_hold_weeks)

    if stage2_ready and context.low_recovery_confirm_count >= int(low_recovery_stage2_confirm_weeks):
        context.low_recovery_stage = 2
        context.low_recovery_hold_left = int(low_recovery_hold_weeks)

    if context.low_recovery_hold_left > 0:
        stage_target = float(low_recovery_stage1_target)
        mode = "stage1"
        if int(context.low_recovery_stage) >= 2:
            stage_target = float(low_recovery_stage2_target)
            mode = "stage2"

        target = max(guarded, min(source, stage_target))
        deadband_override = target > current + 1e-9 and (
            int(context.low_recovery_stage) > previous_stage or context.low_recovery_hold_left > 0
        )
        if confirmation_ready:
            context.low_recovery_hold_left = int(low_recovery_hold_weeks)
        else:
            context.low_recovery_hold_left -= 1
        return {
            "target": target,
            "mode": mode,
            "confirm_count": context.low_recovery_confirm_count,
            "hold_left": context.low_recovery_hold_left,
            "stage": context.low_recovery_stage,
            "deadband_override": deadband_override,
        }

    context.low_recovery_stage = 0
    if armed:
        return {
            "target": guarded,
            "mode": "confirming" if context.low_recovery_confirm_count > 0 else "armed",
            "confirm_count": context.low_recovery_confirm_count,
            "hold_left": context.low_recovery_hold_left,
            "stage": context.low_recovery_stage,
            "deadband_override": False,
        }

    return {
        "target": guarded,
        "mode": "none",
        "confirm_count": context.low_recovery_confirm_count,
        "hold_left": context.low_recovery_hold_left,
        "stage": context.low_recovery_stage,
        "deadband_override": False,
    }


def apply_high_heat_reentry_state(
    context,
    current_target,
    desired_target,
    percentile_ref,
    valuation_trend,
    hot_count,
    heat_mode,
    high_heat_reentry_arm_percentile,
    high_heat_reentry_reset_percentile,
    high_heat_reentry_watch_weeks,
    high_heat_reentry_stage1_confirm_weeks,
    high_heat_reentry_stage2_confirm_weeks,
    high_heat_reentry_stage1_max_hot_count,
    high_heat_reentry_stage2_max_hot_count,
    high_heat_reentry_stage1_max_valuation_trend,
    high_heat_reentry_stage2_max_valuation_trend,
    high_heat_reentry_stage1_target,
    high_heat_reentry_stage2_target,
    high_heat_reentry_hold_weeks,
):
    current = float(current_target)
    desired = float(desired_target)
    percentile = float(percentile_ref)
    valuation = float(valuation_trend)
    hot = int(hot_count)
    previous_stage = int(context.high_heat_reentry_stage)

    if heat_mode not in ("none", "cooldown"):
        context.high_heat_reentry_watch_left = int(high_heat_reentry_watch_weeks)
        context.high_heat_reentry_confirm_count = 0
        context.high_heat_reentry_stage = 0
        context.high_heat_reentry_hold_left = 0
        return {
            "target": desired,
            "mode": "triggered",
            "confirm_count": context.high_heat_reentry_confirm_count,
            "hold_left": context.high_heat_reentry_hold_left,
            "watch_left": context.high_heat_reentry_watch_left,
            "stage": context.high_heat_reentry_stage,
            "deadband_override": False,
        }

    if desired <= current:
        if percentile < float(high_heat_reentry_reset_percentile):
            context.high_heat_reentry_watch_left = 0
            context.high_heat_reentry_confirm_count = 0
            context.high_heat_reentry_stage = 0
            context.high_heat_reentry_hold_left = 0
        return {
            "target": desired,
            "mode": "none",
            "confirm_count": context.high_heat_reentry_confirm_count,
            "hold_left": context.high_heat_reentry_hold_left,
            "watch_left": context.high_heat_reentry_watch_left,
            "stage": context.high_heat_reentry_stage,
            "deadband_override": False,
        }

    armed = percentile >= float(high_heat_reentry_arm_percentile) and (
        context.high_heat_reentry_watch_left > 0 or heat_mode == "cooldown"
    )
    if heat_mode == "cooldown":
        context.high_heat_reentry_watch_left = int(high_heat_reentry_watch_weeks)
        armed = True

    if not armed or percentile < float(high_heat_reentry_reset_percentile):
        context.high_heat_reentry_watch_left = 0
        context.high_heat_reentry_confirm_count = 0
        context.high_heat_reentry_stage = 0
        context.high_heat_reentry_hold_left = 0
        return {
            "target": desired,
            "mode": "none",
            "confirm_count": context.high_heat_reentry_confirm_count,
            "hold_left": context.high_heat_reentry_hold_left,
            "watch_left": context.high_heat_reentry_watch_left,
            "stage": context.high_heat_reentry_stage,
            "deadband_override": False,
        }

    stage1_ready = (
        hot <= int(high_heat_reentry_stage1_max_hot_count)
        and valuation <= float(high_heat_reentry_stage1_max_valuation_trend)
    )
    stage2_ready = (
        hot <= int(high_heat_reentry_stage2_max_hot_count)
        and valuation <= float(high_heat_reentry_stage2_max_valuation_trend)
    )

    if stage1_ready:
        context.high_heat_reentry_confirm_count += 1
    else:
        context.high_heat_reentry_confirm_count = 0

    if context.high_heat_reentry_confirm_count >= int(high_heat_reentry_stage1_confirm_weeks):
        context.high_heat_reentry_stage = max(int(context.high_heat_reentry_stage), 1)
        context.high_heat_reentry_hold_left = int(high_heat_reentry_hold_weeks)

    if stage2_ready and context.high_heat_reentry_confirm_count >= int(high_heat_reentry_stage2_confirm_weeks):
        context.high_heat_reentry_stage = 2
        context.high_heat_reentry_hold_left = int(high_heat_reentry_hold_weeks)

    if context.high_heat_reentry_hold_left > 0:
        stage_target = float(high_heat_reentry_stage1_target)
        mode = "stage1"
        if int(context.high_heat_reentry_stage) >= 2:
            stage_target = float(high_heat_reentry_stage2_target)
            mode = "stage2"
        target = min(desired, max(current, stage_target))
        deadband_override = target > current + 1e-9 and (
            int(context.high_heat_reentry_stage) > previous_stage or context.high_heat_reentry_hold_left > 0
        )
        if stage1_ready:
            context.high_heat_reentry_hold_left = int(high_heat_reentry_hold_weeks)
        else:
            context.high_heat_reentry_hold_left -= 1
        if context.high_heat_reentry_watch_left > 0:
            context.high_heat_reentry_watch_left -= 1
        return {
            "target": target,
            "mode": mode,
            "confirm_count": context.high_heat_reentry_confirm_count,
            "hold_left": context.high_heat_reentry_hold_left,
            "watch_left": context.high_heat_reentry_watch_left,
            "stage": context.high_heat_reentry_stage,
            "deadband_override": deadband_override,
        }

    context.high_heat_reentry_stage = 0
    if context.high_heat_reentry_watch_left > 0:
        context.high_heat_reentry_watch_left -= 1
    return {
        "target": current,
        "mode": "confirming" if context.high_heat_reentry_confirm_count > 0 else "armed",
        "confirm_count": context.high_heat_reentry_confirm_count,
        "hold_left": context.high_heat_reentry_hold_left,
        "watch_left": context.high_heat_reentry_watch_left,
        "stage": context.high_heat_reentry_stage,
        "deadband_override": False,
    }
