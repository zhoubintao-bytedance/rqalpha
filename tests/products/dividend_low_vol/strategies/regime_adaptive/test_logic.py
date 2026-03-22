import numpy as np
import pytest
from types import SimpleNamespace

from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.allocation import combine_targets
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.confidence_guard import (
    apply_cheap_trend_cap,
    confidence_adjusted_target,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.heat_guard import (
    apply_tactical_heat_ceiling,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.market_regime import (
    compute_market_state,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.allocation import (
    trend_support_target,
    valuation_target_from_percentile,
)
from skyeye.products.dividend_low_vol.strategies.regime_adaptive.logic.reentry_guard import (
    apply_high_heat_reentry_state,
    apply_high_reentry_guard,
    apply_low_percentile_add_guard,
    apply_low_percentile_recovery_state,
)


def test_compute_market_state_recognizes_strong_uptrend():
    prices = np.linspace(1.0, 2.2, 140)

    state = compute_market_state(
        close_history=prices,
        short_window=20,
        mid_window=60,
        long_window=120,
        breakout_window=20,
        core_floor_min=0.18,
        core_floor_max=0.92,
    )

    assert state["mode"] == "bull"
    assert state["trend_strength"] > 0.75
    assert state["core_floor"] > 0.80


def test_valuation_target_uses_more_defensive_high_percentile_curve():
    assert valuation_target_from_percentile(0.92, 1.0) == pytest.approx(0.3222, rel=1e-3)


def test_trend_support_stays_moderate_in_strong_expensive_market():
    support = trend_support_target(
        percentile_ref=0.91,
        trend_strength=0.79,
        drawdown=0.0,
        hot_count=2,
        trend_support_threshold=0.74,
        trend_support_min_percentile=0.80,
        trend_support_max_percentile=0.96,
        trend_support_shallow_drawdown=-0.015,
        trend_support_max_hot_count=2,
        trend_support_min_target=0.36,
        trend_support_max_target=0.46,
        trend_support_extreme_percentile=0.90,
        trend_support_extreme_max_target=0.38,
    )

    assert support["mode"] == "trend_support"
    assert support["target"] == pytest.approx(0.3663, rel=1e-3)


def test_combine_targets_keeps_only_small_support_floor():
    result = combine_targets(
        valuation_target=0.35,
        support_target=0.38,
    )

    assert result["mode"] == "trend_support"
    assert result["active_floor"] == pytest.approx(0.38)
    assert result["target"] == pytest.approx(0.38)


def test_low_confidence_only_allows_adding_back_to_core_floor():
    adjusted = confidence_adjusted_target(
        current_target=0.40,
        desired_target=0.88,
        confidence="low",
        core_floor=0.65,
    )

    assert adjusted == pytest.approx(0.65)


def test_cheap_entry_cap_limits_low_percentile_adds_in_weak_trend():
    capped = apply_cheap_trend_cap(
        desired_target=0.88,
        percentile_ref=0.22,
        trend_strength=0.38,
        cheap_entry_cap_percentile=0.35,
        cheap_entry_cap_trend_threshold=0.45,
        cheap_entry_cap_target=0.72,
    )

    assert capped["mode"] == "cheap_trend_cap"
    assert capped["target"] == pytest.approx(0.72)


def test_high_reentry_guard_waits_for_confirmation_in_expensive_zone():
    context = SimpleNamespace(high_reentry_clear_count=0)

    waiting = apply_high_reentry_guard(
        context=context,
        current_target=0.20,
        desired_target=0.33,
        percentile_ref=0.91,
        valuation_trend=0.025,
        hot_count=1,
        heat_mode="none",
        high_reentry_guard_percentile=0.88,
        high_reentry_confirm_weeks=2,
        high_reentry_confirm_trend=-0.01,
    )

    assert waiting["mode"] == "waiting"
    assert waiting["target"] == pytest.approx(0.20)
    assert waiting["clear_count"] == 0

    first_clear = apply_high_reentry_guard(
        context=context,
        current_target=0.20,
        desired_target=0.33,
        percentile_ref=0.91,
        valuation_trend=-0.02,
        hot_count=1,
        heat_mode="none",
        high_reentry_guard_percentile=0.88,
        high_reentry_confirm_weeks=2,
        high_reentry_confirm_trend=-0.01,
    )
    second_clear = apply_high_reentry_guard(
        context=context,
        current_target=0.20,
        desired_target=0.33,
        percentile_ref=0.91,
        valuation_trend=-0.03,
        hot_count=0,
        heat_mode="none",
        high_reentry_guard_percentile=0.88,
        high_reentry_confirm_weeks=2,
        high_reentry_confirm_trend=-0.01,
    )

    assert first_clear["mode"] == "waiting"
    assert first_clear["clear_count"] == 1
    assert second_clear["mode"] == "confirmed"
    assert second_clear["target"] == pytest.approx(0.33)
    assert second_clear["clear_count"] == 2


def test_low_percentile_add_guard_blocks_top_up_while_signal_still_deteriorates():
    guarded = apply_low_percentile_add_guard(
        current_target=0.78,
        desired_target=0.90,
        percentile_ref=0.19,
        valuation_trend=-0.20,
        trend_strength=0.47,
        low_entry_guard_percentile=0.30,
        low_entry_guard_current_target=0.70,
        low_entry_guard_max_valuation_trend=-0.03,
        low_entry_guard_max_trend_strength=0.55,
    )

    assert guarded["mode"] == "waiting"
    assert guarded["target"] == pytest.approx(0.78)


def test_low_percentile_recovery_state_releases_after_two_confirmed_weeks():
    context = SimpleNamespace(low_recovery_confirm_count=0, low_recovery_stage=0, low_recovery_hold_left=0)

    first = apply_low_percentile_recovery_state(
        context=context,
        current_target=0.72,
        guarded_target=0.72,
        recovery_source_target=0.86,
        percentile_ref=0.26,
        valuation_trend=0.04,
        trend_strength=0.32,
        hot_count=0,
        low_recovery_arm_percentile=0.35,
        low_recovery_reset_percentile=0.45,
        low_recovery_min_current_target=0.66,
        low_recovery_confirm_weeks=2,
        low_recovery_confirm_valuation_trend=0.03,
        low_recovery_confirm_trend_strength=0.30,
        low_recovery_stage1_target=0.76,
        low_recovery_stage2_target=0.80,
        low_recovery_stage2_confirm_weeks=3,
        low_recovery_stage2_confirm_valuation_trend=0.05,
        low_recovery_stage2_confirm_trend_strength=0.36,
        low_recovery_hold_weeks=2,
        low_recovery_fail_valuation_trend=-0.08,
        low_recovery_max_hot_count=1,
    )
    second = apply_low_percentile_recovery_state(
        context=context,
        current_target=0.72,
        guarded_target=0.72,
        recovery_source_target=0.86,
        percentile_ref=0.24,
        valuation_trend=0.05,
        trend_strength=0.31,
        hot_count=0,
        low_recovery_arm_percentile=0.35,
        low_recovery_reset_percentile=0.45,
        low_recovery_min_current_target=0.66,
        low_recovery_confirm_weeks=2,
        low_recovery_confirm_valuation_trend=0.03,
        low_recovery_confirm_trend_strength=0.30,
        low_recovery_stage1_target=0.76,
        low_recovery_stage2_target=0.80,
        low_recovery_stage2_confirm_weeks=3,
        low_recovery_stage2_confirm_valuation_trend=0.05,
        low_recovery_stage2_confirm_trend_strength=0.36,
        low_recovery_hold_weeks=2,
        low_recovery_fail_valuation_trend=-0.08,
        low_recovery_max_hot_count=1,
    )

    assert first["mode"] == "confirming"
    assert first["target"] == pytest.approx(0.72)
    assert first["confirm_count"] == 1
    assert first["deadband_override"] is False
    assert second["mode"] == "stage1"
    assert second["target"] == pytest.approx(0.76)
    assert second["confirm_count"] == 2
    assert second["hold_left"] == 2
    assert second["stage"] == 1
    assert second["deadband_override"] is True


def test_low_percentile_recovery_state_can_upgrade_to_stage2():
    context = SimpleNamespace(low_recovery_confirm_count=2, low_recovery_stage=1, low_recovery_hold_left=2)

    result = apply_low_percentile_recovery_state(
        context=context,
        current_target=0.76,
        guarded_target=0.72,
        recovery_source_target=0.86,
        percentile_ref=0.22,
        valuation_trend=0.06,
        trend_strength=0.37,
        hot_count=0,
        low_recovery_arm_percentile=0.35,
        low_recovery_reset_percentile=0.45,
        low_recovery_min_current_target=0.66,
        low_recovery_confirm_weeks=2,
        low_recovery_confirm_valuation_trend=0.03,
        low_recovery_confirm_trend_strength=0.30,
        low_recovery_stage1_target=0.76,
        low_recovery_stage2_target=0.80,
        low_recovery_stage2_confirm_weeks=3,
        low_recovery_stage2_confirm_valuation_trend=0.05,
        low_recovery_stage2_confirm_trend_strength=0.36,
        low_recovery_hold_weeks=2,
        low_recovery_fail_valuation_trend=-0.08,
        low_recovery_max_hot_count=1,
    )

    assert result["mode"] == "stage2"
    assert result["target"] == pytest.approx(0.80)
    assert result["confirm_count"] == 3
    assert result["stage"] == 2
    assert result["deadband_override"] is True


def test_low_percentile_recovery_state_resets_on_deterioration():
    context = SimpleNamespace(low_recovery_confirm_count=2, low_recovery_stage=2, low_recovery_hold_left=2)

    result = apply_low_percentile_recovery_state(
        context=context,
        current_target=0.80,
        guarded_target=0.72,
        recovery_source_target=0.86,
        percentile_ref=0.22,
        valuation_trend=-0.10,
        trend_strength=0.34,
        hot_count=0,
        low_recovery_arm_percentile=0.35,
        low_recovery_reset_percentile=0.45,
        low_recovery_min_current_target=0.66,
        low_recovery_confirm_weeks=2,
        low_recovery_confirm_valuation_trend=0.03,
        low_recovery_confirm_trend_strength=0.30,
        low_recovery_stage1_target=0.76,
        low_recovery_stage2_target=0.80,
        low_recovery_stage2_confirm_weeks=3,
        low_recovery_stage2_confirm_valuation_trend=0.05,
        low_recovery_stage2_confirm_trend_strength=0.36,
        low_recovery_hold_weeks=2,
        low_recovery_fail_valuation_trend=-0.08,
        low_recovery_max_hot_count=1,
    )

    assert result["mode"] == "armed"
    assert result["target"] == pytest.approx(0.72)
    assert result["hold_left"] == 0
    assert result["deadband_override"] is False


def test_high_heat_reentry_state_steps_back_in_after_cooling():
    context = SimpleNamespace(
        high_heat_reentry_watch_left=0,
        high_heat_reentry_confirm_count=0,
        high_heat_reentry_stage=0,
        high_heat_reentry_hold_left=0,
    )

    triggered = apply_high_heat_reentry_state(
        context=context,
        current_target=0.26,
        desired_target=0.18,
        percentile_ref=0.97,
        valuation_trend=0.01,
        hot_count=3,
        heat_mode="extreme_strong",
        high_heat_reentry_arm_percentile=0.86,
        high_heat_reentry_reset_percentile=0.78,
        high_heat_reentry_watch_weeks=4,
        high_heat_reentry_stage1_confirm_weeks=1,
        high_heat_reentry_stage2_confirm_weeks=2,
        high_heat_reentry_stage1_max_hot_count=1,
        high_heat_reentry_stage2_max_hot_count=1,
        high_heat_reentry_stage1_max_valuation_trend=0.06,
        high_heat_reentry_stage2_max_valuation_trend=-0.01,
        high_heat_reentry_stage1_target=0.20,
        high_heat_reentry_stage2_target=0.24,
        high_heat_reentry_hold_weeks=2,
    )
    stage1 = apply_high_heat_reentry_state(
        context=context,
        current_target=0.18,
        desired_target=0.26,
        percentile_ref=0.95,
        valuation_trend=-0.02,
        hot_count=1,
        heat_mode="cooldown",
        high_heat_reentry_arm_percentile=0.86,
        high_heat_reentry_reset_percentile=0.78,
        high_heat_reentry_watch_weeks=4,
        high_heat_reentry_stage1_confirm_weeks=1,
        high_heat_reentry_stage2_confirm_weeks=2,
        high_heat_reentry_stage1_max_hot_count=1,
        high_heat_reentry_stage2_max_hot_count=1,
        high_heat_reentry_stage1_max_valuation_trend=0.06,
        high_heat_reentry_stage2_max_valuation_trend=-0.01,
        high_heat_reentry_stage1_target=0.20,
        high_heat_reentry_stage2_target=0.24,
        high_heat_reentry_hold_weeks=2,
    )
    stage2 = apply_high_heat_reentry_state(
        context=context,
        current_target=0.20,
        desired_target=0.30,
        percentile_ref=0.90,
        valuation_trend=-0.02,
        hot_count=1,
        heat_mode="none",
        high_heat_reentry_arm_percentile=0.86,
        high_heat_reentry_reset_percentile=0.78,
        high_heat_reentry_watch_weeks=4,
        high_heat_reentry_stage1_confirm_weeks=1,
        high_heat_reentry_stage2_confirm_weeks=2,
        high_heat_reentry_stage1_max_hot_count=1,
        high_heat_reentry_stage2_max_hot_count=1,
        high_heat_reentry_stage1_max_valuation_trend=0.06,
        high_heat_reentry_stage2_max_valuation_trend=-0.01,
        high_heat_reentry_stage1_target=0.20,
        high_heat_reentry_stage2_target=0.24,
        high_heat_reentry_hold_weeks=2,
    )

    assert triggered["mode"] == "triggered"
    assert triggered["deadband_override"] is False
    assert stage1["mode"] == "stage1"
    assert stage1["target"] == pytest.approx(0.20)
    assert stage1["deadband_override"] is True
    assert stage2["mode"] == "stage2"
    assert stage2["target"] == pytest.approx(0.24)
    assert stage2["deadband_override"] is True


def test_extreme_heat_keeps_cooldown_and_strong_trend_cap_moderate():
    context = SimpleNamespace(
        heat_cooldown_weeks=2,
        heat_cooldown_left=0,
        heat_cooldown_cap=None,
    )
    result = apply_tactical_heat_ceiling(
        context=context,
        base_target=0.95,
        support_floor=0.0,
        percentile_ref=0.92,
        hot_signals={"hot_count": 3},
        trend_strength=0.80,
        drawdown=0.0,
        heat_trigger_percentile=0.80,
        heat_extreme_percentile=0.90,
        heat_cooldown_min_percentile=0.80,
        heat_cooldown_release_buffer=0.08,
        heat_constructive_trend_threshold=0.62,
        heat_strong_trend_threshold=0.76,
        heat_shallow_drawdown=-0.015,
        heat_cap_hot_weak=0.10,
        heat_cap_hot_constructive=0.18,
        heat_cap_hot_strong=0.32,
        heat_cap_extreme_weak=0.05,
        heat_cap_extreme_constructive=0.12,
        heat_cap_extreme_strong=0.24,
    )

    assert result["mode"] == "extreme_strong"
    assert result["target"] == pytest.approx(0.24)
    assert result["cooldown_left"] == 2

    cooldown = apply_tactical_heat_ceiling(
        context=context,
        base_target=0.50,
        support_floor=0.0,
        percentile_ref=0.88,
        hot_signals={"hot_count": 1},
        trend_strength=0.60,
        drawdown=-0.01,
        heat_trigger_percentile=0.80,
        heat_extreme_percentile=0.90,
        heat_cooldown_min_percentile=0.80,
        heat_cooldown_release_buffer=0.08,
        heat_constructive_trend_threshold=0.62,
        heat_strong_trend_threshold=0.76,
        heat_shallow_drawdown=-0.015,
        heat_cap_hot_weak=0.10,
        heat_cap_hot_constructive=0.18,
        heat_cap_hot_strong=0.32,
        heat_cap_extreme_weak=0.05,
        heat_cap_extreme_constructive=0.12,
        heat_cap_extreme_strong=0.24,
    )

    assert cooldown["mode"] == "cooldown"
    assert cooldown["target"] == pytest.approx(0.32)
    assert cooldown["cooldown_left"] == 1
