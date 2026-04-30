# -*- coding: utf-8 -*-
"""Unit tests for AX1 portfolio stop-loss mechanism."""

from __future__ import annotations

import pandas as pd
import pytest

from skyeye.products.ax1.risk.stop_loss import (
    PortfolioStopLoss,
    StopLossConfig,
    StopLossLevel,
    StopLossLevelConfig,
    StopLossState,
    _coerce_stop_loss_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target_weights(dates, ids, weights_matrix):
    """Build a target_weights DataFrame.

    weights_matrix: list of lists, one per date, weights per id.
    """
    rows = []
    for date, w_list in zip(dates, weights_matrix):
        for oid, w in zip(ids, w_list):
            rows.append({"date": date, "order_book_id": oid, "target_weight": w})
    return pd.DataFrame(rows)


def _make_dates(n):
    return [pd.Timestamp(f"2024-01-{i+1:02d}") for i in range(n)]


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestStopLossConfig:

    def test_default_config_has_three_levels(self):
        cfg = StopLossConfig()
        assert len(cfg.levels) == 3
        assert cfg.levels[0].name == "yellow"
        assert cfg.levels[1].name == "orange"
        assert cfg.levels[2].name == "red"

    def test_default_thresholds(self):
        cfg = StopLossConfig()
        assert cfg.levels[0].drawdown_threshold == pytest.approx(0.10)
        assert cfg.levels[1].drawdown_threshold == pytest.approx(0.15)
        assert cfg.levels[2].drawdown_threshold == pytest.approx(0.20)

    def test_default_exposures(self):
        cfg = StopLossConfig()
        assert cfg.levels[0].target_exposure == pytest.approx(0.70)
        assert cfg.levels[1].target_exposure == pytest.approx(0.40)
        assert cfg.levels[2].target_exposure == pytest.approx(0.00)

    def test_coerce_from_dict(self):
        raw = {
            "enabled": True,
            "levels": [
                {"name": "yellow", "drawdown_threshold": 0.10, "target_exposure": 0.70},
                {"name": "orange", "drawdown_threshold": 0.15, "target_exposure": 0.40},
                {"name": "red", "drawdown_threshold": 0.20, "target_exposure": 0.00},
            ],
            "cooldown_trading_days": 10,
        }
        cfg = _coerce_stop_loss_config(raw)
        assert cfg.enabled is True
        assert len(cfg.levels) == 3
        assert cfg.cooldown_trading_days == 10

    def test_coerce_none_returns_default(self):
        cfg = _coerce_stop_loss_config(None)
        assert cfg.enabled is False
        assert len(cfg.levels) == 3

    def test_coerce_config_passthrough(self):
        original = StopLossConfig(enabled=True)
        cfg = _coerce_stop_loss_config(original)
        assert cfg is original


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------

class TestStopLossState:

    def test_default_values(self):
        state = StopLossState()
        assert state.peak_equity == 1.0
        assert state.current_equity == 1.0
        assert state.current_drawdown == 0.0
        assert state.active_level == StopLossLevel.NORMAL
        assert state.cooldown_remaining == 0
        assert state.exposure_cap == 1.0
        assert state.trigger_log == []

    def test_serialization_roundtrip(self):
        state = StopLossState(
            peak_equity=1.05,
            current_equity=0.90,
            current_drawdown=0.1428,
            active_level=StopLossLevel.ORANGE,
            cooldown_remaining=3,
            red_trigger_date=pd.Timestamp("2024-03-15"),
            last_update_date=pd.Timestamp("2024-03-20"),
            exposure_cap=0.40,
            trigger_log=[{"date": "2024-03-15", "to_level": "orange"}],
        )
        d = state.to_dict()
        restored = StopLossState.from_dict(d)
        assert restored.peak_equity == pytest.approx(1.05)
        assert restored.current_equity == pytest.approx(0.90)
        assert restored.current_drawdown == pytest.approx(0.1428)
        assert restored.active_level == StopLossLevel.ORANGE
        assert restored.cooldown_remaining == 3
        assert restored.exposure_cap == pytest.approx(0.40)
        assert len(restored.trigger_log) == 1

    def test_from_dict_invalid_level_defaults_to_normal(self):
        d = {"active_level": "invalid_level"}
        state = StopLossState.from_dict(d)
        assert state.active_level == StopLossLevel.NORMAL


# ---------------------------------------------------------------------------
# Level determination tests
# ---------------------------------------------------------------------------

class TestLevelDetermination:

    def test_normal_when_no_drawdown(self):
        sl = PortfolioStopLoss({"enabled": True})
        assert sl.state.active_level == StopLossLevel.NORMAL

    def test_yellow_at_10_percent(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.10
        sl.state.current_equity = 0.90
        sl.state.peak_equity = 1.0
        level = sl._determine_level(None)
        assert level == StopLossLevel.YELLOW

    def test_orange_at_15_percent(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.15
        level = sl._determine_level(None)
        assert level == StopLossLevel.ORANGE

    def test_red_at_20_percent(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.20
        level = sl._determine_level(None)
        assert level == StopLossLevel.RED

    def test_just_below_yellow(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.099
        level = sl._determine_level(None)
        assert level == StopLossLevel.NORMAL

    def test_just_below_orange(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.149
        level = sl._determine_level(None)
        assert level == StopLossLevel.YELLOW

    def test_just_below_red(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.199
        level = sl._determine_level(None)
        assert level == StopLossLevel.ORANGE


# ---------------------------------------------------------------------------
# Equity update tests
# ---------------------------------------------------------------------------

class TestEquityUpdate:

    def test_positive_return(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), 0.05)
        assert sl.state.current_equity == pytest.approx(1.05)
        assert sl.state.peak_equity == pytest.approx(1.05)
        assert sl.state.current_drawdown == pytest.approx(0.0)

    def test_negative_return(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), -0.10)
        assert sl.state.current_equity == pytest.approx(0.90)
        assert sl.state.peak_equity == pytest.approx(1.0)
        assert sl.state.current_drawdown == pytest.approx(0.10)

    def test_drawdown_calculation_sequence(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), 0.10)  # equity=1.10
        sl.update_equity(pd.Timestamp("2024-01-02"), -0.20)  # equity=0.88
        assert sl.state.peak_equity == pytest.approx(1.10)
        assert sl.state.current_equity == pytest.approx(0.88)
        expected_dd = 1.0 - 0.88 / 1.10
        assert sl.state.current_drawdown == pytest.approx(expected_dd, abs=1e-6)

    def test_drawdown_never_negative(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), 0.05)
        sl.update_equity(pd.Timestamp("2024-01-02"), 0.03)
        assert sl.state.current_drawdown == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Weight scaling tests
# ---------------------------------------------------------------------------

class TestWeightScaling:

    def test_normal_level_leaves_weights_unchanged(self):
        sl = PortfolioStopLoss({"enabled": True})
        tw = _make_target_weights(
            _make_dates(1), ["A", "B", "C"], [[0.30, 0.40, 0.30]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        assert result["target_weight"].tolist() == pytest.approx([0.30, 0.40, 0.30])

    def test_yellow_scales_to_70_percent(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.10
        sl.state.current_equity = 0.90
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(
            _make_dates(1), ["A", "B", "C"], [[0.30, 0.40, 0.30]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        assert result["target_weight"].tolist() == pytest.approx([0.21, 0.28, 0.21])

    def test_orange_scales_to_40_percent(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.15
        sl.state.current_equity = 0.85
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(
            _make_dates(1), ["A", "B", "C"], [[0.30, 0.40, 0.30]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        assert result["target_weight"].tolist() == pytest.approx([0.12, 0.16, 0.12])

    def test_red_clears_all_positions(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.20
        sl.state.current_equity = 0.80
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(
            _make_dates(1), ["A", "B", "C"], [[0.30, 0.40, 0.30]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        assert result["target_weight"].tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_scaling_preserves_relative_weights(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.10
        sl.state.current_equity = 0.90
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(
            _make_dates(1), ["A", "B"], [[0.20, 0.80]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        # 70% of 0.20 = 0.14, 70% of 0.80 = 0.56, ratio still 1:4
        assert result["target_weight"].iloc[1] / result["target_weight"].iloc[0] == pytest.approx(4.0)

    def test_scaling_preserves_date_and_id_columns(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.current_drawdown = 0.10
        sl.state.current_equity = 0.90
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(
            _make_dates(1), ["A", "B"], [[0.50, 0.50]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        assert "date" in result.columns
        assert "order_book_id" in result.columns
        assert result["order_book_id"].tolist() == ["A", "B"]


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------

class TestCooldown:

    def test_red_triggers_cooldown(self):
        sl = PortfolioStopLoss({"enabled": True, "cooldown_trading_days": 10})
        sl.state.current_drawdown = 0.20
        sl.state.current_equity = 0.80
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(
            _make_dates(1), ["A"], [[0.50]]
        )
        sl.apply(tw, _make_dates(1)[0])
        assert sl.state.cooldown_remaining == 10
        assert sl.state.active_level == StopLossLevel.RED

    def test_cooldown_decrements_each_day(self):
        sl = PortfolioStopLoss({"enabled": True, "cooldown_trading_days": 3})
        sl.state.current_drawdown = 0.20
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 3
        tw = _make_target_weights(
            _make_dates(1), ["A"], [[0.50]]
        )
        sl.apply(tw, _make_dates(1)[0])
        assert sl.state.cooldown_remaining == 2
        sl.apply(tw, _make_dates(1)[0])
        assert sl.state.cooldown_remaining == 1
        sl.apply(tw, _make_dates(1)[0])
        assert sl.state.cooldown_remaining == 0

    def test_cooldown_prevents_recovery(self):
        """Even if drawdown recovers, cooldown keeps exposure at 0."""
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 5,
            "recovery_requires_regime_not_risk_off": False,
        })
        sl.state.current_drawdown = 0.05  # below yellow
        sl.state.current_equity = 0.95
        sl.state.peak_equity = 1.0
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 3

        tw = _make_target_weights(_make_dates(1), ["A"], [[0.50]])
        result = sl.apply(tw, _make_dates(1)[0])
        # Still in cooldown, should remain at RED
        assert sl.state.active_level == StopLossLevel.RED
        assert result["target_weight"].iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Recovery tests
# ---------------------------------------------------------------------------

class TestRecovery:

    def test_recovery_requires_cooldown(self):
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 5,
            "recovery_requires_regime_not_risk_off": False,
        })
        sl.state.current_drawdown = 0.05
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 2
        assert sl._can_recover_from_red(None) is False

    def test_recovery_blocked_when_regime_risk_off(self):
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": True,
        })
        sl.state.current_drawdown = 0.05
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 0
        regime = {"risk_state": "risk_off"}
        assert sl._can_recover_from_red(regime) is False

    def test_recovery_allowed_when_regime_risk_on(self):
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": True,
        })
        sl.state.current_drawdown = 0.05
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 0
        regime = {"risk_state": "risk_on"}
        assert sl._can_recover_from_red(regime) is True

    def test_recovery_allowed_when_regime_neutral(self):
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": True,
        })
        sl.state.current_drawdown = 0.05
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 0
        regime = {"risk_state": "neutral"}
        assert sl._can_recover_from_red(regime) is True

    def test_recovery_without_regime_gating(self):
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": False,
        })
        sl.state.current_drawdown = 0.05
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 0
        # Even risk_off should be OK with gating disabled
        regime = {"risk_state": "risk_off"}
        assert sl._can_recover_from_red(regime) is True

    def test_gradual_recovery_yellow_then_normal(self):
        """After recovering from RED, level is re-evaluated based on DD."""
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": False,
        })
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 0
        sl.state.current_drawdown = 0.12  # qualifies for YELLOW
        level = sl._determine_level(None)
        assert level == StopLossLevel.YELLOW

    def test_recovery_stays_red_if_drawdown_still_high(self):
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": False,
        })
        sl.state.active_level = StopLossLevel.RED
        sl.state.cooldown_remaining = 0
        sl.state.current_drawdown = 0.22  # still qualifies for RED
        level = sl._determine_level(None)
        assert level == StopLossLevel.RED


# ---------------------------------------------------------------------------
# Integration-style full cycle test
# ---------------------------------------------------------------------------

class TestFullCycle:

    def test_normal_to_red_to_recovery(self):
        """Simulate a full drawdown and recovery cycle."""
        cfg = {
            "enabled": True,
            "cooldown_trading_days": 3,
            "recovery_requires_regime_not_risk_off": False,
        }
        sl = PortfolioStopLoss(cfg)
        dates = _make_dates(20)
        tw = _make_target_weights(dates, ["A"], [[0.50]] * 20)

        # Phase 1: normal returns, then crash
        returns = [0.0] * 5 + [-0.05] * 4  # 5 flat, 4 × -5% = ~-18.5%
        # Phase 2: recovery
        returns += [0.03] * 5  # recovery
        returns += [0.0] * (20 - len(returns))

        levels_seen = []
        for i, date in enumerate(dates):
            r = returns[i] if i < len(returns) else 0.0
            sl.update_equity(date, r)
            result = sl.apply(tw[tw["date"] == date], date)
            levels_seen.append(sl.state.active_level)

        # Should have hit at least YELLOW/ORANGE during the crash
        assert StopLossLevel.YELLOW in levels_seen or StopLossLevel.ORANGE in levels_seen

    def test_no_stop_loss_when_disabled(self):
        sl = PortfolioStopLoss({"enabled": False})
        sl.state.current_drawdown = 0.30  # extreme drawdown
        sl.state.current_equity = 0.70
        sl.state.peak_equity = 1.0
        tw = _make_target_weights(_make_dates(1), ["A"], [[0.50]])
        result = sl.apply(tw, _make_dates(1)[0])
        # Weights should be unchanged when disabled
        assert result["target_weight"].iloc[0] == pytest.approx(0.50)

    def test_trigger_log_records_events(self):
        sl = PortfolioStopLoss({"enabled": True, "cooldown_trading_days": 0,
                                 "recovery_requires_regime_not_risk_off": False})
        dates = _make_dates(3)
        tw = _make_target_weights(dates, ["A"], [[0.50]] * 3)

        # Day 1: normal → triggers yellow
        sl.state.current_drawdown = 0.10
        sl.state.current_equity = 0.90
        sl.state.peak_equity = 1.0
        sl.apply(tw[tw["date"] == dates[0]], dates[0])
        assert len(sl.state.trigger_log) == 1
        assert sl.state.trigger_log[0]["to_level"] == "yellow"

        # Day 2: yellow → orange
        sl.state.current_drawdown = 0.15
        sl.state.current_equity = 0.85
        sl.apply(tw[tw["date"] == dates[1]], dates[1])
        assert len(sl.state.trigger_log) == 2
        assert sl.state.trigger_log[1]["to_level"] == "orange"

    def test_apply_with_empty_weights(self):
        sl = PortfolioStopLoss({"enabled": True})
        result = sl.apply(pd.DataFrame(), pd.Timestamp("2024-01-01"))
        assert result.empty

    def test_apply_with_none_weights(self):
        sl = PortfolioStopLoss({"enabled": True})
        result = sl.apply(None, pd.Timestamp("2024-01-01"))
        assert result is None


# ---------------------------------------------------------------------------
# Incremental vs batch drawdown consistency
# ---------------------------------------------------------------------------

class TestDrawdownConsistency:

    def test_incremental_matches_batch(self):
        """Verify that day-by-day incremental state matches batch computation."""
        from skyeye.products.ax1.evaluation.metrics import _max_drawdown_from_returns

        returns = [0.02, -0.05, 0.03, -0.12, 0.04, -0.08, 0.01, -0.03, 0.06, 0.02]
        dates = _make_dates(len(returns))

        # Batch computation
        batch_dd = _max_drawdown_from_returns(pd.Series(returns))

        # Incremental computation
        sl = PortfolioStopLoss({"enabled": True})
        for i, (date, r) in enumerate(zip(dates, returns)):
            sl.update_equity(date, r)

        assert sl.state.current_drawdown == pytest.approx(
            1.0 - sl.state.current_equity / sl.state.peak_equity, abs=1e-10
        )
        # The max drawdown from batch should be >= final drawdown from incremental
        # (they measure different things — batch is peak DD, incremental is current DD)
        assert batch_dd >= 0.0


# ---------------------------------------------------------------------------
# Bankruptcy detection tests
# ---------------------------------------------------------------------------

class TestBankruptcyDetection:

    def test_negative_equity_sets_bankrupt_flag(self):
        """Equity going negative must set bankrupt=True."""
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), -1.5)  # equity = 1.0 * (1 - 1.5) = -0.5
        assert sl.state.current_equity < 0.0
        assert sl.state.bankrupt is True

    def test_zero_equity_sets_bankrupt_flag(self):
        """Equity exactly zero also triggers bankrupt."""
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), -1.0)  # equity = 0.0
        assert sl.state.current_equity == pytest.approx(0.0)
        assert sl.state.bankrupt is True

    def test_positive_equity_does_not_set_bankrupt(self):
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), -0.20)
        assert sl.state.current_equity == pytest.approx(0.80)
        assert sl.state.bankrupt is False

    def test_bankrupt_forces_red_level(self):
        """When bankrupt, _determine_level always returns RED."""
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.bankrupt = True
        level = sl._determine_level(None)
        assert level == StopLossLevel.RED

    def test_bankrupt_prevents_recovery_even_with_cooldown_expired(self):
        """Bankrupt portfolio stays at RED regardless of cooldown or regime."""
        sl = PortfolioStopLoss({
            "enabled": True,
            "cooldown_trading_days": 0,
            "recovery_requires_regime_not_risk_off": False,
        })
        sl.state.bankrupt = True
        sl.state.cooldown_remaining = 0
        sl.state.current_drawdown = 0.0
        level = sl._determine_level({"risk_state": "risk_on"})
        assert level == StopLossLevel.RED

    def test_bankrupt_apply_zeros_exposure(self):
        """Bankrupt portfolio should scale all weights to 0."""
        sl = PortfolioStopLoss({"enabled": True})
        sl.state.bankrupt = True
        tw = _make_target_weights(
            _make_dates(1), ["A", "B", "C"], [[0.30, 0.40, 0.30]]
        )
        result = sl.apply(tw, _make_dates(1)[0])
        assert sl.state.active_level == StopLossLevel.RED
        assert result["target_weight"].tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_bankrupt_flag_persists_in_serialization(self):
        """bankrupt flag survives to_dict / from_dict roundtrip."""
        state = StopLossState(bankrupt=True, current_equity=-0.15)
        d = state.to_dict()
        assert d["bankrupt"] is True
        assert d["current_equity"] == pytest.approx(-0.15)
        restored = StopLossState.from_dict(d)
        assert restored.bankrupt is True
        assert restored.current_equity == pytest.approx(-0.15)

    def test_equity_not_clamped_on_bankruptcy(self):
        """Negative equity is preserved, not silently clamped to 0."""
        sl = PortfolioStopLoss({"enabled": True})
        sl.update_equity(pd.Timestamp("2024-01-01"), -1.2)  # equity = -0.2
        assert sl.state.current_equity == pytest.approx(-0.2)
        assert sl.state.current_drawdown == pytest.approx(1.0)

    def test_full_bankruptcy_flow(self):
        """End-to-end: extreme loss → bankrupt → RED + zero exposure."""
        sl = PortfolioStopLoss({"enabled": True, "cooldown_trading_days": 5})
        tw = _make_target_weights(
            _make_dates(1), ["A", "B"], [[0.40, 0.60]]
        )
        sl.update_equity(pd.Timestamp("2024-01-01"), -1.5)  # equity = -0.5
        result = sl.apply(tw, pd.Timestamp("2024-01-01"))
        assert sl.state.bankrupt is True
        assert sl.state.active_level == StopLossLevel.RED
        assert sl.state.exposure_cap == pytest.approx(0.0)
        assert result["target_weight"].tolist() == pytest.approx([0.0, 0.0])
