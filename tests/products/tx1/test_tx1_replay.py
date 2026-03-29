import pytest

from skyeye.products.tx1.strategies.rolling_score.replay import (
    build_execution_universe,
    compute_turnover_ratio,
    sanitize_target_weights,
    smooth_target_weights,
)


def test_sanitize_target_weights_drops_invalid_values_and_applies_cap():
    sanitized = sanitize_target_weights(
        {
            "000001.XSHE": 0.15,
            "000002.XSHE": 0.05,
            "000003.XSHE": 0.0,
            "000004.XSHE": -0.01,
            "000005.XSHE": None,
        },
        single_stock_cap=0.10,
    )

    assert sanitized == {
        "000001.XSHE": 0.10,
        "000002.XSHE": 0.05,
    }


def test_compute_turnover_ratio_uses_union_of_current_and_target_weights():
    turnover = compute_turnover_ratio(
        {"000001.XSHE": 0.40, "000002.XSHE": 0.60},
        {"000002.XSHE": 0.20, "000003.XSHE": 0.80},
    )

    assert turnover == 0.80


def test_build_execution_universe_merges_holdings_and_targets():
    universe = build_execution_universe(
        {"000003.XSHE": 0.2, "000001.XSHE": 0.3},
        {"000002.XSHE": 0.5, "000001.XSHE": 0.4},
    )

    assert universe == ["000001.XSHE", "000002.XSHE", "000003.XSHE"]


# --- smooth_target_weights tests ---


class TestSmoothTargetWeights:
    def test_first_call_without_state_returns_raw_weights(self):
        raw = {"A": 0.5, "B": 0.5}
        smoothed, state = smooth_target_weights(raw, None, halflife=5)

        assert smoothed == pytest.approx(raw)
        assert sum(smoothed.values()) == pytest.approx(1.0)

    def test_first_call_with_empty_state_returns_raw_weights(self):
        raw = {"A": 0.5, "B": 0.5}
        smoothed, state = smooth_target_weights(raw, {}, halflife=5)

        assert smoothed == pytest.approx(raw)

    def test_basic_ema_blending(self):
        ema_state = {"A": 0.6, "B": 0.4}
        raw = {"A": 0.4, "B": 0.6}
        halflife = 5
        alpha = 2.0 / (halflife + 1.0)  # 1/3

        smoothed, new_state = smooth_target_weights(raw, ema_state, halflife=halflife)

        expected_a = alpha * 0.4 + (1 - alpha) * 0.6  # 0.4667
        expected_b = alpha * 0.6 + (1 - alpha) * 0.4  # 0.5333
        total = expected_a + expected_b
        assert smoothed["A"] == pytest.approx(expected_a / total)
        assert smoothed["B"] == pytest.approx(expected_b / total)
        assert sum(smoothed.values()) == pytest.approx(1.0)

    def test_new_stock_enters_at_raw_weight(self):
        ema_state = {"A": 1.0}
        raw = {"A": 0.5, "B": 0.5}
        halflife = 5
        alpha = 2.0 / (halflife + 1.0)

        smoothed, _ = smooth_target_weights(raw, ema_state, halflife=halflife)

        # B is new, so its EMA starts at raw value
        expected_b_raw = 0.5
        expected_a_raw = alpha * 0.5 + (1 - alpha) * 1.0
        total = expected_a_raw + expected_b_raw
        assert smoothed["B"] == pytest.approx(expected_b_raw / total)
        assert sum(smoothed.values()) == pytest.approx(1.0)

    def test_exiting_stock_decays_toward_zero(self):
        ema_state = {"A": 0.5, "B": 0.5}
        raw = {"A": 1.0}  # B dropped
        halflife = 5
        alpha = 2.0 / (halflife + 1.0)

        smoothed, state = smooth_target_weights(
            raw, ema_state, halflife=halflife, min_weight=0.001,
        )

        expected_b = (1 - alpha) * 0.5  # decaying
        assert "B" in smoothed  # still above min_weight
        assert smoothed["B"] < 0.5  # decayed

    def test_exiting_stock_removed_below_min_weight(self):
        ema_state = {"A": 0.5, "B": 0.001}  # B already tiny
        raw = {"A": 1.0}  # B dropped
        halflife = 5

        smoothed, state = smooth_target_weights(
            raw, ema_state, halflife=halflife, min_weight=0.005,
        )

        # B should decay below min_weight and be removed
        assert "B" not in smoothed
        assert smoothed == pytest.approx({"A": 1.0})

    def test_normalization_sums_to_one(self):
        ema_state = {"A": 0.3, "B": 0.3, "C": 0.4}
        raw = {"A": 0.5, "B": 0.3, "D": 0.2}

        smoothed, _ = smooth_target_weights(raw, ema_state, halflife=3)

        assert sum(smoothed.values()) == pytest.approx(1.0)

    def test_ema_state_stores_pre_normalization_values(self):
        raw = {"A": 0.7, "B": 0.3}
        halflife = 5
        alpha = 2.0 / (halflife + 1.0)

        # First call: state = raw values (pre-normalization = post-normalization here)
        _, state1 = smooth_target_weights(raw, None, halflife=halflife)

        # Second call with different raw
        raw2 = {"A": 0.3, "B": 0.7}
        smoothed2, state2 = smooth_target_weights(raw2, state1, halflife=halflife)

        # State should be pre-normalization EMA values
        expected_a_state = alpha * 0.3 + (1 - alpha) * state1["A"]
        expected_b_state = alpha * 0.7 + (1 - alpha) * state1["B"]
        assert state2["A"] == pytest.approx(expected_a_state)
        assert state2["B"] == pytest.approx(expected_b_state)

    def test_empty_raw_weights_returns_empty(self):
        smoothed, state = smooth_target_weights({}, {"A": 0.5}, halflife=5, min_weight=0.005)

        # All stocks decay; if they stay above min_weight they remain
        # With only one step of decay, 0.5 * (1 - 1/3) = 0.333 > 0.005
        assert "A" in state

    def test_all_decay_below_min_returns_empty(self):
        ema_state = {"A": 0.001}  # already tiny
        smoothed, state = smooth_target_weights(
            {}, ema_state, halflife=5, min_weight=0.005,
        )

        assert smoothed == {}
        assert state == {}
