import pandas as pd
import pytest

from skyeye.products.ax1.execution.smoother import ExecutionSmoother


def test_smoother_aggregates_components_filters_min_weight_and_normalizes():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-02", "2024-01-02"]
            ),
            "order_book_id": ["000001.XSHE", "000001.XSHE", "000002.XSHE", "600000.XSHG"],
            "target_weight": [0.35, 0.25, 0.35, 0.05],
            "component": ["base", "overlay", "base", "overlay"],
        }
    )

    smoothed = ExecutionSmoother(min_weight=0.1).smooth(target_weights)

    assert list(smoothed["order_book_id"]) == ["000001.XSHE", "000002.XSHE"]
    assert smoothed["target_weight"].sum() == pytest.approx(1.0)
    assert smoothed.set_index("order_book_id")["target_weight"].to_dict() == {
        "000001.XSHE": pytest.approx(0.6 / 0.95),
        "000002.XSHE": pytest.approx(0.35 / 0.95),
    }
    assert smoothed["component"].eq("smoothed").all()


def test_smoother_keeps_current_weight_when_delta_is_inside_buffer():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "target_weight": [0.32, 0.68],
        }
    )
    current_weights = {"000001.XSHE": 0.30, "000002.XSHE": 0.70}

    smoothed = ExecutionSmoother(buffer_weight=0.05).smooth(target_weights, current_weights)

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    assert weights["000001.XSHE"] == pytest.approx(0.30)
    assert weights["000002.XSHE"] == pytest.approx(0.70)


def test_smoother_accepts_no_trade_buffer_weight_alias():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "target_weight": [0.32, 0.68],
        }
    )
    current_weights = {"000001.XSHE": 0.30, "000002.XSHE": 0.70}

    smoothed = ExecutionSmoother(no_trade_buffer_weight=0.05).smooth(
        target_weights, current_weights
    )

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    assert weights["000001.XSHE"] == pytest.approx(0.30)
    assert weights["000002.XSHE"] == pytest.approx(0.70)


def test_smoother_ignores_max_industry_weight_when_all_unknown():
    """When all ETFs have industry='Unknown', max_industry_weight should be ignored."""
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["ETF_A", "ETF_B", "ETF_C"],
            "target_weight": [0.40, 0.35, 0.25],
            "industry": ["Unknown", "Unknown", "Unknown"],
        }
    )

    # With max_industry_weight=0.20, if applied it would cap portfolio at 20%
    smoothed = ExecutionSmoother(max_industry_weight=0.20).smooth(target_weights)

    # Total weight should be normalized to 1.0, not capped at 0.20
    assert smoothed["target_weight"].sum() == pytest.approx(1.0)

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    # Weights should be allocated proportionally
    assert weights["ETF_A"] > 0.35
    assert weights["ETF_B"] > 0.30


def test_smoother_keeps_current_weight_when_trade_value_is_too_small():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "target_weight": [0.305, 0.695],
        }
    )
    current_weights = pd.DataFrame(
        {
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "current_weight": [0.30, 0.70],
        }
    )

    smoothed = ExecutionSmoother(min_trade_value=10_000, portfolio_value=1_000_000).smooth(
        target_weights, current_weights
    )

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    assert weights["000001.XSHE"] == pytest.approx(0.30)
    assert weights["000002.XSHE"] == pytest.approx(0.70)


def test_smoother_scales_trade_toward_current_when_max_turnover_is_exceeded():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "target_weight": [0.80, 0.20],
        }
    )
    current_weights = {"000001.XSHE": 0.20, "000002.XSHE": 0.80}

    smoothed = ExecutionSmoother(max_turnover=0.30).smooth(target_weights, current_weights)

    turnover = (
        (smoothed["target_weight"] - smoothed["order_book_id"].map(current_weights))
        .abs()
        .sum()
        / 2
    )
    assert turnover == pytest.approx(0.30, abs=1e-10)


def test_smoother_max_turnover_preserves_high_alpha_over_low_alpha():
    """When turnover is constrained, high-alpha trades should be preserved
    while low-alpha trades are cut first — not uniformly scaled."""
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 3),
            "order_book_id": ["HIGH_ALPHA", "LOW_ALPHA", "SELL"],
            "target_weight": [0.50, 0.30, 0.10],
            "net_alpha": [0.10, 0.01, 0.01],
        }
    )
    current_weights = {"HIGH_ALPHA": 0.10, "LOW_ALPHA": 0.10, "SELL": 0.70}

    smoothed = ExecutionSmoother(
        max_turnover=0.15, net_alpha_column="net_alpha"
    ).smooth(target_weights, current_weights)

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    # HIGH_ALPHA (alpha=0.10) should get more budget than LOW_ALPHA (alpha=0.01)
    high_delta = weights["HIGH_ALPHA"] - 0.10
    low_delta = weights["LOW_ALPHA"] - 0.10
    assert high_delta > low_delta
    # Turnover should be within budget
    turnover = sum(abs(weights.get(k, 0.0) - current_weights.get(k, 0.0)) for k in current_weights) / 2
    assert turnover == pytest.approx(0.15, abs=1e-10)


def test_smoother_applies_net_alpha_gate_only_to_increases_before_turnover_cap():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "order_book_id": [
                "LOW_BUY",
                "HIGH_BUY",
                "LOW_SELL",
                "TURNOVER_SINK",
            ],
            "target_weight": [0.40, 0.60, 0.05, 0.15],
            "net_alpha": [0.01, 0.08, 0.01, 0.10],
        }
    )
    current_weights = {
        "LOW_BUY": 0.20,
        "HIGH_BUY": 0.10,
        "LOW_SELL": 0.40,
        "TURNOVER_SINK": 0.30,
    }

    smoothed = ExecutionSmoother(
        net_alpha_threshold=0.05,
        net_alpha_column="net_alpha",
        max_turnover=0.20,
    ).smooth(target_weights, current_weights)

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    assert weights["LOW_BUY"] == pytest.approx(0.20)
    assert weights["HIGH_BUY"] > current_weights["HIGH_BUY"]
    assert weights["LOW_SELL"] < current_weights["LOW_SELL"]
    turnover = (
        (smoothed["target_weight"] - smoothed["order_book_id"].map(current_weights))
        .abs()
        .sum()
        / 2
    )
    assert turnover == pytest.approx(0.20)


def test_smoother_preserves_target_gross_weight_and_single_name_cap():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE", "000003.XSHE"],
            "target_weight": [0.50, 0.25, 0.15],
        }
    )

    smoothed = ExecutionSmoother(target_gross_weight=0.90, max_weight=0.40).smooth(target_weights)

    assert smoothed["target_weight"].sum() == pytest.approx(0.90)
    assert smoothed["target_weight"].max() <= 0.40 + 1e-12


def test_smoother_preserves_industry_cap_when_normalizing():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 4),
            "order_book_id": ["BANK_A", "BANK_B", "TECH_A", "TECH_B"],
            "target_weight": [0.30, 0.30, 0.20, 0.20],
            "industry": ["bank", "bank", "tech", "tech"],
        }
    )

    smoothed = ExecutionSmoother(
        target_gross_weight=1.0,
        max_industry_weight=0.50,
    ).smooth(target_weights)

    by_industry = smoothed.groupby("industry")["target_weight"].sum()
    assert by_industry["bank"] <= 0.50 + 1e-12
    assert smoothed["target_weight"].sum() == pytest.approx(1.0)


def test_smoother_returns_empty_frame_for_empty_input():
    smoothed = ExecutionSmoother(max_turnover=0.1, buffer_weight=0.02).smooth(
        pd.DataFrame(columns=["date", "order_book_id", "target_weight"])
    )

    assert smoothed.empty
    assert list(smoothed.columns) == ["date", "order_book_id", "target_weight", "component"]


def test_smoother_t_plus_one_lock_prevents_selling_today_buy_weight():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["A", "B"],
            "target_weight": [0.1, 0.9],
            "dollar_volume": [1_000_000.0, 1_000_000.0],
        }
    )
    current_weights = pd.DataFrame(
        {
            "order_book_id": ["A", "B"],
            "current_weight": [0.30, 0.70],
            "today_buy_weight": [0.25, 0.0],
        }
    )

    smoothed = ExecutionSmoother(
        t_plus_one_lock=True,
        today_buy_weight_column="today_buy_weight",
        portfolio_value=1_000_000,
    ).smooth(target_weights, current_weights)

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    assert weights["A"] >= 0.25 - 1e-12


def test_smoother_capacity_caps_trade_by_dollar_volume_participation_rate():
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["ILLQ", "LIQ"],
            "target_weight": [0.9, 0.1],
            "dollar_volume": [100_000.0, 10_000_000.0],
        }
    )
    current_weights = {"ILLQ": 0.0, "LIQ": 0.0}

    smoothed = ExecutionSmoother(
        portfolio_value=1_000_000,
        participation_rate=0.05,
        liquidity_column="dollar_volume",
    ).smooth(target_weights, current_weights)

    weights = smoothed.set_index("order_book_id")["target_weight"].to_dict()
    # 100k * 5% = 5k max trade value => max delta weight = 0.005
    assert weights["ILLQ"] == pytest.approx(0.005)
