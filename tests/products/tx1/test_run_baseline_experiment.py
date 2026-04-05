import pandas as pd

from skyeye.products.tx1 import run_baseline_experiment as baseline_runner


def test_get_liquid_universe_applies_market_cap_floor_before_liquidity_ranking(monkeypatch):
    instruments = pd.DataFrame(
        {
            "order_book_id": ["A", "B", "C"],
            "circulating_market_cap": [10.0, 100.0, 90.0],
        }
    )
    dates = pd.bdate_range("2020-01-01", periods=520)
    rows = []
    volume_map = {"A": 300.0, "B": 200.0, "C": 100.0}
    for order_book_id, volume in volume_map.items():
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "volume": volume,
                }
            )
    bars = pd.DataFrame(rows)

    class FakeFacade:
        def get_daily_bars(self, order_book_ids, *args, **kwargs):
            return bars[bars["order_book_id"].isin(order_book_ids)].copy()

    monkeypatch.setattr(
        baseline_runner,
        "_load_stock_instruments",
        lambda active_only=True: instruments.copy(),
    )
    monkeypatch.setattr(baseline_runner, "DATA_FACADE", FakeFacade())

    assert baseline_runner.get_liquid_universe(2) == ["A", "B"]
    assert baseline_runner.get_liquid_universe(2, market_cap_floor_quantile=0.5) == ["B", "C"]


def test_build_experiment_config_enables_multi_output_and_portfolio_overrides():
    config = baseline_runner.build_experiment_config(
        "lgbm",
        experiment_name="tx1_combo_guarded_b25_h45",
        multi_output_enabled=True,
        volatility_weight=0.15,
        max_drawdown_weight=0.20,
        enable_reliability_score=True,
        holding_bonus=0.2,
        rebalance_interval=15,
    )

    assert config["experiment_name"] == "tx1_combo_guarded_b25_h45"
    assert config["portfolio"]["holding_bonus"] == 0.2
    assert config["portfolio"]["rebalance_interval"] == 15
    assert config["multi_output"]["enabled"] is True
    assert config["multi_output"]["volatility"]["enabled"] is True
    assert config["multi_output"]["max_drawdown"]["enabled"] is True
    assert config["multi_output"]["prediction"]["combine_auxiliary"] is True
    assert config["multi_output"]["prediction"]["volatility_weight"] == 0.15
    assert config["multi_output"]["prediction"]["max_drawdown_weight"] == 0.20
    assert config["multi_output"]["reliability_score"]["enabled"] is True
