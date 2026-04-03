import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from skyeye.evaluation import rolling_score as strategy_scorer
from skyeye.evaluation.rolling_score import engine as rolling_engine


def _trade(order_book_id, symbol, side, price, quantity, trade_dt):
    return {
        "order_book_id": order_book_id,
        "symbol": symbol,
        "side": side,
        "last_price": price,
        "last_quantity": quantity,
        "trade_dt": pd.Timestamp(trade_dt),
    }


def _trades_frame(rows):
    frame = pd.DataFrame(rows)
    frame = frame.set_index("trade_dt")
    return frame


def test_build_trade_log_handles_multiple_windows_and_preserves_original_idx(capsys):
    trades_w5 = _trades_frame(
        [
            _trade("000001.XSHE", "平安银行", "BUY", 10.0, 100, "2024-01-03"),
            _trade("000001.XSHE", "平安银行", "SELL", 12.0, 100, "2024-01-04"),
        ]
    )
    trades_w7 = _trades_frame(
        [
            _trade("000002.XSHE", "万 科A", "BUY", 20.0, 100, "2024-07-03"),
            _trade("000002.XSHE", "万 科A", "SELL", 18.0, 100, "2024-07-04"),
        ]
    )
    window_results = [
        {
            "idx": 5,
            "start": datetime.date(2024, 1, 1),
            "end": datetime.date(2024, 3, 31),
            "summary": {"total_returns": 0.10},
            "trades": trades_w5,
        },
        {
            "idx": 7,
            "start": datetime.date(2024, 7, 1),
            "end": datetime.date(2024, 9, 30),
            "summary": {"total_returns": 0.20},
            "trades": trades_w7,
        },
    ]

    strategy_scorer.build_trade_log(window_results, level="high", cash=100000)

    output = capsys.readouterr().out
    assert "窗口 #5" in output
    assert "窗口 #7" in output
    assert "窗口 #1" not in output
    assert "窗口 #2" not in output


def test_score_window_and_project_to_quarters_ignore_numpy_nan():
    nan_summary = {name: np.float64(np.nan) for name in strategy_scorer.WEIGHTS}
    assert strategy_scorer.score_window(nan_summary) == 0.0

    window_results = [
        {
            "idx": 1,
            "start": datetime.date(2024, 1, 1),
            "end": datetime.date(2024, 12, 31),
            "score": 12.5,
            "summary": {
                "annualized_returns": np.float64(np.nan),
                "max_drawdown": np.float64(-0.12),
                "sharpe": np.float64(np.nan),
                "win_rate": np.float64(0.60),
            },
        }
    ]

    quarterly_scores, quarterly_raw = strategy_scorer.project_to_quarters(window_results)

    assert quarterly_scores
    for raw_values in quarterly_raw.values():
        assert raw_values["annualized_returns"] == 0.0
        assert raw_values["max_drawdown"] == -0.12
        assert raw_values["sharpe"] == 0.0
        assert raw_values["win_rate"] == 0.60


def test_flatten_trades_tracks_portfolio_level_realized_pnl_across_symbols():
    trades = _trades_frame(
        [
            _trade("000002.XSHE", "万 科A", "BUY", 20.0, 50, "2024-01-02"),
            _trade("000001.XSHE", "平安银行", "BUY", 10.0, 100, "2024-01-03"),
            _trade("000001.XSHE", "平安银行", "SELL", 12.0, 100, "2024-01-04"),
            _trade("000002.XSHE", "万 科A", "SELL", 18.0, 50, "2024-01-05"),
        ]
    )

    records = strategy_scorer.flatten_trades(trades)

    assert records[2]["realized_pnl"] == 200.0
    assert records[2]["total_realized_pnl"] == 200.0
    assert records[3]["realized_pnl"] == -100.0
    assert records[3]["total_realized_pnl"] == 100.0


def test_parse_runtime_config_args_splits_mod_and_extra_scopes():
    mod_configs, extra_config = strategy_scorer.parse_runtime_config_args(
        [
            ("dividend_scorer.prior_blend", "0.7"),
            ("extra.strategy_profile", "baseline"),
            ("extra.tx1_artifact_line", "baseline_tree"),
        ]
    )

    assert mod_configs == {"dividend_scorer": {"prior_blend": 0.7}}
    assert extra_config == {
        "strategy_profile": "baseline",
        "tx1_artifact_line": "baseline_tree",
    }


def test_get_benchmark_quarterly_returns_handles_multiindex_daily_bars(monkeypatch):
    index = pd.MultiIndex.from_tuples(
        [
            ("000300.XSHG", pd.Timestamp("2024-03-29")),
            ("000300.XSHG", pd.Timestamp("2024-06-28")),
            ("000300.XSHG", pd.Timestamp("2024-09-30")),
        ],
        names=["order_book_id", "date"],
    )
    bars = pd.DataFrame({"close": [100.0, 110.0, 99.0]}, index=index)

    class FakeFacade:
        def get_daily_bars(self, *args, **kwargs):
            return bars

    monkeypatch.setattr(rolling_engine, "DataFacade", lambda: FakeFacade())

    result = strategy_scorer.get_benchmark_quarterly_returns()

    assert result[(2024, 2)] == pytest.approx(0.10)
    assert result[(2024, 3)] == pytest.approx(-0.10)


def test_run_rolling_backtests_routes_extra_config_to_top_level_extra_and_uses_strategy_benchmark(
    monkeypatch,
    tmp_path,
):
    strategy_path = tmp_path / "strategy.py"
    strategy_path.write_text("def init(context):\n    pass\n", encoding="utf-8")

    observed = {}

    def fake_run(config, source_code):
        observed["config"] = config
        observed["source_code"] = source_code
        return {
            "sys_analyser": {
                "summary": {name: 0.0 for name in strategy_scorer.WEIGHTS},
                "trades": pd.DataFrame(),
                "portfolio": pd.DataFrame(
                    {"market_value": [0.0]},
                    index=[pd.Timestamp("2016-02-01")],
                ),
            }
        }

    monkeypatch.setattr(rolling_engine, "run", fake_run)
    monkeypatch.setattr(
        rolling_engine,
        "find_strategy_spec_by_file",
        lambda strategy_file: SimpleNamespace(benchmark="512890.XSHG"),
    )

    results = strategy_scorer.run_rolling_backtests(
        str(strategy_path),
        cash=100000,
        selected_indices=[1],
        mod_configs={"dividend_scorer": {"prior_blend": 0.7}},
        extra_config={"strategy_profile": "baseline"},
    )

    assert len(results) == 1
    assert observed["source_code"] == "def init(context):\n    pass\n"
    assert observed["config"]["extra"]["strategy_profile"] == "baseline"
    assert observed["config"]["mod"]["sys_analyser"]["benchmark"] == "512890.XSHG"
    assert observed["config"]["mod"]["dividend_scorer"]["enabled"] is True
    assert observed["config"]["mod"]["dividend_scorer"]["prior_blend"] == 0.7
