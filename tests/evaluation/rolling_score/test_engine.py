import datetime

import numpy as np
import pandas as pd

from skyeye.evaluation import rolling_score as strategy_scorer


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
