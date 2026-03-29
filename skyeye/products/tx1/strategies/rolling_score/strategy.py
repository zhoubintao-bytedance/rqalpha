"""TX1 rolling-score strategy adapter.

This strategy replays frozen out-of-sample target weights produced by the TX1
research pipeline. It is the shortest production-usable bridge from TX1
artifacts into the standard RQAlpha execution and rolling_score evaluation
stack.
"""

from __future__ import annotations

from rqalpha.apis import *
from rqalpha.environment import Environment

from skyeye.products.tx1.strategies.rolling_score.replay import (
    build_execution_universe,
    compute_turnover_ratio,
    sanitize_target_weights,
    smooth_target_weights,
)
from skyeye.products.tx1.strategies.rolling_score.runtime import build_runtime


STRATEGY_ID = "tx1.rolling_score"
ROLLING_SCORE_START_DATE = "2019-02-01"


def init(context):
    runtime = build_runtime(
        strategy_id=STRATEGY_ID,
        artifact_line=None,
        load_signal_book=True,
    )
    signal_book = runtime.get("signal_book", {})
    signal_dates = sorted(signal_book)

    context.tx1_strategy_id = STRATEGY_ID
    context.tx1_runtime = runtime
    context.tx1_profile = runtime["profile"]
    context.tx1_signal_book = signal_book
    context.tx1_pending_signal = None
    context.tx1_pending_signal_date = None
    context.tx1_pending_target = None
    context.tx1_last_execution_date = None
    context.tx1_ema_state = {}
    context.tx1_ema_halflife = float(context.tx1_profile.get("ema_halflife", 0) or 0)
    context.tx1_ema_min_weight = float(
        context.tx1_profile.get("ema_min_weight", 0.005) or 0.005
    )
    context.tx1_diagnostics = {
        "rebalance_checks": 0,
        "executed_rebalances": 0,
        "missing_signal_days": 0,
        "turnover_skips": 0,
    }

    logger.info(
        (
            "TX1 rolling_score initialized: artifact_line={} profile={} signals={} "
            "signal_range={}~{} model_kind={}"
        ).format(
            runtime["artifact_line_id"],
            runtime["profile"].get("profile"),
            len(signal_book),
            signal_dates[0] if signal_dates else "-",
            signal_dates[-1] if signal_dates else "-",
            runtime["artifact"].get("model_kind"),
        )
    )


def before_trading(context):
    context.tx1_diagnostics["rebalance_checks"] += 1

    signal_date = _previous_trading_date_text()
    context.tx1_pending_signal_date = signal_date
    signal = context.tx1_signal_book.get(signal_date)
    context.tx1_pending_signal = signal

    if signal is None:
        context.tx1_pending_target = None
        context.tx1_diagnostics["missing_signal_days"] += 1
        return

    raw_target = sanitize_target_weights(
        signal.target_weights,
        single_stock_cap=context.tx1_profile.get("single_stock_cap"),
    )
    tradable_target = _filter_tradable_target_weights(raw_target)

    if context.tx1_ema_halflife > 0:
        tradable_target, context.tx1_ema_state = smooth_target_weights(
            tradable_target,
            context.tx1_ema_state,
            halflife=context.tx1_ema_halflife,
            min_weight=context.tx1_ema_min_weight,
        )

    context.tx1_pending_target = tradable_target

    current_weights = _current_portfolio_weights(context)
    context.tx1_pending_turnover = compute_turnover_ratio(current_weights, tradable_target)
    universe = build_execution_universe(tradable_target, current_weights)
    if universe:
        update_universe(universe)


def handle_bar(context, bar_dict):
    trade_date = _current_trade_date().strftime("%Y-%m-%d")
    if context.tx1_last_execution_date == trade_date:
        return
    context.tx1_last_execution_date = trade_date

    target_portfolio = context.tx1_pending_target
    if target_portfolio is None:
        return

    current_weights = _current_portfolio_weights(context)
    turnover = compute_turnover_ratio(current_weights, target_portfolio)
    turnover_threshold = float(context.tx1_profile.get("turnover_threshold", 0.0) or 0.0)

    if turnover < turnover_threshold:
        context.tx1_diagnostics["turnover_skips"] += 1
        return

    order_target_portfolio(target_portfolio)
    context.tx1_diagnostics["executed_rebalances"] += 1


def after_trading(context):
    pass


def _current_portfolio_weights(context) -> dict[str, float]:
    total_value = float(context.portfolio.total_value or 0.0)
    if total_value <= 0:
        return {}

    weights = {}
    positions = context.portfolio.positions
    for order_book_id in positions:
        position = positions[order_book_id]
        quantity = getattr(position, "quantity", 0) or 0
        market_value = float(getattr(position, "market_value", 0.0) or 0.0)
        if quantity <= 0 or market_value <= 0:
            continue
        weights[order_book_id] = market_value / total_value
    return weights


def _filter_tradable_target_weights(target_weights: dict[str, float]) -> dict[str, float]:
    if not target_weights:
        return {}

    trade_date = _current_trade_date()
    tradable = {}
    for order_book_id, weight in target_weights.items():
        instrument = instruments(order_book_id)
        if instrument is None:
            continue
        if instrument.listed_date.date() > trade_date:
            continue
        if instrument.de_listed_date.date() <= trade_date:
            continue
        if is_st_stock(order_book_id):
            continue
        if is_suspended(order_book_id):
            continue
        tradable[order_book_id] = weight
    return tradable


def _current_trade_date():
    return Environment.get_instance().trading_dt.date()


def _previous_trading_date_text() -> str:
    previous = Environment.get_instance().data_proxy.get_previous_trading_date(
        Environment.get_instance().trading_dt
    )
    return previous.date().strftime("%Y-%m-%d")
