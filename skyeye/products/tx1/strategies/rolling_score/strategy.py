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
    check_stop_loss,
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
    context.tx1_last_rebalance_date = None
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
        "interval_skips": 0,
        "stop_loss_triggers": 0,
        "stop_loss_sells": 0,
        "filtered_trades": 0,
        "filtered_value": 0,
    }

    # Stop-loss configuration
    context.tx1_stop_loss_pct = float(
        context.tx1_profile.get("stop_loss_pct", 0) or 0
    )
    context.tx1_stop_loss_cooldown_days = int(
        context.tx1_profile.get("stop_loss_cooldown_days", 0) or 0
    )
    context.tx1_stop_loss_cooldown: dict[str, str] = {}

    # Minimum weight change threshold configuration
    context.tx1_min_weight_change = float(
        context.tx1_profile.get("min_weight_change", 0.003) or 0.003
    )
    context.tx1_min_trade_value = float(
        context.tx1_profile.get("min_trade_value", 15000) or 15000
    )

    # Minimum rebalance interval configuration
    context.tx1_min_rebalance_interval = int(
        context.tx1_profile.get("min_rebalance_interval", 15) or 15
    )

    logger.info(
        (
            "TX1 rolling_score initialized: artifact_line={} profile={} signals={} "
            "signal_range={}~{} model_kind={} turnover_threshold={} min_rebalance_interval={} min_trade_value={}"
        ).format(
            runtime["artifact_line_id"],
            runtime["profile"].get("profile"),
            len(signal_book),
            signal_dates[0] if signal_dates else "-",
            signal_dates[-1] if signal_dates else "-",
            runtime["artifact"].get("model_kind"),
            context.tx1_profile.get("turnover_threshold", 0.0),
            context.tx1_min_rebalance_interval,
            context.tx1_min_trade_value,
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

    # Filter out stocks in stop-loss cooldown period
    if context.tx1_stop_loss_cooldown_days > 0 and context.tx1_stop_loss_cooldown:
        trade_date_str = _current_trade_date().strftime("%Y-%m-%d")
        cooled = []
        for stock_id, stop_date in context.tx1_stop_loss_cooldown.items():
            days_since = _trading_days_between(stop_date, trade_date_str)
            if days_since >= context.tx1_stop_loss_cooldown_days:
                cooled.append(stock_id)
        for stock_id in cooled:
            del context.tx1_stop_loss_cooldown[stock_id]

        if context.tx1_stop_loss_cooldown:
            tradable_target = {
                sid: w
                for sid, w in tradable_target.items()
                if sid not in context.tx1_stop_loss_cooldown
            }

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

    # ① Stop-loss check: sell before regular rebalance
    if context.tx1_stop_loss_pct > 0:
        positions = context.portfolio.positions
        stop_ids = check_stop_loss(positions, context.tx1_stop_loss_pct)
        if stop_ids:
            context.tx1_diagnostics["stop_loss_triggers"] += len(stop_ids)
            for order_book_id in stop_ids:
                position = positions[order_book_id]
                quantity = getattr(position, "quantity", 0) or 0
                if quantity > 0:
                    order_shares(order_book_id, -quantity)
                    context.tx1_diagnostics["stop_loss_sells"] += 1
                    if context.tx1_stop_loss_cooldown_days > 0:
                        context.tx1_stop_loss_cooldown[order_book_id] = trade_date
            logger.info(
                "TX1 stop-loss triggered: {} stocks on {}".format(
                    len(stop_ids), trade_date
                )
            )

    # ② Regular rebalance
    target_portfolio = context.tx1_pending_target
    if target_portfolio is None:
        return

    # Check minimum rebalance interval
    if context.tx1_min_rebalance_interval > 0 and context.tx1_last_rebalance_date:
        days_since_last = _trading_days_between(context.tx1_last_rebalance_date, trade_date)
        if days_since_last < context.tx1_min_rebalance_interval:
            context.tx1_diagnostics["interval_skips"] += 1
            return

    current_weights = _current_portfolio_weights(context)
    turnover = compute_turnover_ratio(current_weights, target_portfolio)
    turnover_threshold = float(context.tx1_profile.get("turnover_threshold", 0.0) or 0.0)

    if turnover < turnover_threshold:
        context.tx1_diagnostics["turnover_skips"] += 1
        return

    # Filter out small weight changes to avoid inefficient tiny trades
    total_value = context.portfolio.total_value
    filtered_target = {}
    filtered_count = 0
    filtered_value = 0.0

    for stock_id, target_weight in target_portfolio.items():
        current_weight = current_weights.get(stock_id, 0)
        weight_change = abs(target_weight - current_weight)
        trade_value = weight_change * total_value

        # Keep trade if:
        # 1. Weight change is large enough
        # 2. Trade value is large enough
        # 3. Closing position (target_weight == 0)
        # 4. Opening position (current_weight == 0)
        need_trade = (
            weight_change >= context.tx1_min_weight_change
            or trade_value >= context.tx1_min_trade_value
            or target_weight == 0
            or current_weight == 0
        )

        if need_trade:
            filtered_target[stock_id] = target_weight
        else:
            filtered_count += 1
            filtered_value += trade_value

    # Update diagnostics
    context.tx1_diagnostics["filtered_trades"] += filtered_count
    context.tx1_diagnostics["filtered_value"] += filtered_value

    if filtered_count > 0:
        logger.info(
            "TX1 filtered {} small trades, avg value {:.0f} yuan".format(
                filtered_count, filtered_value / filtered_count
            )
        )

    # Skip if all trades are filtered
    if not filtered_target:
        logger.info("TX1 skip rebalance: all trades filtered")
        return

    order_target_portfolio(filtered_target)
    context.tx1_diagnostics["executed_rebalances"] += 1
    context.tx1_last_rebalance_date = trade_date


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


def _trading_days_between(start_date_str: str, end_date_str: str) -> int:
    """Count trading days between two date strings (exclusive of both endpoints)."""
    from datetime import datetime

    env = Environment.get_instance()
    start = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end = datetime.strptime(end_date_str, "%Y-%m-%d").date()

    # Count trading dates from start+1 to end-1 (exclusive both ends)
    count = 0
    current = start
    while current < end:
        next_td = env.data_proxy.get_next_trading_date(current)
        if next_td is None or next_td.date() >= end:
            break
        count += 1
        current = next_td.date()
    return count
