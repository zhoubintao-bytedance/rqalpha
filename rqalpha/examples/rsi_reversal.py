from rqalpha.apis import *

import talib


def init(context):
    context.stock = "000338.XSHE"
    context.rsi_period = 14
    context.overbought = 70
    context.oversold = 30
    context.target_percent = 0.95
    context.fired = False


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    prices = history_bars(context.stock, context.rsi_period + 1, "1d", "close")
    if prices is None or len(prices) < context.rsi_period + 1:
        return

    rsi_value = talib.RSI(prices, timeperiod=context.rsi_period)[-1]

    if rsi_value <= context.oversold and not context.fired:
        order_target_percent(context.stock, context.target_percent)
        context.fired = True
    elif rsi_value >= context.overbought and context.fired:
        order_target_percent(context.stock, 0)
        context.fired = False


def after_trading(context):
    pass
