"""
双均线交叉策略 + 趋势过滤（适配沪深300ETF）

策略逻辑：
- 趋势过滤: 价格在120日均线之上才允许买入（顺大势）
- 金叉买入: 10日均线上穿50日均线，连续2天确认后全仓买入
- 死叉卖出: 10日均线下穿50日均线，连续2天确认后清仓
- 止损保护: 持仓亏损超过6%强制止损，止损后等待20天冷却
"""
from rqalpha.apis import *


def init(context):
    """策略初始化"""
    context.stock = "600218.XSHG"  # 沪深300ETF

    # 均线参数
    context.short_window = 10      # 短期均线
    context.long_window = 30       # 长期均线
    context.trend_window = 60      # 趋势过滤均线（缩短，更快捕捉趋势转折）

    # 信号确认
    context.confirm_bars = 2       # 连续2天确认

    # 仓位管理（ETF波动小，一步到位）
    context.target_percent = 0.95

    # 风控
    context.stop_loss_pct = 0.06   # 止损线6%
    context.cooldown = 20          # 止损后冷却天数
    context.bar_count = 0
    context.cooldown_until = 0     # 冷却截止bar


def before_trading(context):
    pass


def handle_bar(context, bar_dict):
    context.bar_count += 1

    need_bars = max(context.trend_window, context.long_window + context.confirm_bars)
    prices = history_bars(context.stock, need_bars, "1d", "close")
    if prices is None or len(prices) < need_bars:
        return

    # 计算趋势均线
    total = len(prices)
    trend_ma = prices[-context.trend_window:].mean()
    last_price = bar_dict[context.stock].close

    # 信号确认：连续 confirm_bars 天短均线在长均线同侧
    up_confirmed = True
    down_confirmed = True
    for offset in range(context.confirm_bars):
        end_idx = total - offset
        s_ma = prices[end_idx - context.short_window:end_idx].mean()
        l_ma = prices[end_idx - context.long_window:end_idx].mean()
        if s_ma <= l_ma:
            up_confirmed = False
        if s_ma >= l_ma:
            down_confirmed = False

    # 当前持仓状态
    position = get_position(context.stock)
    has_position = position.quantity > 0
    in_cooldown = context.bar_count < context.cooldown_until

    # 止损检查
    if has_position and position.avg_price > 0:
        drawdown = (position.avg_price - last_price) / position.avg_price
        if drawdown >= context.stop_loss_pct:
            order_target_percent(context.stock, 0)
            context.cooldown_until = context.bar_count + context.cooldown
            return

    # 买入条件：金叉确认 + 价格在趋势均线之上 + 不在冷却期
    if up_confirmed and not has_position and last_price > trend_ma and not in_cooldown:
        order_target_percent(context.stock, context.target_percent)

    # 卖出条件：死叉确认
    elif down_confirmed and has_position:
        order_target_percent(context.stock, 0)


def after_trading(context):
    pass
