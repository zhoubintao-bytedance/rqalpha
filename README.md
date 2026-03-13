# RQAlpha 回测系统使用文档

## 目录

- [快速开始](#快速开始)
- [运行示例](#运行示例)
- [策略生命周期](#策略生命周期)
- [下单 API](#下单-api)
- [数据查询 API](#数据查询-api)
- [持仓与账户](#持仓与账户)
- [定时任务（Scheduler）](#定时任务scheduler)
- [Bar / Tick 数据对象](#bar--tick-数据对象)
- [合约信息](#合约信息)
- [配置系统](#配置系统)
- [编程式调用](#编程式调用)
- [内置 Mod 模块](#内置-mod-模块)
- [回测结果指标](#回测结果指标)
- [CLI 命令参考](#cli-命令参考)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 安装

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 从源码安装（开发模式）
pip install -e .

# 或从 PyPI 安装
pip install rqalpha
```

### 2. 下载数据包

```bash
rqalpha download-bundle
```

数据包默认存储在 `~/.rqalpha/bundle/`，包含 A 股日线行情、合约信息、交易日历等。

### 3. 编写策略

创建 `strategy.py`：

```python
from rqalpha.apis import *

def init(context):
    context.stock = "000001.XSHE"  # 平安银行

def handle_bar(context, bar_dict):
    # 获取 20 日均价
    prices = history_bars(context.stock, 20, "1d", "close")
    if prices is None:
        return

    avg = prices.mean()
    cur = bar_dict[context.stock].close

    position = get_position(context.stock)
    if cur < avg * 0.97 and position.quantity == 0:
        order_target_percent(context.stock, 0.95)
    elif cur > avg * 1.03 and position.quantity > 0:
        order_target_percent(context.stock, 0)
```

### 4. 运行回测

```bash
rqalpha run -f strategy.py -s 2024-01-01 -e 2024-12-31 --account stock 100000
```

---

## 运行示例

### 示例策略列表

项目内置了多个示例策略，位于 `rqalpha/examples/` 目录：

| 策略文件 | 说明 | 依赖 |
|----------|------|------|
| `buy_and_hold.py` | 买入并持有（满仓一只股票） | 无 |
| `golden_cross.py` | 均线金叉/死叉策略 | talib |
| `macd.py` | MACD 指标策略 | talib |
| `demo_strategy.py` | 双均线交叉策略（带中文日志） | 无 |
| `IF_macd.py` | 股指期货 MACD 策略 | talib |

> 使用 talib 策略前需先安装：`pip install ta-lib`

### 买入持有策略

```bash
rqalpha run -f ./rqalpha/examples/buy_and_hold.py \
  -s 2016-06-01 -e 2016-12-01 \
  --account stock 100000 \
  --benchmark 000300.XSHG \
  --plot-save ./backtest_plot.png
```

### 金叉策略

```bash
rqalpha run -f ./rqalpha/examples/golden_cross.py \
  -s 2016-06-01 -e 2016-12-01 \
  --account stock 100000 \
  --benchmark 000300.XSHG \
  --plot-save ./backtest_plot.png
```

### MACD 策略

```bash
rqalpha run -f ./rqalpha/examples/macd.py \
  -s 2025-06-01 -e 2026-03-01 \
  --account stock 100000 \
  --benchmark 000300.XSHG \
  --plot-save ./backtest_plot.png
```

### 双均线交叉策略

```bash
rqalpha run -f ./rqalpha/examples/demo_strategy.py \
  -s 2025-06-01 -e 2026-03-01 \
  --account stock 100000 \
  --benchmark 000300.XSHG \
  --plot-save ./demo.png
```

### 查看图表

`--plot-save <路径>` 将回测结果图保存为 PNG 文件，`--plot` 弹出交互式窗口（需要图形界面）。

```bash
# 保存为图片（推荐，适用于无图形界面的服务器环境）
rqalpha run -f strategy.py ... --plot-save ./result.png

# 弹出交互式窗口（需要 TkAgg 等交互式后端）
rqalpha run -f strategy.py ... --plot

# 两者同时使用
rqalpha run -f strategy.py ... --plot --plot-save ./result.png
```

**配置 matplotlib 交互式后端：**

如果 `--plot` 无法弹出窗口，可能是 matplotlib 后端为非交互式的 `agg`。设置为 `TkAgg`：

```bash
# 方式一：临时生效
MPLBACKEND=TkAgg rqalpha run -f strategy.py ... --plot

# 方式二：永久生效
mkdir -p ~/.config/matplotlib
echo "backend: TkAgg" > ~/.config/matplotlib/matplotlibrc
```

### 中文图表标注

默认根据系统语言环境显示中英文。如果图表显示英文标注，设置 `LANG` 环境变量即可切换为中文：

```bash
# 临时生效
LANG=zh_CN.UTF-8 rqalpha run -f strategy.py ... --plot-save ./result.png

# 永久生效（写入 shell 配置）
echo 'export LANG=zh_CN.UTF-8' >> ~/.zshrc  # zsh 用户
echo 'export LANG=zh_CN.UTF-8' >> ~/.bashrc # bash 用户
```

---

## 策略生命周期

一个策略由以下函数组成，按顺序在每个交易日被调用：

```python
def init(context):
    """初始化（只执行一次）

    用途：设置变量、订阅合约、注册定时任务
    context 可添加自定义属性，在所有函数间共享
    """
    context.stock = "000001.XSHE"
    context.counter = 0

def before_trading(context):
    """每日开盘前执行

    用途：日级别的准备工作，此时不能下单
    """
    pass

def handle_bar(context, bar_dict):
    """每个 bar 执行一次（核心交易逻辑）

    日线模式：每天收盘时调用一次
    分钟线模式：每分钟调用一次
    bar_dict 包含所有订阅合约的当前 bar 数据
    """
    bar = bar_dict["000001.XSHE"]
    logger.info(f"当前价格: {bar.close}")

def handle_tick(context, tick):
    """Tick 模式下每个 tick 执行一次（与 handle_bar 二选一）"""
    pass

def after_trading(context):
    """每日收盘后执行

    用途：日终统计，此时不能下单
    """
    pass
```

**context 常用属性：**

| 属性 | 说明 |
|------|------|
| `context.now` | 当前时间（datetime） |
| `context.portfolio` | 投资组合对象 |
| `context.run_info` | 回测配置信息 |
| 自定义属性 | 在 init 中自由添加 |

---

## 下单 API

所有下单函数均支持 `price_or_style` 参数，不传默认为市价单。

### 股票下单

```python
# 按股数下单（正数买入，负数卖出）
order_shares("000001.XSHE", 500)       # 买入 500 股
order_shares("000001.XSHE", -300)      # 卖出 300 股

# 按手数下单（1手 = 100股）
order_lots("000001.XSHE", 5)           # 买入 5 手（500 股）

# 按金额下单
order_value("000001.XSHE", 50000)      # 买入 5 万元

# 按持仓占比下单（相对于总资产）
order_percent("000001.XSHE", 0.3)      # 加仓至总资产 30%

# 目标持仓 - 金额
order_target_value("000001.XSHE", 80000)  # 调整持仓市值至 8 万

# 目标持仓 - 占比（最常用）
order_target_percent("000001.XSHE", 0.5)  # 调整至总资产 50%
order_target_percent("000001.XSHE", 0)    # 清仓

# 取消订单
order = order_shares("000001.XSHE", 500)
cancel_order(order)

# 查看未成交订单
open_orders = get_open_orders()
```

### 期货下单

```python
# 开多仓
buy_open("IF2401", 1)

# 平多仓
sell_close("IF2401", 1)
sell_close("IF2401", 1, close_today=True)  # 平今仓

# 开空仓
sell_open("IF2401", 1)

# 平空仓
buy_close("IF2401", 1)

# 智能下单（自动判断开平方向）
order("IF2401", 3)     # 正数 → 买入方向，自动处理开/平
order("IF2401", -3)    # 负数 → 卖出方向

# 调整至目标手数
order_to("IF2401", 5)  # 调整至多头 5 手
```

### 订单类型

```python
from rqalpha.apis import *

# 市价单（默认）
order_shares("000001.XSHE", 500)

# 限价单
order_shares("000001.XSHE", 500, style=LimitOrder(10.50))

# VWAP 算法单（按成交量加权均价执行）
order_shares("000001.XSHE", 5000, style=VWAPOrder(start_min=30, end_min=60))

# TWAP 算法单（按时间加权均价执行）
order_shares("000001.XSHE", 5000, style=TWAPOrder(start_min=30, end_min=60))
```

### 底层通用下单

```python
from rqalpha.const import SIDE, POSITION_EFFECT

# 精确控制买卖方向和开平标志
submit_order("000001.XSHE", 500, SIDE.BUY, style=LimitOrder(10.5))
submit_order("IF2401", 1, SIDE.SELL, position_effect=POSITION_EFFECT.OPEN)
```

---

## 数据查询 API

### 历史行情

```python
# 获取最近 20 根日线收盘价（返回 numpy array）
closes = history_bars("000001.XSHE", 20, "1d", "close")

# 获取多个字段（返回 numpy structured array）
bars = history_bars("000001.XSHE", 20, "1d", ["open", "high", "low", "close", "volume"])
# 访问：bars["close"], bars["volume"]

# 分钟线（需要 frequency="1m" 模式）
bars_1m = history_bars("000001.XSHE", 60, "1m", "close")

# 参数说明
history_bars(
    order_book_id,               # 合约代码
    bar_count,                   # 获取根数
    frequency,                   # "1d" 日线 | "1m" 分钟线 | "1w" 周线
    fields=None,                 # 字段，None 返回所有
    skip_suspended=True,         # 跳过停牌日
    include_now=False,           # 是否包含当前未完成的 bar
    adjust_type="pre",           # "pre" 前复权 | "post" 后复权 | "none" 不复权
)
```

**可用字段：** `datetime`, `open`, `high`, `low`, `close`, `volume`, `total_turnover`, `open_interest`（期货持仓量）, `settlement`（期货结算价）, `prev_settlement`, `limit_up`, `limit_down`

### 交易日历

```python
# 获取日期区间内的交易日
dates = get_trading_dates("2024-01-01", "2024-12-31")

# 上一个 / 下一个交易日
prev = get_previous_trading_date("2024-03-01", n=1)
next = get_next_trading_date("2024-03-01", n=1)
```

### 实时快照（Tick 模式）

```python
snap = current_snapshot("000001.XSHE")
# snap.last, snap.open, snap.high, snap.low
# snap.bid_prices, snap.ask_prices, snap.bid_volumes, snap.ask_volumes
```

---

## 持仓与账户

### 持仓查询

```python
# 获取单个持仓
pos = get_position("000001.XSHE")
pos.quantity         # 持仓数量
pos.avg_price        # 持仓均价
pos.last_price       # 最新价
pos.market_value     # 市值
pos.pnl              # 累计盈亏
pos.trading_pnl      # 当日交易盈亏
pos.position_pnl     # 持仓盈亏
pos.transaction_cost # 累计手续费
pos.closable         # 可平仓数量（期货）

# 期货空头持仓
from rqalpha.const import POSITION_DIRECTION
pos = get_position("IF2401", POSITION_DIRECTION.SHORT)

# 获取所有持仓
positions = get_positions()
for pos in positions:
    logger.info(f"{pos.order_book_id}: {pos.quantity}股, 市值{pos.market_value:.0f}")
```

### 投资组合（Portfolio）

```python
p = context.portfolio

p.total_value         # 总资产
p.cash                # 可用现金
p.market_value        # 持仓总市值
p.frozen_cash         # 冻结资金（挂单中）
p.unit_net_value      # 单位净值
p.total_returns       # 累计收益率
p.annualized_returns  # 年化收益率
p.daily_pnl           # 当日盈亏
p.daily_returns       # 当日收益率
p.transaction_cost    # 累计手续费
p.pnl                 # 累计盈亏金额
p.positions           # 持仓字典 {order_book_id: Position}
```

### 账户（多账户）

```python
# 股票账户
stock_acct = context.portfolio.stock_account
stock_acct.cash          # 股票账户可用现金
stock_acct.total_value   # 股票账户总资产
stock_acct.market_value  # 股票持仓市值

# 期货账户
future_acct = context.portfolio.future_account
future_acct.margin       # 保证金占用
future_acct.buy_margin   # 多头保证金
future_acct.sell_margin  # 空头保证金
```

### 出入金 & 融资

```python
deposit(account_type="stock", amount=100000)   # 追加资金
withdraw(account_type="stock", amount=50000)   # 提取资金
finance(amount=200000, account_type="stock")   # 融资借入
repay(amount=200000, account_type="stock")     # 偿还融资
```

---

## 定时任务（Scheduler）

在 `init()` 中注册，替代在 `handle_bar` 里手动判断时间：

```python
def init(context):
    # 每日开盘后 30 分钟执行（仅分钟线模式生效）
    scheduler.run_daily(rebalance, time_rule=market_open(minute=30))

    # 每日收盘前 5 分钟执行
    scheduler.run_daily(close_check, time_rule=market_close(minute=5))

    # 每周第一个交易日执行
    scheduler.run_weekly(weekly_rebalance, tradingday=1)

    # 每周最后一个交易日执行
    scheduler.run_weekly(weekly_report, tradingday=-1)

    # 每月第一个交易日执行
    scheduler.run_monthly(monthly_rebalance, tradingday=1)

    # 每月倒数第二个交易日执行
    scheduler.run_monthly(month_end, tradingday=-2)

def rebalance(context, bar_dict):
    order_target_percent("000001.XSHE", 0.5)

def close_check(context, bar_dict):
    logger.info(f"收盘检查: 总资产 {context.portfolio.total_value:.0f}")

def weekly_rebalance(context, bar_dict):
    logger.info("每周调仓")

def weekly_report(context, bar_dict):
    logger.info(f"本周收益: {context.portfolio.daily_returns:.4f}")

def monthly_rebalance(context, bar_dict):
    logger.info("每月调仓")

def month_end(context, bar_dict):
    pass
```

---

## Bar / Tick 数据对象

### Bar 对象（handle_bar 中的 bar_dict）

```python
def handle_bar(context, bar_dict):
    bar = bar_dict["000001.XSHE"]

    bar.open             # 开盘价
    bar.high             # 最高价
    bar.low              # 最低价
    bar.close            # 收盘价
    bar.volume           # 成交量
    bar.total_turnover   # 成交额
    bar.datetime         # 时间戳
    bar.limit_up         # 涨停价
    bar.limit_down       # 跌停价
    bar.prev_close       # 昨收价
    bar.is_trading       # 是否有成交
    bar.suspended        # 是否停牌

    # 期货专用
    bar.open_interest    # 持仓量
    bar.settlement       # 结算价
    bar.prev_settlement  # 昨结算价

    # 便捷方法
    bar.mavg(10, "1d")   # 10 日移动均线
    bar.vwap(10, "1d")   # 10 日 VWAP
```

### Tick 对象（handle_tick 中）

```python
def handle_tick(context, tick):
    tick.last            # 最新价
    tick.open            # 开盘价
    tick.high            # 最高价
    tick.low             # 最低价
    tick.volume          # 累计成交量
    tick.total_turnover  # 累计成交额
    tick.prev_close      # 昨收价

    # 五档行情
    tick.a1_price        # 卖一价
    tick.a1_volume       # 卖一量
    tick.b1_price        # 买一价
    tick.b1_volume       # 买一量
    # ... 到 a5/b5
```

---

## 合约信息

### 查询合约

```python
# 获取单个合约信息
inst = instruments("000001.XSHE")
inst.symbol             # "平安银行"
inst.order_book_id      # "000001.XSHE"
inst.type               # "CS"（股票）
inst.exchange           # "XSHE"（深交所）
inst.round_lot          # 100（一手股数）
inst.listed_date        # 上市日期
inst.de_listed_date     # 退市日期
inst.board_type         # "MainBoard" | "GEM" | ...
inst.sector_code_name   # 行业名称
inst.special_type       # "Normal" | "ST" | "StarST"
inst.status             # "Active" | "Delisted" | ...

# 获取所有某类型合约
all_stocks = all_instruments(type="CS")        # 所有股票
all_etf = all_instruments(type="ETF")          # 所有 ETF
all_futures = all_instruments(type="Future")   # 所有期货
all_index = all_instruments(type="INDX")       # 所有指数

# 合约代码 → 中文名
name = symbol("000001.XSHE")  # → "平安银行"
```

**合约代码格式：**

| 类型 | 格式 | 示例 |
|------|------|------|
| 深交所股票 | XXXXXX.XSHE | 000001.XSHE |
| 上交所股票 | XXXXXX.XSHG | 600519.XSHG |
| 期货 | 品种+合约月份 | IF2401, RB2405 |
| ETF | XXXXXX.XSHG/XSHE | 510300.XSHG |
| 指数 | XXXXXX.XSHG/XSHE | 000300.XSHG（沪深300） |

**合约类型（type 参数）：** `CS`（股票）、`ETF`、`LOF`、`INDX`（指数）、`Future`（期货）、`OPTION`（期权）、`BOND`（债券）、`CONVERTIBLE`（可转债）、`PUBLIC_FUND`（公募基金）、`FUND`（基金）、`REITs`

---

## 配置系统

### 命令行参数

```bash
rqalpha run \
  -f strategy.py \               # 策略文件
  -s 2024-01-01 \                 # 开始日期
  -e 2024-12-31 \                 # 结束日期
  --account stock 100000 \        # 股票账户 10 万
  --account future 500000 \       # 期货账户 50 万（可多账户）
  -fq 1d \                        # 频率：1d 日线 | 1m 分钟线 | tick
  -d ~/.rqalpha/bundle \          # 数据包路径
  -l info \                       # 日志级别：verbose|debug|info|warning|error
  --plot \                        # 显示收益图
  -o result.pkl \                 # 保存结果到文件
  --config config.yml \           # 自定义配置文件
  --enable-profiler               # 性能分析
```

### 配置文件（YAML）

```yaml
base:
  start_date: "2024-01-01"
  end_date: "2024-12-31"
  accounts:
    stock: 100000
    future: 500000
  frequency: "1d"                   # 1d | 1m | tick
  margin_multiplier: 1              # 保证金倍率
  round_price: false                # 是否对价格取整到 tick
  persist: false                    # 是否启用持久化（支持暂停恢复）

extra:
  log_level: info

mod:
  sys_analyser:
    enabled: true
    benchmark: "000300.XSHG"        # 基准（沪深300）
    plot: true                      # 显示图表
    output_file: result.pkl         # 输出文件
    report_save_path: ./report      # 报告目录

  sys_simulation:
    matching_type: current_bar_close  # 撮合方式
    slippage: 0.01                    # 滑点（比例）
    slippage_model: PriceRatioSlippage
    price_limit: true                 # 涨跌停限制
    volume_limit: true                # 成交量限制
    volume_percent: 0.25              # 最大成交占比

  sys_transaction_cost:
    stock_min_commission: 5           # 最低佣金（元）
    stock_commission_multiplier: 1    # 佣金倍率
    tax_multiplier: 1                 # 印花税倍率

  sys_risk:
    validate_price: true              # 价格校验
    validate_is_trading: true         # 是否交易校验
    validate_cash: true               # 资金校验

  sys_accounts:
    stock_t1: true                    # 股票 T+1
    dividend_reinvestment: false      # 分红再投资
    auto_switch_order_value: false    # 资金不足时用剩余资金
```

### 策略内配置

```python
# 在策略文件中定义 __config__ 字典，会合并到全局配置
__config__ = {
    "base": {
        "accounts": {"stock": 200000},
    },
    "mod": {
        "sys_analyser": {"benchmark": "000300.XSHG"},
    }
}
```

**配置优先级：** 命令行参数 > 策略 `__config__` > 配置文件 > 默认值

---

## 编程式调用

除了命令行，也可以在 Python 代码中直接调用回测：

```python
from rqalpha import run

# 方式1：指定策略文件
result = run(
    config={
        "base": {
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "accounts": {"stock": 100000},
            "frequency": "1d",
        },
        "mod": {
            "sys_analyser": {
                "benchmark": "000300.XSHG",
                "plot": False,
            },
            "sys_progress": {"enabled": False},
        },
    },
    strategy_file="strategy.py",
)

# 方式2：传入源码字符串
result = run(config, source_code="""
from rqalpha.apis import *

def init(context):
    context.stock = "000001.XSHE"

def handle_bar(context, bar_dict):
    order_target_percent(context.stock, 0.95)
""")

# 方式3：传入函数
def my_init(context):
    context.stock = "000001.XSHE"

def my_handle_bar(context, bar_dict):
    order_target_percent(context.stock, 0.95)

result = run(config, user_funcs={
    "init": my_init,
    "handle_bar": my_handle_bar,
})

# 解析结果
if result and "sys_analyser" in result:
    summary = result["sys_analyser"]["summary"]
    print(f"总收益率: {summary['total_returns'] * 100:.2f}%")
    print(f"年化收益: {summary['annualized_returns'] * 100:.2f}%")
    print(f"最大回撤: {summary['max_drawdown'] * 100:.2f}%")
    print(f"夏普比率: {summary['sharpe']:.3f}")

    trades = result["sys_analyser"]["trades"]       # 交易记录 DataFrame
    portfolio = result["sys_analyser"]["portfolio"]  # 每日净值 DataFrame
```

---

## 内置 Mod 模块

| Mod | 说明 | 关键配置 |
|-----|------|----------|
| **sys_accounts** | 账户与持仓管理 | `stock_t1`（T+1）, `dividend_reinvestment`（分红再投） |
| **sys_simulation** | 模拟撮合引擎 | `matching_type`, `slippage`, `volume_percent` |
| **sys_analyser** | 绩效分析与报告 | `benchmark`, `plot`, `output_file` |
| **sys_risk** | 风控校验 | `validate_price`, `validate_cash` |
| **sys_scheduler** | 定时任务 | 无需额外配置 |
| **sys_transaction_cost** | 手续费计算 | `stock_min_commission`, `tax_multiplier` |
| **sys_progress** | 进度条显示 | 无需额外配置 |

### Mod 管理命令

```bash
rqalpha mod list               # 查看所有 mod
rqalpha mod enable mod_name    # 启用
rqalpha mod disable mod_name   # 禁用
```

---

## 回测结果指标

通过 `result["sys_analyser"]["summary"]` 获取：

### 收益指标

| 指标 | 字段 | 说明 |
|------|------|------|
| 总收益率 | `total_returns` | 累计收益率 |
| 年化收益率 | `annualized_returns` | 年化后的收益率 |
| 超额收益 | `excess_returns` | 相对基准的超额收益 |
| 总资产 | `total_value` | 期末总资产 |
| 可用现金 | `cash` | 期末可用现金 |

### 风险指标

| 指标 | 字段 | 说明 |
|------|------|------|
| 最大回撤 | `max_drawdown` | 最大净值回撤 |
| 波动率 | `volatility` | 年化波动率 |
| 在险价值 | `var` | Value at Risk |
| 下行风险 | `downside_risk` | 下行偏差 |
| 回撤持续天数 | `max_drawdown_duration_days` | 最大回撤持续天数 |

### 风险调整指标

| 指标 | 字段 | 说明 |
|------|------|------|
| 夏普比率 | `sharpe` | 风险调整后收益 |
| 索提诺比率 | `sortino` | 下行风险调整收益 |
| 信息比率 | `information_ratio` | 主动管理收益/跟踪误差 |
| Alpha | `alpha` | 超额收益（CAPM） |
| Beta | `beta` | 市场敏感度 |
| 跟踪误差 | `tracking_error` | 与基准偏离度 |

### 交易统计

| 指标 | 字段 | 说明 |
|------|------|------|
| 胜率 | `win_rate` | 盈利交易占比 |
| 盈亏比 | `profit_loss_rate` | 平均盈利/平均亏损 |
| 换手率 | `turnover` | 累计换手率 |
| 日均换手率 | `avg_daily_turnover` | 日均换手率 |

### 详细数据（DataFrame）

```python
result["sys_analyser"]["trades"]              # 所有交易记录
result["sys_analyser"]["portfolio"]           # 每日投资组合数据
result["sys_analyser"]["benchmark_portfolio"] # 每日基准数据
result["sys_analyser"]["stock_positions"]     # 每日持仓明细
result["sys_analyser"]["positions_weight"]    # 每日持仓权重
```

---

## CLI 命令参考

```bash
# 运行回测
rqalpha run -f strategy.py -s 2024-01-01 -e 2024-12-31 --account stock 100000

# 下载数据包
rqalpha download-bundle

# 从 RQData 创建/更新数据包
rqalpha create-bundle --rqdatac-uri tcp://user:pass@host:port
rqalpha update-bundle --rqdatac-uri tcp://user:pass@host:port

# 检查数据包完整性
rqalpha check-bundle

# 生成默认配置文件
rqalpha generate-config

# 生成示例策略
rqalpha examples -d ./examples

# 查看版本
rqalpha version

# Mod 管理
rqalpha mod list
rqalpha mod enable mod_name
rqalpha mod disable mod_name
```

---

## 常见问题

### 1. "Bundle not found" 错误
```bash
rqalpha download-bundle
```

### 2. 茅台等高价股买不了
10 万本金买不起 1 手茅台（1 手 = 100 股 × 1700+ 元 ≈ 17万），需要增加初始资金：
```bash
rqalpha run ... --account stock 500000
```

### 3. "Order Creation Failed: 0 order quantity"
资金不足以购买最小交易单位（1 手 = 100 股），增大资金或换标的。

### 4. 股票 T+1 限制
A 股当日买入不能当日卖出，这是 `sys_accounts` 模块的 `stock_t1=true` 默认行为。

### 5. 涨跌停无法成交
策略触发买入时恰好涨停，订单会被取消。日志显示：`Order Cancelled: reach the limit_up price`

### 6. 策略中使用第三方库
```python
import numpy as np
import talib  # 需要 pip install ta-lib

def handle_bar(context, bar_dict):
    closes = history_bars("000001.XSHE", 30, "1d", "close")
    ma = talib.SMA(closes, timeperiod=20)
```

### 7. 自定义绘图

```python
def handle_bar(context, bar_dict):
    closes = history_bars("000001.XSHE", 20, "1d", "close")
    plot("MA20", closes.mean())   # 会显示在回测结果图上
```

### 8. 调试策略
```bash
# 开启详细日志
rqalpha run -f strategy.py ... --log-level debug

# 在策略中打日志
logger.info(f"当前现金: {context.portfolio.cash}")
logger.info(f"持仓: {list(context.portfolio.positions.keys())}")
```
