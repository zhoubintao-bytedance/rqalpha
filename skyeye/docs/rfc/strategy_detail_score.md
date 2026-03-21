# 策略打分器实现计划

> 迁移说明：本文档当前归档在 `skyeye/docs/rfc/`，策略打分器主实现已迁至 `skyeye/strategy_scorer.py`，旧根目录入口已移除。

## Context

用户设计了一套完整的策略打分系统（详见 `strategy_score.md`），需要将设计落地为可运行的代码。打分器对输入的策略文件进行37个滚动窗口回测，通过13个指标的多项式评分 → 季度网格投影 → 时间衰减加权，输出综合得分、稳定性得分和市场环境适应性三个维度的评估结果。核心宗旨：准确、易用、稳定。

## 产出文件

创建单文件 `skyeye/strategy_scorer.py`，作为当前主实现。

## 用法

```bash
# 基本用法：对策略文件一键打分
python skyeye/strategy_scorer.py rqalpha/examples/demo_strategy.py

# 指定初始资金（默认100万）
python skyeye/strategy_scorer.py rqalpha/examples/demo_strategy.py --cash 200000

# 指定交易日志详细程度（默认 low）
python skyeye/strategy_scorer.py rqalpha/examples/demo_strategy.py --log low   # 最近1个窗口
python skyeye/strategy_scorer.py rqalpha/examples/demo_strategy.py --log mid   # 最近1年（4个窗口）
python skyeye/strategy_scorer.py rqalpha/examples/demo_strategy.py --log high  # 全部37个窗口
```

## 实现结构

### 1. 常量与配置

所有从 `strategy_score.md` 提取的参数硬编码为常量：

```python
# 13个指标配置：多项式系数(c0~c4)、锚点范围、外推斜率、是否取反
# coeffs 顺序: [c0, c1, c2, c3, c4]，即 score = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4
INDICATORS = {
    # --- 收益类 ---
    "total_returns": {
        "coeffs": [0.0, 385.873016, 36.772487, -5579.365079, 16243.386243],
        "x_min": -0.15, "x_max": 0.3,
        "slope_left": 144.4444, "slope_right": 655.7937,
        "negate": False,
    },
    "annualized_returns": {
        "coeffs": [-17.142857, 306.190476, 1033.333333, -6476.190476, 9523.809524],
        "x_min": -0.1, "x_max": 0.4,
        "slope_left": 130.0, "slope_right": 462.381,
        "negate": False,
    },
    "excess_annual_returns": {
        "coeffs": [19.312452, 329.895595, -690.603514, 8041.762159, -13750.954927],
        "x_min": -0.1, "x_max": 0.2,
        "slope_left": 764.273, "slope_right": 578.6351,
        "negate": False,
    },
    # --- 风险类（negate=True，越低越好，输入时取反 x_input = -x_raw）---
    "max_drawdown": {
        "coeffs": [166.875, 1702.916667, 8391.666667, 22833.333333, 23333.333333],
        "x_min": -0.35, "x_max": -0.05,
        "slope_left": 218.3333, "slope_right": 1023.3333,
        "negate": True,
    },
    "max_drawdown_duration_days": {
        "coeffs": [549.2455242967089, 9.365380789997912, 0.06447013829687052, 0.00019543904518330954, 2.1194468125416533e-07],
        "x_min": -350, "x_max": -90,
        "slope_left": 0.25, "slope_right": 1.8919,
        "negate": True,
    },
    "excess_max_drawdown": {
        "coeffs": [229.705882, 6249.509804, 76062.091503, 419934.640523, 816993.464052],
        "x_min": -0.2, "x_max": -0.03,
        "slope_left": 382.3529, "slope_right": 2731.3725,
        "negate": True,
    },
    "tracking_error": {
        "coeffs": [160.0, 1423.333333, 4633.333333, 2666.666667, -13333.333333],
        "x_min": -0.3, "x_max": -0.05,
        "slope_left": 803.3333, "slope_right": 986.6667,
        "negate": True,
    },
    # --- 风险调整收益类 ---
    "sharpe": {
        "coeffs": [0.0, 70.434783, -80.942029, 55.652174, -10.144928],
        "x_min": -0.3, "x_max": 2.0,
        "slope_left": 135.1217, "slope_right": 89.8551,
        "negate": False,
    },
    "sortino": {
        "coeffs": [0.0, 73.145743, -77.676768, 37.822671, -5.451339],
        "x_min": -0.3, "x_max": 3.0,
        "slope_left": 130.5527, "slope_right": 39.5527,
        "negate": False,
    },
    "information_ratio": {
        "coeffs": [0.0, 76.260684, -59.850427, 59.496676, -15.906933],
        "x_min": -0.3, "x_max": 1.5,
        "slope_left": 129.953, "slope_right": 83.5684,
        "negate": False,
    },
    # --- 交易统计类 ---
    "win_rate": {
        "coeffs": [-1920.0, 13426.666667, -34366.666667, 37333.333333, -13333.333333],
        "x_min": 0.35, "x_max": 0.6,
        "slope_left": 803.3333, "slope_right": 986.6667,
        "negate": False,
    },
    "profit_loss_rate": {
        "coeffs": [-121.671068, 253.344465, -164.707265, 52.293645, -5.776856],
        "x_min": 0.5, "x_max": 3.0,
        "slope_left": 124.969, "slope_right": 53.1288,
        "negate": False,
    },
    # --- 月度指标 ---
    "monthly_excess_win_rate": {
        "coeffs": [-111.666667, 215.15873, 1013.492063, -2825.396825, 2222.222222],
        "x_min": 0.25, "x_max": 0.7,
        "slope_left": 331.0317, "slope_right": 529.6032,
        "negate": False,
    },
}

# 13个指标的最终权重（加总 = 100%）
WEIGHTS = {
    "total_returns": 0.12,
    "annualized_returns": 0.12,
    "excess_annual_returns": 0.06,
    "max_drawdown": 0.10,
    "max_drawdown_duration_days": 0.075,
    "excess_max_drawdown": 0.0375,
    "tracking_error": 0.0375,
    "sharpe": 0.10,
    "sortino": 0.10,
    "information_ratio": 0.05,
    "win_rate": 0.04,
    "profit_loss_rate": 0.06,
    "monthly_excess_win_rate": 0.10,
}

LAMBDA = 0.03          # 时间衰减参数
BENCHMARK = "000300.XSHG"
```

### 2. 评分函数 `score_indicator(name, value)` → float

- 输入：指标名 + 原始值
- 对 `negate=True` 的指标取反：`x_input = -value`（共 4 个：`max_drawdown`, `max_drawdown_duration_days`, `excess_max_drawdown`, `tracking_error`）
- 锚点范围内用4次多项式：`score = c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4`
- 范围外用线性外推：

```python
if x_input < x_min:
    score = y_min + slope_left * (x_input - x_min)     # y_min = poly(x_min) = -30
elif x_input > x_max:
    score = y_max + slope_right * (x_input - x_max)    # y_max = poly(x_max) = 100
```

- 外推斜率的设计保障（已预计算在 INDICATORS 中）：
  `slope = max(poly_deriv(boundary), avg_slope * 0.5)`
  其中 `avg_slope = (100 - (-30)) / (x_max - x_min)`，确保外推不会比整体趋势的一半还平
- 返回该指标的分数（不封顶不封底）

### 3. 窗口分数 `score_window(summary)` → float

- 输入：单个窗口回测的 summary dict
- 从 summary 中提取13个指标值（key 名与 rqalpha 输出完全一致）
- 对每个指标调 `score_indicator()`，乘以权重，求和
- 处理 NaN：`profit_loss_rate` 可能为 NaN（无亏损天时），用 100 分替代
- 返回该窗口的加权综合分

### 4. 滚动窗口回测 `run_rolling_backtests(strategy_file, cash)` → list[dict]

- 生成37个窗口的起止日期，规则如下：
  ```python
  from dateutil.relativedelta import relativedelta
  import datetime

  base_start = datetime.date(2016, 2, 1)
  windows = []
  for i in range(37):
      start = base_start + relativedelta(months=3 * i)     # 步长3个自然月
      end = start + relativedelta(years=1) - relativedelta(days=1)  # 窗口1年，闭区间
      windows.append((start, end))
  # 第1个窗口: 2016-02-01 ~ 2017-01-31
  # 第2个窗口: 2016-05-01 ~ 2017-04-30
  # ...
  # 第37个窗口: 2025-02-01 ~ 2026-01-31
  ```
- 对每个窗口调用 `rqalpha.run()` 执行回测
  - 关键 config：`benchmark="000300.XSHG"`, `log_level="error"`, `plot=False`, `sys_progress.enabled=False`
  - 参考 `batch_experiments.py` 的 `run_backtest()` 模式
  - 结果从 `result["sys_analyser"]["summary"]` 取
- 打印进度（第N/37个窗口）
- 失败的窗口跳过并警告，不中断整个流程
- 返回 `[{"start": date, "end": date, "summary": dict, "score": float, "trades": DataFrame}, ...]`（summary 保留完整原始指标值供季度投影使用，trades 保留原始交易记录供交易日志使用）

### 5. 交易日志 `build_trade_log(windows, level)` → list[dict]

- 根据 level 筛选窗口：
  - `"low"`（默认）：仅最后 1 个窗口
  - `"mid"`：最后 4 个窗口（约最近 1 年）
  - `"high"`：全部窗口
- 对筛选出的窗口，将逐笔 BUY/SELL 配对为**完整交易（round-trip）**：

**注意**：rqalpha 的 trades DataFrame 使用 `datetime`（DatetimeIndex）作为索引，不是普通列。遍历时需用 `row.name` 获取时间戳。

```python
def pair_trades(trades_df):
    """将逐笔 BUY/SELL 按 FIFO 配对为 round-trip 交易"""
    round_trips = []
    # trades_df 的 index 是 DatetimeIndex（即 datetime），列包含:
    #   order_book_id, symbol, side("BUY"/"SELL"), last_price, last_quantity 等
    for ob_id, group in trades_df.groupby("order_book_id"):
        symbol = group.iloc[0]["symbol"]  # 股票名称，如"平安银行"
        buy_queue = []  # FIFO 队列
        for _, row in group.iterrows():
            trade_dt = row.name  # datetime 是索引，通过 row.name 获取
            if row["side"] == "BUY":
                buy_queue.append({
                    "datetime": trade_dt,
                    "price": row["last_price"],
                    "quantity": row["last_quantity"],
                })
            elif row["side"] == "SELL" and buy_queue:
                sell_qty = row["last_quantity"]
                # FIFO 匹配：逐个消耗 buy_queue
                while sell_qty > 0 and buy_queue:
                    buy = buy_queue[0]
                    matched_qty = min(buy["quantity"], sell_qty)
                    pnl_per_share = row["last_price"] - buy["price"]
                    pnl_amount = pnl_per_share * matched_qty
                    pnl_pct = pnl_per_share / buy["price"]
                    round_trips.append({
                        "order_book_id": ob_id,
                        "symbol": symbol,
                        "buy_datetime": buy["datetime"],
                        "buy_price": buy["price"],
                        "sell_datetime": trade_dt,
                        "sell_price": row["last_price"],
                        "quantity": matched_qty,
                        "pnl_amount": pnl_amount,      # 盈亏金额
                        "pnl_pct": pnl_pct,             # 盈亏比例
                        "label": "盈利" if pnl_amount > 0 else "亏损",
                    })
                    buy["quantity"] -= matched_qty
                    sell_qty -= matched_qty
                    if buy["quantity"] <= 0:
                        buy_queue.pop(0)
        # 窗口结束时 buy_queue 中可能有未卖出的持仓，标记为"持仓中"
        for remaining in buy_queue:
            if remaining["quantity"] > 0:
                round_trips.append({
                    "order_book_id": ob_id,
                    "symbol": symbol,
                    "buy_datetime": remaining["datetime"],
                    "buy_price": remaining["price"],
                    "sell_datetime": None,              # 未卖出
                    "sell_price": None,
                    "quantity": remaining["quantity"],
                    "pnl_amount": None,                 # 未实现盈亏不计入
                    "pnl_pct": None,
                    "label": "持仓中",
                })
    return round_trips
```

- 配对后按 sell_datetime 排序（"持仓中"记录排在最后），计算累计盈亏（cumulative_pnl，仅统计已完成交易）
- 每个 round-trip 记录包含：

| 字段 | 说明 | 示例 |
|---|---|---|
| `window` | 所属窗口编号 | `#35` |
| `order_book_id` | 证券代码 | `000001.XSHE` |
| `symbol` | 股票名称 | `平安银行` |
| `buy_datetime` | 买入时间 | `2025-03-15` |
| `buy_price` | 买入价格 | `12.50` |
| `sell_datetime` | 卖出时间（持仓中为 `—`） | `2025-04-20` |
| `sell_price` | 卖出价格（持仓中为 `—`） | `13.80` |
| `quantity` | 交易数量 | `1000` |
| `pnl_amount` | 盈亏金额（持仓中为 `—`） | `+1300.00` |
| `pnl_pct` | 盈亏比例（持仓中为 `—`） | `+10.4%` |
| `label` | 盈亏标签 | `盈利` / `亏损` / `持仓中` |
| `cumulative_pnl` | 累计盈亏（持仓中不计入） | `+5200.00` |

#### 输出格式

打印为表格，标题行包含汇总统计：
```
【交易日志】(最近1个窗口, 共12笔交易, 胜率 58.3%, 持仓中 2笔)

 #   证券代码       股票名称   买入日期     买入价   卖出日期     卖出价    数量   盈亏金额    盈亏比例  标签   累计盈亏
 1   000001.XSHE  平安银行   2025-02-03   12.50   2025-03-15   13.80   1000   +1300.00   +10.4%  盈利   +1300.00
 2   000001.XSHE  平安银行   2025-03-20   14.00   2025-04-10   13.20   1000   -800.00    -5.7%   亏损   +500.00
 ...
12   600519.XSHG  贵州茅台   2025-12-10   1850.00   —          —       100    —          —      持仓中  +5200.00
```

**汇总统计计算**：
- "共N笔交易"：仅统计已完成交易（排除"持仓中"）
- "胜率"：已完成交易中 pnl_amount > 0 的笔数 / 已完成交易总笔数
- "持仓中 N笔"：仅当有未配对买入时显示

### 6. 季度网格投影 `project_to_quarters(windows)` → tuple[dict, dict]

- 建立季度网格（从最早窗口的起始季度到最晚窗口的结束季度）

#### 季度定义与覆盖判定

使用自然季度：Q1(1-3月)、Q2(4-6月)、Q3(7-9月)、Q4(10-12月)。
每个季度用 `(year, quarter)` 元组标识，如 `(2016, 1)` 表示 2016年Q1。

窗口 W 覆盖季度 Q 的条件：窗口的时间范围与该季度有**任意重叠**。

```python
def get_covered_quarters(start_date, end_date):
    """返回窗口 [start_date, end_date] 覆盖的所有季度列表"""
    quarters = []
    # 起始季度
    q_start = (start_date.year, (start_date.month - 1) // 3 + 1)
    # 结束季度
    q_end = (end_date.year, (end_date.month - 1) // 3 + 1)
    # 遍历 q_start 到 q_end 的所有季度
    year, q = q_start
    while (year, q) <= q_end:
        quarters.append((year, q))
        q += 1
        if q > 4:
            q = 1
            year += 1
    return quarters

# 示例：窗口 2016-02-01 ~ 2017-01-31 覆盖 (2016,1), (2016,2), (2016,3), (2016,4), (2017,1)
```

- 对每个季度，找出所有覆盖它的窗口：
  - **分数投影**：取覆盖窗口的综合分数平均值
  - **原始指标投影**：对每个原始指标（annualized_returns, max_drawdown, sharpe, win_rate），取覆盖窗口的原始值平均值
- 返回两个有序字典：
  - `quarterly_scores`: `{(2016, 1): score, ...}`
  - `quarterly_raw_indicators`: `{(2016, 1): {"annualized_returns": x, "max_drawdown": x, "sharpe": x, "win_rate": x}, ...}`

### 7. 综合得分与核心指标 `compute_composite_score(quarterly_scores, quarterly_raw_indicators)` → tuple[float, dict]

- 按时间从近到远排序季度，**i=0 为最近季度**（权重最大），i 递增越远衰减越多
- 对每个季度施加 `exp(-0.03 * i)` 衰减权重（最近 vs 最远约 3:1，近2年贡献约50%权重）
- 加权平均综合分数，得到最终综合得分：
```python
composite_score = sum(quarterly_score[i] * exp(-0.03 * i) for i in range(n)) / sum(exp(-0.03 * i) for i in range(n))
```
- 对4个核心指标（annualized_returns, max_drawdown, sharpe, win_rate）同样做时间衰减加权平均，用于输出展示
- 返回 `(composite_score, {"annualized_returns": x, "max_drawdown": x, "sharpe": x, "win_rate": x})`

### 8. 稳定性评分 `compute_stability_score(quarterly_scores)` → float

- 输入：季度得分列表（时间衰减前的原始值）
- 三个维度分别计算子分数（分段线性插值，超出边界按 0 或 100）：

```python
# --- 维度1: CV（变异系数），权重 50% ---
# 锚点: CV=0.5→0分, CV=0.25→50分, CV=0.1→100分（越小越好）
if abs(mean(scores)) < 1e-6:
    cv = float('inf')  # mean≈0 时 CV 极大，会被 cv>=0.5 捕获给 0 分
else:
    cv = std(scores) / abs(mean(scores))
if cv >= 0.5:
    cv_score = 0
elif cv >= 0.25:
    cv_score = (0.5 - cv) / (0.5 - 0.25) * 50        # 0→50
elif cv >= 0.1:
    cv_score = 50 + (0.25 - cv) / (0.25 - 0.1) * 50   # 50→100
else:
    cv_score = 100

# --- 维度2: 最差季度得分，权重 30% ---
# 锚点: worst=-10→0分, worst=25→50分, worst=50→100分（越高越好）
worst = min(scores)
if worst <= -10:
    worst_score = 0
elif worst <= 25:
    worst_score = (worst - (-10)) / (25 - (-10)) * 50        # 0→50
elif worst <= 50:
    worst_score = 50 + (worst - 25) / (50 - 25) * 50          # 50→100
else:
    worst_score = 100

# --- 维度3: 最长连续低分（<30分）季度数，权重 20% ---
# 锚点: consec=5→0分, consec=2→50分, consec=0→100分（越少越好）
consec = max_consecutive_below(scores, threshold=30)
if consec >= 5:
    consec_score = 0
elif consec >= 2:
    consec_score = (5 - consec) / (5 - 2) * 50        # 0→50
elif consec >= 0:
    consec_score = 50 + (2 - consec) / (2 - 0) * 50    # 50→100
else:
    consec_score = 100

# --- 加权合成 ---
stability_score = 0.5 * cv_score + 0.3 * worst_score + 0.2 * consec_score
```

### 9. 市场环境适应性 `compute_market_env_scores(quarterly_scores, benchmark_quarterly_returns)` → dict

- 用沪深300的季度涨跌幅给每个季度打标签（>8% 牛市，<-8% 熊市，其他震荡）
- 按标签分组，取该环境下所有季度得分的简单平均值（不做时间衰减）
- 某环境无季度时输出 "N/A"
- 返回 `{"牛市": score|"N/A", "震荡": score|"N/A", "熊市": score|"N/A"}`

**注意**：`benchmark_quarterly_returns` 的季度标识 `(year, quarter)` 必须与 `quarterly_scores` 使用相同的自然季度定义，以便正确关联。两者都使用 `(year, quarter)` 元组作为 key。

#### 沪深300季度涨跌幅的获取

```python
import h5py, numpy as np, pandas as pd, os

# 读取 bundle 中的 indexes.h5
bundle_path = os.path.expanduser("~/.rqalpha/bundle")
with h5py.File(os.path.join(bundle_path, "indexes.h5"), "r") as h5:
    bars = h5["000300.XSHG"][:]
    # bars 是结构化数组，字段: datetime(uint64, YYYYMMDD), open, close, high, low, volume

# 转为 DataFrame
df = pd.DataFrame(bars)
df["date"] = pd.to_datetime(df["datetime"].astype(str), format="%Y%m%d%H%M%S")
df = df.set_index("date").sort_index()

# 按季度 resample，取每季度最后一个交易日的 close
quarterly_close = df["close"].resample("QE").last()

# 季度涨跌幅 = 本季末 close / 上季末 close - 1
quarterly_returns = quarterly_close.pct_change().dropna()

# 转为 {(year, quarter): return_value} 字典，与 quarterly_scores 对齐
benchmark_quarterly_returns = {}
for date, ret in quarterly_returns.items():
    benchmark_quarterly_returns[(date.year, (date.month - 1) // 3 + 1)] = ret

# 打标签
# > 0.08 → "牛市", < -0.08 → "熊市", 其他 → "震荡"
```

### 10. 主函数 `main()`

- argparse 解析命令行：strategy_file, --cash, --log (low/mid/high, 默认 low)
- 调用滚动回测
- 季度投影（同时投影分数和原始指标值）
- 综合得分 + 核心指标（时间衰减加权）
- 稳定性评分
- 市场环境
- 按以下格式输出最终报告：
```
【策略综合得分】 xx.x 分 | 收益 + 风险 + 指标加权总分
【策略稳定得分】  xx 分 | 策略综合得分相同时，反映策略的泛化性
【核心指标】年化 xx.x% | 回撤 xx.x% | 夏普 x.xx | 胜率 xx.x%
【市场环境】牛市 xx.x | 震荡 xx.x | 熊市 xx.x
```
核心指标（年化收益率、最大回撤、夏普率、日胜率）为各季度原始值经时间衰减加权后的均值，与综合得分同口径。

- 构建交易日志（根据 --log 参数筛选窗口，FIFO 配对，输出表格）
- 交易日志紧跟在评分报告之后输出

## 关键实现细节

### rqalpha 调用方式
```python
from rqalpha import run

config = {
    "base": {
        "start_date": start_date,
        "end_date": end_date,
        "accounts": {"stock": cash},
        "frequency": "1d",
    },
    "extra": {"log_level": "error"},
    "mod": {
        "sys_analyser": {"benchmark": "000300.XSHG", "plot": False},
        "sys_progress": {"enabled": False},
    },
}
# 读取策略文件源码，用 run() + source_code 方式执行
# 注意：使用 source_code 参数时无需设置 config.base.strategy_file
with open(strategy_file) as f:
    source_code = f.read()
result = run(config, source_code=source_code)

# run() 在回测异常时可能返回 None，必须检查
if result is None:
    # 该窗口回测失败，跳过
    print(f"警告: 窗口 {start_date}~{end_date} 回测失败，跳过")
    continue

summary = result["sys_analyser"]["summary"]
trades = result["sys_analyser"]["trades"]  # DataFrame，index 为 DatetimeIndex
```

**配置说明**：
- `benchmark` 设置在 `sys_analyser` mod 下，不是 `base` 下（`base.benchmark` 已废弃）
- 设置了 `benchmark` 后，`record` 会自动启用，无需显式设置 `"record": True`
- `plot=False` 防止弹出图表窗口

### 13个指标的 summary key 映射
全部为 summary dict 的顶层 key，直接用 `summary["total_returns"]` 等取值，无嵌套。

完整 key 对照：
| 指标 | summary key | 说明 |
|---|---|---|
| total_returns | `total_returns` | 累计收益率 |
| annualized_returns | `annualized_returns` | 年化收益率 |
| excess_annual_returns | `excess_annual_returns` | 几何超额年化收益率 |
| max_drawdown | `max_drawdown` | 最大回撤，**正数**（如 0.15 表示 15% 回撤） |
| max_drawdown_duration_days | `max_drawdown_duration_days` | 最大回撤持续天数，整数 |
| excess_max_drawdown | `excess_max_drawdown` | 超额最大回撤，正数 |
| tracking_error | `tracking_error` | 跟踪误差，正数 |
| sharpe | `sharpe` | 夏普率 |
| sortino | `sortino` | 索提诺比率 |
| information_ratio | `information_ratio` | 信息比率 |
| win_rate | `win_rate` | 日胜率（正收益天数/总天数） |
| profit_loss_rate | `profit_loss_rate` | 盈亏比，mean(盈利日PnL)/abs(mean(亏损日PnL))；无亏损天时为 NaN |
| monthly_excess_win_rate | `monthly_excess_win_rate` | 月度超额胜率，月度频率下跑赢基准的月份占比（如12个月中8个月跑赢 → 0.667） |

**符号约定**：`max_drawdown`、`excess_max_drawdown`、`tracking_error` 在 rqalpha 中均为**正数**。评分时 `negate=True` 的指标先取反 `x_input = -x_raw`，使其落入多项式定义域（负半轴）。例如 max_drawdown=0.15 → x_input=-0.15 → 落入 [-0.35, -0.05] 范围内。

### 沪深300季度涨跌幅
从 `~/.rqalpha/bundle/indexes.h5` 读取 000300.XSHG 的日线数据，resample 到季度，计算涨跌幅。

### 错误处理
- 窗口回测失败：`rqalpha.run()` 可能返回 `None`（异常中断）或抛出异常，均视为该窗口失败。用 `try/except` 包裹，跳过并打印警告，最终报告中注明失败窗口数
- 成功窗口不足：若成功窗口 < 5 个，直接报错退出，不输出评分（数据量不足以做出可靠评估）
- 某指标为 NaN：`profit_loss_rate` 无亏损天时为 NaN，视为满分（100分）；其他 NaN 给 0 分
- 季度无窗口覆盖：不纳入计算
- 策略无交易：某窗口策略完全未交易时，`trades` DataFrame 为空，交易日志中该窗口无记录，但评分指标仍正常计算（回测结果仍包含 total_returns=0 等）

## 验证

用两个已有策略测试：
```bash
python skyeye/strategy_scorer.py rqalpha/examples/demo_strategy.py
python skyeye/strategy_scorer.py rqalpha/examples/golden_cross.py
```

预期：
- 37个窗口都能跑通（或大部分跑通）
- 两个策略得分不同，能区分好坏
- 输出格式符合设计文档
- 单次运行时间合理（37个1年窗口，每个几秒，总计约2-5分钟）
