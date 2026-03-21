# strategy_scorer.py Bug 记录

当前主实现：`skyeye/strategy_scorer.py`

## Bug

### 1. [Critical] `window_results.index(w)` 导致多窗口时直接崩溃

`strategy_scorer.py:402`、`504` 用 `window_results.index(w)` 取窗口编号。`list.index()` 会逐个比较 dict，而 dict 里含 `trades` 字段（DataFrame），触发 `ValueError: The truth value of a DataFrame is ambiguous`。**只要成功窗口数 > 1，交易日志就会崩溃**。

**修复**：在 `run_rolling_backtests` 的 results 中保存原始窗口编号：

```python
results.append({
    "idx": idx,        # ← 添加这个
    "start": start,
    "end": end,
    ...
})
```

然后所有地方用 `w["idx"]` 代替 `window_results.index(w) + 1`（同时修复窗口编号不匹配问题）。

### 2. [Critical] 窗口编号不匹配

即便修掉上面的 DataFrame 崩溃，`window_results.index(w) + 1` 取的是结果列表中的位置，不是原始窗口号。在"有失败窗口"或"只跑指定窗口"时，日志的 `#` 编号会对不上真实窗口编号。同样通过存储 `idx` 字段解决。

### 3. [High] NaN 处理只识别 Python float，遗漏 `np.float64(np.nan)`

`strategy_scorer.py:208`、`612`：`isinstance(raw, float) and math.isnan(raw)` 只能捕获 Python 原生 `float` 的 NaN。pandas 返回的 `np.float64(np.nan)` 不是 `float` 的实例，会漏过检查，导致窗口/季度均值变成 NaN 并向后传播，最终综合得分变为 NaN。

**修复**：改用 `pd.isna(raw)` 或 `math.isnan(float(raw))`。

### 4. [High] "累计盈亏"是单股票维度，多股票时明显错误

`strategy_scorer.py:348`、`448`、`516`：`realized_pnl` 存在每个 `holding` 下，是单只股票的累计已实现盈亏。表格里直接显示成"累计盈亏"，窗口汇总又用最后一条记录的 `realized_pnl` 作为整个窗口已实现盈亏。当策略交易多只股票时，这个值只反映最后一只股票的盈亏，而非组合整体。

**修复**：维护一个组合级别的 `total_realized_pnl`，每笔卖出交易时累加，表格和汇总都使用该值。

### 5. [Medium] `flatten_trades` 没有按时间排序

`strategy_scorer.py:340`：直接遍历 `trades_df.iterrows()` 但未先排序。如果 `trades_df` 索引非单调递增，持仓与盈亏计算会被打乱，进而影响"累计盈亏/持仓成本/窗口汇总"。

**修复**：在遍历前加 `trades_df = trades_df.sort_index()`。

### 6. [Medium] `get_benchmark_quarterly_returns()` 缺少错误处理

`strategy_scorer.py:836` 直接 `h5py.File(...)` 打开文件，如果 bundle 不存在或 `000300.XSHG` 不在里面会直接崩溃。而同文件的 `read_daily_bars()` 做了很好的容错处理。

### 7. 着色替换逻辑脆弱

`strategy_scorer.py:527`：

```python
padded = padded.replace(plain_row[i], colored_row[i], 1)
```

如果 `plain_row[i]` 的值恰好是 padding 内容的子串（比如值为 `"1"` 而 padded 的空格数恰好让某个位置出现 `"1"`），就会替换到错误的位置。建议改为先 pad 再拼接 ANSI 码，而不是用 replace。

### 8. `flatten_trades` 未处理卖空超量

`strategy_scorer.py:372-373`：如果因为数据问题导致 SELL 的 qty > 当前持仓量，`h["quantity"]` 会变负，后续的 avg_cost 计算会出错。建议加个保护：

```python
sell_qty = min(qty, h["quantity"])
```

## 实现不完整

### 9. `--plot` 描述和实际行为不一致

`strategy_scorer.py:896` 帮助文本写的是"弹窗显示"，但 `line 765` 实际用的是 `matplotlib.use("Agg")` 非交互后端，只保存 PNG 不弹窗。帮助文本需要改为"保存为 PNG 文件"。

### 10. `max_drawdown` 显示可能有符号问题

`strategy_scorer.py:1073` 和 `1123`：回撤用强制红色显示，但 RQAlpha 的 `max_drawdown` 本身就是**负值**（如 -0.15）。乘以 100 后显示 `-15.0%`，视觉上没问题，但语义上"回撤 -15.0%"有点奇怪，通常展示为正数。建议用 `abs(max_dd)`。

## 测试缺口

- 缺少多窗口交易日志测试（能直接暴露 `list.index` + DataFrame 比较崩溃）
- 缺少 NaN 处理的单测（`np.float64(np.nan)` 传入 `score_window` / `project_to_quarters`）
- 缺少多股票交易日志累计盈亏的单测（验证组合级别 vs 单股票级别）
- 缺少 `flatten_trades` 在非时间排序输入下的行为测试
