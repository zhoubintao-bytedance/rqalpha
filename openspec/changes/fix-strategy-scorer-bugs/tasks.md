## 1. Critical: 窗口编号修复（Bug #1, #2）

- [x] 1.1 在 `run_rolling_backtests` 的 `results.append()` 中新增 `"idx": idx` 字段（line 337）
- [x] 1.2 修改 `build_trade_log` 中 `idx = window_results.index(w) + 1` 为 `idx = w["idx"]`（line 428, 567）
- [x] 1.3 修改 `plot_trades` 中 `idx = window_results.index(w) + 1` 为 `idx = w["idx"]`（line 850）

## 2. NaN 处理修复（Bug #3）

- [x] 2.1 修改 `score_window` 中 NaN 检测为 `pd.isna(raw)`（line 232）
- [x] 2.2 修改 `project_to_quarters` 中 NaN 检测为 `pd.isna(val)`（line 676）

## 3. 组合级别累计盈亏（Bug #4）

- [x] 3.1 在 `flatten_trades` 中新增 `total_realized_pnl` 变量，每笔卖出时累加所有股票盈亏
- [x] 3.2 在每条 record 中写入 `total_realized_pnl` 字段
- [x] 3.3 修改 `build_trade_log` 表格中"累计盈亏"列使用 `total_realized_pnl`
- [x] 3.4 修改 `build_trade_log` 窗口汇总中 `realized_pnl` 使用 `total_realized_pnl`

## 4. 交易记录排序（Bug #5）

- [x] 4.1 在 `flatten_trades` 遍历前添加 `trades_df = trades_df.sort_index()`（line 366 前）

## 5. 基准数据容错（Bug #6）

- [x] 5.1 在 `get_benchmark_quarterly_returns` 中添加文件存在性检查，不存在时返回空 dict
- [x] 5.2 添加数据集 `000300.XSHG` 存在性检查

## 6. 着色替换逻辑修复（Bug #7）

- [x] 6.1 重写表格输出的着色逻辑：对需要着色的列，在 pad 后直接拼接 ANSI 码，不使用 `str.replace()`

## 7. 卖空超量保护（Bug #8）

- [x] 7.1 在 `flatten_trades` SELL 分支中添加 `sell_qty = min(qty, h["quantity"])` 保护

## 8. 帮助文本与显示修正（Bug #9, #10）

- [x] 8.1 修改 `--plot` 帮助文本从"弹窗显示"改为"保存为 PNG 文件"（HELP_TEXT line 929）
- [x] 8.2 修改 `max_drawdown` 显示使用 `abs()` 取绝对值（line 1098, 1135 附近）
