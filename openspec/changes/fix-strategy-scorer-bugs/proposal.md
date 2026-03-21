## Why

`skyeye/strategy_scorer.py` 存在多个 bug，包括两个 Critical 级别的崩溃问题（多窗口时 `list.index()` 比较 DataFrame 导致 ValueError、窗口编号不匹配），以及 NaN 处理遗漏、多股票累计盈亏错误、排序缺失、错误处理缺失等问题。这些 bug 影响了打分器在多窗口和多股票场景下的正确性和稳定性。

## What Changes

- 修复 `list.index(w)` 导致多窗口崩溃：在 results 中保存原始窗口编号 `idx`，所有引用处改用 `w["idx"]`
- 修复窗口编号不匹配：统一使用存储的 `idx` 字段而非列表位置
- 修复 NaN 处理遗漏 `np.float64(np.nan)`：改用 `pd.isna()` 替代 `isinstance(float) + math.isnan`
- 修复多股票时"累计盈亏"只反映单只股票：引入组合级别 `total_realized_pnl`
- 修复 `flatten_trades` 未按时间排序：遍历前 `sort_index()`
- 修复 `get_benchmark_quarterly_returns()` 缺少错误处理：添加文件/数据集存在性检查
- 修复着色替换逻辑脆弱的 `str.replace` 问题：改为先 pad 再拼接 ANSI 码
- 修复 `flatten_trades` 未处理卖空超量：添加 `min(qty, holding)` 保护
- 修复 `--plot` 帮助文本描述不准确：改为"保存为 PNG 文件"
- 修复 `max_drawdown` 显示符号问题：使用 `abs()` 显示正数回撤

## Capabilities

### New Capabilities
- `scorer-bug-fixes`: 修复 `skyeye/strategy_scorer.py` 中的 10 个已知 bug，涵盖崩溃修复、数据正确性、健壮性增强和显示修正

### Modified Capabilities

## Impact

- 仅影响 `skyeye/strategy_scorer.py` 单文件
- 不涉及 API 变更或依赖变化
- `flatten_trades` 返回的 records 结构新增 `total_realized_pnl` 字段
- `run_rolling_backtests` 返回的 results 新增 `idx` 字段
