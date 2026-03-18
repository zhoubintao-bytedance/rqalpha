## Context

`strategy_scorer.py` 是策略打分器的核心文件，用于对交易策略进行滚动窗口回测并综合评分。当前版本存在 10 个已知 bug，记录在 `bugs.md` 中。其中 2 个 Critical 级别 bug 导致多窗口场景直接崩溃，其余 bug 影响数据正确性和用户体验。

当前代码约 1175 行，所有修改都在这一个文件内完成。

## Goals / Non-Goals

**Goals:**
- 修复所有 10 个已知 bug，确保多窗口、多股票场景下正常运行
- 保持现有接口和输出格式不变
- 每个修复点尽量最小化变更范围

**Non-Goals:**
- 不重构整体架构
- 不新增功能
- 不修改评分算法或权重参数

## Decisions

### D1: 窗口编号追踪方案
**决定**: 在 `run_rolling_backtests` 的 results dict 中新增 `idx` 字段存储原始窗口编号（1-based）。所有使用 `window_results.index(w)` 的地方改用 `w["idx"]`。

**理由**: 这是 bugs.md 推荐的方案，改动最小，同时解决 Bug #1（DataFrame 比较崩溃）和 Bug #2（编号不匹配）。

### D2: NaN 检测统一使用 pd.isna()
**决定**: 将 `isinstance(raw, float) and math.isnan(raw)` 替换为 `pd.isna(raw)`。

**理由**: `pd.isna()` 能处理 Python float NaN、`np.float64(nan)`、`None` 等所有情况，是最健壮的方案。项目已依赖 pandas，无需引入新依赖。

### D3: 组合级别累计盈亏
**决定**: 在 `flatten_trades` 中新增一个函数级别的 `total_realized_pnl` 变量，每笔卖出交易时累加所有股票的已实现盈亏，写入每条 record 的 `total_realized_pnl` 字段。表格和汇总使用该字段替代单股票的 `realized_pnl`。

**理由**: 保留单股票 `realized_pnl` 用于内部计算，新增组合级别字段避免破坏现有逻辑。

### D4: 着色方案改为索引定位
**决定**: 在输出着色表格时，不使用 `str.replace()` 做文本替换，而是在 pad 完成后，根据列索引位置直接拼接 ANSI 码。

**理由**: 避免值恰好是 padding 子串时替换到错误位置的问题。

### D5: get_benchmark_quarterly_returns 错误处理
**决定**: 参照同文件 `read_daily_bars()` 的模式，添加文件存在性和数据集存在性检查，不存在时返回空 dict。

**理由**: 与现有代码风格保持一致，调用方已能处理空返回。

## Risks / Trade-offs

- **[Risk] `total_realized_pnl` 字段新增改变了 `flatten_trades` 的返回结构** → 仅在内部使用，不影响外部 API。旧的 `realized_pnl` 字段保留用于单股票维度。
- **[Risk] `pd.isna()` 对非数值类型也返回 True（如 None）** → 当前代码中 `raw` 来自 `summary.get(name)` 或 `w["summary"].get(ind)`，None 本身就应该被跳过，行为一致。
- **[Risk] `sort_index()` 可能改变原有输出顺序** → 这正是期望行为，按时间排序是正确的交易流水顺序。
