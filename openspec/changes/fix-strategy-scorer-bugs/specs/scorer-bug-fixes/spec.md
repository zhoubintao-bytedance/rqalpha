## ADDED Requirements

### Requirement: 多窗口场景不崩溃
`run_rolling_backtests` 返回的每个结果 dict SHALL 包含 `idx` 字段（原始窗口编号，1-based）。`build_trade_log` 和 `plot_trades` 中 SHALL 使用 `w["idx"]` 获取窗口编号，不得使用 `list.index(w)`。

#### Scenario: 多窗口回测成功完成
- **WHEN** 运行包含多个成功窗口的回测（窗口数 > 1），且存在交易记录
- **THEN** `build_trade_log` 和 `plot_trades` 正常执行，不抛出 ValueError

#### Scenario: 有失败窗口时编号正确
- **WHEN** 37 个窗口中部分失败（如窗口 #5 失败），成功窗口 results 列表中不连续
- **THEN** 交易日志和图表中显示的窗口编号 SHALL 与原始窗口编号一致（如 #4, #6），而非结果列表中的位置编号

### Requirement: NaN 值正确处理
`score_window` 和 `project_to_quarters` 中的 NaN 检测 SHALL 使用 `pd.isna()` 而非 `isinstance(raw, float) and math.isnan(raw)`。

#### Scenario: np.float64(nan) 被正确识别
- **WHEN** `summary` 中某指标值为 `np.float64(np.nan)`
- **THEN** `score_window` SHALL 将其视为缺失值，使用默认分数（profit_loss_rate 用 100，其他用 0）

#### Scenario: 季度投影跳过 NaN
- **WHEN** 窗口 summary 中某核心指标为 `np.float64(np.nan)`
- **THEN** `project_to_quarters` SHALL 不将该值加入季度平均计算

### Requirement: 组合级别累计盈亏
`flatten_trades` SHALL 维护组合级别的 `total_realized_pnl`，在每条 record 中包含该字段。表格和窗口汇总 SHALL 使用 `total_realized_pnl` 而非单股票的 `realized_pnl`。

#### Scenario: 多股票交易的累计盈亏
- **WHEN** 策略在同一窗口内交易股票 A 和股票 B，A 盈利 1000，B 亏损 500
- **THEN** 最终 record 的 `total_realized_pnl` SHALL 为 500，而非仅反映最后一只股票的盈亏

### Requirement: 交易记录按时间排序
`flatten_trades` 在遍历 `trades_df` 前 SHALL 先调用 `sort_index()` 确保按时间顺序处理。

#### Scenario: 非单调递增索引
- **WHEN** `trades_df` 的索引非时间排序
- **THEN** `flatten_trades` 输出的 records SHALL 按交易时间升序排列

### Requirement: 基准数据读取容错
`get_benchmark_quarterly_returns` SHALL 在 bundle 文件不存在或数据集缺失时返回空 dict，不得抛出异常。

#### Scenario: bundle 文件不存在
- **WHEN** `~/.rqalpha/bundle/indexes.h5` 文件不存在
- **THEN** 函数 SHALL 返回空 dict `{}`

#### Scenario: 数据集缺失
- **WHEN** `indexes.h5` 存在但不包含 `000300.XSHG` 数据集
- **THEN** 函数 SHALL 返回空 dict `{}`

### Requirement: 着色替换不错位
表格输出中 ANSI 着色 SHALL 不使用 `str.replace()` 进行文本替换，而是通过索引定位直接拼接颜色码，避免子串匹配错误。

#### Scenario: 值为 padding 子串
- **WHEN** 某列值为 "1"，而 padded 字符串中其他位置也包含 "1"
- **THEN** 着色 SHALL 只应用于正确的列位置，不影响其他列

### Requirement: 卖出超量保护
`flatten_trades` 在处理 SELL 时 SHALL 将卖出数量限制为不超过当前持仓量。

#### Scenario: 卖出量大于持仓量
- **WHEN** SELL 的 qty > 当前 holding quantity
- **THEN** 实际使用的卖出量 SHALL 为 `min(qty, holding_quantity)`，持仓不会变为负数

### Requirement: --plot 帮助文本准确
`--plot` 选项的帮助文本 SHALL 描述为"保存为 PNG 文件"，不得描述为"弹窗显示"。

#### Scenario: 查看帮助信息
- **WHEN** 用户运行 `--help`
- **THEN** `--plot` 的描述 SHALL 包含"保存为 PNG"相关文字

### Requirement: max_drawdown 显示为正数
最大回撤的百分比显示 SHALL 使用 `abs()` 取绝对值，显示为正数。

#### Scenario: 回撤值为负数
- **WHEN** `max_drawdown` 原始值为 -0.15
- **THEN** 显示 SHALL 为 `15.0%` 而非 `-15.0%`
