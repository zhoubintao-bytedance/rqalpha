# TX1 操作手册

TX1 横截面选股策略的因子研究、模型训练、回测评分全流程指南。

## 当前最优配置（2026-03-31 验证）

**4 因子 + rank labels + 月度重平衡**，已通过 rqalpha 真实回测验证。

| 参数 | 值 | 说明 |
|------|-----|------|
| 因子 | mom_40d, volatility_20d, reversal_5d, amihud_20d | 定义在 `evaluator.py:FEATURE_COLUMNS` |
| Label | rank transform, 20d horizon | `labels.transform: "rank"`（DEFAULT_CONFIG 默认值） |
| 模型 | LightGBM | early_stopping=20, num_leaves=24, max_depth=5 |
| 重平衡 | 每 20 个交易日 | `portfolio.rebalance_interval: 20` |
| Hold buffer | Top50 | `portfolio.hold_top_k: 50` |
| 持仓加分 | 0.5 × pred_std | `portfolio.holding_bonus: 0.5` |
| 成本 | 佣金 8bps + 印花税 5bps + 滑点 5bps | 单边 ~13bps，双边 ~31bps |

### 最新回测结果（2023-03 ~ 2025-12，rqalpha 真实回测）

| 指标 | 值 |
|------|-----|
| 策略收益率 | **73.05%** |
| 基准收益率（沪深300） | 11.33% |
| 超额收益率 | **61.7%** |
| 年化收益率 | 22.87% |
| 年化超额收益 | 18.0% |
| 信息比率 | 1.069 |
| 夏普率 | 0.89 |
| 最大回撤 | 22.3% |

### Walk-Forward 训练指标

| 指标 | 值 |
|------|-----|
| Rank IC | **0.0474** |
| NetRet（日均） | +0.000390 |
| Folds | 14 |

## 操作流程

### 1. 挖因子

参考策略库在 `skyeye/ref_src_strategy/`（592 个策略）。

```bash
# 阅读参考策略（重点文件见 CLAUDE.md "Reference Strategy Library" 段）
cat skyeye/ref_src_strategy/README.md
```

**因子计算**在 `dataset_builder.py` 的 `DatasetBuilder.build()` 中实现。新增因子步骤：
1. 在 per-asset 循环内添加计算逻辑
2. 加入 `required_non_null` 列表
3. 加入 `ordered_columns` 列表
4. 更新 `evaluator.py:FEATURE_COLUMNS`

**数据来源**：
- 量价数据：`~/.rqalpha/bundle/stocks.h5`（OHLCV + total_turnover）
- 基准数据：`~/.rqalpha/bundle/indexes.h5`
- 行业数据：`~/.rqalpha/bundle/instruments.pk`
- 北向数据：akshare `stock_hsgt_hist_em()`（沪股通+深股通，数据截止 2024-08-16）
- 基本面数据：需 rqdatac（未接入 TX1，接入评估见下方"数据源扩展"）

**已验证因子历史**（2026-03-30）：

| 因子 | 单因子 IC | IC IR | 最终去留 | 原因 |
|------|----------|-------|---------|------|
| volatility_20d | -0.0537 | -0.2354 | **保留** | IC 最强 |
| mom_40d | -0.0489 | -0.2380 | **保留** | IR 好，动量代表 |
| amihud_20d | +0.0327 | +0.1647 | **保留** | 正交流动性信号 |
| reversal_5d | +0.0129 | +0.0705 | **保留** | 降 MaxDD，反动量对冲 |
| tw_mom_40d | -0.0598 | -0.3324 | 删除 | 与 mom_40d 相关 0.82 |
| idio_vol_60d | -0.0607 | -0.2823 | 删除 | 与 volatility_20d 相关 0.80 |
| cgo_60d | -0.0326 | -0.1627 | 删除 | 与 mom_40d 相关 0.78 |
| north_net_flow_20d | — | — | 删除 | 市场级特征，NaN 相关，同 regime_support 问题 |
| excess_mom_20d | -0.0400 | -0.2071 | 删除 | 与 mom_20d 完全冗余（r=1.00） |
| regime_support | — | — | 删除 | gain importance 虚高，市场择时非选股 |
| volume_ratio_20d | -0.0028 | -0.0203 | 删除 | IC 约为零 |
| close_position_20d | -0.0137 | -0.0771 | 删除 | IC 弱，与动量相关 0.46 |

**因子筛选标准**：
- 单因子 IC 绝对值 > 0.03
- 与现有因子的截面相关 < 0.5
- 用 `run_feature_experiment.py` 的相关矩阵和单因子 IC 表验证

### 2. 因子对比实验

```bash
# 跑特征工程实验（对比不同因子组合）
PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_feature_experiment \
  --universe-size 300 \
  --output-dir skyeye/artifacts/experiments/tx1_feature_eng_new
```

**输出**：
- `feature_experiment_report.txt` — 可读报告（变体对比 + 单因子 IC + 相关矩阵）
- `feature_experiment_results.json` — 结构化结果
- `single_factor_ic.json` — 各因子单因子 IC
- `correlation_matrix.csv` — 截面相关矩阵

**关键指标解读**：

| 指标 | 含义 | 及格线 | 优秀线 |
|------|------|--------|--------|
| Rank IC | 预测排序与实际收益排序的相关性 | 0.03 | >0.08 |
| IC IR | IC均值/IC标准差，信号稳定性 | 0.2 | >0.6 |
| Spread | 多头-空头收益差 | >0 | >0.01 |
| Hit% | 多头组合正收益率 | >50% | >55% |
| NetRet | 扣费后日均收益 | >0 | >0.0005 |
| MaxDD | 最大回撤 | <10% | <5% |

**修改实验变体**：编辑 `run_feature_experiment.py` 中的 `VARIANTS` 列表。

### 3. 训练模型

```bash
# 用当前最优配置训练 LightGBM 模型
python -m skyeye.products.tx1.run_baseline_experiment \
  --model lgbm \
  --output-dir skyeye/artifacts/experiments/tx1
```

**注意**：`run_baseline_experiment.py` 中必须配置 `"labels": {"transform": "rank"}`，否则走默认 raw labels，IC 会低很多（0.015 vs 0.047）。

**训练参数**来自 `config.py` 和 `baseline_models.py:LightGBMModel.DEFAULT_PARAMS`：
- Walk-forward: 3年训练 + 6月验证 + 6月测试，20日 embargo，14 个 fold
- LightGBM: 200 棵树，early_stopping=20，num_leaves=24

**输出**到 `skyeye/artifacts/experiments/tx1/tx1_baseline_lgbm/`：
- `experiment.json` — 元数据
- `folds/` — 每个 fold 的预测结果和权重（signal book 来源）

### 4. 回测评分

```bash
# 37 个滚动窗口全量评分（约 30-60 分钟）
python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py

# 快速测试（只跑最近几个窗口）
python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py -w 35-37

# 带交易日志
python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py --log mid
```

**评分体系**：13 项指标加权，37 个窗口季度投影，时间衰减（近期权重高）。

| 评级 | 分数 | 含义 |
|------|------|------|
| E | >=60 | 优秀 |
| M+ | >=30 | 良好 |
| M | >=0 | 一般 |
| M- | >=-20 | 较差 |
| I | <-20 | 淘汰 |

**前提**：必须先完成第 3 步训练，生成 artifacts。打分器从 `skyeye/artifacts/experiments/tx1/tx1_baseline_lgbm/` 读取 signal book。

### 5. 对比沪深300（可视化）

```bash
# 最近3年 vs 沪深300，保存图表
PYTHONPATH="$PWD" rqalpha run \
  -f skyeye/products/tx1/strategies/rolling_score/strategy.py \
  -s 2023-03-01 -e 2025-12-03 \
  --account stock 1000000 \
  -bm 000300.XSHG \
  --plot-save tx1_vs_hs300.png \
  -l info
```

**前提**：必须先完成第 3 步训练。图表包含策略收益、基准收益、超额收益曲线和关键指标汇总。

**注意**：训练命令的 `--output-dir` 必须从项目根目录 `rqalpha/` 运行，或使用绝对路径，否则 artifacts 会输出到错误位置。

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `dataset_builder.py` | 因子计算（所有特征工程在这里） |
| `evaluator.py` | FEATURE_COLUMNS 定义 + 预测/组合评估 |
| `label_builder.py` | 标签构建（forward excess return） |
| `baseline_models.py` | 模型定义（Linear/Tree/LightGBM） |
| `portfolio_proxy.py` | 组合构建（重平衡 + hold buffer + 持仓加分） |
| `config.py` | 默认参数配置 |
| `cost_layer.py` | 交易成本模型 |
| `splitter.py` | Walk-forward 分割器 |
| `run_feature_experiment.py` | 因子对比实验脚本 |
| `run_baseline_experiment.py` | 模型训练脚本 |
| `experiment_runner.py` | 实验执行引擎 |
| `strategies/rolling_score/strategy.py` | 回测策略文件（打分器用） |
| `strategies/rolling_score/spec.yaml` | 策略元信息 |

## 经验教训

### 因子不是越多越好
从 4→8→6 因子的实验证明，冗余因子（相关>0.5）会降低 IC 和 NetRet。截面相关矩阵是最重要的筛选工具。

### 换手率比 IC 更重要
B2（11因子）靠换手率从 0.308 降到 0.208，成本拖累从 24.1% 降到 16.3%，这比 IC 的边际改进贡献更大。月度重平衡 + 宽 hold buffer 是当前最有效的降成本手段。

### rank labels 显著优于 raw labels
rank transform 把 IC 从 0.013 提升到 0.047（3.7倍），因为 LightGBM 的 tree split 在 rank 化的均匀分布上更高效。rqalpha 真实回测中，rank labels 的策略收益率从 58% 提升到 73%，最大回撤从 33% 降到 22%。`config.py:DEFAULT_CONFIG` 已将 rank 设为默认值。

### 市场级特征要谨慎
`regime_support` 和 `north_net_flow_20d` 都是所有股票同日取值相同的特征。LightGBM 的 gain importance 会虚高（单次 split 覆盖全部样本），但它们不提供截面选股信息。

### amihud 必须用 total_turnover
之前用 `volume × close` 近似，引入了价格水平偏差。修复为直接使用 H5 中的 `total_turnover` 字段。

## 数据源扩展（备忘）

### rqdatac 基本面数据
- 已有完整封装（`rqalpha/apis/api_rqdatac.py`，42 个字段）
- 需要：`pip install rqdatac` + 凭证配置
- 评估工作量：~660 行代码，1-2 天
- 关键因子：扣非净利率、FCF/市值、ROE

### akshare 北向资金
- 个股级：`stock_hsgt_hold_stock_em()`（数据截止 2024-08-16，东方财富数据源断供）
- 市场级：`stock_hsgt_hist_em(symbol="沪股通")`（同样截止 2024-08-16）
- 根因：东方财富 API 2024-08-19 后不再返回资金流字段
