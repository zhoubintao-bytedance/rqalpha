# TX1 操作手册

TX1 当前默认基线已经切换为 `baseline_5f`。本手册只保留 2026-04-03 之后仍然有效的默认配置、实验结论和操作方法。

## 当前默认基线

来源实验目录：`skyeye/artifacts/experiments/tx1_feature_eng_session3/ab_focus`

| 项目 | 当前默认值 |
|------|------------|
| 默认因子 | `mom_40d`, `volatility_20d`, `reversal_5d`, `amihud_20d`, `turnover_stability_20d` |
| 默认特征别名 | `skyeye/products/tx1/evaluator.py:FEATURE_COLUMNS` |
| 标签 | `rank` transform, `20d` horizon |
| 默认组合 | Top20 买入, Top50 持有, 20 交易日调仓, `holding_bonus=0.5` |
| 成本假设 | 佣金 `0.0008` + 印花税 `0.0005` + 滑点 `5bps` |
| 默认策略 spec | `skyeye/products/tx1/strategies/rolling_score/spec.yaml` |
| 默认策略 profile | `skyeye/products/tx1/strategies/rolling_score/profiles/smooth.yaml` |

默认 5 因子是当前 TX1 研究链路和策略说明里的唯一 baseline。`baseline_4f` 仅保留为历史对照，不再作为默认策略。

## 最新特征实验结论

实验设置：

- 模型：`lgbm`
- 标签：`rank`
- 数据区间：`2015-03-09` 到 `2026-01-29`
- 数据集形状：`(760681, 35)`
- Walk-forward：14 folds

### 4 个核心变体结果

| Variant | #Feat | Rank IC | IC IR | Spread | NetRet(日均) | MaxDD | 结论 |
|---------|------:|--------:|------:|-------:|-------------:|------:|------|
| `baseline_5f` | 5 | 0.0503 | 0.3526 | 0.0115 | 0.000375 | 6.00% | 当前默认 baseline |
| `baseline_4f` | 4 | 0.0375 | 0.2353 | 0.0056 | 0.000319 | 5.68% | 历史参考线 |
| `baseline_5f_fundamental_filter` | 5 | 0.0188 | 0.1208 | -0.0028 | 0.000069 | 5.14% | 不采用 |
| `baseline_4f_fundamental_filter` | 4 | 0.0151 | 0.0973 | -0.0041 | 0.000026 | 5.31% | 不采用 |

### 为什么默认切到 `baseline_5f`

- 相比 `baseline_4f`，`Rank IC` 提升 `+0.0128`，相对增幅约 `34%`
- `IC IR` 提升约 `50%`
- `Spread` 从 `0.0056` 提到 `0.0115`，增幅约 `106%`
- `NetRet` 从 `0.000319` 提到 `0.000375`，相对增幅约 `17.7%`
- `fold_rank_ic_std` 从 `0.0617` 降到 `0.0553`，说明折间更稳
- 风险代价有限：`MaxDD` 只从 `5.68%` 升到 `6.00%`

对应新增因子 `turnover_stability_20d` 的单因子表现也支持保留：

| 因子 | IC Mean | IC IR | 备注 |
|------|--------:|------:|------|
| `turnover_stability_20d` | 0.0390 | 0.3023 | 当前最有效的新增增强因子 |
| `ep_ratio_ttm` | 0.0360 | 0.1631 | 下一个最值得测试的基本面连续因子 |
| `return_on_equity_ttm` | 0.0178 | 0.0948 | 可做弱增强，不适合硬过滤 |

### 明确不再采用的方向

- 不再把 `EP + ROE above_median` 作为默认 universe filter
  - 样本仅保留约 `39%`
  - 两个 fundamental-filter 变体的 `Spread` 都为负
  - `NetRet` 也显著低于未过滤版本
- 不优先添加与现有 baseline 高度共线的候选因子
  - `vol_adj_turnover_20d` 与 `volatility_20d` 相关约 `-0.99`
  - `mom_20d` 与 `excess_mom_20d` 相关约 `1.00`
  - `mom_60d` 与 `excess_mom_60d` 相关约 `0.99`
  - `ma_gap_60d` 与 `mom_40d` 相关约 `0.91`

## 默认文件约定

下面这些文件现在都以 `baseline_5f` 为默认基线：

- `skyeye/products/tx1/evaluator.py`
  - `FEATURE_COLUMNS` 指向 5 因子默认 baseline
- `skyeye/products/tx1/run_feature_experiment.py`
  - baseline delta 的基准变体改为 `baseline_5f`
  - `baseline_4f` 只保留为历史对照
- `skyeye/products/tx1/run_label_experiment.py`
  - 默认组合缓冲统一为 Top50 持有
- `skyeye/products/tx1/strategies/rolling_score/spec.yaml`
  - `signal_inputs` 更新为 5 因子
- `skyeye/products/tx1/strategies/rolling_score/profiles/baseline.yaml`
- `skyeye/products/tx1/strategies/rolling_score/profiles/smooth.yaml`
  - 默认 `hold_top_k=50`

## 常用命令

### 1. 重新跑特征实验

```bash
PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_feature_experiment \
  --universe-size 300 \
  --output-dir skyeye/artifacts/experiments/tx1_feature_eng_new
```

输出文件：

- `feature_experiment_report.txt`
- `feature_experiment_results.json`
- `variant_metrics.csv`
- `single_factor_ic.json`
- `correlation_matrix.csv`

### 2. 训练默认 baseline 模型

```bash
python -m skyeye.products.tx1.run_baseline_experiment \
  --model lgbm \
  --output-dir skyeye/artifacts/experiments/tx1
```

训练时默认使用：

- 5 因子 baseline
- `rank` 标签
- 20 日 horizon
- 20 日调仓 / Top50 持仓缓冲

### 3. 对比新实验和当前 baseline

```bash
PYTHONPATH="$PWD" python -m skyeye.products.tx1.compare_experiments \
  skyeye/artifacts/experiments/tx1/tx1_new_lgbm \
  --baseline baseline_lgbm \
  --artifacts-root skyeye/artifacts/experiments/tx1
```

比较时重点看这 4 项：

1. `rank_ic_mean` 是否继续提升
2. `net_mean_return` 是否仍然赢当前 baseline
3. 折间波动是否恶化
4. 提升是否来自大多数 folds，而不是少数窗口

### 4. 运行 rolling-score 评分

```bash
python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py
```

## 当前研发建议

- 默认主线保持 `baseline_5f`
- 下一轮优先测试：
  - `baseline_5f + ep_ratio_ttm`
  - `baseline_5f + ep_ratio_ttm + return_on_equity_ttm`
- 不要先把基本面做成硬过滤条件
- 不要优先继续堆高共线动量/趋势/波动率镜像因子
- 如果要进一步提升收益兑现，先检查组合层参数，而不是盲目继续加因子

## 关键文件

| 文件 | 用途 |
|------|------|
| `dataset_builder.py` | 因子计算 |
| `evaluator.py` | 默认 baseline 特征列与评估函数 |
| `label_builder.py` | 标签构建 |
| `portfolio_proxy.py` | TopK 组合构建 |
| `config.py` | 研究默认配置 |
| `cost_layer.py` | 成本模型 |
| `run_feature_experiment.py` | 特征实验入口 |
| `run_baseline_experiment.py` | 默认 baseline 训练入口 |
| `compare_experiments.py` | 多实验对比 |
| `strategies/rolling_score/spec.yaml` | 默认策略说明 |

## 不再保留的旧内容

本手册已经删除以下过时内容：

- 旧的 4 因子“当前最优配置”描述
- 早期 rqalpha 真实回测收益表
- 历史冗余因子长表和旧筛选结论
- 过时的经验阈值表和旧阶段结论

后续如有新的 baseline 替换，直接更新本页顶部“当前默认基线”和“最新特征实验结论”两节，不再追加旧版本结论。
