# TX1 操作手册

TX1 当前默认线已经收敛到 `baseline_5f + lgbm + Top25 买入 / Top45 持有`，执行入口是 `tx1.rolling_score@combo_b25_h45`。这份手册只保留 2026-04-05 之后仍然有效的默认配置、运行方式、定版依据和研发建议；更早的方案讨论只作为历史背景，不再视为当前口径。

如果只是找入口，先看这几份文档：

- 产品线入口：[README.md](./README.md)
- 当前可执行策略说明：[strategies/rolling_score/README.md](./strategies/rolling_score/README.md)
- 如果任务涉及实盘辅助或 `run_live_advisor`，直接跳到下文“TX1 实盘辅助使用说明”
- 历史立项 RFC：[../../docs/rfc/tianyan_tx1_strategy_v1.md](../../docs/rfc/tianyan_tx1_strategy_v1.md)
- 历史研究方法学 RFC：[../../docs/rfc/tianyan_tx1_research_design_v1_1.md](../../docs/rfc/tianyan_tx1_research_design_v1_1.md)

## 当前默认线

TX1 现在跑的不是“回测时在线重新训练模型”，而是一条冻结 artifact line 的 replay 策略。研究侧先产出 `experiment.json + weights.parquet`，策略侧再把这条样本外 signal book 回放到标准 `rolling_score` 执行栈里。

| 项目 | 当前默认值 |
|------|------------|
| 策略 ID | `tx1.rolling_score` |
| 默认 artifact line | `combo_b25_h45` |
| 默认 profile | `smooth` |
| 默认模型 | `lgbm` |
| 默认因子 | `mom_40d`, `volatility_20d`, `reversal_5d`, `amihud_20d`, `turnover_stability_20d` |
| 默认特征别名 | `skyeye/products/tx1/evaluator.py:FEATURE_COLUMNS` |
| 标签 | `rank` transform, `20d horizon` |
| 默认组合 | Top25 买入，Top45 持有，跌出 Top45 退出 |
| 调仓与组合层补丁 | 20 交易日重平衡，`holding_bonus=0.5`，`single_stock_cap=0.10`，`turnover_threshold=0.20` |
| 平滑 | `EMA halflife=5`，`ema_min_weight=0.005` |
| 成本假设 | 佣金 `0.0008` + 印花税 `0.0005` + 滑点 `5bps` |
| benchmark | `000300.XSHG` |
| 研究冻结项 | `input_window=60`，`horizon=20` |

默认 5 因子是当前 TX1 唯一保留的信号 baseline。`baseline_4f`、历史 `Top20/Top50` 组合层和多头输出保护版都还在代码里，但现在都只作为历史对照或研究支线，不是默认执行口径。

## 当前运行口径

默认线的运行顺序很简单，但有几个边界条件必须记住。

1. `build_runtime(...)` 读取 `spec.yaml`、解析 artifact line、加载 profile，并校验冻结字段。
2. `load_replay_signal_book(...)` 从实验目录的 `weights.parquet` 生成按交易日索引的 frozen target weights。
3. `before_trading(...)` 读取上一个交易日对应的 signal，做权重清洗、单票上限裁剪、可交易性过滤和 EMA 平滑。
4. `handle_bar(...)` 只在目标组合与当前组合偏离超过阈值时执行 `order_target_portfolio(...)`。

当前这条默认线有几个固定约束：

- `benchmark` 和 `artifact_line_id` 冻结在 `strategy_id + artifact_line` 这一层，profile 不能覆盖。
- 多个 fold 的 signal 日期如果重叠，以更晚 `test_start` 的 fold 为准。
- `listed_date`、`de_listed_date`、`ST`、停牌会在执行层再次过滤。
- `smooth` 是当前默认执行档位，`turnover_threshold=0.20`；`baseline` 保留旧默认 `turnover_threshold=0.30` 作为显式历史对照。
- `soft_sticky`、`sticky`、`ultra_sticky` 是历史 `Top20/Top50` 档位，只保留给对照实验，不是默认参数。

## 运行时覆盖方式

默认解析顺序是“显式传参优先，其次环境变量，然后配置文件，最后才是策略默认值”。

### artifact line 覆盖顺序

1. `build_runtime(..., artifact_line=...)`
2. 环境变量 `SKYEYE_TX1_ARTIFACT_LINE`
3. `config.extra.tx1_artifact_line`
4. `spec.yaml` 里的 `artifact_line_id`

### profile 覆盖顺序

1. `build_runtime(..., profile_name=...)`
2. 环境变量 `SKYEYE_TX1_PROFILE`
3. `config.extra.strategy_profile`
4. 默认值 `smooth`

最常见的覆盖方式如下。

覆盖 artifact line：

```bash
env SKYEYE_TX1_ARTIFACT_LINE=baseline_tree \
  uv run python -m skyeye.evaluation.rolling_score.cli \
    skyeye/products/tx1/strategies/rolling_score/strategy.py
```

覆盖 profile：

```bash
env SKYEYE_TX1_PROFILE=baseline \
  uv run python -m skyeye.evaluation.rolling_score.cli \
    skyeye/products/tx1/strategies/rolling_score/strategy.py
```

两者一起覆盖：

```bash
env SKYEYE_TX1_ARTIFACT_LINE=baseline_linear SKYEYE_TX1_PROFILE=baseline \
  uv run python -m skyeye.evaluation.rolling_score.cli \
    skyeye/products/tx1/strategies/rolling_score/strategy.py
```

## 常用命令

如果只是验证当前默认线，先跑执行链路；如果要推进研究，再跑特征实验、训练和对比。

### 1. 直接跑当前默认 rolling-score

```bash
uv run python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py
```

### 2. 用 RQAlpha 跑单区间并出图

```bash
env UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/mplconfig \
  uv run rqalpha run \
    -f skyeye/products/tx1/strategies/rolling_score/strategy.py \
    -s 2024-01-02 \
    -e 2025-12-31 \
    --account stock 1000000 \
    -bm 000300.XSHG \
    --plot-save tx1_rolling_score.png
```

### 3. 重新跑特征实验

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

### 4. 训练默认 baseline 模型

```bash
python -m skyeye.products.tx1.run_baseline_experiment \
  --model lgbm \
  --output-dir skyeye/artifacts/experiments/tx1
```

训练时默认使用：

- 5 因子 baseline
- `rank` 标签
- 20 日 horizon
- 20 日调仓 / Top25 买入 / Top45 持仓缓冲

### 5. 训练保护版多头输出试验线

```bash
python -m skyeye.products.tx1.run_baseline_experiment \
  --model lgbm \
  --output-dir skyeye/artifacts/experiments/tx1 \
  --experiment-name combo_guarded_b25_h45 \
  --enable-multi-output \
  --volatility-weight 0.15 \
  --max-drawdown-weight 0.2 \
  --enable-reliability-score \
  --holding_bonus 0.2 \
  --rebalance-interval 15
```

### 6. 对比新实验和当前训练侧基线

```bash
PYTHONPATH="$PWD" python -m skyeye.products.tx1.compare_experiments \
  skyeye/artifacts/experiments/tx1/tx1_new_lgbm \
  --baseline baseline_lgbm \
  --artifacts-root skyeye/artifacts/experiments/tx1
```

这条命令先拿训练侧基线 `baseline_lgbm` 做对照；如果候选实验准备往执行默认线推进，还要继续看它的 `rolling-score` 是否超过 `combo_b25_h45`。比较时重点看这 4 项：

1. `rank_ic_mean` 是否继续提升
2. `net_mean_return` 是否仍然赢当前 baseline
3. 折间波动是否恶化
4. 真实 `rolling-score` 是否超过当前默认 `combo_b25_h45`

## 为什么默认线定成这样

当前默认线不是一次性拍脑袋定下来的，而是分两步收敛的：先把信号 baseline 从 `baseline_4f` 切到 `baseline_5f`，再在不改动信号的前提下单独迭代组合层，最后才收敛到 `combo_b25_h45` 这条执行线。

### 组合层为什么定成 `Top25/Top45`

在不改动 5 因子信号的前提下，针对 `baseline_lgbm` 的组合层参数做过一轮独立迭代。完整 `rolling-score` 结果如下：

| Artifact line | 组合规则 | Rolling-score | 结论 |
|---------------|----------|--------------:|------|
| `baseline_lgbm` | Top20 / Top50 / bonus 0.5 | 47.1 | 历史默认 |
| `combo_h45_bonus1` | Top20 / Top45 / bonus 1.0 | 49.0 | 有提升，但不是最优 |
| `combo_h40_bonus1` | Top20 / Top40 / bonus 1.0 | 50.0 | 明显提升 |
| `combo_b25_h45` | Top25 / Top45 / bonus 0.5 | 50.5 | 当前默认策略线 |

这轮迭代给出的结论很明确：

- 当前最有效的改进不是继续堆因子，而是把组合入口从 `Top20/Top50` 调整为 `Top25/Top45`。
- `combo_b25_h45` 相比 `baseline_lgbm` 的全量 `rolling-score` 提升 `+3.4` 分。
- 组合层升级不需要回写默认特征列，所以 5 因子 baseline 继续保留为信号主线。

### 为什么保护版没有切成默认线

`combo_guarded_b25_h45` 是 2026-04-05 新增的多头输出保护版试验线。它在收益头之外又叠加了波动率头、最大回撤头和 reliability score，配置如下：

- 多头输出：收益 + 波动率 + 最大回撤
- prediction blend：`volatility_weight=0.15`，`max_drawdown_weight=0.2`
- reliability score：开启
- 组合层：`Top25/Top45`，但调仓间隔收紧到 `15` 天，`holding_bonus` 降到 `0.2`

当前结果：

| 线别 | `rank_ic_mean` | `top_bucket_spread_mean` | `net_mean_return` | `max_drawdown` | 结论 |
|------|---------------:|-------------------------:|------------------:|---------------:|------|
| `combo_b25_h45` | `0.0518` | `0.0113` | `0.000437` | `4.50%` | 当前默认 |
| `combo_guarded_b25_h45` | `0.0486` | `-0.0003` | `0.000081` | `5.85%` | 仅保留为研究支线 |

这条线说明风险标签不是没价值，而是现在还没走到可以替换默认线的阶段。它确实把 Top20 样本的未来波动率和未来回撤压低了，但收益兑现明显变弱，`spread` 转负，鲁棒性里还触发了 `flag_spread_decay=true`。因此它现在更适合继续做“收益-风险联合排序”研究，不适合替换当前默认执行口径。

### 为什么信号 baseline 是 `baseline_5f`

信号 baseline 的定版对照来自 `skyeye/artifacts/experiments/tx1_feature_eng_session3/ab_focus`。实验设置如下：

- 模型：`lgbm`
- 标签：`rank`
- 数据区间：`2015-03-09` 到 `2026-01-29`
- 数据集形状：`(760681, 35)`
- Walk-forward：14 folds

4 个核心变体结果如下：

| Variant | #Feat | Rank IC | IC IR | Spread | NetRet(日均) | MaxDD | 结论 |
|---------|------:|--------:|------:|-------:|-------------:|------:|------|
| `baseline_5f` | 5 | 0.0503 | 0.3526 | 0.0115 | 0.000375 | 6.00% | 当前默认 baseline |
| `baseline_4f` | 4 | 0.0375 | 0.2353 | 0.0056 | 0.000319 | 5.68% | 历史参考线 |
| `baseline_5f_fundamental_filter` | 5 | 0.0188 | 0.1208 | -0.0028 | 0.000069 | 5.14% | 不采用 |
| `baseline_4f_fundamental_filter` | 4 | 0.0151 | 0.0973 | -0.0041 | 0.000026 | 5.31% | 不采用 |

默认切到 `baseline_5f` 的原因同样很直接：

- 相比 `baseline_4f`，`Rank IC` 提升 `+0.0128`，相对增幅约 `34%`
- `IC IR` 提升约 `50%`
- `Spread` 从 `0.0056` 提到 `0.0115`，增幅约 `106%`
- `NetRet` 从 `0.000319` 提到 `0.000375`，相对增幅约 `17.7%`
- `fold_rank_ic_std` 从 `0.0617` 降到 `0.0553`，说明折间更稳
- 风险代价有限，`MaxDD` 只从 `5.68%` 升到 `6.00%`

对应新增因子 `turnover_stability_20d` 的单因子表现也支持它留在默认 baseline 里：

| 因子 | IC Mean | IC IR | 备注 |
|------|--------:|------:|------|
| `turnover_stability_20d` | 0.0390 | 0.3023 | 当前最有效的新增增强因子 |
| `ep_ratio_ttm` | 0.0360 | 0.1631 | 单因子有信息量，但在当前 5 因子栈上的增量测试未通过 |
| `return_on_equity_ttm` | 0.0178 | 0.0948 | 单因子有弱信息量，但最小增量实验未通过 |

### `ep_ratio_ttm` 这轮为什么没有往前推进

`ep_ratio_ttm` 的单因子相关性不差，所以它值得做最小增量验证；但 2026-04-06 新补跑的 `baseline_5f` vs `baseline_5f_ep` 对照没有支持它进入下一轮。对应产物在 `skyeye/artifacts/experiments/tx1_feature_ep_min/`，关键结果如下：

| Variant | Rank IC | NetRet(日均) | Row Ratio | 结论 |
|---------|--------:|-------------:|----------:|------|
| `baseline_5f` | 0.0503 | 0.000324 | 0.9854 | 保持默认 baseline |
| `baseline_5f_ep` | 0.0467 | 0.000295 | 0.9832 | 不推进 |

这轮实验的教训比“因子没用”更重要：

- 单因子 `IC Mean` 可用，不等于加到现有 baseline 后还能继续抬高组合层收益兑现。
- 基本面连续因子即使没有退化成硬过滤，也可能轻微压缩有效样本，导致比较不再完全等价。
- 在当前 `baseline_5f + lgbm + rank label` 组合里，`ep_ratio_ttm` 提供的增量信息没有强到足以覆盖模型噪声和样本损失。
- 研究层 `rank_ic_mean` 和 `net_mean_return` 同时变差时，就不应该继续往 `combo_b25_h45` 的 executable 对比推进。

### `return_on_equity_ttm` 这轮为什么也没有往前推进

`return_on_equity_ttm` 在单因子表里不算完全没信息量，所以 2026-04-07 又补跑了一轮最小增量验证。对应产物在 `skyeye/artifacts/experiments/tx1_feature_roe_min/`，关键结果如下：

| Variant | Rank IC | Spread | NetRet(日均) | Row Ratio | Fold IC Std | Fold NetRet Std | 结论 |
|---------|--------:|-------:|-------------:|----------:|------------:|----------------:|------|
| `baseline_5f` | 0.0503 | 0.0115 | 0.000324 | 0.9854 | 0.0553 | 0.000843 | 保持默认 baseline |
| `baseline_5f_roe` | 0.0476 | 0.0074 | 0.000324 | 0.9731 | 0.0537 | 0.000870 | 不推进 |

这轮结果的关键信号也很清楚：

- `net_mean_return` 只有几乎可以忽略的微弱抬升，不能覆盖 `rank_ic_mean` 和 `top_bucket_spread_mean` 的同步退化。
- `row_ratio` 从 `0.9854` 降到 `0.9731`，说明即使 `ROE` 覆盖率已经接近完整，增量实验仍然带来了样本压缩，比较不再完全等价。
- `fold_rank_ic_std` 虽然略好，但 `fold_net_return_std` 反而变差，不能把它解释成“稳定提升”。
- 这类“收益几乎不变，但 IC / spread 明显变差”的结果，不应该再继续推进到 executable artifact line 对比。

### 哪些方向已经明确不再采用

- 不再把 `EP + ROE above_median` 当默认 universe filter。样本只剩约 `39%`，两个 fundamental-filter 变体的 `Spread` 都为负，`NetRet` 也明显落后于未过滤版本。
- 不把 `baseline_5f + ep_ratio_ttm` 视为当前默认线候选。最新最小增量实验里，它的 `rank_ic_mean`、`net_mean_return` 和 `top_bucket_spread_mean` 都低于 `baseline_5f`。
- 不把 `baseline_5f + return_on_equity_ttm` 视为当前默认线候选。它没有带来稳定增益，`rank_ic_mean` 和 `top_bucket_spread_mean` 都明显低于 `baseline_5f`。
- 不在 `missing_features` 非空时解读基本面增量实验。只要日志里出现 `fundamental factors unavailable / returned empty`，或者变体详情里出现 `Missing features skipped`，就把这轮实验视为无效，不对结果做任何研究结论。
- 不优先继续添加与现有 baseline 高度共线的候选因子。当前已知高相关组合包括：
  - `vol_adj_turnover_20d` 与 `volatility_20d` 相关约 `-0.99`
  - `mom_20d` 与 `excess_mom_20d` 相关约 `1.00`
  - `mom_60d` 与 `excess_mom_60d` 相关约 `0.99`
  - `ma_gap_60d` 与 `mom_40d` 相关约 `0.91`

## 文件与默认约定

TX1 的默认口径分散在几处文件里，但职责是清楚的。看当前默认线时，优先从这些入口开始：

| 文件 | 当前职责 / 默认约定 |
|------|---------------------|
| `skyeye/products/tx1/README.md` | 产品线入口，先说明当前默认线是什么。 |
| `skyeye/products/tx1/PLAYBOOK.md` | 这份手册，集中记录默认配置、运行方式、定版依据和研发建议。 |
| `skyeye/products/tx1/strategies/rolling_score/README.md` | 当前可执行策略的运行细节和覆盖规则。 |
| `skyeye/products/tx1/evaluator.py` | `FEATURE_COLUMNS` 指向 5 因子默认 baseline。 |
| `skyeye/products/tx1/config.py` | 冻结 `input_window=60`、`horizon=20`，并保留默认组合层参数；多头输出和 reliability score 默认关闭。 |
| `skyeye/products/tx1/run_feature_experiment.py` | baseline delta 的对照基准已经切到 `baseline_5f`。 |
| `skyeye/products/tx1/run_label_experiment.py` | 默认组合缓冲统一到 `Top25/Top45` 口径。 |
| `skyeye/products/tx1/run_baseline_experiment.py` | 训练默认 baseline 或研究支线的统一入口。 |
| `skyeye/products/tx1/portfolio_proxy.py` | TopK 组合构建逻辑。 |
| `skyeye/products/tx1/strategies/rolling_score/spec.yaml` | 冻结 `benchmark`、默认 artifact line 和 5 因子 `signal_inputs`。 |
| `skyeye/products/tx1/strategies/rolling_score/profiles/baseline.yaml` | 历史默认对照档位，保留 `turnover_threshold=0.30`。 |
| `skyeye/products/tx1/strategies/rolling_score/profiles/smooth.yaml` | 当前默认档位，对应 `Top25/Top45`、温和 EMA 平滑和 `turnover_threshold=0.20`。 |

后续如果只是调参数，优先新增或修改 profile；如果改变核心交易假设，优先新建 artifact line，必要时新建策略目录，不要回写这条默认线的定义。

## 当前研发建议

- 默认主线继续保持 `baseline_5f` 信号 + `Top25/Top45` 组合层。
- 如果继续看基本面增强，先避免“单因子看起来不错就直接并入 baseline”的路径，优先做更小、更可归因的增量实验。
- 先把基本面增量实验的前置闸门固定下来。这不是候选方向本身，而是研究纪律：
  - `requested_features == available_features`
  - `missing_features=[]`
  - `row_ratio` 没有异常压缩
  - 以上任一条件不满足时，不解读结果，也不更新研究结论
- 下一轮最该投资源的候选方向优先级调整为：
  - 先围绕 `Top25/Top45` 附近做小范围组合层微调。当前已有 `combo_h40_bonus1`、`combo_h45_bonus1` 两条线相对 `combo_b25_h45` 仍有小幅正向 `net_mean_return` 增量，且没有新增 warnings，更像短期最可能继续产出可执行收益改进的方向。
  - 再验证“其他低共线基本面候选 + 现有 5 因子”的最小增量实验，但不再把 `ep_ratio_ttm` 或 `return_on_equity_ttm` 的单因子直加放在前排。
  - 如果还想重看估值 / 质量因子，优先测试更强约束的交互设计，例如只在收益兑现明确改善时才考虑 `ep_ratio_ttm + quality` 组合。
- 继续研究多头输出时，先把 `combo_guarded_b25_h45` 的收益兑现问题和 `spread decay` 解释清楚，再决定是否继续往默认线推进。
- 不要先把基本面做成硬过滤条件。
- 不要优先继续堆高共线的动量、趋势或波动率镜像因子。
- 如果后续组合层微调的边际收益也开始收窄，再回到新的因子候选，而不是重复验证已经被最小增量实验否掉的 `EP` / `ROE` 单因子路线。

后续如果默认线再次切换，直接更新这份手册顶部的“当前默认线”“常用命令”和“为什么默认线定成这样”三段，不再把旧版本结论继续往后追加。

## TX1 实盘辅助使用说明

### 这套东西到底是什么

TX1 现在分成两条线：

- `research / replay`：继续用于严格的历史 OOS 验证和 rolling-score 回放。
- `live advisor`：用于日频实盘辅助，回答“今天这只股票在当前股票池里排第几、同类历史信号的胜率和收益分布怎样、风险区间怎样”。

`live advisor` 不是“直接替你下单”的执行策略，也不是“模型原始分数等于未来收益”的承诺。它给的是历史 OOS 校准后的解释结果。

### 实盘使用前先做什么

先把研究实验提升成一个 live package。推荐直接从当前研究主线 `tx1_refresh_20260331_lgbm` 提升：

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH="$PWD" python - <<'PY'
from skyeye.products.tx1.live_advisor.promotion import promote_experiment_to_live_package

payload = promote_experiment_to_live_package(
    experiment_ref="skyeye/artifacts/experiments/tx1/tx1_refresh_20260331_lgbm",
    package_id="tx1_live_canary_refresh_20260331_lgbm",
    packages_root="skyeye/artifacts/packages/tx1",
)

print(payload["package_root"])
print(payload["manifest"]["gate_summary"]["gate_level"])
print(payload["manifest"]["fit_end_date"])
print(payload["manifest"]["data_end_date"])
PY
```

这条命令会做 4 件事：

- 读取研究实验的 OOS `predictions_df`
- 生成 calibration bundle
- 按最新可得标签数据训练一份 final-fit live model
- 落盘一个标准 live package 到 `skyeye/artifacts/packages/tx1/<package_id>/`

如果输出的 `gate_level` 是：

- `default_live`：说明稳定性门槛也通过了，可以作为默认 live 包使用
- `canary_live`：说明方向有效，但稳定性还不够强，只建议作为辅助参考
- `blocked`：说明 promotion gate 没过，这个包不应该用于实盘辅助

### 每天怎么打分

package 生成后，每天直接跑：

下面命令默认使用你当前依赖完整的 Python 环境。如果仓库里有多套解释器，使用那套已经带齐 `pandas / h5py / rqdatac` 的环境。

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_live_advisor \
  --package-id tx1_live_canary_refresh_20260331_lgbm \
  --packages-root skyeye/artifacts/packages/tx1 \
  --trade-date 2026-03-31 \
  --top-k 25 \
  --format table
```

建议：

- `--top-k 25` 对应当前 `Top25 买入` 的完整候选池；如果只是快速浏览 shortlist，再改回 `10` 或 `20`
- `--format table` 会输出 3 段：头部元信息、`结果解读`、`指标说明 + 明细表`

如果你想看机器可读结果：

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_live_advisor \
  --package-id tx1_live_canary_refresh_20260331_lgbm \
  --packages-root skyeye/artifacts/packages/tx1 \
  --trade-date 2026-03-31 \
  --top-k 20 \
  --format json \
  --include-dropped
```

可选参数：

- `--top-k`：输出前多少只股票
- `--include-dropped`：把被剔除的股票和原因一起输出
- `--universe-size`：临时缩小 live universe，适合 smoke test
- `--market-cap-floor-quantile`：给 universe 加市值底线
- `--market-cap-column`：指定市值列

### 输出怎么读

建议按下面顺序看表：

1. 先看 `status / score_date / gate_level`
2. 再看 `结果解读`
3. 最后看明细表

表格输出里最重要的是这些字段：

- `代码 / 股票名`：当日进入 top-k 的股票代码和简称
- `排名`：这只股票在当天 live universe 里的名次
- `分位`：名次对应的横截面分位，`100.0%` 表示当日最高分
- `胜率`：历史 OOS 中，相同 bucket 信号未来 `20` 个交易日收益大于 `0` 的比例
- `均值收益`：历史 OOS 中，相同 bucket 的未来 `20` 日平均收益
- `中位收益`：历史 OOS 中，相同 bucket 的未来 `20` 日收益中位数
- `P25~P75`：历史 OOS 中，相同 bucket 的收益分布中间区间
- `样本数`：这个 bucket 的历史 OOS 样本数

如果 `结果解读` 提示“右偏收益排序器，不是高胜率信号”，通常意味着：

- `均值收益 > 0`
- 但 `中位收益 < 0`
- 同时 `胜率` 也未必超过 `50%`

这不是 bug，而是历史收益分布右偏：多数样本小亏小赚，少数大涨样本把均值拉正。所以实盘上要把 TX1 当作“组合排序器”，不要按单票命中率去理解。

在支持 ANSI 颜色的终端里，`status / gate_level`、`均值收益`、`中位收益` 会带颜色；如果你的终端或重定向环境不支持颜色，数值本身仍然按百分比格式输出。

实盘使用时，优先同时看这三件事：

- 排名和分位是否足够靠前
- `样本数` 是否足够大
- `均值收益 / 中位收益 / P25~P75` 是否还值得承担风险

不要把它理解成“今天排第一就一定买”，而是“在今天可交易股票池里，这些股票落在历史上相对更强的一档”。

### 如果输入了未来日期会怎样

如果你输入的 `--trade-date` 晚于当前本地 bundle 的最新可用交易日，系统不会伪造未来数据，而是自动回退到“不晚于你请求日期的最近可用交易日”。

例如当前本地数据如果只到 `2026-03-31`，你输入：

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_live_advisor \
  --package-id tx1_live_canary_refresh_20260331_lgbm \
  --packages-root skyeye/artifacts/packages/tx1 \
  --trade-date 2026-04-10 \
  --top-k 20 \
  --format table
```

这时你要以输出里的 `score_date` 为准。如果 `score_date=2026-03-31`，说明实际打分还是基于 `2026-03-31`。

### 什么情况下不要用当天结果

只要出现下面任一情况，当天就不要按 TX1 建议做决策：

- 输出 `status=stopped`
- 输出里有 runtime gate 的 stop reason
- package 还是 `canary_live`，但你想把它当成主决策依据
- `samples` 太小，或者 `median_ret` 已经没有明显优势
- 你明知 bundle 数据还没更新到你关心的日期

### 日常推荐工作流

建议固定成下面这套流程：

1. bundle 数据更新后，先重新生成一次 live package
2. 看 package 的 `gate_level`
3. 每天开盘前或盘后，用 `run_live_advisor` 跑当日结果
4. 先看 `status / score_date / gate_level`，如果 `status != ok` 就停止使用当天结果
5. 再看 `结果解读`，确认它是高胜率型还是右偏收益型排序
6. 再看前 `10-25` 名，而不是只盯着第 `1` 名
7. 最后结合你自己的仓位、行业暴露和风险预算决定是否交易

### 一个已经验收过的 smoke test 包

开发验收时已经实际跑通过一个小股票池 smoke package：

- `skyeye/artifacts/packages/tx1/tx1_live_canary_refresh_20260331_lgbm_u10`

对应验证命令：

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_live_advisor \
  --package-id tx1_live_canary_refresh_20260331_lgbm_u10 \
  --packages-root skyeye/artifacts/packages/tx1 \
  --trade-date 2026-03-31 \
  --top-k 5 \
  --universe-size 10 \
  --format table
```

这个包只用于开发验收，不建议直接拿来做实盘参考，因为它只在 `10` 只股票的小 universe 上做了 smoke test。
