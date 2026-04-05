# Rolling Score

`tx1.rolling_score` 是 TX1 当前默认可执行策略。它不会在回测时在线重新训练模型，而是把研究链路产出的冻结样本外目标权重回放到标准 RQAlpha / `rolling_score` 执行栈里。

## 默认运行口径

| 项目 | 当前默认值 |
| --- | --- |
| `strategy_id` | `tx1.rolling_score` |
| 默认 artifact line | `combo_b25_h45` |
| 默认 profile | `smooth` |
| 基准 | `000300.XSHG` |
| 信号输入 | `mom_40d`, `volatility_20d`, `reversal_5d`, `amihud_20d`, `turnover_stability_20d` |
| 标签 | `20d horizon + rank transform` |
| 组合规则 | `Top25 买入 / Top45 持有 / 跌出 Top45 退出` |
| 调仓节奏 | `20` 交易日重平衡 |
| 组合层补丁 | `holding_bonus=0.5`, `single_stock_cap=0.10`, `turnover_threshold=0.30` |
| 平滑 | `EMA halflife=5`, `ema_min_weight=0.005` |

## 运行流程

1. `build_runtime(...)` 读取 `spec.yaml`、解析 artifact line、加载 profile，并校验冻结字段。
2. `load_replay_signal_book(...)` 从实验目录的 `weights.parquet` 生成按交易日索引的 frozen target weights。
3. `before_trading(...)` 读取上一个交易日对应的 signal，做权重清洗、单票上限裁剪、可交易性过滤和 EMA 平滑。
4. `handle_bar(...)` 只在目标组合与当前组合偏离超过阈值时执行 `order_target_portfolio(...)`。

几个关键约束：

- 重叠日期的 signal book 以更晚 `test_start` 的 fold 为准。
- `listed_date` / `de_listed_date` / `ST` / 停牌会在执行层再次过滤。
- `benchmark` 和 `artifact_line_id` 是冻结字段，profile 不能改。

## 覆盖顺序

### artifact line

解析优先级从高到低：

1. `build_runtime(..., artifact_line=...)`
2. 环境变量 `SKYEYE_TX1_ARTIFACT_LINE`
3. `config.extra.tx1_artifact_line`
4. `spec.yaml` 中的 `artifact_line_id`

### profile

解析优先级从高到低：

1. `build_runtime(..., profile_name=...)`
2. 环境变量 `SKYEYE_TX1_PROFILE`
3. `config.extra.strategy_profile`
4. 默认值 `smooth`

### 当前 profile 约定

- `smooth`
  当前默认档位，对应 `Top25/Top45` 和温和 EMA 平滑。
- `baseline`
  与 `smooth` 口径一致，保留给显式 baseline 命名使用。
- `soft_sticky` / `sticky` / `ultra_sticky`
  历史 `Top20/Top50` 研究档位，只用于对照，不是当前默认策略。

## 常用命令

### 1. 直接跑当前默认 rolling-score

```bash
uv run python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py
```

### 2. 临时切到其他 artifact line / profile

```bash
env SKYEYE_TX1_ARTIFACT_LINE=baseline_tree SKYEYE_TX1_PROFILE=baseline \
  uv run python -m skyeye.evaluation.rolling_score.cli \
    skyeye/products/tx1/strategies/rolling_score/strategy.py
```

### 3. 直接用 RQAlpha 跑单区间并出图

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

## 相关文档

- [TX1 产品线说明](../../README.md)
- [TX1 PLAYBOOK](../../PLAYBOOK.md)
- [策略 spec](./spec.yaml)
