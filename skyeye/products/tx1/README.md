# TX1

`skyeye/products/tx1/` 承载 TX1 这一条产品线下的研究、实验产物解析和当前可执行策略。

TX1 已经从“验证 Transformer 是否值得成为主线”的研究阶段，收敛到当前默认可执行基线：

- 信号：`baseline_5f`
- 模型：`lgbm`
- 标签：`20d horizon + rank transform`
- 组合层：`Top25 买入 / Top45 持有 / 20 交易日调仓 / holding_bonus=0.5`
- 运行方式：回放冻结的样本外目标权重，而不是在线重新打分
- 默认策略线：`tx1.rolling_score@combo_b25_h45`

## 阅读顺序

如果你是第一次接手 TX1，默认按下面顺序阅读：

1. [PLAYBOOK.md](./PLAYBOOK.md)
   当前默认线、实盘辅助、常用命令和实验结论的单一真相源。
2. [strategies/rolling_score/README.md](./strategies/rolling_score/README.md)
   默认 replay 策略的运行细节、覆盖规则和执行口径。
3. [../../docs/rfc/tianyan_tx1_strategy_v1.md](../../docs/rfc/tianyan_tx1_strategy_v1.md)
   只在需要追溯历史立项背景时再读。
4. [../../docs/rfc/tianyan_tx1_research_design_v1_1.md](../../docs/rfc/tianyan_tx1_research_design_v1_1.md)
   只在需要追溯研究方法学约束时再读。

如果任务涉及 `run_live_advisor`、实盘辅助、`package_id`、`canary_live` 或表格输出解释，直接优先看 [PLAYBOOK.md](./PLAYBOOK.md) 里的“TX1 实盘辅助使用说明”。

## 当前默认线

| 项目 | 当前默认值 |
| --- | --- |
| 策略 ID | `tx1.rolling_score` |
| 基准 | `000300.XSHG` |
| 默认因子 | `mom_40d`, `volatility_20d`, `reversal_5d`, `amihud_20d`, `turnover_stability_20d` |
| 默认 artifact line | `combo_b25_h45` |
| 默认 profile | `smooth` |
| 研究冻结项 | `input_window=60`, `horizon=20` |
| 当前状态 | 默认 replay 执行线 + canary 级 live advisor 辅助链路 |

## 目录职责

- `dataset_builder.py`
  因子计算和横截面数据集构建。
- `label_builder.py`
  收益 / 波动率 / 最大回撤标签工程。
- `config.py`
  TX1 研究冻结项和默认配置入口。
- `experiment_runner.py`、`main.py`
  Walk-forward 训练、评估、持久化主链路。
- `run_feature_experiment.py`
  特征实验入口，当前默认 baseline 已切到 `baseline_5f`。
- `run_label_experiment.py`
  标签变体对比入口，当前默认组合缓冲已经统一到 `Top25/Top45`。
- `run_baseline_experiment.py`
  用 bundle + 因子数据重跑 TX1 基线或试验组合层。
- `artifacts.py`
  `strategy_id@artifact_line_id` 解析、实验目录解析和冻结 signal book 加载。
- `strategies/rolling_score/`
  当前可执行 TX1 策略目录。
- `PLAYBOOK.md`
  当前默认线、实验结论、常用命令和后续研发建议。

## 当前保留的策略线

| 策略线 | 说明 | 当前定位 |
| --- | --- | --- |
| `combo_b25_h45` | `lgbm` 单头收益预测，`Top25/Top45` 月度重平衡 | 当前默认线 |
| `baseline_lgbm` | 历史默认 `Top20/Top50` 组合层 | 历史对照线 |
| `combo_guarded_b25_h45` | 多头输出保护版，叠加波动率 / 最大回撤预测与 reliability score | 研究分支，当前不切默认 |

`combo_guarded_b25_h45` 的最新结果说明：它确实能挑出更低未来波动和更低未来回撤的样本，但当前 `net_mean_return`、`spread` 和 `max_drawdown` 都不如 `combo_b25_h45`，因此只保留为实验支线。

## 策略目录约定

- `strategy.py`
  只保留 RQAlpha 适配和 frozen signal replay，不在这里重写研究训练逻辑。
- `spec.yaml`
  固定记录策略 ID、artifact line、信号输入、适用环境、风险控制和滚动打分起始日期。
- `profiles/*.yaml`
  只承载参数档位。`benchmark` 和 `artifact_line_id` 这类冻结字段不能被 profile 覆盖。

后续如果只是调参数，新增或修改 profile。
后续如果改变核心交易假设，新建新的策略目录，不回写这条默认线。

## 文档入口

- [PLAYBOOK](./PLAYBOOK.md)
  默认配置、常用命令、实盘辅助、实验结论和后续研发建议。
- [rolling_score 策略说明](./strategies/rolling_score/README.md)
  默认 replay 执行链路、profile 覆盖规则和回放约束。
- [天眼 TX1 正式方案 V1（历史立项 RFC）](../../docs/rfc/tianyan_tx1_strategy_v1.md)
- [天眼 TX1 v1.1 研究设计文档 RFC（历史研究方法学）](../../docs/rfc/tianyan_tx1_research_design_v1_1.md)
