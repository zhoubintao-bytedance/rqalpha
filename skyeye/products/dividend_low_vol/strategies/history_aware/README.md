# History Aware

这份策略是红利低波产品线当前的主力基线版本。

- 假设：历史分位 + 趋势能提供比单日分位更稳的主仓位参考
- 结构：`strategy.py` 只保留 RQAlpha 适配层，真实逻辑拆到 `logic/`
- 参数：默认 profile 放在 `profiles/baseline.yaml`

后续如果只是调参数，新增 profile。
后续如果改变核心交易假设，新建新的策略目录，不回写这份基线。
