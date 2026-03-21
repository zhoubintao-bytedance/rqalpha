# SkyEye

`skyeye/` 是当前仓库里所有自研能力的主目录。

当前约定：

- 策略回测打分器主实现：`skyeye/strategy_scorer.py`
- 红利低波打分器主实现：`skyeye/dividend_scorer/`
- 红利低波交易策略主实现：`skyeye/examples/`
- 自研 mod 主实现：`skyeye/mod/rqalpha_mod_dividend_scorer/`
- 文档主目录：`skyeye/docs/`
- 实验图表主目录：`skyeye/experiments/`

当前状态：

- 旧兼容壳和旧自研入口已从活动路径删除
- 后续自研迭代只在 `skyeye/` 下进行
- 后续自研迭代、文档、验证与运行全部以 `skyeye/` 为准
