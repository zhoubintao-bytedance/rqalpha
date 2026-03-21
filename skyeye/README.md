# SkyEye

`skyeye/` 是当前仓库里所有自研研究、策略和评估能力的主目录。

当前目录约定：

- `skyeye/evaluation/`
  评估层。`rolling_score/` 负责滚动窗口打分，`single_run/` 负责单区间结果归一化。
- `skyeye/products/`
  产品线层。每个品类独立维护 scorer、strategies、可选 tools、mod 和 registry。
- `skyeye/products/dividend_low_vol/`
  红利低波产品线主目录。
- `skyeye/artifacts/`
  运行产物和导出图片的约定目录说明。
- `skyeye/reports/`
  评估报告和阶段性结论的约定目录说明。
- `skyeye/docs/`
  使用手册、RFC 和问题记录。

红利低波产品线当前约定：

- `scorer/`
  估值打分器、缓存同步和参数审计。
- `strategies/<strategy_id>/`
  单个策略目录，必须包含 `strategy.py`、`spec.yaml`、`profiles/`，复杂逻辑继续拆到 `logic/`。
- `tools/`
  可选诊断和可视化工具；如果后续需要，再加，不和策略放在同一目录。
- `mod/`
  RQAlpha 对外暴露的产品级 mod。

常用入口：

- [docs/manuals/README.md](./docs/manuals/README.md)
- [products/dividend_low_vol/README.md](./products/dividend_low_vol/README.md)
