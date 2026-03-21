# Dividend Low Vol

`skyeye/products/dividend_low_vol/` 承载红利低波这一条产品线下的全部长期资产：

- `scorer/`：估值打分器和数据准备链路
- `strategies/`：交易策略，按策略目录隔离
- `tools/`：可选诊断和可视化工具，有需要再加
- `mod/`：给 RQAlpha 暴露的产品级 mod
- `registry.py`：策略注册和元信息入口

策略目录约定：

- `strategy.py`
  只保留 RQAlpha 适配层，不堆大段交易细节。
- `spec.yaml`
  固定记录策略 ID、假设、适用环境、风险控制、滚动打分起始日期等结论信息。
- `profiles/*.yaml`
  参数档位。只调参数时新增或修改 profile，不回写核心逻辑。
- `logic/*.py`
  可单测的纯逻辑模块。

迭代原则：

- 新假设，新建新的策略目录，不覆盖现有基线。
- 同一策略的调参，只新增 profile，不回退到单文件硬编码。
- 测试目录镜像到 `tests/products/dividend_low_vol/`，避免逻辑和回测验证混在一起。

常用说明：

- [红利低波打分器使用说明](../../docs/manuals/红利低波打分器使用说明.md)
- [策略回测打分器](../../docs/manuals/策略回测打分器.md)
