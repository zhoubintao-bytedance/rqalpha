# skyeye 数据访问点盘点（用于迁移）

## 目标
- 记录 skyeye 内所有直接数据读取点（bundle 直读、rqdatac 直连、DataProxy 直用、AKShare 直用）
- 为分阶段替换提供清单与验收依据

## 清单

| 文件 | 行范围 | 访问方式 | 迁移状态 |
|---|---:|---|---|
| [engine.py](file:///home/tiger/rqalpha/skyeye/evaluation/rolling_score/engine.py#L946-L960) | 946-960 | bundle 日线读取 | 已切换到 DataFacade |
| [engine.py](file:///home/tiger/rqalpha/skyeye/evaluation/rolling_score/engine.py#L1034-L1055) | 1034-1055 | bundle 指数日线读取 | 已切换到 DataFacade |
| [engine.py](file:///home/tiger/rqalpha/skyeye/evaluation/rolling_score/engine.py#L1137-L1152) | 1137-1152 | bundle instruments.pk 读取 | 已切换到 DataFacade |
| [run_baseline_experiment.py](file:///home/tiger/rqalpha/skyeye/products/tx1/run_baseline_experiment.py) | - | bundle h5 / instruments.pk / AKShare | 已切换到 DataFacade（AKShare 逻辑保留为策略实验外部数据） |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L17-L20) | 17-20 | DataProxy 依赖导入 | 待迁移 |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L779-L812) | 779-812 | DataProxy.history_bars(bundle) | 待迁移 |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L1256-L1267) | 1256-1267 | 构造 bundle DataProxy | 待迁移 |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L639-L644) | 639-644 | AKShare 指数行情 | 待迁移（可由 rqdatac.get_price 替代） |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L817-L822) | 817-822 | AKShare ETF 净值 | 待迁移（rqdatac ETF 估值能力评估后替换/保留） |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L848-L848) | 848 | AKShare 国债利率 | 待迁移（rqdatac.get_yield_curve 替代） |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L870-L870) | 870 | AKShare 指数权重 | 待迁移（rqdatac.index_weights 替代） |
| [data_fetcher.py](file:///home/tiger/rqalpha/skyeye/products/dividend_low_vol/scorer/data_fetcher.py#L1163-L1178) | 1163-1178 | AKShare ETF 行情 | 待迁移（rqdatac.get_price 替代） |
| [compat.py](file:///home/tiger/rqalpha/skyeye/data/compat.py#L20-L40) | 20-40 | AKShare 北向资金 | 待迁移（rqdatac.get_stock_connect 覆盖度评估后替换/保留） |
| [provider.py](file:///home/tiger/rqalpha/skyeye/data/provider.py) | - | rqdatac 统一接口（现存） | 逐步融合到 DataFacade（作为 rqdatac 适配层） |

