# SkyEye 因子层设计文档

## 目标

在 `skyeye` 中新增一个独立的公共因子层，为多个策略提供与市场状态相关的补充指标信息。

该层的核心职责是：

- 接收 `MarketRegime` 作为输入
- 基于市场状态动态选择要输出的指标类别
- 以统一格式返回当前指标值与历史分位
- 在数据不足或单指标失败时稳定降级，但始终返回结果对象

该层服务于 `TX1`、红利低波策略及后续其他需要市场指标补充信息的策略模块。

## 设计边界

### 保持独立

- 因子层新增为独立包：`skyeye/factor_layer/`
- 因子层不修改现有 `skyeye/market_regime_layer.py`
- 因子层只依赖 `MarketRegime` 输入，而不反向影响市场分类逻辑

### 与市场分类层的协作

- `compute_factors(...)` 为纯函数主入口，必须接收 `MarketRegime`
- `compute_factors_from_data_facade(...)` 为便捷入口
- 当便捷入口未显式传入 `regime` 时，内部调用现有 `compute_market_regime_from_data_facade(...)`

### 输出语义

- `FactorLayerResult` 始终包含五大类别字段
- 只计算并填充当前 `regime` 对应的两类重点指标
- 未命中的其他类别统一返回空字典 `{}`，而不是返回默认值或伪造结果

## 模块结构

建议目录结构如下：

```text
skyeye/
└── factor_layer/
    ├── __init__.py
    ├── core.py
    ├── config.py
    ├── result.py
    ├── utils.py
    └── indicators/
        ├── __init__.py
        ├── trend.py
        ├── momentum.py
        ├── oscillator.py
        ├── volatility.py
        └── volume.py
```

各模块职责如下：

- `config.py`
  - 定义 `FactorLayerConfig`
  - 管理分位窗口和各指标参数
- `result.py`
  - 定义 `FactorValue` 与 `FactorLayerResult`
- `utils.py`
  - 放置公共工具函数
  - 包括 bars 预处理、字段校验、当前值提取、分位计算、异常包装等
- `indicators/*.py`
  - 每个文件负责一个指标大类
  - 对外暴露“类别级入口函数”
  - 内部保留“单指标历史序列函数”
- `core.py`
  - 实现 `compute_factors(...)`
  - 负责 `regime -> 类别` 映射
  - 负责调用指标模块、组织结果、处理整体降级
- `__init__.py`
  - 暴露公共 API，避免调用方依赖内部文件路径

## 对外接口

### 纯函数入口

```python
def compute_factors(
    benchmark_bars: pd.DataFrame,
    regime: MarketRegime,
    cfg: FactorLayerConfig | None = None,
) -> FactorLayerResult:
    ...
```

行为约定：

- 接收单标的基准指数 OHLCV 数据
- 接收上游已经计算好的 `MarketRegime`
- 只根据 `regime.regime` 输出对应类别的指标结果
- 不依赖外部全局状态

### 便捷入口

```python
def compute_factors_from_data_facade(
    end_date: str | pd.Timestamp,
    benchmark_id: str = "000300.XSHG",
    regime: MarketRegime | None = None,
    cfg: FactorLayerConfig | None = None,
) -> FactorLayerResult:
    ...
```

行为约定：

- 通过 `DataFacade.get_daily_bars(...)` 拉取基准指数日线
- 当 `regime is None` 时，内部调用现有 `compute_market_regime_from_data_facade(...)`
- 获取 `benchmark_bars` 与 `regime` 后，再委托给 `compute_factors(...)`

## 配置对象

`FactorLayerConfig` 保持与设计草案一致，统一管理：

- `percentile_window`
- 趋势类参数：`macd_fast`、`macd_slow`、`macd_signal`、`ma_periods`、`ema_periods`
- 动量类参数：`mom_period`、`roc_period`、`bias_period`
- 震荡类参数：`rsi_period`、`kdj_n`、`kdj_m1`、`kdj_m2`、`cci_period`
- 波动类参数：`atr_period`、`boll_period`、`boll_std`、`dc_period`
- 量价类参数：`obv_ma_period`、`mfi_period`

首版不引入策略专属配置，保证该层保持通用。

## 输出结构

建议输出对象如下：

```python
from dataclasses import dataclass
from typing import Literal

FactorValue = dict[Literal["value", "percentile"], float]

@dataclass
class FactorLayerResult:
    regime: MarketRegime
    trend_factors: dict[str, FactorValue]
    momentum_factors: dict[str, FactorValue]
    oscillator_factors: dict[str, FactorValue]
    volatility_factors: dict[str, FactorValue]
    volume_factors: dict[str, FactorValue]
```

输出语义：

- `regime` 原样返回
- 五大类别字段始终存在，保证调用方读取稳定
- 当前 `regime` 对应的两类填充结果
- 其余类别返回空字典

## 指标命名约定

为保证策略直接消费方便，因子键统一使用“最终可消费名称”：

### 趋势类

- `MACD`
- `ADX`
- `RSRS`
- `MA_5`、`MA_10`、`MA_20`、`MA_60`
- `EMA_12`、`EMA_26`

### 动量类

- `MOM`
- `ROC`
- `BIAS`

### 震荡类

- `RSI`
- `KDJ_K`
- `KDJ_D`
- `KDJ_J`
- `CCI`

### 波动类

- `ATR`
- `BBANDS_UPPER`
- `BBANDS_MIDDLE`
- `BBANDS_LOWER`
- `BBANDS_WIDTH`
- `DC_UPPER`
- `DC_MIDDLE`
- `DC_LOWER`

### 量价类

- `OBV`
- `OBV_MA`
- `MFI`

其中：

- `MACD` 输出 `hist` 最新值
- `RSRS` 输出 rolling slope 的 zscore 最新值
- `BBANDS` 与 `DC` 不只返回单一指标，而是拆成可直接用于策略判断的轨道字段

## 市场状态映射

固定映射表放在 `core.py`：

```python
REGIME_FACTOR_CATEGORIES = {
    "bull_co_move": ("trend", "volume"),
    "bull_rotation": ("momentum", "trend"),
    "range_co_move": ("oscillator", "volatility"),
    "range_rotation": ("momentum", "oscillator"),
    "bear_co_move": ("volatility", "volume"),
    "bear_rotation": ("momentum", "volatility"),
}
```

建议再提供一个内部函数：

```python
def _select_factor_categories(regime_label: str) -> tuple[str, str]:
    ...
```

用于：

- 统一封装映射逻辑
- 便于单独测试

## 指标计算策略

### 总体原则

- 每个单指标函数都优先输出完整历史序列，而不是只算最后一个值
- `value` 直接取最新有效值
- `percentile` 在同一指标历史序列上滚动计算

例如：

```python
def compute_rsi_series(close: pd.Series, period: int) -> pd.Series:
    ...
```

这样可保证：

- 当前值与分位来源一致
- 测试更直接
- 各类别实现风格统一

### 趋势类

- `MACD`
  - 使用 EMA 差值与信号线构建
  - 对外输出 histogram 最新值
- `ADX`
  - 使用 Wilder 平滑
- `RSRS`
  - 使用 rolling regression slope 并做 rolling zscore
- `MA`
  - 对 `cfg.ma_periods` 中每个周期输出一条均线
- `EMA`
  - 对 `cfg.ema_periods` 中每个周期输出一条指数均线

### 动量类

- `MOM = close_t - close_t-n`
- `ROC = close_t / close_t-n - 1`
- `BIAS = close_t / MA(n) - 1`

### 震荡类

- `RSI`
  - 使用标准 Wilder RSI
- `KDJ`
  - 输出 `KDJ_K`、`KDJ_D`、`KDJ_J`
- `CCI`
  - 使用典型价格与平均绝对偏差公式

### 波动类

- `ATR`
  - 使用 Wilder ATR
- `BBANDS`
  - 输出上轨、中轨、下轨和宽度
  - `BBANDS_WIDTH = (upper - lower) / middle`
- `DC`
  - 输出上轨、中轨、下轨
  - `DC_MIDDLE = (upper + lower) / 2`

### 量价类

- `OBV`
  - 输出最新 OBV
- `OBV_MA`
  - 对 OBV 做移动平均
- `MFI`
  - 使用标准资金流量指数定义

## 与市场分类层的复用策略

### 建议复用

- `market_regime_layer` 的 facade 使用方式
- `normalize_single_instrument_bars(...)`
- 一些纯基础数值函数，如果确认语义稳定，可按需复用

### 不建议直接复用

- 任何为市场标签判定而服务的离散化函数
- 任何直接产出 `bull/bear/range` 或 `rotation/co_move` 的聚合函数

原因：

- 市场分类层的目标是生成标签
- 因子层的目标是输出连续指标值
- 两者虽然共享部分技术指标，但语义不同，不应强耦合

因此首版实现中，`MACD`、`ADX`、`RSRS`、`ATR` 等指标可以在因子层内独立实现或显式封装，避免未来被市场分类层内部改动牵连。

## 分位计算

### 统一规则

- 所有指标的历史分位都使用 `cfg.percentile_window`
- 默认窗口为 `252`
- 分位定义为当前值在最近窗口历史值中的经验分位，结果在 `0.0 ~ 1.0`

### 边界规则

- 如果当前值无法计算，则该指标不进入输出结果
- 如果当前值可算，但历史有效样本不足，则保留 `value`
- 此时 `percentile` 返回 `nan`

建议在 `utils.py` 中提供统一包装函数，例如：

```python
def build_factor_value(
    current: float | None,
    history: pd.Series,
    window: int,
) -> FactorValue | None:
    ...
```

或更进一步：

```python
def safe_build_factor(
    name: str,
    series_builder: Callable[[], pd.Series | pd.DataFrame | None],
    window: int,
) -> tuple[str, FactorValue] | None:
    ...
```

以统一处理：

- 异常捕获
- 最新值提取
- 空序列过滤
- 历史分位计算

## 数据预处理

进入指标计算前，`compute_factors(...)` 应统一做以下预处理：

- 空表直接降级
- 缺 `close` 直接降级
- `index` 统一转换为日期索引
- 统一 `sort_index()`
- 列名按小写 OHLCV 约定读取

这样各指标模块可以假设输入已经过基本规范化，减少重复逻辑。

## 降级与容错

降级分为三层。

### 输入级降级

当发生以下情况时，直接返回空结果对象：

- `benchmark_bars is None`
- `benchmark_bars.empty`
- 缺少 `close`

返回结果中：

- `regime` 仍然保留输入值
- 所有类别字典均为空

### 类别级降级

如果某个命中类别中部分指标无法计算：

- 能算出的指标正常返回
- 算不出的指标跳过
- 该类别允许返回部分结果

如果该类别所有指标都失败，则该类别返回空字典。

### 单指标级降级

- 任一单指标报错不向上抛出
- 在类别函数内部或统一包装器中捕获
- 不影响同类其他指标
- 不影响另一类指标的输出

该规则保证因子层始终能返回 `FactorLayerResult`。

## 建议的数据流

### `compute_factors(...)`

1. 规范化 `benchmark_bars`
2. 根据 `regime.regime` 选择两类目标类别
3. 调用对应类别计算函数
4. 将结果写入 `FactorLayerResult`
5. 未命中的类别写空字典

### `compute_factors_from_data_facade(...)`

1. 使用 `DataFacade` 拉取 benchmark OHLCV
2. 若未传 `regime`，内部调用现有 market regime facade
3. 调用 `compute_factors(...)`
4. 返回统一结果对象

## 测试设计

首版建议新增 `tests/unittest/test_skyeye_factor_layer.py`，至少覆盖以下场景。

### 1. 结果结构测试

- 输入每一种 `regime`
- 验证只填充对应两类
- 其余类别为空字典
- `result.regime is regime`

### 2. 基础指标计算测试

使用可控合成数据验证：

- 趋势行情下 `MOM`、`ROC`、`MA`、`EMA`、`MACD` 可正常出值
- 震荡行情下 `RSI`、`KDJ`、`CCI` 可正常出值
- 包含完整 OHLCV 时 `ATR`、`MFI`、`OBV` 可正常出值

### 3. 降级稳定性测试

- 缺 `volume` 时：
  - `OBV`、`OBV_MA`、`MFI` 缺失
  - 其他不依赖量的指标仍返回
- 缺 `high/low` 时：
  - `ADX`、`ATR`、`RSRS`、`KDJ`、`CCI`、`DC` 可缺失
  - `MACD`、`MA`、`EMA`、`MOM`、`ROC`、`BIAS`、`RSI` 仍可返回

### 4. 分位范围测试

- 所有产生的 `percentile`
  - 要么是 `nan`
  - 要么落在 `0.0 ~ 1.0`

### 5. facade 行为测试

- 当显式传入 `regime` 时，不重复计算市场状态
- 当 `regime is None` 时，内部调用现有 `compute_market_regime_from_data_facade(...)`
- benchmark 数据为空时，返回空结果而不是抛错

### 6. 映射逻辑测试

- 单独测试 `_select_factor_categories(...)`
- 保证 6 种 `regime` 与类别映射固定且可预期

## 预期实现范围

本轮实现范围控制在因子层首版可用，不做额外扩张。

### 本轮包含

- 新增 `skyeye/factor_layer/` 包
- 新增纯函数与 facade 入口
- 实现五大类中的首版核心指标
- 实现历史分位逻辑
- 补充对应单元测试

### 本轮不包含

- 修改 `market_regime_layer.py`
- 多标的批量因子计算
- 缓存与性能优化
- 更多扩展指标如 `KAMA`、`AROON`、`UO`
- 策略层直接集成改造

## 后续迭代方向

- 扩展更多趋势与震荡指标
- 支持行业指数、个股等多标的统一因子输出
- 支持批量计算和缓存
- 评估是否需要把某些基础数值工具进一步抽到共享模块

## 结论

首版因子层采用独立包的方案 A：

- 保持与市场分类层解耦
- 通过 `MarketRegime` 驱动因子类别输出
- 用纯函数保证可测试性和可复用性
- 用统一结果结构和统一分位规则保证策略消费体验
- 用分层降级机制保证稳定性

该设计足以支持后续进入实现计划，并在不改动现有 `market_regime_layer.py` 的前提下完成首版开发。
