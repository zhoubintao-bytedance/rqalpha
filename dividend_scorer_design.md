# 红利低波ETF打分器 — 概要设计

## 1. 项目概述

### 1.1 目标

构建一个针对 **512890（华泰柏瑞红利低波ETF）** 的估值打分系统，输出 0-10 分的估值评分。分数越低代表越低估（买入机会），分数越高代表越贵（卖出信号）。实际分布中，几乎不会出现 0-1 分或 9-10 分的极端值。

### 1.2 核心方法

采用 **IC_IR 加权**（Information Coefficient / Information Ratio）作为正式权重方案。该方法零训练成本，在当前数据量约束下（~7 年、单资产）是统计上最可靠的选择。配合线性映射（零参数）和样本外验证，将过拟合风险降至最低。

> **为什么不用机器学习？** 512890 上市仅 ~7 年，每个训练窗口约 2 年（~500 天），砍掉 250 天标签不完整区后仅剩 ~250 天。重叠标签导致相邻样本相关性 >99%，有效独立样本仅约 12 个。用 12 个样本拟合 10 维 Elastic Net，统计功效接近零，学到的权重不具备泛化能力。待数据积累到 10 年以上可重新评估。

保留 **Elastic Net 作为实验性对照**（见附录 A），用于跟踪数据积累后机器学习路线的可行性。

### 1.3 使用场景

| 模式 | 说明 |
|------|------|
| **独立工具** | 命令行运行，输出当日评分及各维度详情 |
| **嵌入策略** | 在 RQAlpha 策略的 `handle_bar` 中实时调用，驱动买卖决策 |

### 1.4 数据来源

| 来源 | 用途 | 具体 API | 降级策略 |
|------|------|----------|----------|
| AKShare — 中证指数 | PE_TTM（历史全量） | `stock_zh_index_hist_csindex(symbol="H30269")` → `滚动市盈率` 列 | 本地 SQLite 缓存 |
| AKShare — 成分股 | 股息率（历史全量） | `index_stock_cons_weight_csindex("H30269")` 获取成分股 + 权重，`stock_a_indicator_lg(symbol)` 获取各股 `dv_ttm`，加权计算 | 本地 SQLite 缓存；权重使用最新值回算历史 |
| AKShare — 国债利率 | 10 年期国债收益率 | `bond_zh_us_rate(start_date)` → `中国国债收益率10年` 列 | 本地 SQLite 缓存，数据从 2002 年起 |
| AKShare — ETF 行情 | 价格、成交量、复权价 | `fund_etf_hist_em(symbol="512890", adjust="hfq")` | 本地 SQLite 缓存 |
| AKShare — ETF 净值 | 溢折价率计算 | `fund_etf_fund_info_em(fund="512890")` → 单位净值；溢折价 = (收盘价 - NAV) / NAV | 本地 SQLite 缓存 |
| RQAlpha 本地 bundle | 价格、成交量（回测模式备用） | DataProxy API | 无需降级（本地数据） |

> **关于 PB 指标**：经调研，AKShare 免费数据源不提供 H30269 指数级别的历史 PB 数据（`stock_index_pb_lg` 不支持该指数，中证指数官网也未提供 PB 列）。由于 PE 与 PB 在红利低波类价值指数上高度相关（同维度信息冗余），**决定去掉 PB 指标**，PE/PB 维度缩减为仅 PE。估值指标从 9 个调整为 8 个。未来若找到 PB 数据源可加回。

> **关于股息率历史数据**：`stock_zh_index_value_csindex("H30269")` 仅返回最近 20 个交易日的股息率，不足以计算 3 年百分位。因此采用**成分股加权计算**方式：用 `stock_a_indicator_lg` 获取各成分股的 `dv_ttm`（滚动股息率），按最新权重加权得到指数级股息率。历史部分用最新权重回算（存在幸存者偏差，但实现简单且 H30269 成分股调整频率低，偏差有限）。

**数据降级规则**：
- 股息率维度缺失 → **不输出分数**（核心维度，不可缺失）
- 其他维度缺失 → 该维度权重归零，剩余维度重新归一化，标记置信度降低
- **最少维度要求**：可用维度数 < 3（含股息率在内）时拒绝出分，避免退化为单因子/双因子模型
- **备用股息率来源**：`stock_a_indicator_lg` 不可用时，尝试从 `stock_zh_index_value_csindex` 的最近 20 天数据 + SQLite 历史缓存拼接。若仍不足则不出分。标注为**已知单点故障**

**缓存新鲜度规则**：
- 所有缓存数据必须带**最后更新时间戳**
- 缓存数据超过 **3 个交易日**未更新 → 该数据源标记为"过期"，对应维度置信度降低
- 缓存数据超过 **5 个交易日**未更新 → 该数据源视为不可用，触发降级策略
- 输出中包含**数据新鲜度指标**，标明各数据源的最后更新时间

### 1.5 数据存储与缓存（SQLite）

回测前需预先拉取全量历史数据到 SQLite，回测时从 SQLite 读取。推理模式下每日增量更新。

**SQLite 数据库路径**：`~/.rqalpha/dividend_scorer/cache.db`

**表结构**：

```sql
-- 指数日线数据（PE_TTM + 行情）
CREATE TABLE index_daily (
    date TEXT NOT NULL,           -- YYYY-MM-DD
    index_code TEXT NOT NULL,     -- H30269
    close REAL,
    pe_ttm REAL,                  -- 滚动市盈率
    volume REAL,
    amount REAL,
    updated_at TEXT NOT NULL,     -- ISO 8601 时间戳
    PRIMARY KEY (date, index_code)
);

-- ETF 日线数据（行情 + 复权价）
CREATE TABLE etf_daily (
    date TEXT NOT NULL,
    etf_code TEXT NOT NULL,       -- 512890
    open REAL, high REAL, low REAL, close REAL,
    close_hfq REAL,              -- 后复权收盘价
    volume REAL,
    amount REAL,
    turnover_rate REAL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (date, etf_code)
);

-- ETF 净值数据（用于计算溢折价）
CREATE TABLE etf_nav (
    date TEXT NOT NULL,
    etf_code TEXT NOT NULL,
    nav REAL,                     -- 单位净值
    acc_nav REAL,                 -- 累计净值
    updated_at TEXT NOT NULL,
    PRIMARY KEY (date, etf_code)
);

-- 国债收益率
CREATE TABLE bond_yield (
    date TEXT NOT NULL,
    china_10y REAL,               -- 中国 10 年期国债收益率 (%)
    updated_at TEXT NOT NULL,
    PRIMARY KEY (date)
);

-- 成分股股息率
CREATE TABLE stock_indicator (
    date TEXT NOT NULL,
    stock_code TEXT NOT NULL,     -- 000001 (不含交易所后缀)
    dv_ttm REAL,                 -- 滚动股息率 (%)
    pe_ttm REAL,
    pb REAL,
    total_mv REAL,               -- 总市值
    updated_at TEXT NOT NULL,
    PRIMARY KEY (date, stock_code)
);

-- 成分股权重（仅最新快照）
CREATE TABLE index_weight (
    index_code TEXT NOT NULL,
    stock_code TEXT NOT NULL,
    stock_name TEXT,
    weight REAL,                  -- 权重 (%)
    snapshot_date TEXT NOT NULL,  -- 快照日期
    updated_at TEXT NOT NULL,
    PRIMARY KEY (index_code, stock_code)
);

-- 数据源元信息（追踪各数据源的最后更新时间）
CREATE TABLE data_source_meta (
    source_name TEXT PRIMARY KEY, -- 'index_daily', 'etf_daily', 等
    last_update_date TEXT,        -- 最后一条数据的日期
    last_fetch_time TEXT,         -- 最后一次拉取的时间戳
    record_count INTEGER
);
```

**更新策略**：
- `data_fetcher.py` 提供 `sync_all(start_date, end_date)` 方法，首次运行拉取全量，后续增量追加
- 使用 `INSERT OR REPLACE` 实现幂等写入
- 成分股数据需调用 50 次 `stock_a_indicator_lg`（每只成分股一次），建议加 0.5s 间隔避免触发频率限制

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────┐
│                   红利低波打分器                            │
│                                                            │
│  ┌────────────┐   ┌───────────┐   ┌────────────────────┐  │
│  │  数据获取层  │──▶│  特征引擎  │──▶│    分数合成器       │  │
│  │ DataFetcher │   │  Feature   │   │  ScoreSynthesizer   │  │
│  │  +本地缓存  │   │  Engine    │   │  (标准化+线性映射)  │  │
│  └────────────┘   └───────────┘   └────────────────────┘  │
│        ▲                                    │              │
│        │          ┌────────────┐             ▼              │
│  ┌──────────┐     │  IC_IR     │     ┌──────────────┐      │
│  │ AKShare  │     │  权重计算   │     │    输出       │      │
│  │ Bundle   │     │ (正式方案) │     │  总分+子分数  │      │
│  │ RQData   │     └────────────┘     │  +维度详情    │      │
│  └──────────┘                        └──────────────┘      │
│                        │                                    │
│                        ▼                                    │
│              ┌──────────────────┐                           │
│              │    滚动验证器      │                           │
│              │  时序分割+gap     │                           │
│              │  +策略打分器闭环  │                           │
│              └──────────────────┘                           │
└──────────────────────────────────────────────────────────┘
```

---

## 3. 模块设计

### 3.1 模块划分

| 模块 | 文件 | 职责 |
|------|------|------|
| 数据获取 | `dividend_scorer/data_fetcher.py` | 从 AKShare/Bundle/RQData 获取数据，带本地缓存 |
| 特征引擎 | `dividend_scorer/feature_engine.py` | 计算 8 个估值特征 + 2 个置信度修正因子，统一标准化为百分位 |
| 权重计算 | `dividend_scorer/weight_calculator.py` | IC_IR 权重计算（正式） + Elastic Net 实验性对照 |
| 分数合成 | `dividend_scorer/score_synthesizer.py` | 线性映射 → 加权合成 → 总分输出 |
| 滚动验证 | `dividend_scorer/validator.py` | 时间序列滚动验证 + 策略打分器闭环 |
| 主入口 | `dividend_scorer/main.py` | 独立运行 + RQAlpha 嵌入两种模式的入口 |

### 3.2 模块依赖关系

```
main.py
  ├── data_fetcher.py          # 数据获取 + 缓存
  ├── feature_engine.py        ← 依赖 data_fetcher
  ├── weight_calculator.py     ← 依赖 feature_engine（IC_IR 计算）
  ├── score_synthesizer.py     ← 依赖 weight_calculator 输出的权重
  └── validator.py             ← 依赖 score_synthesizer, strategy_scorer
```

### 3.3 模块接口契约

#### data_fetcher → feature_engine

`DataFetcher` 提供两种调用模式：

```python
class DataFetcher:
    def sync_all(self, start_date: str, end_date: str) -> None:
        """从 AKShare 拉取全量数据到 SQLite（首次或增量更新）"""

    def load_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 SQLite 读取全量历史数据，返回统一 DataFrame。
        columns: date, etf_close, etf_close_hfq, etf_volume, etf_nav,
                 pe_ttm, dividend_yield, bond_10y, premium_rate
        index: DatetimeIndex
        """

    def load_latest(self, date: str) -> dict:
        """推理模式：返回指定日期的单日数据 dict"""

    def get_data_freshness(self) -> dict:
        """返回各数据源的最后更新时间，用于新鲜度检查"""
```

#### feature_engine → score_synthesizer

```python
class FeatureEngine:
    def precompute(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """
        回测模式：预计算全量百分位矩阵。
        返回 DataFrame，columns = 8 个估值特征名 + 2 个置信度因子名，
        values = 标准化后的百分位 [0, 1]（已完成方向统一）。
        index: DatetimeIndex
        """

    def compute_single(self, date: str, history_df: pd.DataFrame) -> dict:
        """推理模式：计算单日特征百分位"""
```

#### weight_calculator → score_synthesizer

```python
class WeightCalculator:
    def calculate_ic_ir_weights(
        self, feature_matrix: pd.DataFrame, price_series: pd.Series
    ) -> dict:
        """
        计算 IC_IR 权重。
        返回: {"weights": {feature_name: weight}, "method": "ic_ir"|"domain_knowledge_fallback",
               "ic_stats": {feature_name: {"ic_mean": float, "ic_ir": float, "pvalue": float}}}
        """
```

---

## 4. 特征工程

### 4.1 特征清单（4 维度 8 估值指标 + 2 置信度修正因子）

所有指标在进入模型前统一标准化为 **3 年滚动百分位 ∈ [0, 1]**，并统一方向为 **"越高越贵"**。

#### 维度一：股息率（2 个指标）

| # | 特征名 | 原始计算 | 数据来源 | 标准化 | 方向处理 |
|---|--------|----------|----------|--------|----------|
| 1 | `dividend_yield_pct` | 成分股加权股息率（dv_ttm）在近 3 年的历史百分位 | `stock_a_indicator_lg` × 50 只成分股，按 `index_stock_cons_weight_csindex` 权重加权 | 已是百分位 | **反转**（1-p），高股息率=便宜=低分 |
| 2 | `yield_spread` | 加权股息率 - 10 年期国债利率 | 股息率同上 + `bond_zh_us_rate` → `中国国债收益率10年` | 3 年滚动百分位 | **反转**（1-p），高利差=有吸引力=低分 |

> **设计意图**: 股息率是红利资产的核心估值锚。利差反映相对无风险收益率的吸引力。反转后统一为"高=贵"方向。
>
> **已删除 `yield_spread_pct`**：原设计中 `yield_spread_pct`（利差在近 3 年的历史百分位）与 `yield_spread`（利差做 3 年滚动百分位）在数学上几乎完全相同——两者都是对同一个原始值（利差）做 3 年滚动百分位。保留两者等于对同一信息做双倍加权，扭曲维度间的权重平衡。删除后股息率维度从 3 个指标精简为 2 个。

#### 维度二：PE 估值（1 个指标）

| # | 特征名 | 原始计算 | 数据来源 | 标准化 | 方向处理 |
|---|--------|----------|----------|--------|----------|
| 3 | `pe_percentile` | H30269 指数滚动 PE_TTM 的 3 年历史百分位 | `stock_zh_index_hist_csindex("H30269")` → `滚动市盈率` 列 | 已是百分位 | 保持（高 PE=贵） |

> **设计意图**: PE 百分位高=估值贵，与统一方向一致，无需反转。
>
> **已删除 `pb_percentile`**：AKShare 免费数据源不提供 H30269 指数级别的历史 PB（`stock_index_pb_lg` 不支持该指数）。PE 与 PB 在红利低波类价值指数上高度相关，去掉 PB 信息损失有限。未来若找到 PB 数据源可加回。

#### 维度三：价格技术（3 个指标）

| # | 特征名 | 原始计算 | 数据来源 | 标准化 | 方向处理 |
|---|--------|----------|----------|--------|----------|
| 4 | `ma250_deviation` | (当前价格 / 250 日均线 - 1) × 100% | `fund_etf_hist_em("512890")` 收盘价 | 3 年滚动百分位 | 保持（偏离高=涨多=贵） |
| 5 | `price_percentile` | 价格在近 3 年的历史百分位 | 同上 | 已是百分位 | 保持（价格高=贵） |
| 6 | `rsi_20` | 20 日 RSI | 同上 | 3 年滚动百分位 | 保持（RSI 高=超买=贵） |

> **设计意图**: 捕捉"价格已经涨太多"或"跌出价值"的技术信号。

#### 维度四：溢折价（2 个指标）

| # | 特征名 | 原始计算 | 数据来源 | 标准化 | 方向处理 |
|---|--------|----------|----------|--------|----------|
| 7 | `premium_rate` | (ETF 收盘价 / 单位净值 - 1) × 100% | `fund_etf_hist_em` 收盘价 + `fund_etf_fund_info_em` 单位净值 | 3 年滚动百分位 | 保持（溢价高=贵） |
| 8 | `premium_rate_ma20` | 溢折价率的 20 日移动平均 | 由 premium_rate 计算 | 3 年滚动百分位 | 保持（持续溢价=贵） |

> **设计意图**: 持续溢价=市场情绪过热=偏贵；折价=可能低估。使用收盘价 vs 当日单位净值计算，而非实时 IOPV（IOPV 需 Level-2 行情数据，免费数据源不可得）。

#### 维度五：市场状态（2 个置信度修正因子）

| # | 特征名 | 原始计算 | 标准化 | 角色 |
|---|--------|----------|--------|------|
| 10 | `volatility_percentile` | 20 日历史波动率在近 3 年的百分位 | 已是百分位 | 置信度修正因子 |
| 11 | `volume_ratio` | 当日成交量 / 20 日平均成交量 | 3 年滚动百分位 | 置信度修正因子 |

> **设计意图**: 波动率和量比反映市场状态而非估值水平。高波动率在牛市顶部和熊市底部都会出现，简单映射为"高=贵"在概念上存在矛盾。因此**不参与加权评分**，而是作为**置信度修正因子**：当波动率/量比处于极端值（百分位 >90% 或 <10%）时，降低评分置信度，提示用户当前市场状态异常，评分可靠性下降。
>
> **置信度修正规则**：
> - 波动率百分位 ∈ [10%, 90%] 且 量比百分位 ∈ [10%, 90%] → 置信度"正常"
> - 任一因子处于极端区间 → 置信度"降低"，输出中标记警告
> - 两个因子均处于极端区间 → 置信度"低"，建议人工复核

### 4.2 标准化规范

- **回溯窗口**：3 年（约 730 个交易日）
- **计算方式**：滚动窗口内的 `(rank - 0.5) / count`（Hazen 公式），范围对称且避免 0/1 边界。相同值使用**平均 rank**
- **数据不足时**：使用可用数据的全量百分位，标记置信度降低
- **方向统一**：反转指标做 `1 - percentile`，确保所有指标"越高越贵"

---

## 5. 权重方案

### 5.1 正式方案：IC_IR 加权

采用 **IC_IR 加权** 作为正式权重方案，零训练成本、无过拟合风险。

> **重要说明：时序 IC vs 截面 IC**
>
> 标准的 IC 计算是在**截面**上做的（同一时间点对多个资产算 Spearman），但本系统只有 **512890 一个资产**，无法做截面计算。因此本系统使用的是**时序 IC（Time-Series IC）**：在**时间维度**上收集多个数据点，计算特征值序列与未来收益序列的 Spearman 秩相关系数。

#### 时序 IC 的滚动计算

采用**滚动窗口**方式计算时序 IC，按月滚动得到 IC 时间序列：

```
# 滚动时序 IC 计算流程
对每个月末时间点 t:
  1. 收集过去 K 个子采样点的 (feature_i, future_return) 数据对
     子采样点间隔 = S 个交易日（见下文"子采样"小节）
     K = 滚动窗口内的子采样点数（推荐 K ≥ 30）
  2. IC_i(t) = Spearman(feature_i_series, future_return_series)
     其中两个 series 各有 K 个数据点

# 最终权重
IC_IR_i = |mean(IC_i 月度序列)| / std(IC_i 月度序列)
weight_i = IC_IR_i / Σ IC_IR_j    # 归一化
```

- **IC 含义**：滚动窗口内，特征值序列与未来 N 日收益率序列的 Spearman 秩相关系数（时序维度）
- **方向验证**：因为所有特征已统一为"越高越贵"（预期未来收益越低），IC 应为**负数**。若某特征 IC 为正，说明方向处理有误或该特征无效，应排除
- **IC_IR 含义**：`|mean(IC 月度序列)| / std(IC 月度序列)` 衡量 IC 的稳定性。IC_IR 越高说明该特征预测能力越稳定，给予更高权重
- **权重归一化**：`weight_i = weight_i / Σ weight_j`，确保权重之和为 1
- **弱因子剔除**：t-test p-value > 0.1 或 IC_IR < 0.3 的因子权重置零，剩余因子重新归一化（详见"弱因子剔除"小节）
- **适用范围**：仅对 8 个估值指标计算 IC_IR 权重。波动率和量比作为置信度修正因子，不参与权重分配

#### 未来收益标签

```
future_return(t, N) = adjusted_price(t+N) / adjusted_price(t) - 1
```

**N 的选择**：使用 **60 个交易日**（约 3 个月）。相比原方案的 120 天，缩短标签窗口使得子采样间隔 S=60 天时相邻样本标签完全不重叠，确保 IC 统计推断的可靠性。

> **复权价格处理**：512890 会定期分红，计算未来收益率标签时**必须使用后复权价格**。若用不复权价格，分红日会出现虚假负收益，标签存在系统性错误。从 AKShare 或 RQAlpha bundle 获取价格时需明确指定复权类型。

#### IC 计算的时间窗口

- **回溯期**：使用百分位窗口完整的历史数据（丢弃上市后前 3 年数据，仅使用百分位窗口完整的区间，见 P1-2 修正）
- **子采样间隔 S**：**60 个交易日**（约 3 个月），配合标签窗口 N=60 天使用（见下方"标签窗口调整"）。相邻子采样点的标签完全不重叠，确保样本独立性。子采样后有效独立样本约 `可用交易日 / 60`（丢弃前 3 年后约 4 年 ≈ 1000 天 / 60 ≈ 16 个独立样本）
- **滚动 IC 窗口 K**：每次滚动计算 IC 时使用过去 K=30 个子采样点（覆盖约 30×60=1800 天 ≈ 7 年），确保每个 IC 估计有足够的数据点
- **滚动频率**：按月滚动（每月末计算一次 IC），得到 IC 月度时间序列
- **稳定性检查**：要求 IC 在不同年份保持同号（负数），若反转频繁则该因子不可靠

> **标签窗口调整（原 N=120 → 调整为 N=60）**：
>
> 原设计使用 N=120 天标签窗口，子采样间隔仅 20 天，导致相邻样本标签有 83% 重叠（100/120），样本远非独立，IC_IR 被系统性高估。
>
> 调整方案：**将标签窗口缩短为 N=60 个交易日（约 3 个月）**，子采样间隔 S=60 天，确保相邻样本标签完全不重叠。独立样本数从虚假的 87 个减少为真实的约 16 个（丢弃前 3 年后）。虽然单个样本的信号强度略弱（3 个月 vs 6 个月），但换来了样本独立性，IC_IR 和 t-test 的统计推断更可靠。
>
> 若实际验证中发现 N=60 信号太弱（IC 均值过小），可考虑使用 **Newey-West 调整标准差**来替代简单标准差计算 IC_IR，从而允许使用更长标签窗口 + 更短子采样间隔的组合。

#### 弱因子剔除

- **统计检验方式**：对 IC 月度时间序列做 t-test（`scipy.stats.ttest_1samp`），检验 IC 均值是否显著异于零
- **剔除条件**：t-test p-value > 0.1（IC 不显著异于零的因子权重置零）
- **辅助阈值**：IC_IR < 0.3 作为辅助参考，两个条件任一不满足即剔除
- **IC_IR 0.3 的含义**：在独立样本场景下，IC_IR ≈ `mean(IC) / std(IC)` 近似于 `t / √n`。由于子采样后仍可能有残余自相关，0.3 是一个保守阈值

### 5.2 备选方案：领域知识手动调权

如果 IC_IR 数据不足或个别维度 IC_IR 差异不大，可使用基于领域知识的固定权重作为补充：

| 维度 | 建议权重 | 理由 |
|------|----------|------|
| 股息率（2 指标） | 35% | 红利资产核心估值锚 |
| PE 估值（1 指标） | 20% | 传统估值指标，市场共识度高 |
| 价格技术（3 指标） | 30% | 反映短中期趋势，指标数量最多 |
| 溢折价（2 指标） | 15% | ETF 特有指标，反映情绪 |

> 注：波动率和量比为置信度修正因子，不参与权重分配。

维度内各指标等权分配。

#### IC_IR → 领域知识的自动回退规则

当 IC_IR 方案出现以下任一情况时，**自动回退到领域知识手动调权**，并在输出中标记 `method: "domain_knowledge_fallback"`：

| 触发条件 | 量化阈值 | 说明 |
|----------|----------|------|
| **存活因子不足** | IC_IR 筛选后存活因子 < 5 个 | 样本量有限时大量因子被剔除是正常的，但少于 5 个因子的权重过度集中 |
| **权重过度集中** | IC_IR 权重下，任何单一维度的权重占比 > 60% | 单一维度主导评分，失去多维度估值的意义 |
| **核心维度被边缘化** | IC_IR 权重下，股息率维度总权重 < 10% | 红利ETF打分器中股息率维度不应被边缘化 |
| **权重结构严重偏离** | IC_IR 权重向量与领域知识权重向量的余弦相似度 < 0.5 | IC_IR 结果与领域常识严重矛盾 |

> **回退优先级**：数据缺失降级（第 1.4 节）和 IC_IR 筛选降级可能同时发生。处理顺序为：先处理数据缺失（权重归零+重归一化），再进行 IC_IR 筛选。若两步叠加后存活因子 < 3，拒绝出分。

---

## 6. 分数合成

### 6.1 子分数映射：线性映射（默认）

将标准化后的百分位 p ∈ [0, 1] 映射为 0-10 分：

```
sub_score(p) = 10 × p
```

所有 8 个估值指标共用同一线性映射（因为所有指标已统一标准化到 [0,1] 百分位）。波动率和量比作为置信度修正因子，不经过映射。

> **为什么用线性映射而非 Sigmoid？**
>
> 原设计使用 Sigmoid 函数 `10 / (1 + exp(-k × (p - mid)))` 并通过网格搜索选取 (k, mid) 参数（最大化 `|Spearman(total_score, future_return)|`）。但该搜索过程使用全量历史数据，而 IC_IR 权重也使用全量历史数据，构成**对同一份数据的双重拟合**，引入隐性过拟合风险。
>
> 线性映射 `sub_score = 10 × p` 是**零参数方案**，彻底消除分数映射环节的拟合风险。百分位已在 [0,1] 区间，线性映射到 [0,10] 既直观又无信息损失。实际分布中的"几乎不会到 0-1 分或 9-10 分"特性由百分位本身的分布保证——在 3 年窗口内，当前值要排到前 10% 或后 10% 才会产生接近极端的分数。
>
> **保留 Sigmoid 作为可选实验**：若未来数据充足（>10 年），可考虑在**时序外推验证**下（前 N 年搜索参数、后 M 年验证）评估 Sigmoid 是否带来显著改善。

### 6.2 加权合成

```
total_score = Σ (weight_i × sub_score_i)
```

权重由 IC_IR 加权（或领域知识手动调权）计算得到，归一化到 sum=1。8 个估值指标使用扁平权重结构，不区分维度权重和子权重。波动率和量比不参与加权合成，仅用于修正输出置信度。

---

## 7. 推理流程

IC_IR 方案不需要多窗口集成投票（权重是通过全量历史数据计算的统计量，不存在训练窗口差异），推理流程大幅简化：

```
输入: 当日 10 个原始指标值（8 估值 + 2 置信度）
  ↓
标准化: 全部转为百分位（Hazen 公式）+ 方向统一（反转 2 个估值指标）
  ↓
线性映射: 8 个估值指标百分位 → sub_score = 10 × p ∈ [0, 10]
  ↓
加权合成: IC_IR 权重（或领域知识回退权重）× sub_score → total_score
  ↓
置信度修正: 波动率/量比极端值 → 调整输出置信度等级
  ↓
输出: 总分 + 置信度 + 各指标子分数 + 维度详情
```

> **权重更新频率**：每季度重新计算一次 IC_IR 权重（纳入最新数据），或在市场发生重大结构性变化时手动触发。

### 7.1 性能优化（回测模式 vs 推理模式）

| 模式 | 策略 | 说明 |
|------|------|------|
| **回测模式** | **预计算全量百分位矩阵** | 在回测启动前，一次性对全部历史数据向量化计算所有指标的百分位时间序列（`pandas.DataFrame.rolling().rank()`），存入内存矩阵。`handle_bar` 中仅做矩阵查找，O(1) 复杂度 |
| **推理模式** | **增量计算** | 仅计算当日指标值在滚动窗口中的百分位排名，复杂度 O(730)（窗口大小） |

> **回测性能估算**：预计算 8 个指标 × 1750 天 × 730 窗口的滚动百分位，使用 pandas 向量化操作约需 1-3 秒。若不预计算，每天在 `handle_bar` 中实时计算则需 1750 × 8 × O(730) ≈ 数分钟，不可接受。

---

## 8. 验证体系

### 8.1 IC 稳定性验证

IC_IR 方案的核心验证是检查各因子 IC 在不同时间段的稳定性：

```
验证步骤:
1. 将全量历史数据按年份分段
2. 每个年份独立计算各因子的 IC（Spearman 相关系数）
3. 检查 IC 在不同年份是否保持同号（应为负数）
4. IC 反转频繁的因子标记为不可靠
```

**验证指标**：
- 因子 IC 均值 < -0.05（有意义的负相关，"越贵"的特征对应越低的未来收益）
- 因子 IC_IR > 0.3（IC 稳定性足够）
- IC 同号年份占比 > 70%（方向一致性）

### 8.2 外层验证：策略打分器闭环

```
IC_IR 加权输出每日分数
        ↓
基于分数构造交易策略（阈值预先固定，不参与优化）:
  score < 3.5  → 买入/加仓信号（明显低估区间）
  score > 6.5  → 卖出/减仓信号（偏高估区间）
        ↓
RQAlpha 回测执行
        ↓
strategy_scorer.py 对回测结果评分
        ↓
策略分数高 → 打分器参数可信
策略分数低 → 记录问题，留待下一版本改进（不直接反馈调整）
```

> **重要**：交易阈值 `(3.5, 6.5)` 基于领域知识预先固定，**不得通过回测优化**。否则会形成"用回测优化阈值 → 用阈值做回测 → 用回测结果评估模型"的循环论证，验证结果失去意义。
>
> **禁止循环调整**：外层验证是**纯观测性**的——记录结果但**不据此调整当前版本的特征或权重**。若验证结果不理想，应记录问题并在下一版本中改进，且改进后的版本**必须在保留的样本外数据上重新验证**。
>
> **样本外验证**：保留最近 1 年数据作为**完全不参与任何参数计算和调整的最终验证集**。IC_IR 权重计算、弱因子剔除等所有环节均不使用最后 1 年数据。外层回测验证分为两段报告：训练期内回测结果（参考性）和样本外回测结果（决定性）。

**验证指标**：
- 内层：因子 IC 均值 < -0.05 且 IC_IR > 0.3（IC 为负数，方向为"越贵→未来收益越低"）
- 外层：策略打分器综合分 > 6.0（回测表现良好）

### 8.3 过拟合防御

| 措施 | 说明 |
|------|------|
| IC_IR 无训练过程 | 权重来自统计量而非模型拟合，不存在过拟合 |
| 线性映射零参数 | 分数映射 `sub_score = 10 × p` 无需从数据中学习参数，彻底消除映射环节的过拟合 |
| IC 子采样 | 每 60 天取一个数据点计算 IC，子采样间隔 = 标签窗口 N，消除重叠标签导致的自相关膨胀 |
| 弱因子剔除 | t-test p > 0.1 或 IC_IR < 0.3 的因子权重置零 |
| IC 方向一致性检查 | IC 应为负数，正 IC 因子自动排除 |
| 领域知识兜底 | IC_IR 存活因子 < 5 或权重结构异常时自动回退领域知识调权 |
| 复权价格标签 | 使用后复权价格计算未来收益，避免分红日虚假负收益 |
| 外层验证阈值固定 | 交易阈值 (3.5, 6.5) 基于领域知识预先固定，不参与优化 |
| 外层验证纯观测 | 验证结果不反馈调整当前版本参数，避免循环论证 |
| 样本外验证 | 保留最近 1 年数据完全不参与参数计算，作为最终验证集 |

### 8.4 模型监控与漂移检测

上线后需持续监控模型有效性，及时发现市场结构变化导致的模型失效：

| 监控项 | 方法 | 触发条件 |
|--------|------|----------|
| **IC 衰减监控** | 滚动 1 年（约 250 天）计算各因子时序 IC，使用与主 IC 计算一致的子采样策略（间隔 60 天） | 超过 3 个因子 IC 反转（由负转正）持续 1 年 |
| **评分分布偏移** | 滚动 120 天评分分布与历史全量分布做 KS 检验 | KS 统计量 p-value < 0.05（分布显著偏移） |
| **权重稳定性** | 每季度重算 IC_IR 权重，与上一版本比较余弦相似度 | 余弦相似度 < 0.7（权重结构大幅变化） |
| **数据源可用性** | 每日检查各数据源最后更新时间（以 A 股交易日历为准，长假期间不触发） | 任一数据源超过 3 个交易日未更新 |

**自动响应**：
- 触发 1 项 → 记录日志，标记评分置信度为"需关注"
- 触发 2 项以上 → 自动降低评分置信度为"低"，发送告警，建议人工复核
- 权重稳定性触发 → 自动发起重训练（重新计算 IC_IR 权重）

---

## 9. 输出格式

### 9.1 独立运行模式

```
═══════════════════════════════════════════════
  红利低波打分器 | 512890 | 2025-03-18
═══════════════════════════════════════════════
  综合评分:  4.2 / 10   (偏低估)
  置信度:    正常（波动率/量比均在正常区间）
  权重方案:  ic_ir
───────────────────────────────────────────────
  估值指标         原始值    百分位   子分数   权重
───────────────────────────────────────────────
  股息率百分位       5.2%    ←0.38    3.8    18%
  国债利差          +2.1%    ←0.28    2.8    15%
  PE 百分位          -       0.52     5.2    18%
  MA250 偏离率     -3.2%     0.38     3.8    13%
  价格百分位         -       0.42     4.2    12%
  RSI(20)           45       0.45     4.5    10%
  当前溢折价       -0.1%     0.36     3.6    8%
  20日均溢折价    -0.08%     0.40     4.0    6%
───────────────────────────────────────────────
  置信度修正因子   原始值    百分位   状态
───────────────────────────────────────────────
  波动率百分位       -       0.62     正常
  量比              0.85     0.45     正常
═══════════════════════════════════════════════
  评分区间参考:
    1-3 分: 明显低估区间
    3-5 分: 偏低估，可考虑建仓/加仓
    5-7 分: 合理估值区间
    7-9 分: 偏高估，可考虑减仓/止盈
═══════════════════════════════════════════════
```

> 注：`←` 表示该指标已反转（原始高值=便宜，反转后高值=贵）

### 9.2 嵌入策略模式（API 返回）

```python
{
    "date": "2025-03-18",
    "etf": "512890",
    "total_score": 4.2,
    "confidence": "normal",
    "features": {
        "dividend_yield_pct": {"raw": 0.052, "percentile": 0.62, "normalized": 0.38, "sub_score": 3.8, "weight": 0.18, "inverted": True},
        "yield_spread":       {"raw": 0.021, "percentile": 0.72, "normalized": 0.28, "sub_score": 2.8, "weight": 0.15, "inverted": True},
        "pe_percentile":      {"raw": 8.15, "percentile": 0.52, "normalized": 0.52, "sub_score": 5.2, "weight": 0.18, "inverted": False},
        "ma250_deviation":    {"raw": -0.032, "percentile": 0.38, "normalized": 0.38, "sub_score": 3.8, "weight": 0.13, "inverted": False},
        "price_percentile":   {"raw": 1.05, "percentile": 0.42, "normalized": 0.42, "sub_score": 4.2, "weight": 0.12, "inverted": False},
        "rsi_20":             {"raw": 45, "percentile": 0.45, "normalized": 0.45, "sub_score": 4.5, "weight": 0.10, "inverted": False},
        "premium_rate":       {"raw": -0.001, "percentile": 0.36, "normalized": 0.36, "sub_score": 3.6, "weight": 0.08, "inverted": False},
        "premium_rate_ma20":  {"raw": -0.0008, "percentile": 0.40, "normalized": 0.40, "sub_score": 4.0, "weight": 0.06, "inverted": False}
    },
    "confidence_modifiers": {
        "volatility_percentile": {"raw": 0.62, "status": "normal"},
        "volume_ratio":          {"raw": 0.45, "status": "normal"}
    },
    "model_meta": {
        "method": "ic_ir",          # 或 "domain_knowledge_fallback"
        "test_ic_avg": -0.12,
        "test_ic_ir_avg": 0.45,
        "label_window_n": 60,
        "subsample_interval": 60,
        "params_version": "v5.0_20250318"
    }
}
```

> **字段说明**：
> - `percentile`：3 年滚动百分位原始值（反转前）
> - `normalized`：方向统一后的值（反转指标 = 1 - percentile，非反转指标 = percentile）。统一为"越高越贵"方向
> - `sub_score`：`10 × normalized`（线性映射）
> - `weight`：权重之和 = 1.00（0.18+0.15+0.18+0.13+0.12+0.10+0.08+0.06 = 1.00）

---

## 10. 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 数据获取 | **AKShare** | 免费 A 股数据接口，获取 PE、股息率、国债利率、ETF 行情/净值 |
| 本地缓存 | **SQLite** | AKShare 数据缓存，回测时离线读取，推理时增量更新 |
| 数据处理 | **pandas / numpy** | 特征计算与时间序列处理 |
| 统计 | **scipy.stats** | Spearman 相关系数、IC 计算 |
| 回测引擎 | **RQAlpha** | 作为 Mod 嵌入，策略回测与闭环验证 |
| 可视化（可选） | **matplotlib** | 分数分布图、IC 时序图 |
| 实验性对照（可选） | **scikit-learn (ElasticNetCV)** | 仅用于附录 A 的实验性 Elastic Net 对照 |

---

## 11. 文件结构

```
rqalpha/
├── dividend_scorer/
│   ├── __init__.py
│   ├── main.py                # 主入口（独立运行 CLI）
│   ├── data_fetcher.py        # 数据获取层（AKShare + 本地 SQLite 缓存）
│   ├── feature_engine.py      # 特征引擎（8 估值指标 + 2 置信度修正因子 + 标准化 + 方向统一）
│   ├── weight_calculator.py   # IC_IR 权重计算（正式） + Elastic Net 实验性对照
│   ├── score_synthesizer.py   # 线性映射 + 加权合成
│   ├── validator.py           # IC 稳定性验证 + 策略打分器闭环
│   └── config.py              # 打分器配置常量
├── rqalpha/
│   └── mod/
│       └── rqalpha_mod_dividend_scorer/
│           ├── __init__.py    # load_mod() 入口
│           └── mod.py         # AbstractMod 实现（start_up / tear_down）
├── strategy_scorer.py         # 已有的策略打分器（外层验证用）
└── dividend_scorer_design.md  # 本设计文档
```

### 11.1 RQAlpha Mod 嵌入方式

打分器作为 RQAlpha Mod 嵌入，遵循标准 Mod 生命周期：

```python
# rqalpha/mod/rqalpha_mod_dividend_scorer/mod.py

class DividendScorerMod(AbstractMod):
    def start_up(self, env, mod_config):
        """
        回测启动时:
        1. 从 SQLite 加载全量历史数据到内存
        2. 预计算全量百分位矩阵（向量化）
        3. 加载或计算 IC_IR 权重
        4. 订阅 POST_BAR 事件
        """
        self._scorer = DividendScorer(mod_config)
        self._scorer.precompute(env)
        env.event_bus.add_listener(EVENT.POST_BAR, self._on_bar)

    def _on_bar(self, event):
        """每个 bar 后计算当日评分，存入 context 供策略读取"""
        date = event.trading_dt.date()
        score_result = self._scorer.score(date)  # O(1) 矩阵查找
        # 将评分结果挂载到 environment，策略可通过 env 访问
        Environment.get_instance().dividend_score = score_result

    def tear_down(self, success, exception=None):
        pass
```

**策略中使用**：
```python
def handle_bar(context, bar_dict):
    score = context.env.dividend_score
    if score and score['total_score'] < 3.5:
        order_value('512890.XSHG', 30000)
```

### 11.2 配置常量（config.py）

```python
# dividend_scorer/config.py

# === 目标 ETF ===
ETF_CODE = "512890"
INDEX_CODE = "H30269"

# === 百分位计算 ===
PERCENTILE_WINDOW = 730        # 3 年滚动窗口（交易日）
PERCENTILE_MIN_DATA = 120      # 最少数据量，不足则标记置信度降低

# === IC_IR 计算 ===
LABEL_WINDOW_N = 60            # 未来收益标签窗口（交易日）
SUBSAMPLE_INTERVAL_S = 60      # 子采样间隔（交易日）
IC_ROLLING_K = 30              # 滚动 IC 窗口内的子采样点数
IC_IR_THRESHOLD = 0.3          # IC_IR 弱因子剔除阈值
IC_PVALUE_THRESHOLD = 0.1      # t-test p-value 弱因子剔除阈值
IC_MIN_SURVIVING = 5           # 存活因子不足时回退领域知识
WEIGHT_CONCENTRATION_MAX = 0.6 # 单维度最大权重占比
DIVIDEND_WEIGHT_MIN = 0.1      # 股息率维度最低权重
COSINE_SIM_MIN = 0.5           # IC_IR 与领域知识的最低余弦相似度

# === 分数合成 ===
SCORE_BUY_THRESHOLD = 3.5      # 买入信号阈值（固定，不可优化）
SCORE_SELL_THRESHOLD = 6.5     # 卖出信号阈值（固定，不可优化）

# === 置信度修正 ===
CONFIDENCE_EXTREME_LOW = 0.1   # 极端区间下界
CONFIDENCE_EXTREME_HIGH = 0.9  # 极端区间上界

# === 缓存 ===
CACHE_DB_PATH = "~/.rqalpha/dividend_scorer/cache.db"
CACHE_STALE_DAYS = 3           # 缓存过期天数（交易日）
CACHE_EXPIRED_DAYS = 5         # 缓存不可用天数（交易日）
API_CALL_INTERVAL = 0.5        # AKShare API 调用间隔（秒）

# === 领域知识权重（回退方案） ===
DOMAIN_WEIGHTS = {
    "dividend_yield_pct": 0.175,   # 股息率维度 35% / 2 指标
    "yield_spread":       0.175,
    "pe_percentile":      0.20,    # PE 维度 20% / 1 指标
    "ma250_deviation":    0.10,    # 价格技术维度 30% / 3 指标
    "price_percentile":   0.10,
    "rsi_20":             0.10,
    "premium_rate":       0.075,   # 溢折价维度 15% / 2 指标
    "premium_rate_ma20":  0.075,
}

# === 验证 ===
MIN_DIMENSIONS = 3             # 最少可用维度数，不足则拒绝出分
STRATEGY_SCORE_TARGET = 6.0    # 外层验证：策略打分器目标分
SAMPLE_OUT_YEARS = 1           # 样本外验证保留年数

# === 特征元信息 ===
FEATURES = {
    "dividend_yield_pct": {"dimension": "dividend", "inverted": True},
    "yield_spread":       {"dimension": "dividend", "inverted": True},
    "pe_percentile":      {"dimension": "pe",       "inverted": False},
    "ma250_deviation":    {"dimension": "price",    "inverted": False},
    "price_percentile":   {"dimension": "price",    "inverted": False},
    "rsi_20":             {"dimension": "price",    "inverted": False},
    "premium_rate":       {"dimension": "premium",  "inverted": False},
    "premium_rate_ma20":  {"dimension": "premium",  "inverted": False},
}
```

---

## 12. 风险与约束

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 512890 上市仅 ~7 年，历史数据有限 | IC 计算样本量有限，统计显著性不足 | IC 子采样（间隔 60 天 = 标签窗口 N）+ t-test 显著性检验 + 领域知识兜底 + 弱因子剔除 |
| 有效独立样本严重不足 | 重叠标签导致相邻样本相关性高 | 放弃 ML 路线，使用无训练过程的 IC_IR 加权 + 子采样间隔 = 标签窗口确保独立性 |
| 复权价格缺失 | 分红日出现虚假负收益，IC 计算标签存在系统性错误 | 必须使用后复权价格计算未来收益率标签（`fund_etf_hist_em(adjust="hfq")`） |
| AKShare 数据质量不稳定 | 特征计算出错 | 本地 SQLite 缓存（带时间戳）+ 分层降级策略 + 新鲜度监控 |
| 股息率依赖成分股逐只拉取 | 50 次 API 调用，耗时且可能触发频率限制 | 加 0.5s 间隔 + SQLite 缓存减少重复调用 + 批量失败时降级为近 20 天官方数据 |
| 股息率使用最新权重回算历史 | 幸存者偏差 + 历史权重不准 | H30269 半年调整一次，偏差有限；在输出中标记数据局限性 |
| PB 数据不可得 | 估值维度从 2 指标退化为 1 指标 | PE 与 PB 高度相关，信息损失有限；未来找到 PB 数据源可加回 |
| 溢折价使用收盘价 vs NAV 而非实时 IOPV | 日内溢折价信号缺失 | 日频打分器场景下收盘价 vs NAV 足够；实时 IOPV 需 Level-2 数据 |
| 市场结构变化（regime change） | 历史 IC 失效 | 每季度重算 IC_IR + 滚动 1 年 IC 衰减监控 + 评分分布 KS 检验 + 领域知识约束 |
| IC_IR 筛选后因子全军覆没 | 无法出分或单因子退化 | 存活因子 < 5 自动回退领域知识调权；回退后仍 < 3 则拒绝出分 |
| 股息率数据更新延迟 | 实时分数不够准确 | 标注数据时效性，低频指标用最近可用缓存值 + 缓存超过 3 个交易日（A 股日历）标记过期 |
| 多维度同时缺失 | 退化为单因子模型 | 可用维度数 < 3 时拒绝出分 |

---

## 附录 A：Elastic Net 实验性对照（待数据充足后评估）

> **状态**：实验性。当前数据量不足以支撑有效训练，仅作为对照实验保留。待 512890 数据积累到 10 年以上（约 2028 年后）重新评估。

### A.1 已知问题

| 问题 | 说明 | 影响 |
|------|------|------|
| 有效样本量不足 | 每个训练窗口 ~250 天可用，重叠标签导致有效独立样本 ~12 个 | 12 样本拟合 10 维模型，统计功效接近零 |
| 前视偏差风险 | `ElasticNetCV(cv=5)` 默认使用 KFold，将未来数据混入训练集 | 选出的 alpha 和 l1_ratio 无效 |
| 复权价格 | 未处理复权价格则分红日标签有系统性错误 | IC 和权重均受污染 |

### A.2 正确配置（如需实验）

```python
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit

model = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=None,
    cv=TimeSeriesSplit(n_splits=5, gap=250),  # 必须用 TimeSeriesSplit，不能用默认 KFold
    positive=True,
    max_iter=10000,
)
```

### A.3 引入条件

满足以下**全部**条件时可考虑将 Elastic Net 升级为正式方案：
1. 数据积累 > 10 年（有效独立样本 > 30 个）
2. 在 TimeSeriesSplit 验证下，Elastic Net 打分与未来收益的 Spearman 相关系数**显著优于** IC_IR（p < 0.05）
3. 多窗口权重稳定性良好（不同窗口的权重向量余弦相似度 > 0.7）
