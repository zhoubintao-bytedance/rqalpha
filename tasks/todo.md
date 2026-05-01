# AX1 Tradable Outcome + Alpha Transfer 重构计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 AX1 建立一条贯穿训练后评估、confidence、promotion 的可交易净收益唯一真相，并记录 alpha 从模型输出到最终可执行持仓的损耗归因。

**Architecture:** 新增两个一等契约：`TradableOutcome` 负责净收益、成本、净值曲线、回撤、换手、容量和订单负担；`AlphaTransferLedger` 负责记录 `model_expected_return -> adjusted_expected_return -> confidence/risk score -> target_weight -> smoothed_weight -> executable_weight -> realized_net_return` 的 retention 和 blocker reason。现有 evaluation / promotion / confidence 不再各自猜 label 字段，而是统一从契约对象取交易口径字段。

**Tech Stack:** Python, pandas, pytest, existing AX1 modules under `skyeye/products/ax1`, tests under `tests/products/ax1`.

---

## HARD-GATE

- [x] 用户确认本计划后，才允许修改业务代码。
- [ ] 本轮不调默认参数，不改 profile 阈值，不重写模型，不做 live advisor，不做 Black-Litterman。
- [ ] 每个实现任务先写失败测试，再实现，再跑定向测试。
- [ ] 最终必须跑 AX1 全量测试和 CLI smoke。

## Current Evidence

- 训练 target 现在是 `label_relative_net_return_{h}d`，由 `skyeye/products/ax1/research/training.py::target_columns()` 决定。
- `MultiHorizonLabelBuilder` 同时生成 gross return、cost-aware net return、peer-relative net return，成本先验和交易成本语义混在一起。
- signal evaluation 优先 relative label，portfolio evaluation 优先 net label，confidence calibration 默认 relative 20d，calibration bundle / parameter validation 又各自选择 net/gross label。
- portfolio `max_drawdown` 当前基于 gross `portfolio_return` 曲线；`excess_net_mean_return` 使用成本感知 `_net_return`，两者不是同一条净值曲线。
- promotion gate 已经有部分 tradability hard gate，但仍混合 research evidence 和 tradability evidence；Rank IC / bucket spread 是 soft check。
- alpha retention 不透明：confidence、risk penalty、turnover、capacity、rebalance interval、lot、min trade 等机制会影响最终交易，但没有统一归因表。

## File Structure

- Create: `skyeye/products/ax1/tradability.py`
  - Owns `TradableOutcome`, `AlphaTransferLedger`, and pure helper functions for building tradability artifacts.
- Modify: `skyeye/products/ax1/evaluation/metrics.py`
  - Add net-equity based portfolio metric helper and consume `TradableOutcome` when available.
- Modify: `skyeye/products/ax1/research/execution.py`
  - Build `TradableOutcome` and `AlphaTransferLedger` inside portfolio replay.
- Modify: `skyeye/products/ax1/run_experiment.py`
  - Persist `tradable_outcome` and `alpha_transfer_ledger`; pass them to promotion and readiness summaries.
- Modify: `skyeye/products/ax1/confidence.py`
  - Add calibration path based on tradable net success while preserving `confidence_raw`.
- Modify: `skyeye/products/ax1/promotion.py`
  - Split gate output into `tradability_gate` and `research_support_gate`; top-level `passed` depends first on tradability.
- Modify: `skyeye/products/ax1/package_io.py`
  - Ensure new artifact sections round-trip in experiment/package JSON.
- Test: `tests/products/ax1/test_tradability_contract.py`
- Test: `tests/products/ax1/test_evaluation_metrics.py`
- Test: `tests/products/ax1/test_promotion.py`
- Test: `tests/products/ax1/test_config_pipeline.py`
- Test: `tests/products/ax1/test_calibration.py`

---

## Task 1: TradableOutcome Contract

**Files:**
- Create: `skyeye/products/ax1/tradability.py`
- Test: `tests/products/ax1/test_tradability_contract.py`

- [x] **Step 1: Write failing tests for net outcome and double-cost prevention**

Add tests that construct simple target weights, labels, and orders:

```python
import pandas as pd
import pytest

from skyeye.products.ax1.tradability import build_tradable_outcome


def test_tradable_outcome_uses_execution_cost_once_for_net_curve():
    weights = pd.DataFrame(
        [
            {"date": "2024-01-02", "order_book_id": "510300.XSHG", "target_weight": 1.0},
            {"date": "2024-01-03", "order_book_id": "510300.XSHG", "target_weight": 1.0},
        ]
    )
    labels = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "order_book_id": "510300.XSHG",
                "label_return_10d": 0.010,
                "label_net_return_10d": 0.008,
            },
            {
                "date": "2024-01-03",
                "order_book_id": "510300.XSHG",
                "label_return_10d": -0.020,
                "label_net_return_10d": -0.022,
            },
        ]
    )
    orders = pd.DataFrame(
        [
            {
                "date": "2024-01-02",
                "order_book_id": "510300.XSHG",
                "side": "buy",
                "order_value": 1000000.0,
                "estimated_cost": 1000.0,
            }
        ]
    )
    outcome = build_tradable_outcome(
        target_weights=weights,
        labels=labels,
        orders=orders,
        portfolio_value=1_000_000.0,
        gross_label_column="label_return_10d",
    )
    assert outcome["schema_version"] == 1
    assert outcome["return_column"] == "label_return_10d"
    assert outcome["net_return_by_date"]["2024-01-02"] == pytest.approx(0.009)
    assert outcome["net_return_by_date"]["2024-01-03"] == pytest.approx(-0.020)
    assert outcome["net_equity_curve"][-1]["equity"] == pytest.approx(1.009 * 0.98)
    assert outcome["max_net_drawdown"] > 0.0
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_tradability_contract.py::test_tradable_outcome_uses_execution_cost_once_for_net_curve -q
```

Expected: FAIL because `skyeye.products.ax1.tradability` does not exist.

- [x] **Step 3: Implement `build_tradable_outcome`**

Implement a pure function in `skyeye/products/ax1/tradability.py` with this public signature:

```python
def build_tradable_outcome(
    *,
    target_weights: pd.DataFrame,
    labels: pd.DataFrame,
    orders: pd.DataFrame | None,
    portfolio_value: float,
    gross_label_column: str = "label_return_10d",
) -> dict[str, Any]:
    ...
```

Required behavior:
- merge by `date/order_book_id`;
- compute `gross_return_by_date` from final executable `target_weight * gross_label_column`;
- compute `execution_cost_by_date` from `orders.estimated_cost / portfolio_value`;
- compute `net_return_by_date = gross_return_by_date - execution_cost_by_date`;
- compute `net_equity_curve` and `max_net_drawdown`;
- expose `mean_net_return`, `mean_execution_cost`, `mean_turnover`, `date_count`.

- [x] **Step 4: Run contract tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_tradability_contract.py -q
```

Expected: PASS.

---

## Task 2: AlphaTransferLedger

**Files:**
- Modify: `skyeye/products/ax1/tradability.py`
- Test: `tests/products/ax1/test_tradability_contract.py`

- [x] **Step 1: Write failing test for alpha transfer stages**

Add a test that compares model scores, target weights, smoothed weights, executable weights, and realized net returns:

```python
from skyeye.products.ax1.tradability import build_alpha_transfer_ledger


def test_alpha_transfer_ledger_reports_retention_and_blockers():
    predictions = pd.DataFrame(
        [
            {"date": "2024-01-02", "order_book_id": "A", "expected_relative_net_return_10d": 0.020, "confidence": 0.50},
            {"date": "2024-01-02", "order_book_id": "B", "expected_relative_net_return_10d": 0.010, "confidence": 1.00},
        ]
    )
    target = pd.DataFrame(
        [
            {"date": "2024-01-02", "order_book_id": "A", "target_weight": 0.60},
            {"date": "2024-01-02", "order_book_id": "B", "target_weight": 0.40},
        ]
    )
    executable = pd.DataFrame(
        [
            {"date": "2024-01-02", "order_book_id": "A", "target_weight": 0.30, "trade_reason": "capacity"},
            {"date": "2024-01-02", "order_book_id": "B", "target_weight": 0.40, "trade_reason": "trade"},
        ]
    )
    outcome = {"net_return_by_date": {"2024-01-02": 0.006}}
    ledger = build_alpha_transfer_ledger(
        predictions=predictions,
        target_weights=target,
        executable_weights=executable,
        tradable_outcome=outcome,
        score_column="expected_relative_net_return_10d",
    )
    assert ledger["schema_version"] == 1
    assert ledger["summary"]["model_alpha_weighted"] == pytest.approx(0.02 + 0.01)
    assert ledger["summary"]["executable_alpha_weighted"] == pytest.approx(0.30 * 0.02 + 0.40 * 0.01)
    assert ledger["blocker_counts"]["capacity"] == 1
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_tradability_contract.py::test_alpha_transfer_ledger_reports_retention_and_blockers -q
```

Expected: FAIL because `build_alpha_transfer_ledger` is not implemented.

---

# AX1 行业约束异常排查与修复计划

## HARD-GATE

- [x] 本段只做现状分析与证据收集，不修改 AX1 业务代码。
- [x] 待你确认“现状分析”无误后，再继续写“功能点 / 修复方案”段。
- [x] 待你确认完整计划后，再按 TDD 开始实现。

## 现状分析

- 症状已在当前训练产物复现：
  - `/home/tiger/rqalpha/tmp/ax1_train_runs/ax1_20260501_full_20180101_20260430_mt128_nocap2/experiment.json` 中 `constraint_status == "hard_violation"`。
  - 同一产物里 `target_weights` 的 `industry` 全部为 `Unknown`。
  - 同一产物里 `evaluation.portfolio.constraint_violations.max_industry_weight_count == 98`。
- 评估层把 `Unknown` 当成真实行业分组统计：
  - `skyeye/products/ax1/evaluation/metrics.py::_industry_exposure_frame()` 在缺失行业时直接补 `"Unknown"`，随后 `_constraint_violations()` 直接按 `date, industry` 聚合并计数，没有跳过 “全 Unknown” 场景。
- 但执行/优化层对 “全 Unknown” 的语义是“跳过行业约束”：
  - `skyeye/products/ax1/optimizer/allocation.py` 在 `unique_industries == {"Unknown"}` 时显式跳过 `max_industry_weight`。
  - `skyeye/products/ax1/optimizer/executable.py` 也在 `unique_industries == {"Unknown"}` 时显式跳过 group cap。
  - 这说明当前 hard violation 统计与真实执行语义不一致。
- `industry` 之所以全是 `Unknown`，根因在 ETF 行业元数据获取链路：
  - `skyeye/products/ax1/data_builder.py::_attach_provider_industry()` 依赖 `DataFacade.get_industry()`。
  - `skyeye/data/facade.py::get_industry()` 直接透传到 `RQDataProvider.get_industry()`。
  - `skyeye/data/provider.py::get_industry()` 实际调用 `rqdatac.get_instrument_industry(...)`，该接口按股票口径校验；对 ETF 会报 `expect ('CS',) instrument, got ETF(...)`，并返回空结果。
  - 我本地对 `512800.XSHG / 512880.XSHG / 515000.XSHG / 159928.XSHE / 510300.XSHG / 510880.XSHG` 逐个复现，`DataFacade.get_industry(..., date='2026-03-31')` 都返回空。
- `run_experiment` / universe metadata 侧还放大了这个问题：
  - `DynamicUniverseBuilder.build_with_metadata()` 产出的 metadata 目前只保留 `order_book_id / asset_type / universe_layer / universe_pit_status / universe_as_of_date`，没有把 `industry` 当成 universe metadata contract 保下来。
  - `skyeye/products/ax1/run_experiment.py::_attach_universe_metadata()` 也只回填 `asset_type / universe_layer`，不会回填 `industry`。

## 当前判断

- 这不是单点 bug，而是两个连在一起的问题：
  - 上游 ETF 行业分类拿不到，导致训练/执行链路里的 `industry` 退化成全 `Unknown`。
  - 下游评估层没有和执行层保持一致的 “all Unknown => skip industry cap” 语义，导致 98 天硬违规是伪阳性。
- 只修评估层会把伪违规消掉，但行业约束仍然不会真正生效。
- 只修上游行业分类，评估/执行语义仍然存在潜在不一致，未来再次出现全 `Unknown` 时还会误报。

## 功能点

- [x] **功能点 1：给 AX1 ETF universe 补一条稳定的行业分类 fallback**
  - 修改 `skyeye/products/ax1/data_builder.py`
  - 目标：当 `DataFacade.get_industry()` 对 ETF 返回空时，不再把整池 ETF 直接降级成全 `Unknown`。
  - 实现方向：增加 AX1 产品自带的 ETF 元数据 fallback，至少覆盖当前 profile 里的 ETF universe；仅在 provider 没给出有效行业时启用，不覆盖上游显式值。
  - 预期结果：`AX1TrainingDataBuilder.build()` 产出的 raw frame 中，当前 AX1 ETF universe 的 `industry` 不再全为 `Unknown`。

- [x] **功能点 2：把 `industry` 升级成 AX1 universe metadata 的一等字段**
  - 修改 `skyeye/products/ax1/universe.py`
  - 修改 `skyeye/products/ax1/run_experiment.py`
  - 目标：`DynamicUniverseBuilder.build_with_metadata()` 产出的 metadata 要保留 `industry`；`run_experiment` 在回填 universe metadata 时也要持续带着 `industry` 走，避免后续链路丢字段。
  - 预期结果：`universe_metadata`、`scoped`、`fused_predictions`、最终 `target_weights` 的行业字段来源一致，不依赖“碰巧前面还没掉字段”。

- [x] **功能点 3：评估层 hard-violation 统计与执行层语义对齐**
  - 修改 `skyeye/products/ax1/evaluation/metrics.py`
  - 目标：当某个日期的活跃持仓行业集合是纯 `{"Unknown"}` 时，不把它计入 `max_industry_weight` 违规；这要和 `allocation` / `executable` / `smoother` 现有的 all-Unknown skip 语义一致。
  - 预期结果：未来即便上游再次缺行业数据，也不会再把 “缺元数据” 错判成“真实行业硬违规”。

- [x] **功能点 4：补回归测试，锁住真实 contract**
  - 修改 `tests/products/ax1/test_train_entrypoint.py`
  - 修改 `tests/products/ax1/test_evaluation_metrics.py`
  - 视实现需要补 `tests/products/ax1/test_config_pipeline.py` 或新增更窄的 AX1 metadata test
  - 目标：
    - 训练数据构建测试不再断言 `industry` 全 `Unknown`，而是断言 fallback contract 生效；
    - 评估层新增 all-Unknown skip test；
    - 必要时补 `universe metadata` 保留 `industry` 的测试。

## 本段范围边界

- [x] 本轮不改 `max_industry_weight` 默认值，不放松约束阈值。
- [x] 本轮不改 optimizer 的核心分配逻辑；只修元数据来源和评估语义一致性。
- [x] 本轮不重做 profile schema；先在当前 AX1 universe contract 内把 bug 收口。

## 风险与决策

- [x] **决策 1：ETF 行业 fallback 放在 AX1 产品层，不放到通用 `DataFacade`**
  - 推荐：在 `skyeye/products/ax1` 内实现 AX1 自己的 ETF 行业解析 / fallback helper，由 `data_builder` 和 `run_experiment` 复用。
  - 不推荐把这次修复直接塞进 `skyeye/data/facade.py`：
    - 当前问题是 `rqdatac.get_instrument_industry()` 对 ETF 语义不成立，不是通用 data layer 的简单 bug；
    - AX1 对 `industry` 的需求本质上是“组合约束分组标签”，允许与通用证券行业口径不完全相同；
    - 把 AX1 特有 ETF 分类语义下沉到通用层，容易污染其他产品。

- [x] **决策 2：fallback source priority 采用“显式值优先，缺失时再回退”**
  - 推荐优先级：
    - raw/frame 中已有且非空、非 `Unknown` 的 `industry`
    - provider 返回的有效行业值
    - AX1 内置 ETF fallback registry
    - 最后才是 `Unknown`
  - 这样可以保留用户手工构造测试数据和未来上游修好后的真实值，不会让本地 fallback 反向覆盖更可信的来源。

- [x] **决策 3：AX1 内置 fallback 先做 deterministic registry，不做运行时中文名称猜测**
  - 推荐：用稳定的 ETF code -> industry registry 作为 fallback 主体，覆盖当前 AX1 默认 universe。
  - 不推荐在运行时靠 `symbol` 中文分词/模糊匹配猜行业：
    - `科技龙头`、`消费`、`红利`、`5年国债`、`黄金`、`纳指` 这类名称跨“行业 / 风格 / 大类资产”边界，模糊猜测容易把 style/core ETF 误打成行业；
    - 模糊规则一旦写进主链路，未来 profile 扩 universe 时更难验证。
  - 本轮允许 registry 未覆盖的 ETF 继续保留 `Unknown`，但不能让当前默认 AX1 universe 大面积漏标。

- [x] **决策 4：评估层先严格对齐执行层当前 contract，只 skip “all Unknown”**
  - 推荐：`evaluation.metrics` 与 `allocation` / `smoother` / `executable` 保持同一条规则：
    - 若当日活跃持仓行业集合恰好为 `{"Unknown"}`，则该日不计 `max_industry_weight` 违规；
    - 若存在已知行业与 `Unknown` 混合，则保持当前保守语义，不在本轮擅自改成“忽略 Unknown bucket”。
  - 原因：这轮目标是 contract 一致性和真实 bug 收口，不额外引入新的风控语义变化。

- [x] **风险 1：fallback registry 需要跟 universe 演进同步维护**
  - 风险：未来 profile 新增 ETF 时，如果没补 registry，可能再次出现局部 `Unknown`。
  - 控制方式：测试里至少锁住当前默认 profile universe；必要时在 config pipeline test 里断言关键 ETF 的行业标签非 `Unknown`。

- [x] **风险 2：已有测试把“industry 全 Unknown”当成旧行为**
  - 风险：`tests/products/ax1/test_train_entrypoint.py` 当前明确断言 `frame["industry"].eq("Unknown").all()`。
  - 控制方式：把断言改成 fallback contract；避免只改实现不改测试，留下错误旧预期。

- [x] **风险 3：仅修评估层会掩盖上游分类缺口**
  - 处理：本轮必须同时补上游 fallback 与 metadata 传递，不能只把 `98` 天违规数字改小。

## Review

- [x] TDD:
  - 先新增失败测试：
    - `tests/products/ax1/test_train_entrypoint.py::test_ax1_training_data_builder_uses_data_facade_and_attaches_training_contract`
    - `tests/products/ax1/test_train_entrypoint.py::test_ax1_universe_metadata_preserves_and_reapplies_industry_labels`
    - `tests/products/ax1/test_evaluation_metrics.py::test_portfolio_layer_ignores_all_unknown_industry_constraint_violations`
  - 失败后再实现，随后重跑全部转绿。
- [x] 定向验证：
  - `tests/products/ax1/test_train_entrypoint.py` + `tests/products/ax1/test_evaluation_metrics.py` + `tests/products/ax1/test_allocation_optimizer.py` + `tests/products/ax1/test_execution_smoother.py` + `tests/products/ax1/test_executable_optimizer.py` => `67 passed`
- [x] 语法 / CLI smoke：
  - `python -m py_compile skyeye/products/ax1/etf_metadata.py skyeye/products/ax1/data_builder.py skyeye/products/ax1/universe.py skyeye/products/ax1/run_experiment.py skyeye/products/ax1/evaluation/metrics.py ...`
  - `python -m skyeye.products.ax1.run_experiment --help`
  - `python -m skyeye.products.ax1.train --help`
- [x] 真实 DataFacade quick probe：
  - 默认 `personal_etf_core` universe 下，builder 产物 `raw_unknown_count == 0`
  - `_build_industry_map(...)` 的 `industry_map_unknown_count == 0`
- [x] AX1 全量测试：
  - `tests/products/ax1 -q` => `408 passed, 2 failed`
  - 失败项与本次行业修复无关，落在当前工作区已有 profile / implementation_status 断言漂移：
    - `test_default_profile_preserves_ax1_architecture_decisions`
    - `test_run_experiment_writes_lgbm_pipeline_contract`

- [x] **Step 3: Implement `build_alpha_transfer_ledger`**

Public signature:

```python
def build_alpha_transfer_ledger(
    *,
    predictions: pd.DataFrame,
    target_weights: pd.DataFrame,
    executable_weights: pd.DataFrame,
    tradable_outcome: dict[str, Any],
    score_column: str = "expected_relative_net_return_10d",
) -> dict[str, Any]:
    ...
```

Required fields:
- `summary.model_alpha_weighted`;
- `summary.target_alpha_weighted`;
- `summary.executable_alpha_weighted`;
- `summary.target_retention_ratio`;
- `summary.executable_retention_ratio`;
- `blocker_counts`;
- `by_date`;
- `by_asset`.

- [x] **Step 4: Run ledger tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_tradability_contract.py -q
```

Expected: PASS.

---

## Task 3: Wire TradableOutcome Into Portfolio Replay

**Files:**
- Modify: `skyeye/products/ax1/research/execution.py`
- Modify: `skyeye/products/ax1/run_experiment.py`
- Test: `tests/products/ax1/test_config_pipeline.py`

- [x] **Step 1: Write failing pipeline test**

Add a test using the existing lightweight AX1 pipeline fixture or monkeypatch path. It must assert:
- `result["tradable_outcome"]` exists;
- `result["alpha_transfer_ledger"]` exists;
- `evaluation.portfolio.max_drawdown` equals `tradable_outcome.max_net_drawdown` after the migration task is complete.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_config_pipeline.py -q
```

Expected: FAIL on missing `tradable_outcome` / `alpha_transfer_ledger`.

- [x] **Step 3: Build artifacts in `run_portfolio_replay`**

In `skyeye/products/ax1/research/execution.py::run_portfolio_replay()`:
- after `smoothed_weights` and `orders` are built;
- before returning;
- call `build_tradable_outcome(...)`;
- call `build_alpha_transfer_ledger(...)`;
- return both in the replay result.

- [x] **Step 4: Persist artifacts from runner**

In `skyeye/products/ax1/run_experiment.py::run_experiment()`:
- read `tradable_outcome = replay_result["tradable_outcome"]`;
- read `alpha_transfer_ledger = replay_result["alpha_transfer_ledger"]`;
- include both in final `result`.

- [x] **Step 5: Run pipeline test**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_config_pipeline.py -q
```

Expected: PASS for the new artifact assertions.

---

## Task 4: Make Evaluation Use Net Tradable Outcome

**Files:**
- Modify: `skyeye/products/ax1/evaluation/metrics.py`
- Modify: `skyeye/products/ax1/research/execution.py`
- Test: `tests/products/ax1/test_evaluation_metrics.py`

- [x] **Step 1: Write failing test for net drawdown**

Add a test where gross return has shallow drawdown but execution cost creates deeper net drawdown. Assert portfolio `max_drawdown` follows net curve, not gross curve.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_evaluation_metrics.py -q
```

Expected: FAIL because current `max_drawdown` is gross-return based.

- [x] **Step 3: Add explicit tradable outcome input**

Update `evaluate_portfolio_layer(...)` to accept:

```python
tradable_outcome: dict | None = None
```

When provided:
- set `portfolio_return_mean` from tradable gross mean if available;
- set `net_mean_return` from `tradable_outcome["mean_net_return"]`;
- set `max_drawdown` from `tradable_outcome["max_net_drawdown"]`;
- keep gross drawdown as `gross_max_drawdown`;
- keep existing order/cost diagnostics as diagnostics only.

- [x] **Step 4: Pass the artifact from replay**

In `run_portfolio_replay()`, pass `tradable_outcome` into `evaluate_portfolio_layer(...)`.

- [x] **Step 5: Run evaluation tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_evaluation_metrics.py -q
```

Expected: PASS.

---

## Task 5: Split Promotion Gate

**Files:**
- Modify: `skyeye/products/ax1/promotion.py`
- Test: `tests/products/ax1/test_promotion.py`

- [x] **Step 1: Write failing tests for gate precedence**

Add tests:
- research metrics pass but tradability fails -> top-level gate fails;
- tradability passes but Rank IC soft check fails -> top-level gate can pass with warning.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_promotion.py -q
```

Expected: FAIL because `tradability_gate` / `research_support_gate` are absent.

- [x] **Step 3: Refactor gate output shape**

Keep backward-compatible top-level fields:
- `passed`;
- `failed_checks`;
- `checks`;
- `metrics`;
- `thresholds`.

Add:
- `tradability_gate`;
- `research_support_gate`;
- `warning_checks`.

Hard tradability checks include:
- min OOS rows;
- opportunity benchmark available;
- min excess net mean return;
- max net drawdown;
- max excess drawdown;
- max rolling underperformance;
- max mean turnover;
- min active day ratio where configured as hard.

Research support checks include:
- Rank IC;
- top bucket spread;
- rank significance;
- bootstrap CI;
- effective breadth;
- feature diagnostics;
- parameter validation.

- [x] **Step 4: Run promotion tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_promotion.py -q
```

Expected: PASS.

---

## Task 6: Confidence Calibration Uses Tradable Net Success

**Files:**
- Modify: `skyeye/products/ax1/confidence.py`
- Modify: `skyeye/products/ax1/research/training.py`
- Test: `tests/products/ax1/test_calibration.py`

- [x] **Step 1: Write failing test for confidence semantics**

Add a test where relative label is positive but tradable net return is negative after costs. The calibrated confidence bucket must treat it as failure when `outcome_column="tradable_net_success"`.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_calibration.py -q
```

Expected: FAIL because current confidence calibration only labels hit-rate from selected label column.

- [x] **Step 3: Add explicit outcome column support**

Update `fit_confidence_calibrator(...)` so callers can pass an explicit binary or numeric outcome column:

```python
outcome_column: str | None = None
```

If `outcome_column` is present:
- success means `outcome_column > 0`;
- summary records `outcome_column`;
- fallback summary states why outcome column was missing.

- [x] **Step 4: Wire pipeline only after tradable artifact is available**

Do not make this block training if tradable outcome is unavailable during fold-level training. For first implementation:
- keep fold-level `confidence_raw`;
- add aggregate post-replay confidence diagnostic based on tradable net outcomes;
- do not change model predictions used by allocation in this phase.

- [ ] **Step 5: Run confidence tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_calibration.py tests/products/ax1/test_config_pipeline.py -q
```

Expected: PASS.

---

## Task 7: Package Roundtrip And Status

**Files:**
- Modify: `skyeye/products/ax1/package_io.py`
- Modify: `skyeye/products/ax1/config.py`
- Test: `tests/products/ax1/test_persistence_package.py`
- Test: `tests/products/ax1/test_config_pipeline.py`

- [x] **Step 1: Write failing package roundtrip assertions**

Assert saved experiment/package preserves:
- `tradable_outcome`;
- `alpha_transfer_ledger`;
- component manifest status for tradability and alpha transfer.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_persistence_package.py -q
```

Expected: FAIL on missing fields/status.

- [x] **Step 3: Update package and manifest plumbing**

Ensure package save/load preserves new sections without converting date-keyed maps into non-deterministic forms. Add implementation statuses:
- `tradable_outcome: implemented_net_equity_contract`;
- `alpha_transfer_ledger: implemented_alpha_retention_attribution`;
- `promotion_gate: tradability_first_research_support_second`.

- [x] **Step 4: Run package tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_persistence_package.py tests/products/ax1/test_config_pipeline.py -q
```

Expected: PASS.

---

## Task 8: Final Verification

**Files:**
- No new implementation files.
- Update this checklist with exact verification output before marking complete.

- [ ] **Step 1: Run focused tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/products/ax1/test_tradability_contract.py \
  tests/products/ax1/test_evaluation_metrics.py \
  tests/products/ax1/test_promotion.py \
  tests/products/ax1/test_calibration.py \
  tests/products/ax1/test_config_pipeline.py \
  tests/products/ax1/test_persistence_package.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run full AX1 suite**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1 -q
```

Expected: PASS.

- [ ] **Step 3: Run py_compile**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m py_compile \
  skyeye/products/ax1/tradability.py \
  skyeye/products/ax1/evaluation/metrics.py \
  skyeye/products/ax1/research/execution.py \
  skyeye/products/ax1/run_experiment.py \
  skyeye/products/ax1/confidence.py \
  skyeye/products/ax1/promotion.py \
  skyeye/products/ax1/package_io.py \
  skyeye/products/ax1/config.py
```

Expected: exit code 0.

- [ ] **Step 4: Run CLI smoke**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m skyeye.products.ax1.run_experiment --help
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m skyeye.products.ax1.promote_package --help
```

Expected: both commands return 0 and print help text.

---

## Self-Review

- [ ] Spec coverage: pain point 2 is covered by `TradableOutcome`, net-equity drawdown, promotion split, and confidence semantics.
- [ ] Pain point 1 is covered by `AlphaTransferLedger`; no parameters are changed.
- [ ] No new docs are created; this plan lives in `tasks/todo.md` per repo rule and user confirmation.
- [ ] The plan intentionally avoids tuning thresholds or profile defaults.
- [ ] Implementation remains gated until explicit user confirmation.

## Review Log

- Pending implementation.

---

# AX1 Unified Data Access And Train CLI Plan

> HARD-GATE: 本节是修订后的轻量 Spec + implementation checklist。用户确认前只允许调研和更新本计划，不允许修改实现代码。

## Goal

解决两个问题：

1. AX1 训练数据获取必须接入统一 `skyeye.data.DataFacade`，不再要求用户手工准备 raw CSV。
2. AX1 提供简单训练入口：

```bash
python -m skyeye.products.ax1.train --start-date ... --end-date ... --profile personal_etf_core --run-tag ...
```

## Current Facts

- `skyeye.data.DataFacade` 已经是统一数据门面，串联 bundle、SQLite cache、rqdatac。
- TX1 的 `skyeye/products/tx1/run_baseline_experiment.py::build_raw_df()` 已经通过 `DataFacade` 构造训练面板。
- AX1 当前 `skyeye/products/ax1/run_experiment.py` 只暴露 `--raw-csv` CLI；`raw_df/raw_csv` 应降级为内部/测试接口。
- AX1 当前 `PriceVolumeDataSource` 仍要求调用方提供 raw_df，不是完整历史训练数据 builder。
- AX1 `run_experiment()` 在 `promotable_training` 下会把 `data_provider` 传给 PIT universe audit；新 CLI 必须把同一个 `DataFacade` 同时传给 data builder 和 `run_experiment(data_provider=...)`。
- AX1 `save_experiment()` 把 `output_dir` 当最终 experiment 目录，不会自动拼 `experiment_name`；新 CLI 必须显式使用 `output_root / run_tag`，避免不同 run 覆盖同一目录。

## Design

新增 AX1 自己的轻量训练数据 builder，复用统一 `DataFacade`：

`profile -> resolve AX1 universe ids -> DataFacade.get_daily_bars(adjust_type="pre") -> normalize AX1 raw panel -> attach metadata/source diagnostics -> run_experiment(raw_df=..., data_provider=same_facade) -> save artifacts under output_root/run_tag`

边界：

- 不复制 TX1 的股票 universe 逻辑；AX1 按 ETF-first profile 的 `universe.layers` 构建候选池。
- 不把 CSV 作为主入口；CSV 仅保留给测试、debug、离线导入。
- 不引入新的说明文档；计划和结果继续写在 `tasks/todo.md`。
- 不修改 `DataFacade` 的核心语义，除非发现 AX1 必需字段无法通过现有接口取得。
- 不伪造 ETF 行业分类：builder 优先使用 provider 返回的行业分类；拿不到时填 `Unknown`，让现有 ETF 约束路径通过 `exposure_group` 控制风险。
- `adjusted_close = close` 只在请求 `adjust_type="pre"` 后成立，并需要写入 `price_adjustment_status="pre_adjusted_via_data_facade"`，避免静默伪造复权契约。

## Files

- Create: `skyeye/products/ax1/data_builder.py`
- Create: `skyeye/products/ax1/train.py`
- Modify: `skyeye/products/ax1/run_experiment.py`
- Test: `tests/products/ax1/test_train_entrypoint.py`
- Test: `tests/products/ax1/test_config_pipeline.py`

## Task 1: AX1 Training Data Builder

- [x] **Step 1: Write failing builder tests**

Add tests in `tests/products/ax1/test_train_entrypoint.py` covering:

- profile universe layers are flattened into ETF-first ids;
- builder calls `DataFacade.get_daily_bars(..., adjust_type="pre")`;
- output frame has AX1-required columns: `date`, `order_book_id`, `close`, `adjusted_close`, `volume`, `asset_type`, `universe_layer`, `exposure_group`, `industry`, `listed_date`, `is_st`, `is_suspended`, `price_adjustment_status`;
- output uses `industry="Unknown"` when provider industry is unavailable, not layer name masquerading as industry;
- output uses `price_adjustment_status="pre_adjusted_via_data_facade"` when `adjusted_close` is copied from pre-adjusted `close`;
- no raw CSV is required.

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_train_entrypoint.py -q
```

Expected: FAIL because `skyeye.products.ax1.data_builder` does not exist.

- [x] **Step 2: Implement `AX1TrainingDataBuilder`**

Create `skyeye/products/ax1/data_builder.py` with:

- `AX1TrainingDataRequest(profile_config, start_date, end_date, data_source="auto")`;
- `AX1TrainingDataBuilder(data_facade=None).build(request) -> pd.DataFrame`;
- helper to flatten enabled `universe.layers[*].include`;
- helper to normalize `DataFacade.get_daily_bars` output into long panel;
- metadata attachment from profile layer definitions: `asset_type`, `universe_layer`, `exposure_group`;
- optional metadata attachment from `data_facade.all_instruments(type="ETF", date=end_date)` when available: `listed_date`;
- optional provider-backed flags via `data_facade.is_st_stock()` / `data_facade.is_suspended()` when available; otherwise default `False` for ETF rows and let existing PIT audit use `data_provider`;
- `industry` from provider only when available; otherwise `Unknown`;
- `adjusted_close` copied from `close` only when the builder requested `adjust_type="pre"` and no explicit adjusted column exists;
- `price_adjustment_status` set to `pre_adjusted_via_data_facade` for that copied adjusted close path.

- [x] **Step 3: Run builder tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_train_entrypoint.py -q
```

Expected: PASS for data builder tests.

## Task 2: AX1 Train CLI

- [x] **Step 1: Write failing CLI tests**

In `tests/products/ax1/test_train_entrypoint.py`, add tests for:

- `build_parser()` accepts `--start-date`, `--end-date`, `--profile`, `--run-tag`, `--output-root`;
- default profile alias `personal_etf_core` resolves to `skyeye/products/ax1/profiles/personal_etf_core.yaml`;
- `main()` creates one `DataFacade`, builds data through `AX1TrainingDataBuilder(data_facade=same_facade)`, and calls `run_experiment(raw_df=..., data_provider=same_facade)`;
- `main()` passes `output_dir=Path(output_root) / run_tag`, not just `output_root`;
- CLI prints the saved experiment path.

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_train_entrypoint.py -q
```

Expected: FAIL because `skyeye.products.ax1.train` does not exist.

- [x] **Step 2: Implement `skyeye.products.ax1.train`**

Create `skyeye/products/ax1/train.py` with a small CLI:

```bash
python -m skyeye.products.ax1.train \
  --start-date 2018-01-01 \
  --end-date 2026-04-30 \
  --profile personal_etf_core \
  --run-tag ax1_20260501
```

Behavior:

- resolve profile alias or explicit path;
- load and normalize AX1 profile;
- instantiate exactly one `DataFacade`;
- build raw panel via `AX1TrainingDataBuilder(data_facade=facade)`;
- call `run_experiment(profile_path=..., raw_df=..., output_dir=Path(output_root) / run_tag, experiment_name=run_tag, data_provider=facade)`;
- print final `output_dir`;
- return non-zero on data/build/training errors via normal exception propagation.

- [x] **Step 3: Keep `run_experiment` as pipeline API**

Modify `skyeye/products/ax1/run_experiment.py` only if needed to support cleaner internal invocation. The existing `--raw-csv` path remains available for debug, but the new user-facing training path is `train.py`.

- [x] **Step 4: Run CLI tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_train_entrypoint.py -q
```

Expected: PASS.

## Task 3: End-To-End Contract Guard

- [x] **Step 1: Write focused integration test with fake DataFacade**

In `tests/products/ax1/test_train_entrypoint.py`, add a fake facade that implements:

- `get_daily_bars(order_book_ids, start_date, end_date, fields, adjust_type)`;
- `all_instruments(type=None, date=None)`;
- `is_st_stock(order_book_ids, start_date, end_date)`;
- `is_suspended(order_book_ids, start_date, end_date)`.

Patch `skyeye.products.ax1.train.run_experiment` with a fake that records inputs and returns `{"output_dir": ".../run_tag"}`. Assert:

- no raw CSV path is required;
- fake facade instance identity is the same in builder and `run_experiment`;
- generated raw_df includes all required AX1 fields;
- `output_dir` ends with the run tag.

- [x] **Step 2: Run integration test**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider tests/products/ax1/test_train_entrypoint.py -q
```

Expected: PASS.

## Task 4: Verification And Entry Contract

- [x] **Step 1: Run focused AX1 pipeline tests**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m pytest -p no:cacheprovider \
  tests/products/ax1/test_train_entrypoint.py \
  tests/products/ax1/test_config_pipeline.py \
  tests/products/ax1/test_data_quality.py \
  tests/products/ax1/test_universe_liquidity.py \
  -q
```

Expected: PASS.

Actual:

```text
64 passed in 87.60s (0:01:27)
```

- [x] **Step 2: Run py_compile**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m py_compile \
  skyeye/products/ax1/data_builder.py \
  skyeye/products/ax1/train.py \
  skyeye/products/ax1/run_experiment.py
```

Expected: exit code 0.

Actual: exit code 0.

- [x] **Step 3: Run CLI smoke**

Run:

```bash
env PYTHONPATH=/home/tiger/rqalpha PYTHONDONTWRITEBYTECODE=1 /home/tiger/miniconda3/bin/python -m skyeye.products.ax1.train --help
```

Expected: exit code 0 and help contains `--start-date`, `--end-date`, `--profile`, `--run-tag`.

Actual: exit code 0; help contains `--start-date`, `--end-date`, `--profile`, `--run-tag`, and `--output-root`.

- [x] **Step 4: Optional tiny dry-run smoke with monkeypatched/fake data only**

Do not hit online quota in default verification. If needed, run only the fake-data pytest path above. A real data run requires explicit user approval because it may touch bundle/rqdatac and take time.

- [x] **Step 5: Record result**

Append exact verification output summary to this section before marking complete.

## Open Decisions

- 默认 `--output-root`：采用 `skyeye/artifacts/experiments/ax1_train`。
- 默认数据源：不新增 CLI 参数，沿用 `DataFacade` 的 `SKYEYE_DATA_SOURCE` / `RQALPHA_BUNDLE_PATH` / `SKYEYE_BUNDLE_PATH` 环境变量。
- profile alias：支持 `default`、`personal_etf_core`、`lgbm_multi_target` 三个 alias，同时允许传文件路径。
- real data smoke：本轮默认不跑真实在线数据，只跑 fake facade 和 CLI help；需要真实训练时再由用户确认命令和日期范围。

## Review Log

- Revised after design review:
  - added same-`DataFacade` requirement for builder and `run_experiment(data_provider=...)`;
  - fixed `output_root / run_tag` artifact directory contract;
  - prohibited fake industry labels from layer names;
  - made pre-adjusted price contract explicit with `price_adjustment_status`;
  - added fake-facade integration guard and avoided online data use in default verification.
- Implementation complete:
  - added `skyeye/products/ax1/data_builder.py`;
  - added `skyeye/products/ax1/train.py`;
  - added `tests/products/ax1/test_train_entrypoint.py`;
  - verified focused AX1 tests, py_compile, and CLI help smoke.

---

# AX1 训练执行记录（2026-05-01）

## 目标

- 启动一轮真实 AX1 训练，使用 `personal_etf_core` profile。
- 持续观察训练日志，确认没有异常报错、异常停滞或明显异常数据。
- 训练完成后，抽取产物里的最近 3 年区间表现，供用户核对目标跑分区间 `23.430 ~ 26.430`。

## 执行计划

- [x] 确认训练命令、输出目录、run tag，并避免覆盖现有产物。
- [x] 启动真实训练进程，stdout/stderr 全量落盘。
- [x] 轮询日志与进程状态，记录关键阶段与异常检查结论。
- [x] 训练完成后读取 experiment 产物，提取总结果与最近 3 年指标。
- [x] 在本节追加 review 结论与关键证据路径。

## 运行参数

- Python: `/home/tiger/miniconda3/bin/python`
- Module: `skyeye.products.ax1.train`
- Start date: `2018-01-01`
- End date: `2026-04-30`
- Profile: `personal_etf_core`
- Output root: `tmp/ax1_train_runs`
- Run tag: `ax1_20260501_full_20180101_20260430`

## Review

- 训练已完成（ExitCode=0）。
- 关键证据：
  - 日志：`/home/tiger/rqalpha/tmp/ax1_train_logs/ax1_20260501_full_20180101_20260430.log`
  - 产物：`/home/tiger/rqalpha/tmp/ax1_train_runs/ax1_20260501_full_20180101_20260430_mt128_nocap2/experiment.json`
- 根因与修复：capacity 约束在部分 ETF 缺少流动性字段时触发崩溃（`skyeye/products/ax1/execution/smoother.py:308`），已改为缺失 liquidity 时跳过 capacity cap，且 profile 禁用 `execution.participation_rate`。
- 结果摘要：experiment `status=constraint_warning`、`constraint_status=hard_violation`；本次 walk-forward OOS 覆盖区间约为 `2025-09-04 ~ 2026-02-27`，因此无法从该产物提取“最近 3 年”区间指标。

---

# AX1 Fold 2 风险溢价翻转根因排查（2026-05-01）

## 目标

- 只读分析训练产物 `/home/tiger/rqalpha/tmp/ax1_train_runs/ax1_20260501_full_20180101_20260430_mt128_nocap2/experiment.json`。
- 复核 Fold 2 在 `2025-11-06 ~ 2025-12-05` 的 RankIC 反向、risk/style 分层表现和切换点。
- 追踪 `feature_risk_forecast`、`feature_regime_*`、style 特征、训练目标与组合执行链路，限定真实根因。

## 执行计划

- [x] 读取 `fold_002` metadata、model bundle、predictions/orders/weights 和 experiment summary。
- [x] 复算 daily RankIC、`pred_vs_risk`、`label_vs_risk`、分层 IC/spread 和风险分位表现。
- [x] 检查训练窗内风险溢价符号、测试窗中 11 月中下旬切换点，以及 Fold 2 是否暴露在分布外状态。
- [x] 追踪当前 AX1 代码里 risk/regime/style 特征如何生成、进入模型、进入组合。
- [x] 输出根因判断、非根因排除项、以及后续修复方向；本节只读，不改业务代码。

## Review

- 复现 Fold 2：daily RankIC 均值 `-0.150946`；`pred_vs_risk` 均值 `-0.616101`；`label_vs_risk` 均值 `+0.118095`。
- 切换点：`2025-11-13` 后 RankIC 均值 `-0.2839`，`label_vs_risk` 均值 `+0.2763`；`2025-11-19 ~ 2025-11-25` RankIC 均值 `-0.5858`，`label_vs_risk` 均值 `+0.4744`。
- 分层：style daily IC 均值约 `-0.2688`，industry 约 `-0.1226`，core 约 `-0.0091`。
- 执行层不是首因：预测层已经反向；最终持仓只是跟随高分低风险暴露，选中资产风险分位加权均值约 `0.33`，style 组约 `0.20`。
- 根因：`feature_regime_*` 是按日期广播的状态特征，但 AX1 预处理对所有特征做按日横截面标准化；日级常数被压成全 0。Fold 2 bundle 中所有 return/risk heads 的 regime feature_infos 均为 `none`，importance 均为 0。
- 结论：Fold 2 暴露的是 AX1 当前 LGBM 主链路缺少有效 regime-conditioned risk/style premium 翻转能力；不是单纯随机噪音，也不是某几只 ETF 带崩。

---

# AX1 Regime-aware 预处理与交互特征修复计划（2026-05-01）

## 目标

- 修复 `feature_regime_*` 被按日横截面标准化压成全 0 的主链路问题。
- 把不同语义的特征从“全部同一预处理路径”改成显式 `preprocess_policy` contract。
- 让默认 LGBM 主链路具备 risk/style 与 regime 的交互输入，避免只学习静态风险/风格溢价方向。

## 架构

- `FeatureView.metadata["preprocess_policies"]` 记录每个 feature 的预处理策略。
- 默认策略为 `cross_sectional`，保持现有 winsorize + neutralize + per-date z-score。
- `feature_regime_*` 策略为 `passthrough`，不做横截面标准化。
- `feature_interaction_*` 策略为 `cross_sectional`，由 `regime_interaction` scope 提供，并进入默认 LGBM feature scopes。

## 文件范围

- Modify: `skyeye/products/ax1/features/view.py`
- Modify: `skyeye/products/ax1/preprocessing/preprocessor.py`
- Modify: `skyeye/products/ax1/research/training.py`
- Modify: `skyeye/products/ax1/config.py`
- Modify: `skyeye/products/ax1/profiles/default.yaml`
- Modify: `skyeye/products/ax1/profiles/lgbm_multi_target.yaml`
- Modify: `skyeye/products/ax1/profiles/personal_etf_core.yaml`
- Test: `tests/products/ax1/test_preprocessor.py`
- Test: `tests/products/ax1/test_unified_features.py`
- Test: `tests/products/ax1/test_lgbm_config.py`
- Test: `tests/products/ax1/test_config_pipeline.py`

## 执行计划

- [x] 写失败测试：`FeaturePreprocessor.transform(..., preprocess_policies={"feature_regime_risk_on": "passthrough"})` 保留日期级 regime 值，同时普通特征仍按日标准化。
- [x] 写失败测试：`FeatureView.metadata["preprocess_policies"]` 标记 regime 为 `passthrough`、普通/interaction 特征为 `cross_sectional`。
- [x] 写失败测试：训练 fold 的 `preprocessor_bundle` 持久化 `preprocess_policies`，且默认 feature columns 包含 regime interaction。
- [x] 实现 AX1 preprocessor policy-aware transform/to_bundle/from_bundle。
- [x] 实现 FeatureView policy metadata，并在 training pipeline 传入 transform/to_bundle。
- [x] 更新默认 config/profile scope：`[common, etf_zscore, regime, regime_interaction]`。
- [x] 跑定向测试：`test_preprocessor.py`、`test_unified_features.py`、`test_lgbm_config.py`、`test_config_pipeline.py` 相关用例。
- [x] 跑 AX1 全量测试或记录阻塞原因。

## Review

- RED 证据：新增 3 个测试初次运行失败，分别覆盖 `transform()` 不接受 `preprocess_policies`、FeatureView 缺少 policy metadata、默认 config 未启用 `regime_interaction`。
- 实现结果：`FeatureView.metadata["preprocess_policies"]` 记录 feature 级策略；regime scope 为 `passthrough`，其他 feature scope 默认 `cross_sectional`。
- 实现结果：AX1 `FeaturePreprocessor` 支持 policy-aware `transform()`、`to_bundle()`、`from_bundle()`；训练 pipeline 将 policy 传入 train/val/test/diagnostic transform，并写入 fold `preprocessor_bundle`。
- 实现结果：默认 config 和 3 个 AX1 profile 的 LGBM `include_scopes` 已切到 `[common, etf_zscore, regime, regime_interaction]`。
- 验证：
  - RED focused tests -> 初次 `3 failed`，原因符合预期。
  - `tests/products/ax1/test_preprocessor.py tests/products/ax1/test_unified_features.py tests/products/ax1/test_lgbm_config.py -q` -> `49 passed in 3.69s`。
  - `test_default_profile_preserves_ax1_architecture_decisions` / `test_personal_etf_core_profile_declares_etf_first_universe_and_costs` / `test_run_experiment_writes_lgbm_pipeline_contract` / `test_run_experiment_default_lgbm_path_produces_non_constant_predictions` -> `4 passed in 34.22s`。
  - `tests/products/ax1 -q` -> `412 passed, 1 warning in 109.70s`。
  - `py_compile` touched AX1 modules -> exit code 0。
  - `python -m skyeye.products.ax1.train --help` -> exit code 0。
  - `python -m skyeye.products.ax1.run_experiment --help` -> exit code 0。
