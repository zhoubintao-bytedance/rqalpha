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
