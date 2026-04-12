# TX1 EP Ratio Continuous Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal TX1 candidate line `baseline_5f_ep` / `combo_b25_h45_ep` so the team can test whether adding `ep_ratio_ttm` stably beats the current `baseline_5f` and `combo_b25_h45` under unchanged portfolio-layer assumptions.

**Architecture:** Keep the executable strategy adapter unchanged and make the experiment pipeline configurable instead. Add a new research variant to `run_feature_experiment.py`, then thread an explicit feature-set override through `run_baseline_experiment.py` into `ExperimentRunner` so the training pipeline can emit a replayable candidate artifact line without promoting `ep_ratio_ttm` into the global default feature alias.

**Tech Stack:** Python, pandas, LightGBM, pytest, RQAlpha, SkyEye TX1 research pipeline

---

## File Structure

- Modify: `skyeye/products/tx1/run_feature_experiment.py`
  - Add a minimal research-only variant named `baseline_5f_ep`
- Modify: `skyeye/products/tx1/experiment_runner.py`
  - Read optional `config["features"]["columns"]` instead of always using the shared default `FEATURE_COLUMNS`
- Modify: `skyeye/products/tx1/run_baseline_experiment.py`
  - Add a narrow feature-set registry and CLI/config plumbing for `baseline_5f_ep`
- Modify: `tests/products/tx1/test_run_feature_experiment.py`
  - Verify the new variant definition and smoke-run it
- Modify: `tests/products/tx1/test_run_baseline_experiment.py`
  - Verify feature-set resolution and config construction for the candidate line
- Modify: `tests/products/tx1/test_main.py`
  - Verify `ExperimentRunner` really trains on configured feature columns

## Task 1: Add the `baseline_5f_ep` Research Variant

**Files:**
- Modify: `skyeye/products/tx1/run_feature_experiment.py:88-180`
- Modify: `tests/products/tx1/test_run_feature_experiment.py:32-139`
- Test: `tests/products/tx1/test_run_feature_experiment.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/products/tx1/test_run_feature_experiment.py`:

```python
def test_build_variants_contains_ep_candidate():
    variants = build_variants()
    names = [v["name"] for v in variants]

    assert "baseline_5f_ep" in names


def test_baseline_5f_ep_variant_uses_expected_features():
    variants = build_variants()
    baseline_5f_ep = next(v for v in variants if v["name"] == "baseline_5f_ep")

    assert baseline_5f_ep["features"] == [*BASELINE_5F_COLUMNS, "ep_ratio_ttm"]


def test_run_ep_variant_smoke(tmp_path, make_raw_panel):
    raw_df = make_raw_panel(periods=1400, extended=True)

    result = run_feature_experiments(
        raw_df=raw_df,
        output_dir=tmp_path,
        variant_names=["baseline_5f", "baseline_5f_ep"],
        model_kind="linear",
        max_folds=1,
    )

    by_name = {variant["name"]: variant for variant in result["variants"]}

    assert by_name["baseline_5f"]["n_folds"] == 1
    assert by_name["baseline_5f_ep"]["n_folds"] == 1
    assert by_name["baseline_5f_ep"]["features"] == [*BASELINE_5F_COLUMNS, "ep_ratio_ttm"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH="$PWD" pytest -q \
  tests/products/tx1/test_run_feature_experiment.py::test_build_variants_contains_ep_candidate \
  tests/products/tx1/test_run_feature_experiment.py::test_baseline_5f_ep_variant_uses_expected_features \
  tests/products/tx1/test_run_feature_experiment.py::test_run_ep_variant_smoke
```

Expected: FAIL because `baseline_5f_ep` is not yet present in `build_variants()`.

- [ ] **Step 3: Add the minimal production variant**

Insert this variant immediately after the existing `baseline_5f` entry in `skyeye/products/tx1/run_feature_experiment.py`:

```python
        {
            "name": "baseline_5f_ep",
            "purpose": "Current 5-factor TX1 baseline plus continuous EP valuation.",
            "features": _dedupe(list(BASELINE_5F_COLUMNS) + ["ep_ratio_ttm"]),
            "preprocess": None,
        },
```

Keep `baseline_name = "baseline_5f"` unchanged so all candidate deltas still compare back to the current baseline.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH="$PWD" pytest -q tests/products/tx1/test_run_feature_experiment.py
```

Expected: PASS, including the new `baseline_5f_ep` variant tests and the existing smoke coverage.

- [ ] **Step 5: Commit the research-variant change**

```bash
git add tests/products/tx1/test_run_feature_experiment.py skyeye/products/tx1/run_feature_experiment.py
git commit -m "feat(tx1): add ep ratio feature experiment variant"
```

## Task 2: Thread Feature-Set Overrides Through the Training Pipeline

**Files:**
- Modify: `skyeye/products/tx1/experiment_runner.py:6-11`
- Modify: `skyeye/products/tx1/experiment_runner.py:64-118`
- Modify: `skyeye/products/tx1/run_baseline_experiment.py:490-589`
- Modify: `skyeye/products/tx1/run_baseline_experiment.py:624-667`
- Modify: `tests/products/tx1/test_run_baseline_experiment.py:1-63`
- Modify: `tests/products/tx1/test_main.py:1-52`
- Test: `tests/products/tx1/test_run_baseline_experiment.py`
- Test: `tests/products/tx1/test_main.py`

- [ ] **Step 1: Write the failing tests for feature-set plumbing**

Extend `tests/products/tx1/test_run_baseline_experiment.py` with these tests:

```python
from skyeye.products.tx1.evaluator import BASELINE_5F_COLUMNS


def test_resolve_feature_set_returns_ep_candidate():
    feature_set_name, feature_columns = baseline_runner.resolve_feature_set("baseline_5f_ep")

    assert feature_set_name == "baseline_5f_ep"
    assert feature_columns == [*BASELINE_5F_COLUMNS, "ep_ratio_ttm"]


def test_build_experiment_config_stores_feature_metadata():
    config = baseline_runner.build_experiment_config(
        "lgbm",
        experiment_name="tx1_combo_b25_h45_ep",
        feature_set_name="baseline_5f_ep",
        feature_columns=[*BASELINE_5F_COLUMNS, "ep_ratio_ttm"],
    )

    assert config["experiment_name"] == "tx1_combo_b25_h45_ep"
    assert config["features"]["name"] == "baseline_5f_ep"
    assert config["features"]["columns"] == [*BASELINE_5F_COLUMNS, "ep_ratio_ttm"]
```

Extend `tests/products/tx1/test_main.py` with this test:

```python
import numpy as np

from skyeye.products.tx1 import experiment_runner as runner_module
from skyeye.products.tx1.main import main


def test_main_uses_configured_feature_columns(monkeypatch, make_raw_panel):
    raw_df = make_raw_panel(periods=2200, extended=True)
    captured = {}

    class FakeModel:
        def fit(self, train_X, train_y):
            captured["feature_columns"] = list(train_X.columns)
            return self

        def predict(self, test_X):
            return np.zeros(len(test_X), dtype=float)

    monkeypatch.setattr(runner_module, "create_model", lambda kind, params=None: FakeModel())
    monkeypatch.setattr(runner_module, "supports_validation", lambda model: False)

    result = main(
        {
            "model": {"kind": "linear"},
            "features": {"columns": ["mom_40d", "ep_ratio_ttm"]},
        },
        raw_df=raw_df,
    )

    assert result["fold_results"]
    assert captured["feature_columns"] == ["mom_40d", "ep_ratio_ttm"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
PYTHONPATH="$PWD" pytest -q \
  tests/products/tx1/test_run_baseline_experiment.py::test_resolve_feature_set_returns_ep_candidate \
  tests/products/tx1/test_run_baseline_experiment.py::test_build_experiment_config_stores_feature_metadata \
  tests/products/tx1/test_main.py::test_main_uses_configured_feature_columns
```

Expected: FAIL because `run_baseline_experiment.py` does not yet expose `resolve_feature_set`, `build_experiment_config()` does not persist feature metadata, and `ExperimentRunner` still always uses `FEATURE_COLUMNS`.

- [ ] **Step 3: Implement the feature-set override path**

Add a narrow registry and resolver near `build_experiment_config()` in `skyeye/products/tx1/run_baseline_experiment.py`:

```python
from skyeye.products.tx1.evaluator import BASELINE_5F_COLUMNS


FEATURE_SET_REGISTRY = {
    "baseline_5f": list(BASELINE_5F_COLUMNS),
    "baseline_5f_ep": [*BASELINE_5F_COLUMNS, "ep_ratio_ttm"],
}


def resolve_feature_set(feature_set_name: str | None = None) -> tuple[str, list[str]]:
    resolved_name = feature_set_name or "baseline_5f"
    try:
        return resolved_name, list(FEATURE_SET_REGISTRY[resolved_name])
    except KeyError as exc:
        raise ValueError("unknown feature set: {}".format(resolved_name)) from exc
```

Extend `build_experiment_config()` to store explicit feature metadata:

```python
def build_experiment_config(
    model_kind: str,
    *,
    experiment_name: str | None = None,
    feature_set_name: str = "baseline_5f",
    feature_columns: list[str] | None = None,
    multi_output_enabled: bool = False,
    volatility_weight: float = 0.0,
    max_drawdown_weight: float = 0.0,
    enable_reliability_score: bool = False,
    holding_bonus: float = 0.5,
    rebalance_interval: int = 20,
) -> dict:
    multi_output_enabled = bool(
        multi_output_enabled
        or volatility_weight > 0
        or max_drawdown_weight > 0
        or enable_reliability_score
    )
    config = {
        "model": {"kind": model_kind},
        "labels": {"transform": "rank"},
        "features": {
            "name": feature_set_name,
            "columns": list(feature_columns or BASELINE_5F_COLUMNS),
        },
        "robustness": {"enabled": True, "stability_metric": "rank_ic_mean"},
        "costs": {
            "enabled": True,
            "commission_rate": 0.0008,
            "stamp_tax_rate": 0.0005,
            "slippage_bps": 5.0,
        },
        "portfolio": {
            "rebalance_interval": int(rebalance_interval),
            "holding_bonus": float(holding_bonus),
        },
    }
    if experiment_name:
        config["experiment_name"] = experiment_name
    if multi_output_enabled:
        config["multi_output"] = {
            "enabled": True,
            "volatility": {"enabled": True, "transform": "rank"},
            "max_drawdown": {"enabled": True, "transform": "rank"},
            "prediction": {
                "combine_auxiliary": True,
                "volatility_weight": float(volatility_weight),
                "max_drawdown_weight": float(max_drawdown_weight),
            },
            "reliability_score": {"enabled": bool(enable_reliability_score)},
        }
    return config
```

Update `run_experiment()` and `main()` to accept `feature_set_name` and wire it into the config:

```python
def run_experiment(
    model_kind: str,
    output_base: str,
    universe_size: int = UNIVERSE_SIZE,
    *,
    experiment_name: str | None = None,
    feature_set_name: str | None = None,
    market_cap_floor_quantile: float | None = None,
    market_cap_column: str | None = None,
    multi_output_enabled: bool = False,
    volatility_weight: float = 0.0,
    max_drawdown_weight: float = 0.0,
    enable_reliability_score: bool = False,
    holding_bonus: float = 0.5,
    rebalance_interval: int = 20,
) -> dict:
    from skyeye.products.tx1.main import main as tx1_main

    output_dir_name = _resolve_output_dir_name(experiment_name, model_kind)
    output_dir = str(Path(output_base) / output_dir_name)
    resolved_feature_set_name, feature_columns = resolve_feature_set(feature_set_name)
    config = build_experiment_config(
        model_kind,
        experiment_name=output_dir_name,
        feature_set_name=resolved_feature_set_name,
        feature_columns=feature_columns,
        multi_output_enabled=multi_output_enabled,
        volatility_weight=volatility_weight,
        max_drawdown_weight=max_drawdown_weight,
        enable_reliability_score=enable_reliability_score,
        holding_bonus=holding_bonus,
        rebalance_interval=rebalance_interval,
    )
    universe = get_liquid_universe(
        universe_size,
        market_cap_floor_quantile=market_cap_floor_quantile,
        market_cap_column=market_cap_column,
    )
    raw_df = build_raw_df(universe)
    print(
        "\\nRunning TX1 experiment: model={} experiment={} feature_set={} multi_output={} market_cap_floor={}".format(
            model_kind,
            output_dir_name,
            resolved_feature_set_name,
            bool(config.get("multi_output", {}).get("enabled", False)),
            market_cap_floor_quantile,
        )
    )
    result = tx1_main(config=config, raw_df=raw_df, output_dir=output_dir)
    return result
```

```python
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SET_REGISTRY),
        default="baseline_5f",
    )
```

```python
        result = run_experiment(
            model_kind,
            args.output_dir,
            universe_size=universe_size,
            experiment_name=experiment_name,
            feature_set_name=args.feature_set,
            market_cap_floor_quantile=args.market_cap_floor_quantile,
            market_cap_column=args.market_cap_column,
            multi_output_enabled=args.enable_multi_output,
            volatility_weight=args.volatility_weight,
            max_drawdown_weight=args.max_drawdown_weight,
            enable_reliability_score=args.enable_reliability_score,
            holding_bonus=args.holding_bonus,
            rebalance_interval=args.rebalance_interval,
        )
```

Update `skyeye/products/tx1/experiment_runner.py` so configured features override the shared default alias:

```python
    def _resolve_feature_columns(self, train_df):
        configured = self.config.get("features", {}).get("columns")
        requested = list(configured) if configured else list(FEATURE_COLUMNS)
        feature_cols = [column for column in requested if column in train_df.columns]
        if not feature_cols:
            raise ValueError(
                "no configured feature columns available in training data: {}".format(requested)
            )
        return feature_cols
```

```python
            feature_cols = self._resolve_feature_columns(train_df)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
PYTHONPATH="$PWD" pytest -q \
  tests/products/tx1/test_run_baseline_experiment.py \
  tests/products/tx1/test_main.py
```

Expected: PASS, including the new resolver/config tests and the `main()` assertion that the configured feature list is the exact model input.

- [ ] **Step 5: Commit the feature-override pipeline**

```bash
git add \
  tests/products/tx1/test_run_baseline_experiment.py \
  tests/products/tx1/test_main.py \
  skyeye/products/tx1/experiment_runner.py \
  skyeye/products/tx1/run_baseline_experiment.py
git commit -m "feat(tx1): allow feature-set overrides for baseline experiments"
```

## Verification Runbook

After Tasks 1 and 2 are implemented and committed, run this exact sequence to evaluate the candidate.

### 1. Research-Layer Comparison: `baseline_5f` vs `baseline_5f_ep`

- [ ] **Run the minimal feature experiment**

```bash
PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_feature_experiment \
  --variant baseline_5f baseline_5f_ep \
  --model-kind lgbm \
  --label-transform rank \
  --output-dir skyeye/artifacts/experiments/tx1_feature_ep_min
```

Expected: the output directory contains `feature_experiment_results.json`, `feature_experiment_report.txt`, and `variant_metrics.csv`, with rows for both `baseline_5f` and `baseline_5f_ep`.

- [ ] **Inspect the research-layer gates**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("skyeye/artifacts/experiments/tx1_feature_ep_min/feature_experiment_results.json")
data = json.loads(path.read_text())

for item in data["variants"]:
    if item["name"] in {"baseline_5f", "baseline_5f_ep"}:
        print(
            item["name"],
            {
                "rank_ic_mean": item["prediction"]["rank_ic_mean"],
                "net_mean_return": item["portfolio"]["net_mean_return"],
                "row_ratio": item["row_ratio"],
            },
        )
PY
```

Expected: one printed row for `baseline_5f` and one for `baseline_5f_ep`, ready for the first-pass gate check on `rank_ic_mean`, `net_mean_return`, and coverage.

### 2. Train the Candidate Executable Artifact Line

- [ ] **Run the unchanged portfolio-layer experiment with the candidate feature set**

```bash
PYTHONPATH="$PWD" python -m skyeye.products.tx1.run_baseline_experiment \
  --model lgbm \
  --output-dir skyeye/artifacts/experiments/tx1 \
  --experiment-name combo_b25_h45_ep \
  --feature-set baseline_5f_ep \
  --holding-bonus 0.5 \
  --rebalance-interval 20
```

Expected: a new experiment directory `skyeye/artifacts/experiments/tx1/tx1_combo_b25_h45_ep` with `experiment.json` and `folds/*/weights.parquet`.

### 3. Compare Executable-Layer `rolling-score`

- [ ] **Inspect candidate warnings and support metrics at the artifact level**

Run:

```bash
python -m skyeye.products.tx1.compare_experiments \
  combo_b25_h45_ep \
  --baseline combo_b25_h45 \
  --artifacts-root skyeye/artifacts/experiments/tx1
```

Expected: the report includes the candidate `rank_ic_mean`, `net_mean_return`, and whether new overfit flags such as `flag_spread_decay` or `flag_ic_decay` were introduced.

- [ ] **Run the baseline line**

```bash
env SKYEYE_TX1_ARTIFACT_LINE=combo_b25_h45 MPLCONFIGDIR=/tmp/mplconfig \
  python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py \
  > /tmp/tx1_combo_b25_h45_rolling_score.txt
```

Expected: `/tmp/tx1_combo_b25_h45_rolling_score.txt` ends with `成功完成 25/25 个窗口回测`.

- [ ] **Run the candidate line**

```bash
env SKYEYE_TX1_ARTIFACT_LINE=combo_b25_h45_ep MPLCONFIGDIR=/tmp/mplconfig \
  python -m skyeye.evaluation.rolling_score.cli \
  skyeye/products/tx1/strategies/rolling_score/strategy.py \
  > /tmp/tx1_combo_b25_h45_ep_rolling_score.txt
```

Expected: `/tmp/tx1_combo_b25_h45_ep_rolling_score.txt` ends with `成功完成 25/25 个窗口回测`.

- [ ] **Extract the primary score**

```bash
rg "策略综合得分" /tmp/tx1_combo_b25_h45_rolling_score.txt /tmp/tx1_combo_b25_h45_ep_rolling_score.txt
```

Expected: two lines, one baseline and one candidate, each containing the final `rolling-score`.

### 4. Assign the Final Verdict

- [ ] **Apply the agreed gates**

Use:

- `rolling-score` from the rolling-score reports
- `rank_ic_mean`, `net_mean_return`, `flag_spread_decay`, `flag_ic_decay` from the candidate artifact comparison report

Decision rules:

- `PASS`: candidate `rolling-score` is strictly higher, `rank_ic_mean` is not lower, `net_mean_return` is not lower, and there are no new `spread_decay` / `ic_decay` warnings
- `WATCHLIST`: candidate `rolling-score` is higher by at least `+2.0`, exactly one of `rank_ic_mean` or `net_mean_return` is slightly lower, and there are no new `spread_decay` / `ic_decay` warnings
- `REJECT`: any other outcome
