# TX1 `ep_ratio_ttm` Continuous Enhancement Design

## Goal

Evaluate whether adding the continuous fundamental factor `ep_ratio_ttm` to the current TX1 signal baseline can stably outperform the current `baseline_5f` and, after being replayed through the unchanged default portfolio layer, beat the default executable line `combo_b25_h45`.

The primary decision metric is `rolling-score`. Supporting metrics are `rank_ic_mean`, `net_mean_return`, and whether the candidate introduces new `spread_decay` or `ic_decay` warnings.

## Current Context

TX1 has already converged to a frozen out-of-sample replay architecture rather than online retraining at backtest time. The current default line is:

- Signal: `baseline_5f`
- Model: `lgbm`
- Label: `20d horizon + rank transform`
- Portfolio layer: `Top25 buy / Top45 hold / 20-trading-day rebalance`
- Executable artifact line: `tx1.rolling_score@combo_b25_h45`

The current research conclusion is that:

- The biggest completed improvement came from portfolio-layer tuning, not from adding many more factors.
- `turnover_stability_20d` is the most validated incremental OHLCV factor already promoted into the baseline.
- `ep_ratio_ttm` is the next most promising continuous fundamental factor, but it should be tested as a continuous signal enhancement rather than as a hard universe filter.

## Scope

This experiment is intentionally narrow.

### In Scope

- Add exactly one new candidate signal variant: `baseline_5f + ep_ratio_ttm`
- Keep the existing TX1 training and evaluation stack intact
- Compare research-layer performance against the current `baseline_5f`
- If research-layer results are not disqualifying, train a replayable artifact line with the unchanged default portfolio layer and compare full `rolling-score` against `combo_b25_h45`
- Produce a final verdict: `PASS`, `WATCHLIST`, or `REJECT`

### Out of Scope

- No changes to the live/executable strategy adapter
- No changes to the default strategy spec or default profile
- No changes to the frozen research settings: `input_window=60`, `horizon=20`
- No new hard filters such as `EP + ROE above_median`
- No second added factor in the same round
- No simultaneous portfolio-layer retuning

## Comparison Objects

The experiment compares two pairs of objects, one for the research layer and one for the executable layer.

### Research Layer

- Baseline: `baseline_5f`
- Candidate: `baseline_5f_ep`

`baseline_5f_ep` is defined as:

- `mom_40d`
- `volatility_20d`
- `reversal_5d`
- `amihud_20d`
- `turnover_stability_20d`
- `ep_ratio_ttm`

### Executable Layer

- Baseline executable line: `combo_b25_h45`
- Candidate executable line: `combo_b25_h45_ep`

`combo_b25_h45_ep` must use:

- The candidate signal from `baseline_5f_ep`
- The same `lgbm` model family
- The same `rank` label
- The same portfolio-layer assumptions as the current default line:
  - `Top25 buy`
  - `Top45 hold`
  - `20` trading day rebalance cadence
  - unchanged cost assumptions

This isolates the incremental effect of `ep_ratio_ttm` from portfolio-layer changes.

## Fixed Experimental Controls

To keep attribution clean, the following controls remain fixed:

- `input_window=60`
- `horizon=20`
- label transform = `rank`
- model kind = `lgbm`
- walk-forward split settings unchanged
- portfolio layer unchanged for the executable comparison
- benchmark unchanged: `000300.XSHG`

## Experimental Flow

### Step 1: Define the Candidate Feature Variant

Add a minimal feature variant named `baseline_5f_ep` to the research experiment entrypoint. The variant must only differ from `baseline_5f` by adding `ep_ratio_ttm`.

### Step 2: Run Research-Layer Comparison

Run the candidate against the current `baseline_5f` using the existing TX1 research workflow. Compare:

- `rank_ic_mean`
- `net_mean_return`
- `flag_spread_decay`
- `flag_ic_decay`

This step answers whether the additional factor improves cross-sectional ranking quality before replay execution is considered.

### Step 3: Train a Replayable Candidate Artifact Line

If the research-layer result is not obviously disqualifying, run the normal TX1 baseline experiment pipeline with the candidate feature set and the unchanged default portfolio-layer configuration, producing a temporary artifact line named `combo_b25_h45_ep`.

### Step 4: Run Executable-Layer Comparison

Replay `combo_b25_h45_ep` through the existing rolling-score strategy stack and compare it directly with `combo_b25_h45` using the same full `rolling-score` workflow.

### Step 5: Assign Final Verdict

Classify the result as `PASS`, `WATCHLIST`, or `REJECT` according to the gates below.

## Decision Gates

### PASS

The candidate is `PASS` only if all of the following are true:

- `rolling-score` is strictly higher than `combo_b25_h45`
- research-layer `rank_ic_mean` is not lower than `baseline_5f`
- research-layer `net_mean_return` is not lower than `baseline_5f`
- no new `spread_decay`
- no new `ic_decay`

`PASS` means the candidate is eligible to move forward as the next default-line contender.

### WATCHLIST

The candidate is `WATCHLIST` if:

- `rolling-score` improves by at least `+2.0` points versus `combo_b25_h45`
- and exactly one of `rank_ic_mean` or `net_mean_return` shows a small deterioration
- and there are no new `spread_decay` or `ic_decay` warnings

`WATCHLIST` means the candidate is not strong enough to promote, but is good enough for a second review round.

### REJECT

The candidate is `REJECT` if any of the following are true:

- `rolling-score` does not beat `combo_b25_h45`
- a new `spread_decay` warning appears
- a new `ic_decay` warning appears
- both `rank_ic_mean` and `net_mean_return` deteriorate
- the added factor materially reduces effective sample coverage and makes the comparison non-like-for-like

`REJECT` means the experiment ends and the default line remains unchanged.

## Risk Checks

This experiment must explicitly check the following failure modes.

### Factor Availability Risk

`ep_ratio_ttm` must be verified as a usable continuous factor in the candidate dataset:

- acceptable missingness
- acceptable cross-sectional coverage
- no hidden conversion into a hard filter effect

If the factor mostly reduces sample size instead of improving ranking quality, the result must be treated as a failed enhancement.

### Attribution Risk

Signal enhancement and portfolio-layer changes must not be mixed in the same round. If the candidate wins, the claimed source of improvement must still be the added `ep_ratio_ttm` factor, not a changed rebalance or holding rule.

### Concentration Risk

A higher average score driven by a small number of windows is not sufficient. If the candidate wins only through a few isolated windows while introducing new overfit warnings, it cannot be promoted.

## Expected File-Level Impact

The expected implementation scope for this design is intentionally small.

- `skyeye/products/tx1/run_feature_experiment.py`
  - add the minimal research variant `baseline_5f_ep`
- `skyeye/products/tx1/run_baseline_experiment.py`
  - allow explicit feature-set override or a narrow candidate experiment entry for `combo_b25_h45_ep`
- tests under `tests/products/tx1/`
  - verify the candidate variant definition and training entry behavior
- no change to:
  - `skyeye/products/tx1/strategies/rolling_score/strategy.py`
  - `skyeye/products/tx1/strategies/rolling_score/spec.yaml`
  - `skyeye/products/tx1/strategies/rolling_score/profiles/*.yaml`
  - default `FEATURE_COLUMNS` alias in `skyeye/products/tx1/evaluator.py`

## Validation Outputs

The implementation and experiment run should produce three final outputs.

### 1. Research-Layer Comparison Table

Baseline versus candidate:

- `baseline_5f`
- `baseline_5f_ep`

with:

- `rank_ic_mean`
- `net_mean_return`
- `flag_spread_decay`
- `flag_ic_decay`

### 2. Executable-Layer Comparison Table

Baseline executable line versus candidate executable line:

- `combo_b25_h45`
- `combo_b25_h45_ep`

with:

- `rolling-score`
- key supporting summary metrics
- whether new warnings appear

### 3. Final One-Line Verdict

Exactly one of:

- `PASS`
- `WATCHLIST`
- `REJECT`

plus a short reason sentence.

## Non-Goals for This Round

Even if this experiment succeeds, it does not automatically justify:

- promoting `ep_ratio_ttm` into the global default feature alias
- adding `return_on_equity_ttm` in the same change
- converting fundamentals into universe filters
- revisiting the guarded multi-output branch

Those would require separate follow-up design decisions.
