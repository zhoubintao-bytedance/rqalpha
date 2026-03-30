# -*- coding: utf-8 -*-
"""
TX1 Baseline Experiment Runner

Loads A-share data from RQAlpha bundle and runs the TX1 baseline
experiment (linear + tree) with the walk-forward splitter.

Usage:
    python -m skyeye.products.tx1.run_baseline_experiment
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

BUNDLE_DIR = Path.home() / ".rqalpha" / "bundle"
BENCHMARK_ID = "000300.XSHG"   # CSI 300

# Date range for the experiment
TRAIN_START = "2015-01-01"
DATA_END    = "2026-03-06"

# Liquid universe: top-N stocks by median daily turnover (2015-2026)
UNIVERSE_SIZE = 300


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_benchmark_close() -> pd.DataFrame:
    with h5py.File(BUNDLE_DIR / "indexes.h5", "r") as f:
        if BENCHMARK_ID not in f:
            raise KeyError(f"Benchmark {BENCHMARK_ID} not in indexes.h5")
        ds = f[BENCHMARK_ID][:]
    df = pd.DataFrame({
        "date": pd.to_datetime(ds["datetime"].astype(str), format="%Y%m%d%H%M%S"),
        "benchmark_close": ds["close"].astype(float),
    })
    df["date"] = df["date"].dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def _load_all_stocks(universe: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """Batch-load all stocks from h5 in a single file open."""
    start_dt = int(start_date.strftime("%Y%m%d")) * 1_000_000
    end_dt = int(end_date.strftime("%Y%m%d")) * 1_000_000

    rows = []
    skipped = 0
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as f:
        for i, sid in enumerate(universe):
            if i % 50 == 0:
                print(f"  {i}/{len(universe)}...", flush=True)
            if sid not in f:
                skipped += 1
                continue
            ds = f[sid][:]
            mask = (ds["datetime"] >= start_dt) & (ds["datetime"] <= end_dt)
            ds = ds[mask]
            if len(ds) < 300:
                skipped += 1
                continue
            dates = pd.to_datetime(ds["datetime"].astype(str), format="%Y%m%d%H%M%S").normalize()
            df = pd.DataFrame({
                "date": dates,
                "order_book_id": sid,
                "open": ds["open"].astype(float),
                "high": ds["high"].astype(float),
                "low": ds["low"].astype(float),
                "close": ds["close"].astype(float),
                "prev_close": ds["prev_close"].astype(float),
                "volume": ds["volume"].astype(float),
                "total_turnover": ds["total_turnover"].astype(float),
            })
            rows.append(df)

    print(f"Loaded {len(rows)} stocks, skipped {skipped}")
    if not rows:
        raise RuntimeError("No stocks loaded")
    return pd.concat(rows, ignore_index=True)


def _get_liquid_universe(n: int = UNIVERSE_SIZE) -> list[str]:
    """Return top-N stocks by median daily turnover, filtered to active CS."""
    with open(BUNDLE_DIR / "instruments.pk", "rb") as f:
        instruments = pickle.load(f)

    active_cs = {
        inst["order_book_id"]
        for inst in instruments
        if inst["type"] == "CS"
        and inst["exchange"] in ("XSHE", "XSHG")
        and inst.get("status") == "Active"
    }

    start_dt = int(TRAIN_START.replace("-", "")) * 1_000_000
    end_dt = int(DATA_END.replace("-", "")) * 1_000_000

    medians = {}
    with h5py.File(BUNDLE_DIR / "stocks.h5", "r") as f:
        for sid in active_cs:
            if sid not in f:
                continue
            ds = f[sid][:]
            mask = (ds["datetime"] >= start_dt) & (ds["datetime"] <= end_dt)
            vol = ds["volume"][mask].astype(float)
            if len(vol) < 500:          # need enough history
                continue
            medians[sid] = float(np.median(vol))

    sorted_stocks = sorted(medians, key=lambda s: medians[s], reverse=True)
    return sorted_stocks[:n]


# ---------------------------------------------------------------------------
# Build raw_df
# ---------------------------------------------------------------------------

def _load_sector_map() -> dict:
    """Load sector_code mapping from instruments.pk."""
    with open(BUNDLE_DIR / "instruments.pk", "rb") as f:
        instruments = pickle.load(f)
    return {
        inst["order_book_id"]: inst.get("sector_code", "Unknown")
        for inst in instruments
        if inst.get("type") == "CS"
    }


def _load_northbound_flow() -> pd.DataFrame:
    """Load daily northbound net flow from akshare (沪股通+深股通 combined)."""
    try:
        import akshare as ak
        parts = []
        for symbol in ("沪股通", "深股通"):
            df = ak.stock_hsgt_hist_em(symbol=symbol)
            date_col = [c for c in df.columns if "日期" in c]
            flow_col = [c for c in df.columns if "当日成交净买额" in c or "当日净流入" in c]
            if not date_col or not flow_col:
                continue
            part = pd.DataFrame({
                "date": pd.to_datetime(df[date_col[0]], errors="coerce"),
                "flow": pd.to_numeric(df[flow_col[0]], errors="coerce"),
            }).dropna()
            parts.append(part)
        if not parts:
            return pd.DataFrame(columns=["date", "north_net_flow"])
        combined = pd.concat(parts, ignore_index=True)
        result = combined.groupby("date", as_index=False)["flow"].sum()
        result.columns = ["date", "north_net_flow"]
        return result.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"Warning: northbound data unavailable ({e}), skipping")
        return pd.DataFrame(columns=["date", "north_net_flow"])


def build_raw_df(universe: list[str]) -> pd.DataFrame:
    print(f"Loading benchmark ({BENCHMARK_ID})...")
    benchmark = _load_benchmark_close()

    start_date = pd.Timestamp(TRAIN_START)
    end_date = pd.Timestamp(DATA_END)
    benchmark = benchmark[
        (benchmark["date"] >= start_date) & (benchmark["date"] <= end_date)
    ]

    print(f"Loading {len(universe)} stocks (batch)...")
    stocks_df = _load_all_stocks(universe, start_date, end_date)
    raw_df = stocks_df.merge(benchmark, on="date", how="inner")
    raw_df = raw_df.dropna(subset=["close", "volume", "benchmark_close"])

    # Add sector classification
    sector_map = _load_sector_map()
    raw_df["sector"] = raw_df["order_book_id"].map(sector_map).fillna("Unknown")

    # Add northbound net flow (market-level, optional)
    print("Loading northbound flow data...")
    north_df = _load_northbound_flow()
    if not north_df.empty:
        raw_df = raw_df.merge(north_df, on="date", how="left")
        raw_df["north_net_flow"] = raw_df["north_net_flow"].fillna(0.0)
        print(f"  Northbound data: {len(north_df)} days")
    else:
        print("  Northbound data: unavailable, skipping")

    raw_df = raw_df.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    print(f"raw_df shape: {raw_df.shape}  date range: {raw_df['date'].min().date()} – {raw_df['date'].max().date()}")
    return raw_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(model_kind: str, output_base: str, universe_size: int = UNIVERSE_SIZE) -> dict:
    from skyeye.products.tx1.main import main as tx1_main

    output_dir = str(Path(output_base) / f"tx1_baseline_{model_kind}")

    config = {
        "model": {"kind": model_kind},
        "labels": {"transform": "rank"},
        "robustness": {"enabled": True, "stability_metric": "rank_ic_mean"},
        "costs": {
            "enabled": True,
            "commission_rate": 0.0008,
            "stamp_tax_rate": 0.0005,
            "slippage_bps": 5.0,
        },
    }

    universe = _get_liquid_universe(universe_size)
    raw_df = build_raw_df(universe)

    print(f"\nRunning TX1 experiment: model={model_kind}")
    result = tx1_main(config=config, raw_df=raw_df, output_dir=output_dir)
    return result


def print_summary(result: dict) -> None:
    agg = result.get("aggregate_metrics", {})
    pred = agg.get("prediction", {})
    port = agg.get("portfolio", {})
    rob = agg.get("robustness", {})

    print("\n" + "=" * 60)
    print(f"Model: {result.get('model_kind')}")
    print(f"Folds: {len(result.get('fold_results', []))}")
    print("-" * 60)
    print("Prediction (avg across folds):")
    for k, v in pred.items():
        print(f"  {k:35s}: {v:.4f}")
    print("Portfolio (avg across folds):")
    for k, v in port.items():
        print(f"  {k:35s}: {v:.4f}")

    robustness = result.get("aggregate_metrics", {}).get("robustness")
    if robustness:
        stab = robustness.get("stability", {})
        overfit = robustness.get("overfit_flags", {})
        print("Robustness:")
        print(f"  stability_score               : {stab.get('stability_score', 0):.1f}")
        print(f"  cv                            : {stab.get('cv', 0):.4f}")
        print(f"  flag_ic_decay                 : {overfit.get('flag_ic_decay')}")
        print(f"  flag_val_dominant             : {overfit.get('flag_val_dominant')}")

    print(f"\nOutput saved to: {result.get('output_dir', '(not saved)')}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TX1 Baseline Experiment")
    parser.add_argument("--model", choices=["linear", "tree", "lgbm", "both", "all"], default="both")
    parser.add_argument("--output-dir", default="skyeye/artifacts/experiments/tx1")
    parser.add_argument("--universe-size", type=int, default=UNIVERSE_SIZE)
    args = parser.parse_args()

    universe_size = args.universe_size
    if args.model == "both":
        models = ["linear", "tree"]
    elif args.model == "all":
        models = ["linear", "tree", "lgbm"]
    else:
        models = [args.model]
    results = {}
    for model_kind in models:
        result = run_experiment(model_kind, args.output_dir, universe_size=universe_size)
        print_summary(result)
        results[model_kind] = result

    return results


if __name__ == "__main__":
    main()
