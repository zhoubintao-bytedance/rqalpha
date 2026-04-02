# -*- coding: utf-8 -*-
"""
TX1 Baseline Experiment Runner

Loads A-share data from RQAlpha bundle and runs the TX1 baseline
experiment (linear + tree) with the walk-forward splitter.

Usage:
    python -m skyeye.products.tx1.run_baseline_experiment
"""

import argparse
from pathlib import Path

import pandas as pd
from skyeye.data import DataFacade

BENCHMARK_ID = "000300.XSHG"

TRAIN_START = "2015-01-01"
DATA_END    = "2026-03-06"

UNIVERSE_SIZE = 300
DATA_FACADE = DataFacade()
BAR_FIELDS = ["close", "volume", "total_turnover"]


def _chunked(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def _normalize_multi_column_bars(
    bars: pd.DataFrame,
    fields: list[str],
    single_order_book_id: str | None = None,
) -> pd.DataFrame:
    if single_order_book_id is not None:
        data = {}
        for field in fields:
            for level in range(bars.columns.nlevels):
                if field not in set(bars.columns.get_level_values(level)):
                    continue
                field_frame = bars.xs(field, axis=1, level=level, drop_level=True)
                if isinstance(field_frame, pd.DataFrame):
                    if single_order_book_id in field_frame.columns:
                        data[field] = field_frame[single_order_book_id]
                    else:
                        data[field] = field_frame.iloc[:, 0]
                else:
                    data[field] = field_frame
                break
        df = pd.DataFrame(data, index=bars.index).reset_index()
        df = df.rename(columns={df.columns[0]: "date"})
        df.insert(1, "order_book_id", single_order_book_id)
        return df

    merged = None
    for field in fields:
        field_frame = None
        for level in range(bars.columns.nlevels):
            if field in set(bars.columns.get_level_values(level)):
                field_frame = bars.xs(field, axis=1, level=level, drop_level=True)
                break
        if field_frame is None:
            continue
        if isinstance(field_frame, pd.Series):
            part = field_frame.rename(field).reset_index()
            part = part.rename(columns={part.columns[0]: "date"})
            part.insert(1, "order_book_id", single_order_book_id or BENCHMARK_ID)
        else:
            part = field_frame.stack(dropna=False).rename(field).reset_index()
            part = part.rename(columns={part.columns[0]: "date", part.columns[1]: "order_book_id"})
        merged = part if merged is None else merged.merge(part, on=["date", "order_book_id"], how="outer")

    if merged is None:
        return pd.DataFrame(columns=["date", "order_book_id", *fields])
    return merged


def _normalize_daily_bars(
    bars: pd.DataFrame | None,
    fields: list[str],
    single_order_book_id: str | None = None,
) -> pd.DataFrame:
    if bars is None or len(bars) == 0:
        return pd.DataFrame(columns=["date", "order_book_id", *fields])

    df = bars.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = _normalize_multi_column_bars(df, fields, single_order_book_id)
    elif isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif "date" not in df.columns and "datetime" not in df.columns:
        df = df.reset_index()

    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        else:
            for column in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df = df.rename(columns={column: "date"})
                    break

    if "order_book_id" not in df.columns:
        if single_order_book_id is not None:
            insert_at = 1 if "date" in df.columns else 0
            df.insert(insert_at, "order_book_id", single_order_book_id)
        else:
            for column in df.columns:
                if column != "date" and df[column].dtype == object:
                    df = df.rename(columns={column: "order_book_id"})
                    break

    keep = [column for column in ["date", "order_book_id", *fields] if column in df.columns]
    df = df.loc[:, keep].copy()
    if "date" not in df.columns or "order_book_id" not in df.columns:
        return pd.DataFrame(columns=["date", "order_book_id", *fields])

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for field in fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")
    return df.sort_values(["date", "order_book_id"]).reset_index(drop=True)


def _load_stock_instruments(active_only: bool) -> pd.DataFrame:
    instruments = DATA_FACADE.all_instruments(type="CS")
    if instruments is None or instruments.empty:
        raise RuntimeError("No stock instruments loaded")
    instruments = instruments.copy()
    if "exchange" in instruments.columns:
        instruments = instruments[instruments["exchange"].isin(["XSHE", "XSHG"])]
    if active_only and "status" in instruments.columns:
        instruments = instruments[instruments["status"] == "Active"]
    keep = [
        "order_book_id",
        "symbol",
        "exchange",
        "status",
        "sector_code",
        "sector_code_name",
        "industry_code",
        "industry_name",
        "market_cap",
        "market_capitalization",
        "circulating_market_cap",
    ]
    keep = [c for c in keep if c in instruments.columns]
    if keep:
        instruments = instruments.loc[:, keep]
    return instruments.reset_index(drop=True)

def _load_benchmark_close() -> pd.DataFrame:
    start_date = pd.Timestamp(TRAIN_START)
    end_date = pd.Timestamp(DATA_END)
    bars = DATA_FACADE.get_daily_bars(
        BENCHMARK_ID,
        start_date,
        end_date,
        fields=BAR_FIELDS,
        adjust_type="none",
    )
    df = _normalize_daily_bars(bars, BAR_FIELDS, single_order_book_id=BENCHMARK_ID)
    if df.empty or "close" not in df.columns:
        raise RuntimeError(f"Benchmark {BENCHMARK_ID} has no daily bars")
    df = df.loc[:, ["date", "close"]].rename(columns={"close": "benchmark_close"})
    return df.sort_values("date").reset_index(drop=True)


def _load_all_stocks(universe: list[str], start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    bars = DATA_FACADE.get_daily_bars(
        universe,
        start_date,
        end_date,
        fields=BAR_FIELDS,
        adjust_type="pre",
    )
    stocks_df = _normalize_daily_bars(bars, BAR_FIELDS)
    if stocks_df.empty:
        raise RuntimeError("No stocks loaded")
    counts = stocks_df.groupby("order_book_id").size()
    valid_ids = counts[counts >= 300].index
    skipped = len(universe) - len(valid_ids)
    stocks_df = stocks_df[stocks_df["order_book_id"].isin(valid_ids)].copy()
    print(f"Loaded {len(valid_ids)} stocks, skipped {skipped}")
    if stocks_df.empty:
        raise RuntimeError("No stocks loaded")
    return stocks_df.reset_index(drop=True)


def _get_liquid_universe(n: int = UNIVERSE_SIZE) -> list[str]:
    instruments = _load_stock_instruments(active_only=True)
    active_cs = instruments["order_book_id"].tolist()
    medians = {}
    for chunk in _chunked(active_cs, 500):
        bars = DATA_FACADE.get_daily_bars(
            chunk,
            pd.Timestamp(TRAIN_START),
            pd.Timestamp(DATA_END),
            fields=BAR_FIELDS,
            adjust_type="pre",
        )
        stocks_df = _normalize_daily_bars(bars, BAR_FIELDS)
        if stocks_df.empty or "volume" not in stocks_df.columns:
            continue
        grouped = stocks_df.groupby("order_book_id")["volume"].agg(["size", "median"])
        grouped = grouped[grouped["size"] >= 500]
        medians.update(grouped["median"].astype(float).to_dict())

    sorted_stocks = sorted(medians, key=lambda s: medians[s], reverse=True)
    return sorted_stocks[:n]


# ---------------------------------------------------------------------------
# Build raw_df
# ---------------------------------------------------------------------------

def _load_sector_map() -> dict:
    instruments = _load_stock_instruments(active_only=False)
    sector_field = next(
        (
            field
            for field in ["sector_code", "industry_code", "sector_code_name", "industry_name"]
            if field in instruments.columns
        ),
        None,
    )
    if sector_field is None:
        return {}
    return {
        row["order_book_id"]: row.get(sector_field) or "Unknown"
        for _, row in instruments.iterrows()
    }


def _load_northbound_flow() -> pd.DataFrame:
    """Load daily northbound net flow (沪股通+深股通 combined)."""
    try:
        from skyeye.data.provider import RQDataProvider
        provider = RQDataProvider()
        df = provider.get_northbound_flow(TRAIN_START, DATA_END)
        if df is not None and not df.empty:
            # Normalize column names
            date_col = [c for c in df.columns if "date" in c.lower()]
            flow_col = [c for c in df.columns if "flow" in c.lower() or "net" in c.lower()]
            if date_col and flow_col:
                result = pd.DataFrame({
                    "date": pd.to_datetime(df[date_col[0]], errors="coerce"),
                    "north_net_flow": pd.to_numeric(df[flow_col[0]], errors="coerce"),
                }).dropna()
                return result.sort_values("date").reset_index(drop=True)
        return pd.DataFrame(columns=["date", "north_net_flow"])
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
