# -*- coding: utf-8 -*-
"""
TX1 Baseline Experiment Runner

Loads A-share data from RQAlpha bundle and runs the TX1 baseline
experiment (linear + tree) with the walk-forward splitter.

Usage:
    python -m skyeye.products.tx1.run_baseline_experiment
"""

import argparse
import os
from pathlib import Path

import pandas as pd
from skyeye.data import DataFacade

BENCHMARK_ID = "000300.XSHG"

TRAIN_START = "2015-01-01"
DEFAULT_DATA_END = "2026-03-06"

UNIVERSE_SIZE = 300
DATA_FACADE = DataFacade()
BAR_FIELDS = ["close", "volume", "total_turnover"]
MARKET_CAP_COLUMNS = ("circulating_market_cap", "market_capitalization", "market_cap")


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


def resolve_data_end() -> pd.Timestamp:
    """解析 TX1 当前可用的数据终点。

    优先允许用环境变量显式覆盖，便于临时补跑指定日期；
    否则直接读取基准指数真实可用的最后一个交易日，避免依赖手写常量。
    """
    override = os.environ.get("SKYEYE_TX1_DATA_END")
    if override:
        return pd.Timestamp(override).normalize()

    probe_end = pd.Timestamp.today().normalize()
    bars = DATA_FACADE.get_daily_bars(
        BENCHMARK_ID,
        pd.Timestamp(TRAIN_START),
        probe_end,
        fields=["close"],
        adjust_type="none",
    )
    benchmark_df = _normalize_daily_bars(bars, ["close"], single_order_book_id=BENCHMARK_ID)
    if not benchmark_df.empty:
        return pd.Timestamp(benchmark_df["date"].max()).normalize()

    return pd.Timestamp(DEFAULT_DATA_END)


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

def _load_benchmark_close(end_date: pd.Timestamp | None = None) -> pd.DataFrame:
    start_date = pd.Timestamp(TRAIN_START)
    end_date = pd.Timestamp(end_date).normalize() if end_date is not None else resolve_data_end()
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


def _get_liquid_universe(
    n: int = UNIVERSE_SIZE,
    *,
    data_end: pd.Timestamp | None = None,
) -> list[str]:
    instruments = _load_stock_instruments(active_only=True)
    active_cs = instruments["order_book_id"].tolist()
    data_end = pd.Timestamp(data_end).normalize() if data_end is not None else resolve_data_end()
    medians = {}
    for chunk in _chunked(active_cs, 500):
        bars = DATA_FACADE.get_daily_bars(
            chunk,
            pd.Timestamp(TRAIN_START),
            data_end,
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


def _resolve_market_cap_column(
    instruments: pd.DataFrame,
    preferred_column: str | None = None,
) -> str | None:
    candidates = []
    if preferred_column:
        candidates.append(preferred_column)
    candidates.extend([column for column in MARKET_CAP_COLUMNS if column not in candidates])
    for column in candidates:
        if column not in instruments.columns:
            continue
        values = pd.to_numeric(instruments[column], errors="coerce")
        if values.notna().any():
            return column
    return None


def _load_market_cap_snapshot(
    order_book_ids: list[str],
    preferred_column: str | None = None,
    lookback_days: int = 90,
    data_end: pd.Timestamp | None = None,
) -> tuple[str | None, pd.Series | None]:
    if not order_book_ids:
        return None, None

    end_date = pd.Timestamp(data_end).normalize() if data_end is not None else resolve_data_end()
    start_date = end_date - pd.Timedelta(days=int(lookback_days))
    candidates = []
    if preferred_column:
        candidates.append(preferred_column)
    candidates.extend([column for column in MARKET_CAP_COLUMNS if column not in candidates])

    for column in candidates:
        try:
            factor_df = DATA_FACADE.get_factor(order_book_ids, [column], start_date, end_date)
        except Exception:
            continue
        if factor_df is None or factor_df.empty:
            continue
        values = factor_df.reset_index()
        if "date" not in values.columns:
            for candidate in values.columns:
                if pd.api.types.is_datetime64_any_dtype(values[candidate]):
                    values = values.rename(columns={candidate: "date"})
                    break
        if column not in values.columns or "order_book_id" not in values.columns:
            continue
        values["date"] = pd.to_datetime(values["date"], errors="coerce")
        values[column] = pd.to_numeric(values[column], errors="coerce")
        values = values.dropna(subset=["date", column]).sort_values(["order_book_id", "date"])
        if values.empty:
            continue
        snapshot = values.groupby("order_book_id", sort=False)[column].last()
        if snapshot.notna().any():
            return column, snapshot.astype(float)
    return None, None


def _apply_market_cap_floor(
    instruments: pd.DataFrame,
    market_cap_floor_quantile: float | None = None,
    market_cap_column: str | None = None,
    data_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict | None]:
    if market_cap_floor_quantile is None:
        return instruments.reset_index(drop=True), None

    quantile = float(market_cap_floor_quantile)
    if quantile < 0.0 or quantile >= 1.0:
        raise ValueError("market_cap_floor_quantile must be in [0, 1)")

    resolved_column = _resolve_market_cap_column(instruments, preferred_column=market_cap_column)
    source = "instrument"
    if resolved_column is not None:
        values = pd.to_numeric(instruments[resolved_column], errors="coerce")
    else:
        resolved_column, snapshot = _load_market_cap_snapshot(
            instruments["order_book_id"].tolist(),
            preferred_column=market_cap_column,
            data_end=data_end,
        )
        if resolved_column is None or snapshot is None:
            raise ValueError("no market cap column available for market_cap floor")
        values = instruments["order_book_id"].map(snapshot)
        source = "factor_snapshot"

    threshold = float(values.dropna().quantile(quantile))
    filtered = instruments.loc[values >= threshold].copy().reset_index(drop=True)
    summary = {
        "column": resolved_column,
        "source": source,
        "quantile": quantile,
        "threshold": threshold,
        "kept": int(len(filtered)),
        "total": int(len(instruments)),
    }
    return filtered, summary


def get_liquid_universe(
    n: int = UNIVERSE_SIZE,
    market_cap_floor_quantile: float | None = None,
    market_cap_column: str | None = None,
    data_end: pd.Timestamp | None = None,
) -> list[str]:
    data_end = pd.Timestamp(data_end).normalize() if data_end is not None else resolve_data_end()
    instruments = _load_stock_instruments(active_only=True)
    filtered, floor_summary = _apply_market_cap_floor(
        instruments,
        market_cap_floor_quantile=market_cap_floor_quantile,
        market_cap_column=market_cap_column,
        data_end=data_end,
    )
    if floor_summary:
        print(
            "Market-cap floor applied: source={} column={} quantile={:.0%} threshold={:.4g} kept={}/{}".format(
                floor_summary["source"],
                floor_summary["column"],
                floor_summary["quantile"],
                floor_summary["threshold"],
                floor_summary["kept"],
                floor_summary["total"],
            )
        )
    active_cs = filtered["order_book_id"].tolist()
    medians = {}
    for chunk in _chunked(active_cs, 500):
        bars = DATA_FACADE.get_daily_bars(
            chunk,
            pd.Timestamp(TRAIN_START),
            data_end,
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


def _load_northbound_flow(data_end: pd.Timestamp) -> pd.DataFrame:
    """Load daily northbound net flow (沪股通+深股通 combined)."""
    try:
        from skyeye.data.provider import RQDataProvider
        provider = RQDataProvider()
        df = provider.get_northbound_flow(TRAIN_START, data_end.strftime("%Y-%m-%d"))
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


FUNDAMENTAL_FACTORS = [
    "ep_ratio_ttm",
    "return_on_equity_ttm",
    "operating_revenue_growth_ratio_ttm",
    "net_profit_growth_ratio_ttm",
    "pcf_ratio_ttm",
]


def _load_fundamental_factors(
    universe: list[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load daily fundamental factors via rqdatac for the given universe."""
    try:
        factor_df = DATA_FACADE.get_factor(
            universe, FUNDAMENTAL_FACTORS, start_date, end_date
        )
    except Exception as exc:
        print(f"Warning: fundamental factors unavailable ({exc}), skipping")
        return pd.DataFrame(columns=["date", "order_book_id"] + FUNDAMENTAL_FACTORS)

    if factor_df is None or factor_df.empty:
        print("Warning: fundamental factors returned empty, skipping")
        return pd.DataFrame(columns=["date", "order_book_id"] + FUNDAMENTAL_FACTORS)

    result = factor_df.reset_index()
    if "date" not in result.columns:
        for col in result.columns:
            if pd.api.types.is_datetime64_any_dtype(result[col]):
                result = result.rename(columns={col: "date"})
                break
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    return result.sort_values(["date", "order_book_id"]).reset_index(drop=True)


def build_raw_df(
    universe: list[str],
    *,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    print(f"Loading benchmark ({BENCHMARK_ID})...")
    end_date = pd.Timestamp(end_date).normalize() if end_date is not None else resolve_data_end()
    start_date = pd.Timestamp(start_date).normalize() if start_date is not None else pd.Timestamp(TRAIN_START)
    benchmark = _load_benchmark_close(end_date=end_date)

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
    north_df = _load_northbound_flow(end_date)
    if not north_df.empty:
        raw_df = raw_df.merge(north_df, on="date", how="left")
        raw_df["north_net_flow"] = raw_df["north_net_flow"].fillna(0.0)
        print(f"  Northbound data: {len(north_df)} days")
    else:
        print("  Northbound data: unavailable, skipping")

    # Add fundamental factors (daily, via rqdatac)
    print("Loading fundamental factors...")
    fund_df = _load_fundamental_factors(universe, start_date, end_date)
    if not fund_df.empty and len(fund_df) > 0:
        raw_df = raw_df.merge(fund_df, on=["date", "order_book_id"], how="left")
        loaded = [c for c in FUNDAMENTAL_FACTORS if c in raw_df.columns]
        coverage = raw_df[loaded].notna().mean()
        print(f"  Fundamental factors loaded: {loaded}")
        for col in loaded:
            print(f"    {col}: {coverage[col]:.1%} coverage")
    else:
        print("  Fundamental factors: unavailable, skipping")

    raw_df = raw_df.sort_values(["date", "order_book_id"]).reset_index(drop=True)
    print(f"raw_df shape: {raw_df.shape}  date range: {raw_df['date'].min().date()} – {raw_df['date'].max().date()}")
    return raw_df


def build_live_raw_df(
    *,
    trade_date=None,
    universe: list[str] | None = None,
    universe_size: int = UNIVERSE_SIZE,
    market_cap_floor_quantile: float | None = None,
    market_cap_column: str | None = None,
) -> pd.DataFrame:
    """为 live advisor 构建截止指定日期的实时原始面板。"""
    end_date = pd.Timestamp(trade_date).normalize() if trade_date is not None else resolve_data_end()
    if universe is None:
        universe = get_liquid_universe(
            universe_size,
            market_cap_floor_quantile=market_cap_floor_quantile,
            market_cap_column=market_cap_column,
            data_end=end_date,
        )
    return build_raw_df(universe, end_date=end_date)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_experiment_config(
    model_kind: str,
    *,
    experiment_name: str | None = None,
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


def _resolve_output_dir_name(experiment_name: str | None, model_kind: str) -> str:
    suffix = experiment_name or f"baseline_{model_kind}"
    return suffix if suffix.startswith("tx1_") else f"tx1_{suffix}"


def run_experiment(
    model_kind: str,
    output_base: str,
    universe_size: int = UNIVERSE_SIZE,
    *,
    experiment_name: str | None = None,
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
    config = build_experiment_config(
        model_kind,
        experiment_name=output_dir_name,
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
        "\nRunning TX1 experiment: model={} experiment={} multi_output={} market_cap_floor={}".format(
            model_kind,
            output_dir_name,
            bool(config.get("multi_output", {}).get("enabled", False)),
            market_cap_floor_quantile,
        )
    )
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
    parser.add_argument("--experiment-name")
    parser.add_argument("--market-cap-floor-quantile", type=float)
    parser.add_argument("--market-cap-column")
    parser.add_argument("--enable-multi-output", action="store_true")
    parser.add_argument("--volatility-weight", type=float, default=0.0)
    parser.add_argument("--max-drawdown-weight", type=float, default=0.0)
    parser.add_argument("--enable-reliability-score", action="store_true")
    parser.add_argument("--holding-bonus", type=float, default=0.5)
    parser.add_argument("--rebalance-interval", type=int, default=20)
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
        experiment_name = args.experiment_name
        if experiment_name and len(models) > 1:
            experiment_name = f"{experiment_name}_{model_kind}"
        result = run_experiment(
            model_kind,
            args.output_dir,
            universe_size=universe_size,
            experiment_name=experiment_name,
            market_cap_floor_quantile=args.market_cap_floor_quantile,
            market_cap_column=args.market_cap_column,
            multi_output_enabled=args.enable_multi_output,
            volatility_weight=args.volatility_weight,
            max_drawdown_weight=args.max_drawdown_weight,
            enable_reliability_score=args.enable_reliability_score,
            holding_bonus=args.holding_bonus,
            rebalance_interval=args.rebalance_interval,
        )
        print_summary(result)
        results[model_kind] = result

    return results


if __name__ == "__main__":
    main()
