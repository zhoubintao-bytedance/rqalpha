# -*- coding: utf-8 -*-

import argparse
import json
from collections import OrderedDict

import pandas as pd
from scipy.stats import spearmanr

from skyeye.products.dividend_low_vol.scorer.config import (
    SCORE_BUY_PERCENTILE,
    SCORE_SELL_PERCENTILE,
    WEIGHT_PRIOR_BLEND,
)
from skyeye.products.dividend_low_vol.scorer.main import DividendScorer


EVAL_HORIZONS = (20, 60, 120)
MIN_SIGNAL_COUNT = 20
PERIODS = OrderedDict([
    ("2019_2021", ("2019-01-18", "2021-12-31")),
    ("2022_2023", ("2022-01-01", "2023-12-31")),
    ("2024_2026", ("2024-01-01", "2026-12-31")),
])


def _classify_flags(flags):
    if not flags:
        return "accurate"
    if len(flags) <= 3:
        return "mixed"
    return "overfit_risk"


def _safe_float(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _pct(value):
    if value is None or pd.isna(value):
        return None
    return round(float(value) * 100, 3)


def _spearman(left, right):
    if len(left) < 5:
        return None
    corr = spearmanr(left, right, nan_policy="omit").correlation
    return None if pd.isna(corr) else float(corr)


def _evaluate_frame(frame, buy_percentile, sell_percentile):
    metrics = OrderedDict()
    for horizon in EVAL_HORIZONS:
        ret_col = "fwd_{}d".format(horizon)
        valid = frame[["score", "score_percentile", ret_col]].dropna()
        low = valid[valid["score_percentile"] <= buy_percentile]
        high = valid[valid["score_percentile"] >= sell_percentile]
        metrics[str(horizon)] = {
            "sample_count": int(len(valid)),
            "score_spearman": _spearman(valid["score"], valid[ret_col]),
            "percentile_spearman": _spearman(valid["score_percentile"], valid[ret_col]),
            "low_count": int(len(low)),
            "high_count": int(len(high)),
            "low_mean_ret_pct": _pct(low[ret_col].mean()),
            "high_mean_ret_pct": _pct(high[ret_col].mean()),
            "low_minus_high_pct": _pct(low[ret_col].mean() - high[ret_col].mean()) if len(low) and len(high) else None,
            "low_win_rate": _safe_float((low[ret_col] > 0).mean()) if len(low) else None,
            "high_win_rate": _safe_float((high[ret_col] > 0).mean()) if len(high) else None,
        }
    return metrics


def _judge_periods(overall_metrics, period_metrics):
    flags = []

    for horizon, metrics in overall_metrics.items():
        spread = metrics.get("low_minus_high_pct")
        if spread is None or spread <= 0:
            flags.append("overall {}d low_minus_high <= 0".format(horizon))
        if metrics.get("low_count", 0) < MIN_SIGNAL_COUNT:
            flags.append("overall {}d low_count < {}".format(horizon, MIN_SIGNAL_COUNT))
        if metrics.get("high_count", 0) < MIN_SIGNAL_COUNT:
            flags.append("overall {}d high_count < {}".format(horizon, MIN_SIGNAL_COUNT))

    for period_name, metrics_by_horizon in period_metrics.items():
        for horizon, metrics in metrics_by_horizon.items():
            spread = metrics.get("low_minus_high_pct")
            if spread is None or spread <= 0:
                flags.append("{} {}d low_minus_high <= 0".format(period_name, horizon))
            if metrics.get("low_count", 0) < MIN_SIGNAL_COUNT:
                flags.append("{} {}d low_count < {}".format(period_name, horizon, MIN_SIGNAL_COUNT))
            if metrics.get("high_count", 0) < MIN_SIGNAL_COUNT:
                flags.append("{} {}d high_count < {}".format(period_name, horizon, MIN_SIGNAL_COUNT))

    return _classify_flags(flags), flags


def _compare_baseline_overall(current_overall, baseline_overall):
    regressions = []
    for horizon in EVAL_HORIZONS:
        key = str(horizon)
        current_metrics = current_overall.get(key, {})
        baseline_metrics = baseline_overall.get(key, {})
        current_spread = current_metrics.get("low_minus_high_pct")
        baseline_spread = baseline_metrics.get("low_minus_high_pct")
        if current_spread is None or baseline_spread is None:
            continue
        if current_spread < baseline_spread:
            regressions.append(
                "overall {}d low_minus_high regressed: {} < {}".format(
                    key, current_spread, baseline_spread
                )
            )
    return regressions


def run_audit(
    db_path,
    start_date=None,
    end_date=None,
    prior_blend=WEIGHT_PRIOR_BLEND,
    buy_percentile=SCORE_BUY_PERCENTILE,
    sell_percentile=SCORE_SELL_PERCENTILE,
    baseline_report=None,
):
    scorer = DividendScorer(
        db_path=db_path,
        prior_blend=prior_blend,
        dynamic_diagnostic=False,
    )
    scorer.precompute(start_date=start_date, end_date=end_date)
    score_df = scorer.score_history_df.copy()
    history_df = scorer.history_df.copy()
    df = history_df.join(score_df, how="left")
    df = df[df["etf_close_hfq"].notna() & df["total_score"].notna() & df["score_percentile"].notna()].copy()
    df = df.rename(columns={"total_score": "score"})

    for horizon in EVAL_HORIZONS:
        df["fwd_{}d".format(horizon)] = df["etf_close_hfq"].shift(-horizon) / df["etf_close_hfq"] - 1.0

    overall_metrics = _evaluate_frame(df, buy_percentile, sell_percentile)
    period_metrics = OrderedDict()
    score_distribution = OrderedDict()
    for name, (start, end) in PERIODS.items():
        part = df.loc[start:end].copy()
        if part.empty:
            continue
        period_metrics[name] = _evaluate_frame(part, buy_percentile, sell_percentile)
        score_distribution[name] = {
            "sample_count": int(len(part)),
            "score_min": round(float(part["score"].min()), 3),
            "score_q25": round(float(part["score"].quantile(0.25)), 3),
            "score_median": round(float(part["score"].median()), 3),
            "score_q75": round(float(part["score"].quantile(0.75)), 3),
            "score_max": round(float(part["score"].max()), 3),
            "buy_signal_pct": round(float((part["score_percentile"] <= buy_percentile).mean()) * 100, 2),
            "sell_signal_pct": round(float((part["score_percentile"] >= sell_percentile).mean()) * 100, 2),
        }

    verdict, flags = _judge_periods(overall_metrics, period_metrics)
    baseline_regressions = []
    if baseline_report is not None:
        baseline_regressions = _compare_baseline_overall(
            overall_metrics,
            baseline_report.get("overall", {}),
        )
        if baseline_regressions:
            flags = list(flags) + [
                "baseline {}".format(item) for item in baseline_regressions
            ]
            verdict = _classify_flags(flags)

    return {
        "verdict": verdict,
        "flags": flags,
        "config": {
            "prior_blend": float(prior_blend),
            "buy_percentile": float(buy_percentile),
            "sell_percentile": float(sell_percentile),
        },
        "sample_range": {
            "start": df.index[0].strftime("%Y-%m-%d"),
            "end": df.index[-1].strftime("%Y-%m-%d"),
            "count": int(len(df)),
        },
        "score_model": {
            "score_method_counts": score_df["method"].value_counts(dropna=False).to_dict(),
            "confidence_counts": score_df["confidence"].value_counts(dropna=False).to_dict(),
        },
        "baseline_compare": {
            "checked": baseline_report is not None,
            "regressions": baseline_regressions,
            "baseline_sample_range": baseline_report.get("sample_range") if baseline_report is not None else None,
        },
        "overall": overall_metrics,
        "periods": period_metrics,
        "score_distribution": score_distribution,
    }


def _render_text(report):
    lines = []
    lines.append("Dividend Scorer Parameter Audit")
    lines.append("verdict: {}".format(report["verdict"]))
    if report["flags"]:
        lines.append("flags:")
        for flag in report["flags"]:
            lines.append("  - {}".format(flag))
    lines.append(
        "sample: {start} -> {end} ({count})".format(**report["sample_range"])
    )
    lines.append(
        "config: prior_blend={prior_blend:.2f}, buy_pct={buy_percentile:.0%}, sell_pct={sell_percentile:.0%}".format(
            **report["config"]
        )
    )
    baseline_compare = report.get("baseline_compare", {})
    if baseline_compare.get("checked"):
        lines.append(
            "baseline: checked against {start} -> {end} ({count})".format(
                **baseline_compare["baseline_sample_range"]
            )
        )
        regressions = baseline_compare.get("regressions") or []
        if regressions:
            lines.append("baseline regressions:")
            for item in regressions:
                lines.append("  - {}".format(item))
        else:
            lines.append("baseline regressions: none")
    lines.append("")
    lines.append("overall:")
    for horizon, metrics in report["overall"].items():
        lines.append(
            "  {h}d low-high={spread}% low={low}% high={high}% low_n={low_n} high_n={high_n} score_rho={score_rho}".format(
                h=horizon,
                spread=metrics["low_minus_high_pct"],
                low=metrics["low_mean_ret_pct"],
                high=metrics["high_mean_ret_pct"],
                low_n=metrics["low_count"],
                high_n=metrics["high_count"],
                score_rho=round(metrics["score_spearman"], 4) if metrics["score_spearman"] is not None else None,
            )
        )
    lines.append("")
    lines.append("periods:")
    for period_name, metrics_by_horizon in report["periods"].items():
        lines.append("  {}".format(period_name))
        for horizon, metrics in metrics_by_horizon.items():
            lines.append(
                "    {h}d low-high={spread}% low_n={low_n} high_n={high_n}".format(
                    h=horizon,
                    spread=metrics["low_minus_high_pct"],
                    low_n=metrics["low_count"],
                    high_n=metrics["high_count"],
                )
            )
    return "\n".join(lines)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Dividend scorer parameter audit")
    parser.add_argument("--db-path", dest="db_path", default=None, help="SQLite cache path")
    parser.add_argument("--start-date", dest="start_date", default=None, help="history start date")
    parser.add_argument("--end-date", dest="end_date", default=None, help="history end date")
    parser.add_argument("--prior-blend", dest="prior_blend", type=float, default=WEIGHT_PRIOR_BLEND,
                        help="domain prior blend weight, 1.0 means fixed domain prior only")
    parser.add_argument("--buy-percentile", dest="buy_percentile", type=float, default=SCORE_BUY_PERCENTILE,
                        help="buy threshold on rolling score percentile")
    parser.add_argument("--sell-percentile", dest="sell_percentile", type=float, default=SCORE_SELL_PERCENTILE,
                        help="sell threshold on rolling score percentile")
    parser.add_argument("--baseline-json", dest="baseline_json", default=None,
                        help="compare current report against a saved JSON baseline report")
    parser.add_argument("--write-json", dest="write_json", default=None,
                        help="write current report to a JSON file")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    baseline_report = None
    if args.baseline_json:
        with open(args.baseline_json, "r", encoding="utf-8") as f:
            baseline_report = json.load(f)
    report = run_audit(
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        prior_blend=args.prior_blend,
        buy_percentile=args.buy_percentile,
        sell_percentile=args.sell_percentile,
        baseline_report=baseline_report,
    )
    if args.write_json:
        with open(args.write_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(_render_text(report))


if __name__ == "__main__":
    main()
