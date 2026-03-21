# -*- coding: utf-8 -*-

import argparse
import contextlib
import io
import json
import sys
from collections import OrderedDict


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Tune dividend scorer prior_blend with walk-forward strategy scoring",
    )
    parser.add_argument("strategy_file", help="strategy file path")
    parser.add_argument("--cash", type=int, default=1000000, help="initial cash")
    parser.add_argument("--db-path", type=str, default=None, help="override dividend scorer cache db path")
    parser.add_argument("--bundle-path", type=str, default=None, help="override RQAlpha bundle path")
    parser.add_argument(
        "--window",
        "-w",
        type=str,
        default=None,
        help="window spec, e.g. 37 or 35-37",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default="0.7,0.8,0.9",
        help="comma-separated prior_blend candidates",
    )
    parser.add_argument("--verbose", action="store_true", help="show underlying rolling backtest logs")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser


def parse_candidates(raw_value):
    candidates = []
    seen = set()
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        candidate = float(stripped)
        if candidate < 0.0 or candidate > 1.0:
            raise ValueError("prior_blend candidate must be within [0, 1]: {}".format(stripped))
        if candidate not in seen:
            seen.add(candidate)
            candidates.append(candidate)
    if not candidates:
        raise ValueError("at least one prior_blend candidate is required")
    return candidates


def _load_strategy_scorer():
    from skyeye import strategy_scorer
    return strategy_scorer


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _format_metric(value, pattern="{:.2f}", scale=1.0):
    if value == "N/A":
        return "N/A"
    return pattern.format(float(value) * scale)


def _build_result_row(prior_blend, window_results, summary):
    market_env = summary["market_env"]
    core_indicators = summary["core_indicators"]
    return OrderedDict([
        ("prior_blend", float(prior_blend)),
        ("window_count", int(len(window_results))),
        ("composite", float(summary["composite"])),
        ("stability", float(summary["stability"])),
        ("bear_market", market_env.get("熊市")),
        ("sideways_market", market_env.get("震荡")),
        ("bull_market", market_env.get("牛市")),
        ("annualized_returns", float(core_indicators["annualized_returns"])),
        ("max_drawdown", float(core_indicators["max_drawdown"])),
        ("sharpe", float(core_indicators["sharpe"])),
        ("win_rate", float(core_indicators["win_rate"])),
        ("overfit_flags", list(summary["overfit_flags"])),
    ])


def print_summary_table(rows):
    headers = [
        "prior_blend",
        "windows",
        "composite",
        "stability",
        "bear",
        "sideways",
        "bull",
        "ann_ret",
        "max_dd",
        "sharpe",
        "win_rate",
    ]
    rendered_rows = []
    for row in rows:
        rendered_rows.append([
            "{:.2f}".format(row["prior_blend"]),
            str(row["window_count"]),
            _format_metric(row["composite"]),
            _format_metric(row["stability"]),
            _format_metric(row["bear_market"]),
            _format_metric(row["sideways_market"]),
            _format_metric(row["bull_market"]),
            _format_metric(row["annualized_returns"], "{:.1%}"),
            _format_metric(abs(row["max_drawdown"]), "{:.1%}"),
            _format_metric(row["sharpe"]),
            _format_metric(row["win_rate"], "{:.1%}"),
        ])
    widths = [len(header) for header in headers]
    for rendered in rendered_rows:
        for idx, cell in enumerate(rendered):
            widths[idx] = max(widths[idx], len(cell))

    def render_line(values):
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    print(render_line(headers))
    print(render_line(["-" * width for width in widths]))
    for rendered in rendered_rows:
        print(render_line(rendered))


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not os.path.isfile(args.strategy_file):
        parser.error("strategy file does not exist: {}".format(args.strategy_file))

    try:
        candidates = parse_candidates(args.candidates)
    except ValueError as exc:
        parser.error(str(exc))

    strategy_scorer = _load_strategy_scorer()
    selected_indices = None
    if args.window:
        selected_indices = strategy_scorer.parse_window_arg(args.window)
    benchmark_quarterly_returns = strategy_scorer.get_benchmark_quarterly_returns()

    rows = []
    for prior_blend in candidates:
        mod_config = {
            "prior_blend": prior_blend,
        }
        if args.db_path:
            mod_config["db_path"] = args.db_path
        if args.bundle_path:
            mod_config["bundle_path"] = args.bundle_path
        if args.verbose:
            window_results = strategy_scorer.run_rolling_backtests(
                args.strategy_file,
                args.cash,
                selected_indices=selected_indices,
                extra_mods=["dividend_scorer"],
                mod_configs={
                    "dividend_scorer": mod_config,
                },
            )
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                window_results = strategy_scorer.run_rolling_backtests(
                    args.strategy_file,
                    args.cash,
                    selected_indices=selected_indices,
                    extra_mods=["dividend_scorer"],
                    mod_configs={
                        "dividend_scorer": mod_config,
                    },
                )
        if not window_results:
            continue
        summary = strategy_scorer.summarize_window_results(
            window_results,
            benchmark_quarterly_returns=benchmark_quarterly_returns,
        )
        rows.append(_build_result_row(prior_blend, window_results, summary))

    if not rows:
        raise RuntimeError("no successful backtest windows were produced")

    rows.sort(
        key=lambda item: (
            item["composite"],
            item["stability"],
            item["bear_market"] if isinstance(item["bear_market"], (int, float)) else float("-inf"),
        ),
        reverse=True,
    )

    output = OrderedDict([
        ("strategy_file", os.path.abspath(args.strategy_file)),
        ("window", args.window),
        ("cash", int(args.cash)),
        ("candidates", rows),
        ("best", rows[0]),
    ])

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default))
        return

    print("策略文件: {}".format(os.path.abspath(args.strategy_file)))
    print("窗口范围: {}".format(args.window or "all"))
    print("候选 prior_blend: {}".format(", ".join("{:.2f}".format(value) for value in candidates)))
    print()
    print_summary_table(rows)
    print()
    print(
        "最佳 prior_blend: {prior_blend:.2f}  composite={composite:.2f}  stability={stability:.2f}".format(
            **rows[0]
        )
    )
    if rows[0]["overfit_flags"]:
        print("过拟合提示: {}".format("；".join(rows[0]["overfit_flags"])))


if __name__ == "__main__":
    main(sys.argv[1:])
