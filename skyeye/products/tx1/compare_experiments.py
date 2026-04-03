# -*- coding: utf-8 -*-
"""CLI for TX1 experiment comparison."""

import argparse
import json
from pathlib import Path

from skyeye.products.tx1.robustness import compare_experiments, render_comparison_report


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compare TX1 experiment dirs or artifact lines against a baseline"
    )
    parser.add_argument(
        "experiments",
        nargs="+",
        help=(
            "Experiment dirs, experiment.json files, full artifact refs "
            "(tx1.rolling_score@baseline_lgbm), or bare artifact lines (baseline_lgbm). "
            "If only one experiment is given and --baseline is omitted, baseline_lgbm is used."
        ),
    )
    parser.add_argument(
        "--baseline",
        help=(
            "Optional baseline reference. If omitted, the first positional argument is used "
            "as baseline unless only one experiment is supplied."
        ),
    )
    parser.add_argument(
        "--artifacts-root",
        default="skyeye/artifacts/experiments/tx1",
        help="Root directory for resolving bare artifact lines.",
    )
    parser.add_argument(
        "--metric-key",
        default="rank_ic_mean",
        help="Primary fold metric used for stability and fold comparison.",
    )
    parser.add_argument(
        "--return-metric",
        choices=["net_mean_return", "mean_return"],
        default="net_mean_return",
        help="Fold return metric used for uplift concentration analysis.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the text report.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional path to write the structured comparison payload as JSON.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    result = compare_experiments(
        args.experiments,
        baseline_ref=args.baseline,
        artifacts_root=args.artifacts_root,
        metric_key=args.metric_key,
        return_metric=args.return_metric,
    )
    report = render_comparison_report(result)
    print(report, end="")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    if args.json_output:
        json_path = Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True),
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
