#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TX1 4 小时 focused autoresearch 启动脚本。"""

from __future__ import annotations

import argparse

from skyeye.products.tx1.autoresearch.focused_search import (
    build_default_run_tag,
    run_liquidity_focus_search,
)


def build_parser() -> argparse.ArgumentParser:
    """构造 4 小时 focused runner 的命令行参数。"""
    parser = argparse.ArgumentParser(description="Run TX1 focused autoresearch for up to 4 hours.")
    parser.add_argument("--run-tag", default=None, help="本次运行标签；默认自动按时间生成。")
    parser.add_argument(
        "--runs-root",
        default="skyeye/artifacts/experiments/tx1_autoresearch",
        help="autoresearch 产物根目录。",
    )
    parser.add_argument("--hours", type=float, default=4.0, help="最长运行时长（小时）。")
    parser.add_argument("--universe-size", type=int, default=300, help="研究样本 liquid universe 大小。")
    parser.add_argument("--start-date", default=None, help="可选，研究样本起始日。")
    parser.add_argument("--end-date", default=None, help="可选，研究样本截止日。")
    parser.add_argument("--model-kind", choices=["linear", "tree", "lgbm"], default="lgbm")
    parser.add_argument("--label-transform", choices=["raw", "rank", "quantile"], default="rank")
    parser.add_argument("--horizon-days", type=int, default=20, help="标签 horizon 天数。")
    parser.add_argument("--smoke-max-folds", type=int, default=1, help="smoke 阶段最多运行的 fold 数。")
    parser.add_argument("--full-max-folds", type=int, default=5, help="full 阶段最多运行的 fold 数。")
    parser.add_argument(
        "--max-stabilization-rounds",
        type=int,
        default=3,
        help="phase-1 liquidity_plus 稳化搜索最多扩表轮数。",
    )
    parser.add_argument(
        "--max-replay-rounds",
        type=int,
        default=4,
        help="phase-2 rolling_score replay 搜索最多扩表轮数。",
    )
    parser.add_argument(
        "--replay-cash",
        type=float,
        default=1000000.0,
        help="phase-2 rolling_score replay 使用的初始资金。",
    )
    parser.add_argument(
        "--replay-window-indices",
        default=None,
        help="可选，逗号分隔的 rolling_score 窗口编号列表，默认跑全窗口。",
    )
    return parser


def main(argv=None) -> int:
    """解析参数并启动 focused runner。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tag = args.run_tag or build_default_run_tag()
    replay_selected_indices = None
    if args.replay_window_indices:
        replay_selected_indices = [
            int(token.strip())
            for token in str(args.replay_window_indices).split(",")
            if token.strip()
        ]
    result = run_liquidity_focus_search(
        run_tag=run_tag,
        runs_root=args.runs_root,
        universe_size=int(args.universe_size),
        start_date=args.start_date,
        end_date=args.end_date,
        model_kind=args.model_kind,
        label_transform=args.label_transform,
        horizon_days=int(args.horizon_days),
        max_runtime_hours=float(args.hours),
        smoke_max_folds=int(args.smoke_max_folds),
        full_max_folds=int(args.full_max_folds),
        max_stabilization_rounds=int(args.max_stabilization_rounds),
        max_replay_rounds=int(args.max_replay_rounds),
        replay_cash=float(args.replay_cash),
        replay_selected_indices=replay_selected_indices,
    )
    print("run_tag={}".format(result["run_tag"]))
    print("status={}".format(result["status"]))
    print("run_root={}".format(result["run_root"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
