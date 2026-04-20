"""TX1 autoresearch CLI 入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from skyeye.products.tx1.autoresearch.loop import run_loop


def build_parser() -> argparse.ArgumentParser:
    """构造 autoresearch CLI 的参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Run TX1 autoresearch loop (requires a clean dedicated git worktree)"
    )
    parser.add_argument("--run-tag", required=True, help="本次 autoresearch run 的唯一标签")
    parser.add_argument(
        "--runs-root",
        default="skyeye/artifacts/experiments/tx1_autoresearch",
        help="autoresearch 运行目录根路径",
    )
    parser.add_argument("--max-experiments", type=int, default=0, help="最多执行多少轮实验，0 表示不设上限")
    parser.add_argument("--smoke-max-folds", type=int, default=1, help="smoke 阶段最多运行的 fold 数")
    parser.add_argument("--full-max-folds", type=int, default=None, help="full 阶段最多运行的 fold 数")
    parser.add_argument("--build-raw-df", action="store_true", help="按研究侧默认链路构建 raw_df 并运行 baseline")
    parser.add_argument("--evaluate-current", action="store_true", help="对当前工作区脏改动执行一次 smoke/full 候选评估")
    parser.add_argument("--universe-size", type=int, default=300, help="研究侧 raw_df 构建使用的 liquid universe 大小")
    parser.add_argument("--start-date", default=None, help="可选，研究样本起始日")
    parser.add_argument("--end-date", default=None, help="可选，研究样本截止日")
    parser.add_argument("--variant-name", default="baseline_5f", help="候选评估使用的 variant 名称")
    parser.add_argument("--model-kind", choices=["linear", "tree", "lgbm"], default="lgbm", help="候选评估使用的模型类型")
    parser.add_argument("--label-transform", choices=["raw", "rank", "quantile"], default="rank", help="标签变换方式")
    parser.add_argument("--horizon-days", type=int, default=20, help="标签 horizon 天数")
    return parser


def main(argv=None) -> int:
    """解析 CLI 参数并启动 autoresearch 主循环。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_loop(
        run_tag=args.run_tag,
        runs_root=Path(args.runs_root),
        build_raw_df_for_run=bool(args.build_raw_df),
        evaluate_current=bool(args.evaluate_current),
        universe_size=int(args.universe_size),
        start_date=args.start_date,
        end_date=args.end_date,
        variant_name=args.variant_name,
        model_kind=args.model_kind,
        label_transform=args.label_transform,
        horizon_days=int(args.horizon_days),
        max_experiments=args.max_experiments,
        smoke_max_folds=args.smoke_max_folds,
        full_max_folds=args.full_max_folds,
    )
    status = str(result.get("status") or "ok")
    if status in {"invalid", "crash"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
