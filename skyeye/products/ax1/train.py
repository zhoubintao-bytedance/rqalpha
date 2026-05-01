"""Lightweight AX1 training CLI backed by the shared SkyEye data facade."""

from __future__ import annotations

import argparse
from pathlib import Path

from skyeye.data import DataFacade
from skyeye.products.ax1.config import load_profile
from skyeye.products.ax1.data_builder import AX1TrainingDataBuilder, AX1TrainingDataRequest
from skyeye.products.ax1.run_experiment import run_experiment


PROFILE_ALIASES = {
    "default": Path("skyeye/products/ax1/profiles/default.yaml"),
    "personal_etf_core": Path("skyeye/products/ax1/profiles/personal_etf_core.yaml"),
    "lgbm_multi_target": Path("skyeye/products/ax1/profiles/lgbm_multi_target.yaml"),
}
DEFAULT_OUTPUT_ROOT = Path("skyeye/artifacts/experiments/ax1_train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AX1 from the shared SkyEye data layer")
    parser.add_argument("--start-date", required=True, help="Training data start date, e.g. 2018-01-01")
    parser.add_argument("--end-date", required=True, help="Training data end date, e.g. 2026-04-30")
    parser.add_argument(
        "--profile",
        default="personal_etf_core",
        help="AX1 profile alias or YAML path: default, personal_etf_core, lgbm_multi_target",
    )
    parser.add_argument("--run-tag", required=True, help="Unique run tag used as experiment name and output subdir")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root directory for AX1 train artifacts; final output is output-root/run-tag",
    )
    return parser


def resolve_profile_path(profile: str | Path) -> Path:
    value = str(profile)
    return PROFILE_ALIASES.get(value, Path(value))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile_path = resolve_profile_path(args.profile)
    profile_config = load_profile(profile_path)
    facade = DataFacade()
    raw_df = AX1TrainingDataBuilder(data_facade=facade).build(
        AX1TrainingDataRequest(
            profile_config=profile_config,
            start_date=args.start_date,
            end_date=args.end_date,
        )
    )
    output_dir = Path(args.output_root) / str(args.run_tag)
    result = run_experiment(
        profile_path=profile_path,
        output_dir=output_dir,
        raw_df=raw_df,
        raw_csv=None,
        experiment_name=str(args.run_tag),
        data_provider=facade,
    )
    print(result["output_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
