# -*- coding: utf-8 -*-
"""Promote AX1 experiment artifact to a research package manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


def promote_package(
    *,
    experiment_dir: str | Path,
    packages_root: str | Path,
    package_id: str | None = None,
    require_gate: bool = True,
    skip_gate: bool = False,
    gate_level: str = "canary_live",
) -> Path:
    """把 AX1 experiment artifact 转成 research package。"""
    from skyeye.products.ax1.package_io import save_package
    from skyeye.products.ax1.persistence import load_experiment

    experiment = load_experiment(experiment_dir)
    if skip_gate:
        require_gate = False
    if require_gate:
        experiment["gate_summary"] = _resolve_gate_summary(experiment, gate_level=gate_level)
        if not experiment["gate_summary"].get("passed", False):
            failed = experiment["gate_summary"].get("failed_checks") or []
            raise ValueError("promotion gate failed: failed_checks=[{}]".format(", ".join(map(str, failed)) or "unknown"))
    experiment["gate_required"] = bool(require_gate)
    return save_package(experiment, packages_root=packages_root, package_id=package_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote AX1 experiment to research package")
    parser.add_argument("--experiment-dir", required=True, help="AX1 experiment directory")
    parser.add_argument("--packages-root", required=True, help="AX1 packages root")
    parser.add_argument("--package-id", default=None, help="Optional package id")
    parser.add_argument("--require-gate", action="store_true", default=True, help="Require promotion gate before packaging (default)")
    parser.add_argument("--skip-gate", action="store_true", help="Create a research/dev package without requiring promotion gate")
    parser.add_argument(
        "--gate-level",
        choices=("canary_live", "default_live"),
        default="canary_live",
        help="Promotion gate level used unless --skip-gate is set",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_dir = promote_package(
        experiment_dir=args.experiment_dir,
        packages_root=args.packages_root,
        package_id=args.package_id,
        require_gate=args.require_gate,
        skip_gate=args.skip_gate,
        gate_level=args.gate_level,
    )
    print(package_dir)
    return 0


def _resolve_gate_summary(experiment: dict, *, gate_level: str) -> dict:
    gate_summary = experiment.get("gate_summary")
    if isinstance(gate_summary, dict) and gate_summary.get("gate_level") == gate_level:
        return gate_summary

    from skyeye.products.ax1.calibration import build_calibration_bundle
    from skyeye.products.ax1.promotion import evaluate_promotion_gate

    if not isinstance(experiment.get("calibration_bundle"), dict) or not experiment["calibration_bundle"]:
        experiment["calibration_bundle"] = build_calibration_bundle(experiment, bucket_count=10)
    return evaluate_promotion_gate(experiment, gate_level=gate_level)


if __name__ == "__main__":
    raise SystemExit(main())
