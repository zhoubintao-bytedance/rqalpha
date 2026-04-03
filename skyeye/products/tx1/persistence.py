# -*- coding: utf-8 -*-
"""
TX1 Experiment Persistence Module

Provides disk persistence for TX1 experiments with JSON metadata
and Parquet DataFrame storage for efficient round-trip serialization.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExperimentPaths:
    """Paths for experiment artifacts."""

    root: Path
    metadata: Path
    folds_dir: Path

    @classmethod
    def from_root(cls, output_dir: str) -> "ExperimentPaths":
        root = Path(output_dir)
        return cls(
            root=root,
            metadata=root / "experiment.json",
            folds_dir=root / "folds",
        )

    def fold_dir(self, fold_index: int) -> Path:
        return self.folds_dir / f"fold_{fold_index:03d}"


class ExperimentStore:
    """Manages experiment persistence to disk.

    Directory structure:
        output_dir/
            experiment.json          # Metadata, config, aggregate metrics
            folds/
                fold_001/
                    predictions.parquet
                    weights.parquet
                    portfolio_returns.parquet
                    fold_metadata.json
                fold_002/
                    ...
    """

    def __init__(self, output_dir: str):
        self.paths = ExperimentPaths.from_root(output_dir)

    def save(
        self,
        result: dict,
        config: dict | None = None,
        experiment_name: str | None = None,
    ) -> str:
        """Save complete experiment to disk.

        Args:
            result: Experiment result dict from ExperimentRunner.run()
            config: Experiment configuration dict
            experiment_name: Optional experiment identifier

        Returns:
            Absolute path to experiment directory
        """
        self._ensure_directories()

        # Build metadata
        metadata = self._build_metadata(result, config, experiment_name)

        # Save fold artifacts
        for fold_result in result.get("fold_results", []):
            self._save_fold(fold_result)

        # Add fold metadata references
        metadata["folds"] = [
            {
                "index": fr["fold_index"],
                "path": str(self.paths.fold_dir(fr["fold_index"]).relative_to(self.paths.root)),
                "row_counts": fr.get("row_counts", {}),
                "date_range": fr.get("date_range", {}),
            }
            for fr in result.get("fold_results", [])
        ]

        # Save main metadata
        with open(self.paths.metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        return str(self.paths.root.absolute())

    def load(self) -> dict:
        """Load complete experiment from disk.

        Returns:
            Dict matching ExperimentRunner output format with added
            'output_dir' and loaded DataFrames for each fold.
        """
        with open(self.paths.metadata, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        fold_results = []
        for fold_meta in metadata.get("folds", []):
            fold_idx = fold_meta["index"]
            fold_dir = self.paths.fold_dir(fold_idx)

            fold_result = {
                "fold_index": fold_idx,
                "row_counts": fold_meta.get("row_counts", {}),
                "date_range": fold_meta.get("date_range", {}),
            }

            # Load DataFrames if they exist
            predictions_path = fold_dir / "predictions.parquet"
            if predictions_path.exists():
                fold_result["predictions_df"] = pd.read_parquet(predictions_path)

            weights_path = fold_dir / "weights.parquet"
            if weights_path.exists():
                fold_result["weights_df"] = pd.read_parquet(weights_path)

            returns_path = fold_dir / "portfolio_returns.parquet"
            if returns_path.exists():
                fold_result["portfolio_returns_df"] = pd.read_parquet(returns_path)

            # Load fold metadata
            fold_metadata_path = fold_dir / "fold_metadata.json"
            if fold_metadata_path.exists():
                with open(fold_metadata_path, "r", encoding="utf-8") as f:
                    fold_metadata = json.load(f)
                fold_result["prediction_metrics"] = fold_metadata.get("prediction_metrics", {})
                fold_result["validation_metrics"] = fold_metadata.get("validation_metrics", {})
                fold_result["portfolio_metrics"] = fold_metadata.get("portfolio_metrics", {})
                fold_result["stratified_metrics"] = fold_metadata.get("stratified_metrics", {})
                fold_result["head_metrics"] = fold_metadata.get("head_metrics", {})
                fold_result["head_validation_metrics"] = fold_metadata.get("head_validation_metrics", {})
                fold_result["prediction_blend_metrics"] = fold_metadata.get("prediction_blend_metrics", {})
                fold_result["selection_metrics"] = fold_metadata.get("selection_metrics", {})
                fold_result["model_heads"] = fold_metadata.get("model_heads", [])

            fold_results.append(fold_result)

        return {
            "model_kind": metadata.get("model_kind"),
            "model_heads": metadata.get("model_heads", ["return"] if metadata.get("model_kind") else []),
            "prediction_columns": metadata.get("prediction_columns", ["prediction"]),
            "config": metadata.get("config"),
            "experiment_name": metadata.get("experiment_name"),
            "created_at": metadata.get("created_at"),
            "fold_results": fold_results,
            "aggregate_metrics": metadata.get("aggregate_metrics", {}),
            "output_dir": str(self.paths.root.absolute()),
        }

    def _ensure_directories(self) -> None:
        """Create necessary directory structure."""
        self.paths.root.mkdir(parents=True, exist_ok=True)
        self.paths.folds_dir.mkdir(exist_ok=True)

    def _build_metadata(
        self,
        result: dict,
        config: dict | None,
        experiment_name: str | None,
    ) -> dict:
        """Build experiment metadata dict."""
        return {
            "version": "1.1",
            "created_at": datetime.now().isoformat(),
            "experiment_name": experiment_name or f"tx1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_kind": result.get("model_kind"),
            "model_heads": result.get("model_heads", []),
            "prediction_columns": result.get("prediction_columns", ["prediction"]),
            "config": config or {},
            "aggregate_metrics": result.get("aggregate_metrics", {}),
            "num_folds": len(result.get("fold_results", [])),
        }

    def _save_fold(self, fold_result: dict) -> None:
        """Save single fold artifacts."""
        fold_idx = fold_result["fold_index"]
        fold_dir = self.paths.fold_dir(fold_idx)
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions DataFrame
        if "predictions_df" in fold_result:
            fold_result["predictions_df"].to_parquet(
                fold_dir / "predictions.parquet", index=False
            )

        # Save weights DataFrame
        if "weights_df" in fold_result:
            fold_result["weights_df"].to_parquet(
                fold_dir / "weights.parquet", index=False
            )

        # Save portfolio returns DataFrame
        if "portfolio_returns_df" in fold_result:
            fold_result["portfolio_returns_df"].to_parquet(
                fold_dir / "portfolio_returns.parquet", index=False
            )

        # Save fold metadata
        fold_metadata = {
            "fold_index": fold_idx,
            "prediction_metrics": fold_result.get("prediction_metrics", {}),
            "validation_metrics": fold_result.get("validation_metrics", {}),
            "portfolio_metrics": fold_result.get("portfolio_metrics", {}),
            "stratified_metrics": fold_result.get("stratified_metrics", {}),
            "head_metrics": fold_result.get("head_metrics", {}),
            "head_validation_metrics": fold_result.get("head_validation_metrics", {}),
            "prediction_blend_metrics": fold_result.get("prediction_blend_metrics", {}),
            "selection_metrics": fold_result.get("selection_metrics", {}),
            "row_counts": fold_result.get("row_counts", {}),
            "date_range": fold_result.get("date_range", {}),
            "model_heads": fold_result.get("model_heads", []),
        }
        with open(fold_dir / "fold_metadata.json", "w", encoding="utf-8") as f:
            json.dump(fold_metadata, f, indent=2, default=str)


def save_experiment(
    result: dict,
    output_dir: str,
    config: dict | None = None,
    experiment_name: str | None = None,
) -> str:
    """Convenience function to save experiment.

    Args:
        result: Experiment result from ExperimentRunner.run()
        output_dir: Directory to save experiment
        config: Optional configuration dict
        experiment_name: Optional experiment name

    Returns:
        Absolute path to experiment directory
    """
    store = ExperimentStore(output_dir)
    return store.save(result, config, experiment_name)


def load_experiment(output_dir: str) -> dict:
    """Convenience function to load experiment.

    Args:
        output_dir: Experiment directory path

    Returns:
        Experiment result dict with loaded DataFrames
    """
    store = ExperimentStore(output_dir)
    return store.load()
