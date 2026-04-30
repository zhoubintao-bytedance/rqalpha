import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from skyeye.products.ax1.package_io import load_package, save_package
from skyeye.products.ax1.persistence import load_experiment, save_experiment


def test_save_and_load_experiment_round_trips_jsonable_payload(tmp_path):
    result = {
        "experiment_name": "ax1_lgbm_demo",
        "config": {"profile": "lgbm", "lookback_days": np.int64(20)},
        "data_range": {
            "start": pd.Timestamp("2025-01-01"),
            "end": pd.Timestamp("2025-01-31"),
        },
        "artifact_dir": Path("skyeye/artifacts/ax1/lgbm"),
        "evaluation": pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "portfolio_return": [np.float64(0.01), np.float64(-0.002)],
                "turnover": [0.1, 0.2],
            }
        ),
    }

    experiment_dir = save_experiment(
        result,
        tmp_path / "experiments" / "ax1_lgbm_demo",
        experiment_name="ax1_lgbm_demo",
    )
    loaded = load_experiment(experiment_dir)

    assert experiment_dir == tmp_path / "experiments" / "ax1_lgbm_demo"
    assert (experiment_dir / "experiment.json").is_file()
    assert loaded["experiment_name"] == "ax1_lgbm_demo"
    assert loaded["config"]["lookback_days"] == 20
    assert loaded["data_range"]["start"] == "2025-01-01T00:00:00"
    assert loaded["artifact_dir"] == "skyeye/artifacts/ax1/lgbm"
    assert loaded["evaluation"] == [
        {
            "date": "2025-01-02T00:00:00",
            "portfolio_return": 0.01,
            "turnover": 0.1,
        },
        {
            "date": "2025-01-03T00:00:00",
            "portfolio_return": -0.002,
            "turnover": 0.2,
        },
    ]


def test_save_experiment_writes_frozen_fold_artifacts(tmp_path):
    result = {
        "experiment_name": "ax1_lgbm_demo",
        "training_summary": {
            "model_kind": "lgbm_multi_target",
            "feature_columns": ["feature_a", "feature_b"],
            "fold_results": [
                {
                    "fold_id": 0,
                    "train_end": "2024-03-29",
                    "val_start": "2024-04-01",
                    "val_end": "2024-04-30",
                    "test_start": "2024-05-01",
                    "test_end": "2024-05-31",
                    "predictions_df": pd.DataFrame(
                        {
                            "date": pd.to_datetime(["2024-05-06", "2024-05-06"]),
                            "order_book_id": ["510300.XSHG", "510500.XSHG"],
                            "expected_relative_net_return_10d": [0.01, 0.02],
                            "fold_id": [0, 0],
                        }
                    ),
                    "labels": pd.DataFrame(
                        {
                            "date": pd.to_datetime(["2024-05-06", "2024-05-06"]),
                            "order_book_id": ["510300.XSHG", "510500.XSHG"],
                            "label_net_return_10d": [0.011, 0.019],
                            "fold_id": [0, 0],
                        }
                    ),
                    "predictor_bundle": {
                        "schema_version": 1,
                        "product": "ax1",
                        "predictor_kind": "lgbm_multi_target",
                        "feature_columns": ["feature_a", "feature_b"],
                    },
                    "preprocessor_bundle": {
                        "kind": "feature_preprocessor",
                        "feature_columns": ["feature_a", "feature_b"],
                    },
                }
            ],
        },
        "target_weights": [
            {
                "date": pd.Timestamp("2024-05-06"),
                "order_book_id": "510300.XSHG",
                "target_weight": 0.40,
                "fold_id": 0,
            },
            {
                "date": pd.Timestamp("2024-05-06"),
                "order_book_id": "510500.XSHG",
                "target_weight": 0.35,
                "fold_id": 0,
            },
        ],
        "orders": [
            {
                "date": pd.Timestamp("2024-05-06"),
                "order_book_id": "510300.XSHG",
                "order_weight": 0.40,
                "fold_id": 0,
            }
        ],
    }

    experiment_dir = save_experiment(
        result,
        tmp_path / "experiments" / "ax1_lgbm_demo",
        experiment_name="ax1_lgbm_demo",
    )

    fold_dir = experiment_dir / "folds" / "fold_000"
    loaded = load_experiment(experiment_dir)
    fold_meta = loaded["folds"][0]
    persisted_fold = loaded["training_summary"]["fold_results"][0]

    assert loaded["artifact_schema_version"] == 1
    assert fold_meta["fold_id"] == 0
    assert fold_meta["model_bundle_ref"] == "folds/fold_000/model_bundle.json"
    assert persisted_fold["model_bundle_ref"] == "folds/fold_000/model_bundle.json"
    assert persisted_fold["preprocessor_bundle_ref"] == "folds/fold_000/preprocessor_bundle.json"
    assert persisted_fold["predictions_ref"] == "folds/fold_000/predictions.parquet"
    assert persisted_fold["labels_ref"] == "folds/fold_000/labels.parquet"
    assert (fold_dir / "predictions.parquet").is_file()
    assert (fold_dir / "labels.parquet").is_file()
    assert (fold_dir / "weights.parquet").is_file()
    assert (fold_dir / "orders.parquet").is_file()
    assert (fold_dir / "model_bundle.json").is_file()
    assert (fold_dir / "preprocessor_bundle.json").is_file()
    assert (fold_dir / "fold_metadata.json").is_file()
    assert pd.read_parquet(fold_dir / "predictions.parquet")["expected_relative_net_return_10d"].tolist() == [0.01, 0.02]
    assert pd.read_parquet(fold_dir / "weights.parquet")["target_weight"].tolist() == [0.40, 0.35]
    assert json.loads((fold_dir / "model_bundle.json").read_text(encoding="utf-8"))["product"] == "ax1"


def test_save_and_load_package_writes_manifest_contract(tmp_path):
    experiment = {
        "experiment_name": "ax1_lgbm_demo",
        "output_dir": str(tmp_path / "experiments" / "ax1_lgbm_demo"),
        "config": {"profile": "lgbm"},
        "factor_schema": {"features": ["quality_score"]},
        "model_schema": {"kind": "lgbm_multi_target"},
        "view_fusion": {"kind": "noop"},
        "risk_model": {"kind": "none"},
        "optimizer": {"kind": "not_implemented"},
        "constraints": {"max_weight": 0.1},
        "execution": {"kind": "noop_smoother"},
        "implementation_status": {"constraints": "declared_not_enforced"},
        "evaluation": {"total_return": 0.0},
        "tradable_outcome": {"schema_version": 1, "mean_net_return": 0.001, "max_net_drawdown": 0.02},
        "alpha_transfer_ledger": {"schema_version": 1, "summary": {"executable_retention_ratio": 0.75}},
        "confidence_diagnostic": {"schema_version": 1, "outcome_column": "tradable_net_success"},
        "calibration_bundle": {"summary": {"oos_rows": 100}, "bucket_stats": []},
        "gate_summary": {"gate_level": "canary_live", "passed": True},
        "data_range": {"start": "2025-01-01", "end": "2025-01-31"},
    }

    package_root = save_package(
        experiment,
        tmp_path / "packages",
        package_id="ax1_pkg_lgbm_demo",
    )
    loaded_by_path = load_package(package_root)
    loaded_by_id = load_package("ax1_pkg_lgbm_demo", packages_root=tmp_path / "packages")

    manifest = loaded_by_path["manifest"]
    assert package_root == tmp_path / "packages" / "ax1_pkg_lgbm_demo"
    assert (package_root / "manifest.json").is_file()
    assert (package_root / "experiment_ref.json").is_file()
    assert loaded_by_id == loaded_by_path

    expected_manifest_keys = {
        "package_id",
        "product",
        "schema_version",
        "created_at",
        "source_experiment",
        "config",
        "factor_schema",
        "model_schema",
        "view_fusion",
        "risk_model",
        "optimizer",
        "constraints",
        "execution",
        "implementation_status",
        "evaluation",
        "tradable_outcome",
        "alpha_transfer_ledger",
        "confidence_diagnostic",
        "calibration_bundle",
        "gate_summary",
        "data_range",
        "status",
    }
    assert expected_manifest_keys <= manifest.keys()
    assert manifest["package_id"] == "ax1_pkg_lgbm_demo"
    assert manifest["product"] == "ax1"
    assert manifest["source_experiment"] == "ax1_lgbm_demo"
    assert manifest["config"] == {"profile": "lgbm"}
    assert manifest["factor_schema"] == {"features": ["quality_score"]}
    assert manifest["view_fusion"] == {"kind": "noop"}
    assert manifest["execution"] == {"kind": "noop_smoother"}
    assert manifest["implementation_status"] == {"constraints": "declared_not_enforced"}
    assert manifest["tradable_outcome"]["mean_net_return"] == pytest.approx(0.001)
    assert manifest["alpha_transfer_ledger"]["summary"]["executable_retention_ratio"] == pytest.approx(0.75)
    assert manifest["confidence_diagnostic"]["outcome_column"] == "tradable_net_success"
    assert manifest["calibration_bundle"]["summary"]["oos_rows"] == 100
    assert manifest["gate_summary"] == {"gate_level": "canary_live", "passed": True}
    assert manifest["status"] == "research_package"
    assert loaded_by_path["experiment_ref"]["experiment_name"] == "ax1_lgbm_demo"
    assert loaded_by_path["experiment_ref"]["output_dir"] == experiment["output_dir"]


def test_package_manifest_references_frozen_artifact_components(tmp_path):
    experiment = {
        "experiment_name": "ax1_lgbm_demo",
        "artifact_schema_version": 1,
        "folds": [
            {
                "fold_id": 0,
                "path": "folds/fold_000",
                "model_bundle_ref": "folds/fold_000/model_bundle.json",
                "preprocessor_bundle_ref": "folds/fold_000/preprocessor_bundle.json",
                "predictions_ref": "folds/fold_000/predictions.parquet",
                "weights_ref": "folds/fold_000/weights.parquet",
            }
        ],
        "calibration_bundle": {"summary": {"oos_rows": 10}, "bucket_stats": []},
        "gate_summary": {"gate_level": "canary_live", "passed": True},
    }

    package_root = save_package(
        experiment,
        tmp_path / "packages",
        package_id="ax1_pkg_lgbm_demo",
    )
    manifest = load_package(package_root)["manifest"]

    assert manifest["artifact_schema_version"] == 1
    assert manifest["fold_artifacts"][0]["path"] == "folds/fold_000"
    assert manifest["model_bundle_refs"] == ["folds/fold_000/model_bundle.json"]
    assert manifest["preprocessor_bundle_refs"] == ["folds/fold_000/preprocessor_bundle.json"]
    assert manifest["calibration_bundle"]["path"] == "calibration_bundle.json"


def test_load_package_validates_model_and_preprocessor_feature_columns(tmp_path):
    experiment_dir = tmp_path / "experiments" / "ax1_lgbm_demo"
    fold_dir = experiment_dir / "folds" / "fold_000"
    fold_dir.mkdir(parents=True)
    (fold_dir / "model_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "product": "ax1",
                "predictor_kind": "lgbm_multi_target",
                "feature_columns": ["feature_a"],
                "model_bundle": {"model_kind": "multi_head", "feature_columns": ["feature_a"], "state": {"heads": {}}},
                "preprocessor_bundle": {"feature_columns": ["feature_a"]},
            }
        ),
        encoding="utf-8",
    )
    (fold_dir / "preprocessor_bundle.json").write_text(
        json.dumps(
            {
                "kind": "feature_preprocessor",
                "feature_columns": ["feature_b"],
                "required_columns": ["date", "order_book_id", "feature_b"],
            }
        ),
        encoding="utf-8",
    )
    experiment = {
        "experiment_name": "ax1_lgbm_demo",
        "output_dir": str(experiment_dir),
        "artifact_schema_version": 1,
        "folds": [
            {
                "fold_id": 0,
                "path": "folds/fold_000",
                "model_bundle_ref": "folds/fold_000/model_bundle.json",
                "preprocessor_bundle_ref": "folds/fold_000/preprocessor_bundle.json",
            }
        ],
        "gate_summary": {"gate_level": "canary_live", "passed": True},
    }
    save_package(experiment, tmp_path / "packages", package_id="ax1_pkg_mismatch")

    with pytest.raises(ValueError, match="feature_columns.*mismatch"):
        load_package("ax1_pkg_mismatch", packages_root=tmp_path / "packages", validate_artifacts=True)
