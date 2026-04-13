from skyeye.products.tx1.live_advisor.promotion import evaluate_promotion_gate
from skyeye.products.tx1.live_advisor.promotion import promote_experiment_to_live_package


def test_evaluate_promotion_gate_marks_canary_when_stability_is_weak():
    """验证 promotion gate 会把方向有效但稳定性不足的实验标记成 canary。"""
    calibration_bundle = {
        "bucket_stats": [
            {"bucket_id": "b0", "sample_count": 420},
            {"bucket_id": "b1", "sample_count": 410},
        ]
    }
    experiment_result = {
        "num_folds": 14,
        "aggregate_metrics": {
            "prediction": {
                "rank_ic_mean": 0.055,
                "top_bucket_spread_mean": 0.011,
            },
            "portfolio": {
                "net_mean_return": 0.0003,
                "max_drawdown": 0.055,
                "mean_turnover": 0.02,
            },
            "robustness": {
                "stability": {
                    "stability_score": 14.28,
                    "cv": 0.88,
                },
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {
                    "metric_consistency": {
                        "positive_ratio": 0.92,
                    }
                },
            },
        },
    }

    gate = evaluate_promotion_gate(experiment_result, calibration_bundle)

    assert gate["passed"] is True
    assert gate["gate_level"] == "canary_live"
    assert gate["default_live_passed"] is False
    assert gate["checks"]["stability_score"]["passed"] is False
    assert gate["checks"]["positive_ratio"]["passed"] is True


def test_evaluate_promotion_gate_blocks_package_on_missing_bucket_samples():
    """验证 bucket 样本不足时 package 会被直接拦截。"""
    calibration_bundle = {
        "bucket_stats": [
            {"bucket_id": "b0", "sample_count": 420},
            {"bucket_id": "b1", "sample_count": 120},
        ]
    }
    experiment_result = {
        "num_folds": 14,
        "aggregate_metrics": {
            "prediction": {
                "rank_ic_mean": 0.055,
                "top_bucket_spread_mean": 0.011,
            },
            "portfolio": {
                "net_mean_return": 0.0003,
                "max_drawdown": 0.055,
                "mean_turnover": 0.02,
            },
            "robustness": {
                "stability": {
                    "stability_score": 80.0,
                    "cv": 0.30,
                },
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {
                    "metric_consistency": {
                        "positive_ratio": 0.95,
                    }
                },
            },
        },
    }

    gate = evaluate_promotion_gate(experiment_result, calibration_bundle)

    assert gate["passed"] is False
    assert gate["gate_level"] == "blocked"
    assert gate["checks"]["bucket_sample_count"]["passed"] is False


def test_promote_experiment_to_live_package_builds_saved_package(make_raw_panel, tmp_path):
    """验证 promotion 会产出可落盘的 live package。"""
    raw_df = make_raw_panel(
        periods=220,
        assets=("000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE", "000005.XSHE"),
        extended=True,
    )
    prediction_rows = []
    dates = sorted(raw_df["date"].unique())
    assets = sorted(raw_df["order_book_id"].unique())
    for date in dates:
        for idx, order_book_id in enumerate(assets, start=1):
            prediction_rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "prediction": float(idx),
                    "label_return_raw": 0.01 * idx,
                    "label_volatility_horizon": 0.20 + 0.01 * idx,
                    "label_max_drawdown_horizon": 0.10 - 0.01 * idx,
                }
            )
    experiment_result = {
        "model_kind": "linear",
        "config": {
            "model": {"kind": "linear"},
            "preprocessing": {"enabled": False},
        },
        "num_folds": 14,
        "fold_results": [{"predictions_df": __import__("pandas").DataFrame(prediction_rows)}],
        "aggregate_metrics": {
            "prediction": {
                "rank_ic_mean": 0.055,
                "top_bucket_spread_mean": 0.011,
            },
            "portfolio": {
                "net_mean_return": 0.0003,
                "max_drawdown": 0.055,
                "mean_turnover": 0.02,
            },
            "robustness": {
                "stability": {
                    "stability_score": 14.28,
                    "cv": 0.88,
                },
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {
                    "metric_consistency": {
                        "positive_ratio": 0.92,
                    }
                },
            },
        },
    }

    package_payload = promote_experiment_to_live_package(
        experiment_result=experiment_result,
        raw_df=raw_df,
        package_id="tx1_live_promoted_demo",
        packages_root=tmp_path,
        bucket_count=2,
    )

    assert package_payload["manifest"]["package_id"] == "tx1_live_promoted_demo"
    assert package_payload["manifest"]["gate_summary"]["gate_level"] == "canary_live"
    assert package_payload["manifest"]["fit_end_date"] <= package_payload["manifest"]["data_end_date"]
    assert package_payload["recent_canary_bundle"]["summary"]["n_rows"] > 0
    assert package_payload["recent_canary_bundle"]["window"]["end_date"] == package_payload["manifest"]["fit_end_date"]


def test_promote_experiment_to_live_package_records_extended_manifest_metadata(make_raw_panel, tmp_path):
    """验证 promotion 会把 freshness、canary 原因和数据依赖元数据写进 manifest。"""
    raw_df = make_raw_panel(
        periods=220,
        assets=("000001.XSHE", "000002.XSHE", "000003.XSHE", "000004.XSHE", "000005.XSHE"),
        extended=True,
    )
    prediction_rows = []
    dates = sorted(raw_df["date"].unique())
    assets = sorted(raw_df["order_book_id"].unique())
    for date in dates:
        for idx, order_book_id in enumerate(assets, start=1):
            prediction_rows.append(
                {
                    "date": date,
                    "order_book_id": order_book_id,
                    "prediction": float(idx),
                    "label_return_raw": 0.01 * idx,
                    "label_volatility_horizon": 0.20 + 0.01 * idx,
                    "label_max_drawdown_horizon": 0.10 - 0.01 * idx,
                }
            )
    experiment_result = {
        "model_kind": "linear",
        "config": {
            "model": {"kind": "linear"},
            "preprocessing": {"enabled": False},
            "labels": {"horizon": 20},
        },
        "num_folds": 14,
        "fold_results": [{"predictions_df": __import__("pandas").DataFrame(prediction_rows)}],
        "aggregate_metrics": {
            "prediction": {
                "rank_ic_mean": 0.055,
                "top_bucket_spread_mean": 0.011,
            },
            "portfolio": {
                "net_mean_return": 0.0003,
                "max_drawdown": 0.055,
                "mean_turnover": 0.02,
            },
            "robustness": {
                "stability": {
                    "stability_score": 14.28,
                    "cv": 0.88,
                },
                "overfit_flags": {
                    "flag_ic_decay": False,
                    "flag_spread_decay": False,
                    "flag_val_dominant": False,
                },
                "regime_scores": {
                    "metric_consistency": {
                        "positive_ratio": 0.92,
                    }
                },
            },
        },
    }

    package_payload = promote_experiment_to_live_package(
        experiment_result=experiment_result,
        raw_df=raw_df,
        package_id="tx1_live_manifest_meta_demo",
        packages_root=tmp_path,
        bucket_count=2,
    )

    manifest = package_payload["manifest"]

    assert manifest["label_end_date"] == manifest["fit_end_date"]
    assert manifest["evidence_end_date"] == package_payload["recent_canary_bundle"]["window"]["end_date"]
    assert set(manifest["canary_reason"]) == {"stability_score", "cv"}
    assert manifest["data_dependency_summary"]["bundle_price"] is True
    assert manifest["data_dependency_summary"]["bundle_volume"] is True
    assert manifest["data_dependency_summary"]["northbound_flow"] is False
    assert manifest["data_dependency_summary"]["fundamental_factors"] == []
    assert manifest["freshness_policy"]["model_warning_days"] == 20
    assert manifest["freshness_policy"]["model_stop_days"] == 40
