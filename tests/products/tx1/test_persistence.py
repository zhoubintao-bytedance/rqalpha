import json

import pandas as pd

from skyeye.products.tx1.persistence import load_experiment, save_experiment


def test_save_and_load_experiment_preserves_multi_output_metadata(tmp_path):
    output_dir = tmp_path / "tx1_multi_output_save"
    result = {
        "model_kind": "linear",
        "model_heads": ["return", "volatility", "max_drawdown"],
        "prediction_columns": [
            "prediction",
            "prediction_ret",
            "prediction_vol",
            "prediction_mdd",
            "reliability_score",
        ],
        "aggregate_metrics": {
            "prediction": {"rank_ic_mean": 0.1},
            "head_metrics": {"volatility": {"rank_ic_mean": 0.2}},
        },
        "fold_results": [
            {
                "fold_index": 1,
                "prediction_metrics": {"rank_ic_mean": 0.1},
                "validation_metrics": {"rank_ic_mean": 0.08},
                "head_metrics": {"volatility": {"rank_ic_mean": 0.2}},
                "head_validation_metrics": {"volatility": {"rank_ic_mean": 0.15}},
                "prediction_blend_metrics": {"rank_ic_mean_delta_vs_ret": 0.01},
                "selection_metrics": {"prediction_top_k_mean_future_volatility": 0.3},
                "portfolio_metrics": {"mean_return": 0.001},
                "row_counts": {"train": 10, "val": 5, "test": 5},
                "date_range": {"test_start": "2024-01-01 00:00:00", "test_end": "2024-01-05 00:00:00"},
                "model_heads": ["return", "volatility", "max_drawdown"],
                "predictions_df": pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                        "order_book_id": ["000001.XSHE", "000002.XSHE"],
                        "prediction": [0.9, 0.1],
                        "prediction_ret": [0.8, 0.2],
                        "prediction_vol": [0.3, 0.7],
                        "prediction_mdd": [0.2, 0.6],
                        "reliability_score": [0.9, 0.4],
                        "target_label": [0.7, 0.3],
                        "label_return_raw": [0.05, -0.02],
                    }
                ),
                "weights_df": pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02"]),
                        "order_book_id": ["000001.XSHE"],
                        "weight": [1.0],
                        "prediction": [0.9],
                    }
                ),
                "portfolio_returns_df": pd.DataFrame(
                    {
                        "date": pd.to_datetime(["2024-01-02"]),
                        "portfolio_return": [0.01],
                        "turnover": [0.1],
                        "overlap": [1.0],
                    }
                ),
            }
        ],
    }

    save_experiment(result, str(output_dir), config={"multi_output": {"enabled": True}})
    loaded = load_experiment(str(output_dir))

    assert loaded["model_heads"] == ["return", "volatility", "max_drawdown"]
    assert "prediction_vol" in loaded["fold_results"][0]["predictions_df"].columns
    assert loaded["fold_results"][0]["head_metrics"]["volatility"]["rank_ic_mean"] == 0.2
    assert loaded["fold_results"][0]["prediction_blend_metrics"]["rank_ic_mean_delta_vs_ret"] == 0.01
    assert loaded["aggregate_metrics"]["head_metrics"]["volatility"]["rank_ic_mean"] == 0.2


def test_load_experiment_supports_legacy_schema(tmp_path):
    output_dir = tmp_path / "legacy_experiment"
    fold_dir = output_dir / "folds" / "fold_001"
    fold_dir.mkdir(parents=True)

    experiment_json = {
        "version": "1.0",
        "created_at": "2026-03-29T09:36:19.983982",
        "experiment_name": "legacy_tx1",
        "model_kind": "linear",
        "config": {"model": {"kind": "linear"}},
        "aggregate_metrics": {"prediction": {"rank_ic_mean": 0.05}},
        "num_folds": 1,
        "folds": [
            {
                "index": 1,
                "path": "folds/fold_001",
                "row_counts": {"test": 2},
                "date_range": {"test_start": "2024-01-01 00:00:00"},
            }
        ],
    }
    with open(output_dir / "experiment.json", "w", encoding="utf-8") as handle:
        json.dump(experiment_json, handle)

    pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "order_book_id": ["000001.XSHE", "000002.XSHE"],
            "prediction": [0.9, 0.1],
            "target_label": [0.7, 0.3],
            "label_return_raw": [0.05, -0.02],
        }
    ).to_parquet(fold_dir / "predictions.parquet", index=False)

    with open(fold_dir / "fold_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fold_index": 1,
                "prediction_metrics": {"rank_ic_mean": 0.05},
                "validation_metrics": {"rank_ic_mean": 0.04},
                "portfolio_metrics": {"mean_return": 0.001},
                "row_counts": {"test": 2},
                "date_range": {"test_start": "2024-01-01 00:00:00"},
            },
            handle,
        )

    loaded = load_experiment(str(output_dir))

    assert loaded["prediction_columns"] == ["prediction"]
    assert loaded["model_heads"] == ["return"]
    assert loaded["fold_results"][0]["head_metrics"] == {}
    assert loaded["fold_results"][0]["selection_metrics"] == {}
    assert "prediction" in loaded["fold_results"][0]["predictions_df"].columns
