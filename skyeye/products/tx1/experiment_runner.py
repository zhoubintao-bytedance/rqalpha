# -*- coding: utf-8 -*-

from skyeye.products.tx1.baseline_models import create_model
from skyeye.products.tx1.cost_layer import CostConfig
from skyeye.products.tx1.evaluator import FEATURE_COLUMNS, build_portfolio_returns, evaluate_portfolios, evaluate_predictions
from skyeye.products.tx1.persistence import save_experiment
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.robustness import compute_regime_scores, compute_stability_score, detect_overfit_flags


class ExperimentRunner(object):
    def __init__(self, config, dataset_builder, label_builder, splitter, preprocessor=None):
        self.config = config
        self.dataset_builder = dataset_builder
        self.label_builder = label_builder
        self.splitter = splitter
        self.preprocessor = preprocessor

    def _build_cost_config(self):
        costs_cfg = self.config.get("costs", {})
        if not costs_cfg.get("enabled", False):
            return None
        return CostConfig(
            commission_rate=costs_cfg.get("commission_rate", 0.0008),
            stamp_tax_rate=costs_cfg.get("stamp_tax_rate", 0.0005),
            slippage_bps=costs_cfg.get("slippage_bps", 5.0),
        )

    def run(self, raw_df, output_dir=None):
        dataset = self.dataset_builder.build(raw_df)
        labeled = self.label_builder.build(dataset)
        folds = self.splitter.split(labeled)
        fold_results = []
        cost_config = self._build_cost_config()
        portfolio_builder = PortfolioProxy(
            buy_top_k=self.config["portfolio"]["buy_top_k"],
            hold_top_k=self.config["portfolio"]["hold_top_k"],
            rebalance_interval=self.config["portfolio"].get("rebalance_interval", 20),
            holding_bonus=self.config["portfolio"].get("holding_bonus", 0.5),
        )
        for idx, fold in enumerate(folds, start=1):
            train_df = fold["train_df"].copy()
            val_df = fold["val_df"].copy()
            test_df = fold["test_df"].copy()

            # Derive feature columns from what the dataset actually contains
            feature_cols = [c for c in FEATURE_COLUMNS if c in train_df.columns]

            # Preprocessing inside fold to prevent data leakage
            if self.preprocessor is not None:
                train_df = self.preprocessor.transform(train_df, feature_cols)
                val_df = self.preprocessor.transform(val_df, feature_cols)
                test_df = self.preprocessor.transform(test_df, feature_cols)

            model = create_model(self.config["model"]["kind"], params=self.config["model"].get("params"))
            fit_kwargs = {}
            if hasattr(model, "fit") and "val_X" in model.fit.__code__.co_varnames:
                fit_kwargs["val_X"] = val_df[feature_cols]
                fit_kwargs["val_y"] = val_df["target_label"]
            model.fit(train_df[feature_cols], train_df["target_label"], **fit_kwargs)
            val_df["prediction"] = model.predict(val_df[feature_cols])
            test_df["prediction"] = model.predict(test_df[feature_cols])
            prediction_metrics = evaluate_predictions(test_df, top_k=self.config["evaluation"]["top_k"])
            validation_metrics = evaluate_predictions(val_df, top_k=self.config["evaluation"]["top_k"])
            weights_df = portfolio_builder.build(test_df[["date", "order_book_id", "prediction"]])
            portfolio_returns = build_portfolio_returns(
                test_df,
                weights_df,
                horizon_days=self.config["labels"]["horizon"],
            )
            portfolio_metrics = evaluate_portfolios(portfolio_returns, cost_config=cost_config)

            # Build date range info from fold metadata
            date_range = {
                "train_start": fold.get("train_start"),
                "train_end": fold.get("train_end"),
                "val_start": fold.get("val_start"),
                "val_end": fold.get("val_end"),
                "test_start": fold.get("test_start"),
                "test_end": fold.get("test_end"),
            }

            # Prepare DataFrames for persistence
            predictions_df = test_df[["date", "order_book_id", "prediction", "target_label", "label_return_raw"]].copy()

            fold_results.append(
                {
                    "fold_index": idx,
                    "prediction_metrics": prediction_metrics,
                    "validation_metrics": validation_metrics,
                    "portfolio_metrics": portfolio_metrics,
                    "row_counts": {
                        "train": len(train_df),
                        "val": len(val_df),
                        "test": len(test_df),
                    },
                    "date_range": date_range,
                    "predictions_df": predictions_df,
                    "weights_df": weights_df,
                    "portfolio_returns_df": portfolio_returns,
                }
            )
        aggregate = self._aggregate_metrics(fold_results)

        # Robustness analysis
        robustness_cfg = self.config.get("robustness", {})
        if robustness_cfg.get("enabled", False):
            stability_metric = robustness_cfg.get("stability_metric", "rank_ic_mean")
            aggregate["robustness"] = {
                "stability": compute_stability_score(fold_results, metric_key=stability_metric),
                "overfit_flags": detect_overfit_flags(fold_results),
                "regime_scores": compute_regime_scores(fold_results, metric_key=stability_metric),
            }

        result = {
            "model_kind": self.config["model"]["kind"],
            "fold_results": fold_results,
            "aggregate_metrics": aggregate,
        }

        # Persist to disk if output_dir is provided
        if output_dir is not None:
            saved_path = save_experiment(result, output_dir, config=self.config)
            result["output_dir"] = saved_path

        return result

    @staticmethod
    def _aggregate_metrics(fold_results):
        if not fold_results:
            return {"prediction": {}, "portfolio": {}}
        prediction_keys = fold_results[0]["prediction_metrics"].keys()
        portfolio_keys = fold_results[0]["portfolio_metrics"].keys()
        prediction = {
            key: float(sum(result["prediction_metrics"][key] for result in fold_results) / len(fold_results))
            for key in prediction_keys
        }
        portfolio = {
            key: float(sum(result["portfolio_metrics"][key] for result in fold_results) / len(fold_results))
            for key in portfolio_keys
        }
        return {"prediction": prediction, "portfolio": portfolio}
