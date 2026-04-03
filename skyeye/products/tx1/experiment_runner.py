# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from skyeye.products.tx1.baseline_models import create_model, create_multi_head_model, supports_validation
from skyeye.products.tx1.cost_layer import CostConfig
from skyeye.products.tx1.evaluator import FEATURE_COLUMNS, build_portfolio_returns, evaluate_portfolios, evaluate_predictions
from skyeye.products.tx1.persistence import save_experiment
from skyeye.products.tx1.portfolio_proxy import PortfolioProxy
from skyeye.products.tx1.robustness import compute_regime_scores, compute_stability_score, detect_overfit_flags


PREDICTION_OUTPUT_COLUMNS = [
    "prediction",
    "prediction_ret",
    "prediction_vol",
    "prediction_mdd",
    "reliability_score",
]


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

    def _build_head_specs(self):
        specs = {
            "return": {
                "target_column": "target_return",
                "label_column": "label_return_raw",
                "prediction_column": "prediction_ret",
            },
        }
        multi_output_cfg = self.config.get("multi_output", {})
        if multi_output_cfg.get("enabled", False) and multi_output_cfg["volatility"].get("enabled", False):
            specs["volatility"] = {
                "target_column": "target_volatility",
                "label_column": "label_volatility_horizon",
                "prediction_column": "prediction_vol",
            }
        if multi_output_cfg.get("enabled", False) and multi_output_cfg["max_drawdown"].get("enabled", False):
            specs["max_drawdown"] = {
                "target_column": "target_max_drawdown",
                "label_column": "label_max_drawdown_horizon",
                "prediction_column": "prediction_mdd",
            }
        return specs

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
        head_specs = self._build_head_specs()
        use_multi_output = self.config.get("multi_output", {}).get("enabled", False)

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

            if use_multi_output:
                model = create_multi_head_model(
                    self.config["model"]["kind"],
                    head_specs,
                    params=self.config["model"].get("params"),
                )
                target_columns = [spec["target_column"] for spec in head_specs.values()]
                model.fit(
                    train_df[feature_cols],
                    train_df[target_columns],
                    val_X=val_df[feature_cols],
                    val_targets=val_df[target_columns],
                )
                self._attach_head_predictions(val_df, model.predict(val_df[feature_cols]), head_specs)
                self._attach_head_predictions(test_df, model.predict(test_df[feature_cols]), head_specs)
            else:
                model = create_model(self.config["model"]["kind"], params=self.config["model"].get("params"))
                fit_kwargs = {}
                if supports_validation(model):
                    fit_kwargs["val_X"] = val_df[feature_cols]
                    fit_kwargs["val_y"] = val_df["target_label"]
                model.fit(train_df[feature_cols], train_df["target_label"], **fit_kwargs)
                self._initialize_prediction_columns(val_df)
                self._initialize_prediction_columns(test_df)
                val_df["prediction_ret"] = model.predict(val_df[feature_cols])
                test_df["prediction_ret"] = model.predict(test_df[feature_cols])

            self._finalize_prediction_outputs(val_df)
            self._finalize_prediction_outputs(test_df)

            prediction_metrics = evaluate_predictions(test_df, top_k=self.config["evaluation"]["top_k"])
            validation_metrics = evaluate_predictions(val_df, top_k=self.config["evaluation"]["top_k"])
            head_metrics = self._evaluate_head_metrics(test_df, head_specs)
            head_validation_metrics = self._evaluate_head_metrics(val_df, head_specs)
            prediction_blend_metrics = self._evaluate_prediction_blend_metrics(test_df)
            selection_metrics = self._evaluate_selection_metrics(test_df)

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

            predictions_df = self._build_predictions_df(test_df)

            fold_results.append(
                {
                    "fold_index": idx,
                    "prediction_metrics": prediction_metrics,
                    "validation_metrics": validation_metrics,
                    "head_metrics": head_metrics,
                    "head_validation_metrics": head_validation_metrics,
                    "prediction_blend_metrics": prediction_blend_metrics,
                    "selection_metrics": selection_metrics,
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
                    "model_heads": list(head_specs.keys()),
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
            "model_heads": list(head_specs.keys()),
            "prediction_columns": list(PREDICTION_OUTPUT_COLUMNS),
            "fold_results": fold_results,
            "aggregate_metrics": aggregate,
        }

        # Persist to disk if output_dir is provided
        if output_dir is not None:
            saved_path = save_experiment(result, output_dir, config=self.config)
            result["output_dir"] = saved_path

        return result

    @staticmethod
    def _initialize_prediction_columns(frame):
        for column in PREDICTION_OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = np.nan

    def _attach_head_predictions(self, frame, predictions, head_specs):
        self._initialize_prediction_columns(frame)
        for head_name, values in predictions.items():
            prediction_column = head_specs[head_name]["prediction_column"]
            frame[prediction_column] = values

    def _finalize_prediction_outputs(self, frame):
        self._initialize_prediction_columns(frame)
        if frame["prediction_ret"].isna().all() and "prediction" in frame.columns:
            frame["prediction_ret"] = frame["prediction"]
        multi_output_cfg = self.config.get("multi_output", {})
        prediction_cfg = multi_output_cfg.get("prediction", {})
        if prediction_cfg.get("combine_auxiliary", False):
            frame["prediction"] = self._build_combined_prediction(frame)
        else:
            frame["prediction"] = frame["prediction_ret"]
        if multi_output_cfg.get("reliability_score", {}).get("enabled", False):
            frame["reliability_score"] = self._build_reliability_score(frame)
        else:
            frame["reliability_score"] = np.nan

    def _build_combined_prediction(self, frame):
        prediction_cfg = self.config.get("multi_output", {}).get("prediction", {})
        combined = self._cross_section_rank(frame, "prediction_ret")
        if prediction_cfg.get("volatility_weight", 0.0) > 0 and frame["prediction_vol"].notna().any():
            combined = combined - prediction_cfg["volatility_weight"] * self._cross_section_rank(frame, "prediction_vol")
        if prediction_cfg.get("max_drawdown_weight", 0.0) > 0 and frame["prediction_mdd"].notna().any():
            combined = combined - prediction_cfg["max_drawdown_weight"] * self._cross_section_rank(frame, "prediction_mdd")
        return combined

    def _build_reliability_score(self, frame):
        components = [self._cross_section_rank(frame, "prediction_ret")]
        if frame["prediction_vol"].notna().any():
            components.append(1.0 - self._cross_section_rank(frame, "prediction_vol"))
        if frame["prediction_mdd"].notna().any():
            components.append(1.0 - self._cross_section_rank(frame, "prediction_mdd"))
        if len(components) < 2:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        component_frame = pd.concat(components, axis=1)
        dispersion = component_frame.std(axis=1, ddof=0).clip(lower=0.0, upper=0.5)
        return (1.0 - 2.0 * dispersion).clip(lower=0.0, upper=1.0)

    @staticmethod
    def _cross_section_rank(frame, column):
        return frame.groupby("date")[column].rank(method="average", pct=True)

    def _evaluate_head_metrics(self, frame, head_specs):
        metrics = {}
        top_k = self.config["evaluation"]["top_k"]
        for head_name, spec in head_specs.items():
            prediction_column = spec["prediction_column"]
            if prediction_column not in frame.columns or frame[prediction_column].isna().all():
                continue
            hit_rate_mode = "positive" if head_name == "return" else "median"
            metrics[head_name] = self._evaluate_prediction_column(
                frame,
                prediction_column=prediction_column,
                label_column=spec["label_column"],
                top_k=top_k,
                hit_rate_mode=hit_rate_mode,
            )
        return metrics

    def _evaluate_prediction_blend_metrics(self, frame):
        top_k = self.config["evaluation"]["top_k"]
        ret_metrics = self._evaluate_prediction_column(
            frame,
            prediction_column="prediction_ret",
            label_column="label_return_raw",
            top_k=top_k,
            hit_rate_mode="positive",
        )
        blend_metrics = self._evaluate_prediction_column(
            frame,
            prediction_column="prediction",
            label_column="label_return_raw",
            top_k=top_k,
            hit_rate_mode="positive",
        )
        return {
            "rank_ic_mean_delta_vs_ret": float(blend_metrics["rank_ic_mean"] - ret_metrics["rank_ic_mean"]),
            "rank_ic_ir_delta_vs_ret": float(blend_metrics["rank_ic_ir"] - ret_metrics["rank_ic_ir"]),
            "top_bucket_spread_mean_delta_vs_ret": float(
                blend_metrics["top_bucket_spread_mean"] - ret_metrics["top_bucket_spread_mean"]
            ),
            "top_k_hit_rate_delta_vs_ret": float(
                blend_metrics.get("top_k_hit_rate", 0.0) - ret_metrics.get("top_k_hit_rate", 0.0)
            ),
        }

    def _evaluate_selection_metrics(self, frame):
        ret_selection = self._evaluate_selection_profile(frame, "prediction_ret")
        active_selection = self._evaluate_selection_profile(frame, "prediction")
        metrics = {}
        for key, value in ret_selection.items():
            metrics["prediction_ret_{}".format(key)] = value
        for key, value in active_selection.items():
            metrics["prediction_{}".format(key)] = value
        for key in active_selection.keys():
            metrics["prediction_delta_vs_ret_{}".format(key)] = float(active_selection[key] - ret_selection[key])
        return metrics

    def _evaluate_selection_profile(self, frame, score_column):
        top_k = self.config["evaluation"]["top_k"]
        return_means = []
        volatility_means = []
        drawdown_means = []
        reliability_means = []
        for _, day_df in frame.groupby("date", sort=True):
            ranked = day_df.dropna(subset=[score_column]).sort_values(score_column, ascending=False)
            if ranked.empty:
                continue
            top = ranked.head(min(top_k, len(ranked)))
            return_means.append(float(top["label_return_raw"].mean()))
            volatility_means.append(float(top["label_volatility_horizon"].mean()))
            drawdown_means.append(float(top["label_max_drawdown_horizon"].mean()))
            if "reliability_score" in top.columns and top["reliability_score"].notna().any():
                reliability_means.append(float(top["reliability_score"].mean()))
        result = {
            "top_k_mean_future_return": float(np.mean(return_means)) if return_means else 0.0,
            "top_k_mean_future_volatility": float(np.mean(volatility_means)) if volatility_means else 0.0,
            "top_k_mean_future_max_drawdown": float(np.mean(drawdown_means)) if drawdown_means else 0.0,
        }
        if reliability_means:
            result["top_k_mean_reliability_score"] = float(np.mean(reliability_means))
        return result

    @staticmethod
    def _evaluate_prediction_column(frame, prediction_column, label_column, top_k, hit_rate_mode):
        if frame is None or len(frame) == 0:
            raise ValueError("prediction frame must not be empty")
        grouped = frame.groupby("date", sort=True)
        rank_ics = []
        spreads = []
        top_means = []
        bottom_means = []
        hit_rates = []
        for _, day_df in grouped:
            valid = day_df[[prediction_column, label_column]].dropna()
            if len(valid) < 2:
                continue
            pred_rank = valid[prediction_column].rank(method="average")
            label_rank = valid[label_column].rank(method="average")
            rank_ic = float(pred_rank.corr(label_rank, method="pearson"))
            if np.isfinite(rank_ic):
                rank_ics.append(rank_ic)
            ranked = valid.sort_values(prediction_column, ascending=False)
            top = ranked.head(min(top_k, len(ranked)))
            bottom = ranked.tail(min(top_k, len(ranked)))
            top_mean = float(top[label_column].mean())
            bottom_mean = float(bottom[label_column].mean())
            top_means.append(top_mean)
            bottom_means.append(bottom_mean)
            spreads.append(top_mean - bottom_mean)
            if hit_rate_mode == "positive":
                hit_rates.append(float((top[label_column] > 0).mean()))
            elif hit_rate_mode == "median":
                day_threshold = float(valid[label_column].median())
                hit_rates.append(float((top[label_column] >= day_threshold).mean()))
        rank_ic_mean = float(np.mean(rank_ics)) if rank_ics else 0.0
        rank_ic_ir = float(rank_ic_mean / np.std(rank_ics)) if len(rank_ics) > 1 and np.std(rank_ics) > 0 else 0.0
        result = {
            "rank_ic_mean": rank_ic_mean,
            "rank_ic_ir": rank_ic_ir,
            "top_bucket_spread_mean": float(np.mean(spreads)) if spreads else 0.0,
            "top_bucket_label_mean": float(np.mean(top_means)) if top_means else 0.0,
            "bottom_bucket_label_mean": float(np.mean(bottom_means)) if bottom_means else 0.0,
        }
        if hit_rate_mode == "positive":
            result["top_k_hit_rate"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        elif hit_rate_mode == "median":
            result["top_k_above_median_rate"] = float(np.mean(hit_rates)) if hit_rates else 0.0
        return result

    @staticmethod
    def _build_predictions_df(test_df):
        columns = [
            "date",
            "order_book_id",
            "prediction",
            "prediction_ret",
            "prediction_vol",
            "prediction_mdd",
            "reliability_score",
            "target_label",
            "target_return",
            "target_volatility",
            "target_max_drawdown",
            "label_return_raw",
            "label_volatility_horizon",
            "label_max_drawdown_horizon",
        ]
        return test_df[[column for column in columns if column in test_df.columns]].copy()

    @classmethod
    def _aggregate_metrics(cls, fold_results):
        if not fold_results:
            return {"prediction": {}, "portfolio": {}}
        aggregate = {
            "prediction": cls._average_metric_dicts([result.get("prediction_metrics", {}) for result in fold_results]),
            "validation": cls._average_metric_dicts([result.get("validation_metrics", {}) for result in fold_results]),
            "portfolio": cls._average_metric_dicts([result.get("portfolio_metrics", {}) for result in fold_results]),
            "head_metrics": cls._average_metric_dicts([result.get("head_metrics", {}) for result in fold_results]),
            "head_validation_metrics": cls._average_metric_dicts(
                [result.get("head_validation_metrics", {}) for result in fold_results]
            ),
            "prediction_blend": cls._average_metric_dicts(
                [result.get("prediction_blend_metrics", {}) for result in fold_results]
            ),
            "selection": cls._average_metric_dicts([result.get("selection_metrics", {}) for result in fold_results]),
        }
        return aggregate

    @classmethod
    def _average_metric_dicts(cls, metric_dicts):
        metric_dicts = [metric_dict for metric_dict in metric_dicts if metric_dict]
        if not metric_dicts:
            return {}
        keys = set()
        for metric_dict in metric_dicts:
            keys.update(metric_dict.keys())
        averaged = {}
        for key in sorted(keys):
            values = [metric_dict[key] for metric_dict in metric_dicts if key in metric_dict]
            first_value = values[0]
            if isinstance(first_value, dict):
                averaged[key] = cls._average_metric_dicts(values)
                continue
            if isinstance(first_value, (int, float, np.floating, np.integer)) and not isinstance(first_value, bool):
                averaged[key] = float(np.mean(values))
        return averaged
