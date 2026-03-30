# -*- coding: utf-8 -*-

from skyeye.products.tx1.config import normalize_config
from skyeye.products.tx1.dataset_builder import DatasetBuilder
from skyeye.products.tx1.experiment_runner import ExperimentRunner
from skyeye.products.tx1.label_builder import LabelBuilder
from skyeye.products.tx1.preprocessor import FeaturePreprocessor
from skyeye.products.tx1.splitter import WalkForwardSplitter


def main(config=None, raw_df=None, output_dir=None):
    cfg = normalize_config(config)
    if raw_df is None:
        raise ValueError("raw_df is required for the phase-1 TX1 research runner")

    # Use output_dir from config if not provided as argument
    if output_dir is None and cfg.get("output_dir"):
        output_dir = cfg["output_dir"]

    # Build preprocessor if enabled
    preproc_cfg = cfg.get("preprocessing", {})
    preprocessor = None
    if preproc_cfg.get("enabled", False):
        preprocessor = FeaturePreprocessor(
            neutralize=preproc_cfg.get("neutralize", True),
            winsorize_scale=preproc_cfg.get("winsorize_scale", 5.0),
            standardize=preproc_cfg.get("standardize", True),
        )

    runner = ExperimentRunner(
        config=cfg,
        dataset_builder=DatasetBuilder(input_window=cfg["dataset"]["input_window"]),
        label_builder=LabelBuilder(
            horizon=cfg["labels"]["horizon"],
            transform=cfg["labels"]["transform"],
            winsorize=cfg["labels"].get("winsorize"),
        ),
        splitter=WalkForwardSplitter(
            train_years=cfg["splitter"]["train_years"],
            val_months=cfg["splitter"]["val_months"],
            test_months=cfg["splitter"]["test_months"],
            embargo_days=cfg["splitter"]["embargo_days"],
        ),
        preprocessor=preprocessor,
    )
    return runner.run(raw_df, output_dir=output_dir)
