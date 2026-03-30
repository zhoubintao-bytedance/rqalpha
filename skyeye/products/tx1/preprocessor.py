# -*- coding: utf-8 -*-
"""
TX1 Feature Preprocessor

Cross-sectional feature preprocessing pipeline:
  1. Winsorize (5x MAD per date)
  2. Sector + market-cap neutralization (OLS residuals per date)
  3. Z-score standardization (per date)

All operations are per-date cross-sectional, so no information leaks
across time. Safe to apply inside each walk-forward fold.
"""

import numpy as np
import pandas as pd


class FeaturePreprocessor(object):
    """Cross-sectional feature preprocessor.

    Args:
        neutralize: If True, regress out sector dummies + ln(close) per date.
        winsorize_scale: MAD multiplier for winsorization. None to skip.
        standardize: If True, z-score standardize per date after neutralization.
    """

    def __init__(self, neutralize=True, winsorize_scale=5.0, standardize=True):
        self.neutralize = neutralize
        self.winsorize_scale = winsorize_scale
        self.standardize = standardize

    def transform(self, df, feature_columns):
        """Apply preprocessing pipeline to feature columns.

        Args:
            df: DataFrame with 'date', 'order_book_id', feature columns,
                and optionally 'sector' and 'close' for neutralization.
            feature_columns: List of column names to preprocess.

        Returns:
            DataFrame with preprocessed feature columns (other columns unchanged).
        """
        result = df.copy()

        for col in feature_columns:
            if col not in result.columns:
                continue

            # Step 1: Winsorize (5x MAD per date)
            if self.winsorize_scale is not None:
                result[col] = result.groupby("date")[col].transform(
                    self._winsorize_series
                )

            # Step 2: Sector + market-cap neutralization
            if self.neutralize and "sector" in result.columns and "close" in result.columns:
                result[col] = self._neutralize_column(result, col)

            # Step 3: Cross-sectional z-score
            if self.standardize:
                cs_mean = result.groupby("date")[col].transform("mean")
                cs_std = result.groupby("date")[col].transform("std")
                result[col] = (result[col] - cs_mean) / cs_std.replace(0, np.nan)

        return result

    def _winsorize_series(self, series):
        """Winsorize a single cross-section using MAD."""
        median = series.median()
        mad = (series - median).abs().median()
        if mad < 1e-12:
            return series
        lower = median - self.winsorize_scale * mad
        upper = median + self.winsorize_scale * mad
        return series.clip(lower=lower, upper=upper)

    def _neutralize_column(self, df, col):
        """Per-date OLS neutralization: col ~ sector_dummies + ln(close)."""
        residuals = pd.Series(index=df.index, dtype=float)

        for date, day_df in df.groupby("date"):
            y = day_df[col].values
            valid_mask = np.isfinite(y)
            if valid_mask.sum() < 5:
                residuals.loc[day_df.index] = y
                continue

            # Build design matrix: sector dummies + ln(close)
            sector_dummies = pd.get_dummies(day_df["sector"], drop_first=True, dtype=float)
            ln_close = np.log(day_df["close"].clip(lower=0.01)).values.reshape(-1, 1)
            X = np.column_stack([
                np.ones(len(day_df)),
                sector_dummies.values,
                ln_close,
            ])

            # Handle NaN in y: set residual = NaN for those rows
            y_clean = np.where(valid_mask, y, 0.0)
            try:
                coef, _, _, _ = np.linalg.lstsq(X[valid_mask], y_clean[valid_mask], rcond=None)
                predicted = X @ coef
                res = y - predicted
            except np.linalg.LinAlgError:
                res = y

            residuals.loc[day_df.index] = res

        return residuals
