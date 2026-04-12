# -*- coding: utf-8 -*-
"""
TX1 Feature Preprocessor

Cross-sectional feature preprocessing pipeline:
  1. Winsorize (MAD-based clipping per date)
  2. Sector + size neutralization (OLS residuals per date)
  3. Z-score standardization (per date)

All operations are per-date cross-sectional, so no information leaks
across time. Safe to apply inside each walk-forward fold.
"""

import hashlib
import json

import numpy as np
import pandas as pd


class FeaturePreprocessor(object):
    """Cross-sectional feature preprocessor.

    Args:
        neutralize: If True, regress out sector dummies and/or ln(close) per date.
        winsorize_scale: MAD multiplier for winsorization. None to skip.
        standardize: If True, z-score standardize per date after neutralization.
        min_obs: Minimum valid cross-sectional observations required for OLS neutralization.
    """

    def __init__(self, neutralize=True, winsorize_scale=5.0, standardize=True, min_obs=5):
        self.neutralize = neutralize
        self.winsorize_scale = winsorize_scale
        self.standardize = standardize
        self.min_obs = int(min_obs)

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

            if self.winsorize_scale is not None:
                result[col] = result.groupby("date")[col].transform(self._winsorize_series)

            if self.neutralize and ("sector" in result.columns or "close" in result.columns):
                result[col] = self._neutralize_column(result, col)

            if self.standardize:
                result[col] = result.groupby("date")[col].transform(self._standardize_series)

        return result

    def required_columns(self, feature_columns):
        """返回当前预处理配置运行所需的最小列集合。"""
        required = ["date", "order_book_id"]
        if self.neutralize:
            required.extend(["close", "sector"])
        for feature_name in feature_columns or []:
            if feature_name not in required:
                required.append(feature_name)
        return required

    def to_bundle(self, feature_columns):
        """导出 preprocessor bundle，供 promoted package 持久化。"""
        feature_columns = list(feature_columns or [])
        payload = {
            "neutralize": bool(self.neutralize),
            "winsorize_scale": self.winsorize_scale,
            "standardize": bool(self.standardize),
            "min_obs": int(self.min_obs),
            "feature_columns": feature_columns,
            "required_columns": self.required_columns(feature_columns),
        }
        payload["bundle_hash"] = self.bundle_hash(feature_columns)
        return payload

    def bundle_hash(self, feature_columns):
        """生成稳定 hash，避免 runtime 使用了错配的预处理口径。"""
        raw = json.dumps(
            {
                "neutralize": bool(self.neutralize),
                "winsorize_scale": self.winsorize_scale,
                "standardize": bool(self.standardize),
                "min_obs": int(self.min_obs),
                "feature_columns": list(feature_columns or []),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_bundle(cls, bundle):
        """从持久化 bundle 恢复 preprocessor 实例。"""
        if not isinstance(bundle, dict):
            raise ValueError("bundle must be a dict")
        return cls(
            neutralize=bundle.get("neutralize", True),
            winsorize_scale=bundle.get("winsorize_scale", 5.0),
            standardize=bundle.get("standardize", True),
            min_obs=bundle.get("min_obs", 5),
        )

    def _winsorize_series(self, series):
        """Winsorize a single cross-section using MAD."""
        result = series.astype(float).copy()
        valid = result[np.isfinite(result)]
        if valid.empty:
            return result
        median = valid.median()
        mad = (valid - median).abs().median()
        if not np.isfinite(mad) or mad < 1e-12:
            return result
        lower = median - self.winsorize_scale * mad
        upper = median + self.winsorize_scale * mad
        result.loc[valid.index] = valid.clip(lower=lower, upper=upper)
        return result

    def _standardize_series(self, series):
        result = series.astype(float).copy()
        valid = result[np.isfinite(result)]
        if valid.empty:
            return result
        mean = valid.mean()
        std = valid.std()
        if not np.isfinite(std) or std < 1e-12:
            result.loc[valid.index] = 0.0
            return result
        result.loc[valid.index] = (valid - mean) / std
        return result

    def _neutralize_column(self, df, col):
        """Per-date OLS neutralization: col ~ sector_dummies + ln(close)."""
        residuals = pd.Series(index=df.index, dtype=float)

        for _, day_df in df.groupby("date"):
            y = pd.to_numeric(day_df[col], errors="coerce")
            valid_mask = np.isfinite(y.to_numpy(dtype=float))

            design_parts = [np.ones((len(day_df), 1), dtype=float)]

            if "sector" in day_df.columns:
                sectors = day_df["sector"].fillna("Unknown").astype(str)
                sector_dummies = pd.get_dummies(sectors, drop_first=True, dtype=float)
                if not sector_dummies.empty:
                    design_parts.append(sector_dummies.to_numpy(dtype=float))

            if "close" in day_df.columns:
                ln_close = np.log(pd.to_numeric(day_df["close"], errors="coerce").clip(lower=0.01))
                close_values = ln_close.to_numpy(dtype=float).reshape(-1, 1)
                design_parts.append(close_values)
                valid_mask &= np.isfinite(close_values[:, 0])

            X = np.column_stack(design_parts)
            min_required = max(self.min_obs, X.shape[1] + 1)
            if valid_mask.sum() < min_required:
                residuals.loc[day_df.index] = y.to_numpy(dtype=float)
                continue

            y_values = y.to_numpy(dtype=float)
            try:
                coef, _, _, _ = np.linalg.lstsq(X[valid_mask], y_values[valid_mask], rcond=None)
                predicted = X @ coef
                resid = y_values - predicted
                resid[~valid_mask] = np.nan
            except np.linalg.LinAlgError:
                resid = y_values

            residuals.loc[day_df.index] = resid

        return residuals
