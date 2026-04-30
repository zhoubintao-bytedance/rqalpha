"""AX1 risk model skeletons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HistoricalCovarianceRiskModel:
    """基于历史 close 面板估计收益协方差矩阵。"""

    shrinkage: float = 0.0
    min_periods: int = 20
    lookback_days: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.shrinkage) <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1")
        if int(self.min_periods) < 1:
            raise ValueError("min_periods must be positive")
        if self.lookback_days is not None and int(self.lookback_days) <= 0:
            raise ValueError("lookback_days must be positive")
        self.covariance_: pd.DataFrame | None = None

    def fit(self, raw_df: pd.DataFrame) -> "HistoricalCovarianceRiskModel":
        close_pivot = _build_close_pivot(raw_df)
        if self.lookback_days is not None and not close_pivot.empty:
            close_pivot = close_pivot.tail(int(self.lookback_days))
        returns = close_pivot.pct_change(fill_method=None).dropna(how="all")
        if len(returns) < int(self.min_periods):
            covariance = pd.DataFrame(
                0.0,
                index=close_pivot.columns,
                columns=close_pivot.columns,
            )
        else:
            covariance = returns.cov().reindex(index=close_pivot.columns, columns=close_pivot.columns)
            covariance = covariance.fillna(0.0)
        self.covariance_ = self._apply_shrinkage(covariance)
        return self

    def get_covariance_matrix(self) -> pd.DataFrame:
        if self.covariance_ is None:
            raise ValueError("risk model is not fitted")
        return self.covariance_.copy()

    def _apply_shrinkage(self, covariance: pd.DataFrame) -> pd.DataFrame:
        if covariance.empty or self.shrinkage == 0:
            return covariance.copy()
        diagonal = pd.DataFrame(
            np.diag(np.diag(covariance.to_numpy(dtype=float))),
            index=covariance.index,
            columns=covariance.columns,
        )
        return covariance * (1.0 - float(self.shrinkage)) + diagonal * float(self.shrinkage)


@dataclass
class FactorRiskModel:
    """ETF-first statistical factor risk model using PCA on recent returns."""

    n_factors: int = 3
    shrinkage: float = 0.0
    min_periods: int = 20
    lookback_days: int | None = None
    idiosyncratic_floor: float = 1e-8
    cov_method: str = "ledoit_wolf"
    ewm_halflife: int | None = None

    def __post_init__(self) -> None:
        if int(self.n_factors) <= 0:
            raise ValueError("n_factors must be positive")
        if not 0.0 <= float(self.shrinkage) <= 1.0:
            raise ValueError("shrinkage must be between 0 and 1")
        if int(self.min_periods) < 1:
            raise ValueError("min_periods must be positive")
        if self.lookback_days is not None and int(self.lookback_days) <= 0:
            raise ValueError("lookback_days must be positive")
        if float(self.idiosyncratic_floor) < 0:
            raise ValueError("idiosyncratic_floor must be non-negative")
        if self.cov_method not in ("ledoit_wolf", "ewm"):
            raise ValueError("cov_method must be 'ledoit_wolf' or 'ewm'")
        if self.ewm_halflife is not None and int(self.ewm_halflife) <= 0:
            raise ValueError("ewm_halflife must be positive")
        self.covariance_: pd.DataFrame | None = None
        self.factor_loadings_: pd.DataFrame | None = None
        self.factor_variance_: pd.Series | None = None
        self.idiosyncratic_variance_: pd.Series | None = None

    def fit(self, raw_df: pd.DataFrame, factor_exposures: pd.DataFrame | None = None) -> "FactorRiskModel":
        close_pivot = _build_close_pivot(raw_df)
        if self.lookback_days is not None and not close_pivot.empty:
            close_pivot = close_pivot.tail(int(self.lookback_days))
        returns = close_pivot.pct_change(fill_method=None).dropna(how="all")
        assets = close_pivot.columns
        if len(assets) == 0:
            self.covariance_ = pd.DataFrame()
            self.factor_loadings_ = pd.DataFrame()
            self.factor_variance_ = pd.Series(dtype=float)
            self.idiosyncratic_variance_ = pd.Series(dtype=float)
            return self
        if len(returns) < int(self.min_periods):
            covariance = pd.DataFrame(
                np.diag(np.full(len(assets), float(self.idiosyncratic_floor))),
                index=assets,
                columns=assets,
            )
            self.covariance_ = covariance
            self.factor_loadings_ = pd.DataFrame(index=assets)
            self.factor_variance_ = pd.Series(dtype=float)
            self.idiosyncratic_variance_ = pd.Series(np.diag(covariance.to_numpy(dtype=float)), index=assets)
            return self

        sample_cov = self._estimate_covariance(returns, assets)
        sample_matrix = sample_cov.to_numpy(dtype=float)
        sample_matrix = np.nan_to_num(sample_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        sample_matrix = (sample_matrix + sample_matrix.T) / 2.0
        diagonal = np.clip(np.diag(sample_matrix), 0.0, None)
        np.fill_diagonal(sample_matrix, diagonal)

        max_factors = min(int(self.n_factors), len(assets), max(len(returns) - 1, 1))
        factor_matrix = np.zeros_like(sample_matrix)
        factor_values = np.array([], dtype=float)
        factor_vectors = np.empty((len(assets), 0), dtype=float)
        if max_factors > 0 and np.any(sample_matrix):
            eigen_values, eigen_vectors = np.linalg.eigh(sample_matrix)
            order = np.argsort(eigen_values)[::-1]
            positive = [idx for idx in order if eigen_values[idx] > eigen_values.max() * max(1e-12, np.finfo(float).eps)][:max_factors]
            if positive:
                factor_values = eigen_values[positive].astype(float)
                factor_vectors = eigen_vectors[:, positive].astype(float)
                factor_matrix = factor_vectors @ np.diag(factor_values) @ factor_vectors.T

        residual_diagonal = np.diag(sample_matrix - factor_matrix)
        residual_diagonal = np.clip(residual_diagonal, float(self.idiosyncratic_floor), None)
        covariance_matrix = factor_matrix + np.diag(residual_diagonal)
        if float(self.shrinkage) > 0:
            covariance_matrix = (
                covariance_matrix * (1.0 - float(self.shrinkage))
                + np.diag(np.diag(covariance_matrix)) * float(self.shrinkage)
            )
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2.0
        np.fill_diagonal(
            covariance_matrix,
            np.clip(np.diag(covariance_matrix), float(self.idiosyncratic_floor), None),
        )

        covariance = pd.DataFrame(covariance_matrix, index=assets, columns=assets)
        self.covariance_ = covariance
        self.factor_loadings_ = pd.DataFrame(
            factor_vectors,
            index=assets,
            columns=[f"stat_factor_{idx + 1}" for idx in range(factor_vectors.shape[1])],
        )
        self.factor_variance_ = pd.Series(
            factor_values,
            index=[f"stat_factor_{idx + 1}" for idx in range(len(factor_values))],
        )
        self.idiosyncratic_variance_ = pd.Series(residual_diagonal, index=assets)
        return self

    def _estimate_covariance(self, returns: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
        """Estimate covariance matrix using the configured method."""
        if self.cov_method == "ewm":
            halflife = int(self.ewm_halflife) if self.ewm_halflife is not None else 20
            cov = returns.ewm(halflife=halflife).cov().iloc[-len(assets):]
            cov = cov.reindex(index=assets, columns=assets).fillna(0.0)
            return _symmetrize_frame(cov)
        return _ledoit_wolf_shrinkage(returns, assets)

    def get_covariance_matrix(self) -> pd.DataFrame:
        if self.covariance_ is None:
            raise ValueError("risk model is not fitted")
        return self.covariance_.copy()


def _ledoit_wolf_shrinkage(returns: pd.DataFrame, assets: pd.Index) -> pd.DataFrame:
    """Ledoit-Wolf (2004) linear shrinkage toward scaled identity.

    Optimal when N < p (under-determined): shrinks the sample covariance
    toward a structured target, dramatically reducing estimation error.
    """
    X = returns.reindex(columns=assets).fillna(0.0).to_numpy(dtype=float)
    n, p = X.shape
    if n < 2:
        cov = pd.DataFrame(0.0, index=assets, columns=assets)
        return cov

    # Demean
    X = X - X.mean(axis=0)

    # Sample covariance (1/n normalization as in Ledoit-Wolf)
    S = X.T @ X / n
    mu = np.trace(S) / p  # target scale
    F = mu * np.eye(p)     # shrinkage target

    # Compute optimal shrinkage intensity (Ledoit-Wolf 2004, Lemma 3.1)
    # delta = ||S - F||^2  (squared Frobenius distance to target)
    delta = np.sum((S - F) ** 2) / p

    # beta = (1/n^2) * sum_k ||x_k x_k^T - S||^2
    sum_sq = 0.0
    for k in range(n):
        xk = X[k:k + 1, :]  # (1, p)
        outer = xk.T @ xk   # (p, p)
        sum_sq += np.sum((outer - S) ** 2)
    beta = min(sum_sq / (n ** 2 * p), delta)

    # Shrinkage intensity
    if delta == 0.0:
        alpha = 1.0
    else:
        alpha = beta / delta

    # Shrunk covariance (rescale to 1/(n-1) for consistency with pandas .cov())
    S_shrunk = (alpha * F + (1.0 - alpha) * S) * n / (n - 1)

    cov = pd.DataFrame(S_shrunk, index=assets, columns=assets)
    return _symmetrize_frame(cov)


def _symmetrize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    matrix = frame.to_numpy(dtype=float)
    matrix = (matrix + matrix.T) / 2.0
    return pd.DataFrame(matrix, index=frame.index, columns=frame.columns)


def _build_close_pivot(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df is None:
        raise ValueError("raw_df must not be None")
    missing = [column for column in ["date", "order_book_id", "close"] if column not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df missing required columns: {missing}")
    frame = raw_df.dropna(subset=["date", "order_book_id", "close"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["close"])
    pivot = frame.pivot_table(
        index="date",
        columns="order_book_id",
        values="close",
        aggfunc="last",
    )
    return pivot.sort_index().sort_index(axis=1)
