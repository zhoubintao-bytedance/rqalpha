"""AX1 view fusion skeletons."""

from __future__ import annotations

import pandas as pd


class NoOpViewFusion:
    """把模型预期收益原样转成 adjusted_expected_return。"""

    def __init__(self, return_column: str | None = None) -> None:
        self.return_column = return_column

    def fuse(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if predictions is None:
            return pd.DataFrame(columns=["adjusted_expected_return"])
        result = predictions.copy()
        return_column = self.return_column or _infer_return_column(result)
        if return_column not in result.columns:
            raise ValueError(f"predictions missing return column: {return_column}")
        result["adjusted_expected_return"] = pd.to_numeric(result[return_column], errors="coerce")
        return result


class BlackLittermanViewFusion:
    """Black-Litterman 真实数学留给后续版本。"""

    def fuse(self, predictions: pd.DataFrame, covariance: pd.DataFrame | None = None) -> pd.DataFrame:
        raise NotImplementedError("AX1 Black-Litterman view fusion is not implemented yet")

    def _compute_posterior_views(self) -> pd.DataFrame:
        raise NotImplementedError("AX1 Black-Litterman posterior calculation is not implemented yet")


def _infer_return_column(frame: pd.DataFrame) -> str:
    for column in [
        "expected_relative_net_return_10d",
        "expected_relative_net_return_20d",
        "expected_relative_net_return_5d",
    ]:
        if column in frame.columns:
            return column
    raise ValueError("predictions must contain an expected relative net return column")
