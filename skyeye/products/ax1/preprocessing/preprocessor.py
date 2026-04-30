"""AX1 外置预处理器。

子类化复用 TX1 `FeaturePreprocessor`，但对 AX1 当前数据源不带 sector 列的情况做优雅降级：
- `sector_optional=True` 时，`required_columns` 不把 sector 列入必需
- 父类 `_neutralize_column` 本身已通过 `if "sector" in day_df.columns` 的条件分支
  自动退化为 const + ln(close) OLS 残差，无需重写 transform
"""

from __future__ import annotations

from skyeye.products.tx1.preprocessor import FeaturePreprocessor as _TX1Preprocessor


class FeaturePreprocessor(_TX1Preprocessor):
    """AX1 横截面预处理器：MAD winsorize + (sector 可选) + ln(close) 中性化 + z-score。"""

    def __init__(
        self,
        neutralize: bool = True,
        winsorize_scale: float | None = 3.5,
        standardize: bool = True,
        min_obs: int = 5,
        sector_optional: bool = True,
    ):
        super().__init__(
            neutralize=neutralize,
            winsorize_scale=winsorize_scale,
            standardize=standardize,
            min_obs=min_obs,
        )
        self.sector_optional = bool(sector_optional)

    def required_columns(self, feature_columns):
        """返回当前配置运行所需的最小列集合。

        AX1 当前数据源暂无 sector，`sector_optional=True` 时 sector 不作为必需。
        """
        required = ["date", "order_book_id"]
        if self.neutralize:
            required.append("close")
            if not self.sector_optional:
                required.append("sector")
        for feature_name in feature_columns or []:
            if feature_name not in required:
                required.append(feature_name)
        return required

    def to_bundle(self, feature_columns):
        """导出 AX1 预处理配置，保留 sector 可选语义。"""
        payload = super().to_bundle(feature_columns)
        payload["kind"] = "feature_preprocessor"
        payload["sector_optional"] = bool(self.sector_optional)
        return payload

    @classmethod
    def from_bundle(cls, bundle):
        """从持久化 bundle 恢复 AX1 预处理器。"""
        if not isinstance(bundle, dict):
            raise ValueError("bundle must be a dict")
        return cls(
            neutralize=bundle.get("neutralize", True),
            winsorize_scale=bundle.get("winsorize_scale", 3.5),
            standardize=bundle.get("standardize", True),
            min_obs=bundle.get("min_obs", 5),
            sector_optional=bundle.get("sector_optional", True),
        )
