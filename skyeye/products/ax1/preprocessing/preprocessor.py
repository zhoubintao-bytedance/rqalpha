"""AX1 外置预处理器。

子类化复用 TX1 `FeaturePreprocessor`，但对 AX1 当前数据源不带 sector 列的情况做优雅降级：
- `sector_optional=True` 时，`required_columns` 不把 sector 列入必需
- 父类 `_neutralize_column` 本身已通过 `if "sector" in day_df.columns` 的条件分支
  自动退化为 const + ln(close) OLS 残差，无需重写 transform
"""

from __future__ import annotations

from collections.abc import Mapping

from skyeye.products.tx1.preprocessor import FeaturePreprocessor as _TX1Preprocessor

_CROSS_SECTIONAL_POLICY = "cross_sectional"
_PASSTHROUGH_POLICY = "passthrough"
_SUPPORTED_POLICIES = {_CROSS_SECTIONAL_POLICY, _PASSTHROUGH_POLICY}


class FeaturePreprocessor(_TX1Preprocessor):
    """AX1 横截面预处理器：MAD winsorize + (sector 可选) + ln(close) 中性化 + z-score。"""

    def __init__(
        self,
        neutralize: bool = True,
        winsorize_scale: float | None = 3.5,
        standardize: bool = True,
        min_obs: int = 5,
        sector_optional: bool = True,
        preprocess_policies: Mapping[str, str] | None = None,
    ):
        super().__init__(
            neutralize=neutralize,
            winsorize_scale=winsorize_scale,
            standardize=standardize,
            min_obs=min_obs,
        )
        self.sector_optional = bool(sector_optional)
        self.preprocess_policies = _normalize_preprocess_policies([], preprocess_policies)

    def transform(self, df, feature_columns, preprocess_policies: Mapping[str, str] | None = None):
        """Apply policy-aware preprocessing.

        `cross_sectional` preserves the inherited TX1 behavior. `passthrough`
        keeps date-level state features such as regime dummies from being
        collapsed to zero by per-date z-score standardization.
        """
        policies = _normalize_preprocess_policies(
            feature_columns,
            preprocess_policies if preprocess_policies is not None else self.preprocess_policies,
        )
        cross_sectional_columns = [
            feature_name
            for feature_name in feature_columns or []
            if policies.get(str(feature_name), _CROSS_SECTIONAL_POLICY) == _CROSS_SECTIONAL_POLICY
        ]
        if not cross_sectional_columns:
            return df.copy()
        return super().transform(df, cross_sectional_columns)

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

    def to_bundle(self, feature_columns, preprocess_policies: Mapping[str, str] | None = None):
        """导出 AX1 预处理配置，保留 sector 可选语义。"""
        payload = super().to_bundle(feature_columns)
        payload["kind"] = "feature_preprocessor"
        payload["sector_optional"] = bool(self.sector_optional)
        payload["preprocess_policies"] = _normalize_preprocess_policies(
            feature_columns,
            preprocess_policies if preprocess_policies is not None else self.preprocess_policies,
        )
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
            preprocess_policies=bundle.get("preprocess_policies"),
        )


def _normalize_preprocess_policies(
    feature_columns,
    preprocess_policies: Mapping[str, str] | None,
) -> dict[str, str]:
    policies = {str(key): str(value) for key, value in dict(preprocess_policies or {}).items()}
    feature_names = [str(feature_name) for feature_name in feature_columns or []]
    for feature_name in feature_names:
        policies.setdefault(feature_name, _CROSS_SECTIONAL_POLICY)
    unsupported = {
        feature_name: policy
        for feature_name, policy in policies.items()
        if policy not in _SUPPORTED_POLICIES
    }
    if unsupported:
        raise ValueError(f"unsupported AX1 preprocess policies: {unsupported}")
    if not feature_names:
        return policies
    return {feature_name: policies[feature_name] for feature_name in feature_names if feature_name in policies}
