# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import lightgbm as lgb


class LinearBaselineModel(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, train_X, train_y):
        # 线性基线直接对扩展后的设计矩阵做最小二乘拟合。
        X = self._design_matrix(train_X)
        y = np.asarray(train_y, dtype=float)
        self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, test_X):
        # 预测阶段复用训练时相同的设计矩阵拼接方式。
        if self.coef_ is None:
            raise RuntimeError("model must be fit before predict")
        X = self._design_matrix(test_X)
        return X @ self.coef_

    @staticmethod
    def _design_matrix(frame):
        X = frame.to_numpy(dtype=float)
        intercept = np.ones((len(frame), 1), dtype=float)
        return np.concatenate([intercept, X], axis=1)


class TreeBaselineModel(object):
    def __init__(self):
        self.threshold_ = None
        self.left_value_ = None
        self.right_value_ = None
        self.feature_idx_ = 0

    _N_CANDIDATES = 50   # quantile-based threshold candidates per feature

    def fit(self, train_X, train_y):
        # 简化树模型只搜索单层阈值，保持 walk-forward 下的低方差特性。
        X = train_X.to_numpy(dtype=float)
        y = np.asarray(train_y, dtype=float)
        best_loss = None
        quantiles = np.linspace(0, 100, self._N_CANDIDATES + 2)[1:-1]
        for feature_idx in range(X.shape[1]):
            values = np.unique(np.percentile(X[:, feature_idx], quantiles))
            for threshold in values:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_value = float(y[left_mask].mean())
                right_value = float(y[right_mask].mean())
                pred = np.where(left_mask, left_value, right_value)
                loss = float(np.mean((pred - y) ** 2))
                if best_loss is None or loss < best_loss:
                    best_loss = loss
                    self.feature_idx_ = feature_idx
                    self.threshold_ = float(threshold)
                    self.left_value_ = left_value
                    self.right_value_ = right_value
        if self.threshold_ is None:
            self.threshold_ = float(X[:, 0].mean())
            self.left_value_ = float(y.mean())
            self.right_value_ = float(y.mean())
        return self

    def predict(self, test_X):
        # 按训练时找到的最佳阈值做左右叶子打分。
        if self.threshold_ is None:
            raise RuntimeError("model must be fit before predict")
        X = test_X.to_numpy(dtype=float)
        return np.where(X[:, self.feature_idx_] <= self.threshold_, self.left_value_, self.right_value_)


class LightGBMModel(object):
    """Gradient boosting via LightGBM with anti-overfit defaults for walk-forward.

    Conservative defaults: few leaves, high regularization, early stopping
    on validation set to prevent overfitting on noisy financial data.
    """

    DEFAULT_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 24,
        "max_depth": 5,
        "learning_rate": 0.03,
        "n_estimators": 200,
        "subsample": 0.75,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 0.3,
        "reg_lambda": 2.0,
        "min_child_samples": 80,
        "verbose": -1,
        "early_stopping_rounds": 20,
    }

    def __init__(self, params=None):
        merged = dict(self.DEFAULT_PARAMS)
        if params:
            merged.update(params)
        self._n_estimators = merged.pop("n_estimators", 500)
        self._early_stopping = merged.pop("early_stopping_rounds", 50)
        self._params = merged
        self._model = None

    def fit(self, train_X, train_y, val_X=None, val_y=None):
        # LightGBM 保留保守默认参数，并在可用时启用验证集早停。
        X = train_X.to_numpy(dtype=float)
        y = np.asarray(train_y, dtype=float)
        feature_names = list(train_X.columns) if hasattr(train_X, "columns") else None
        train_data = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=False)

        callbacks = []
        valid_sets = [train_data]
        valid_names = ["train"]

        if val_X is not None and val_y is not None:
            vX = val_X.to_numpy(dtype=float)
            vy = np.asarray(val_y, dtype=float)
            val_data = lgb.Dataset(vX, label=vy, reference=train_data, free_raw_data=False)
            valid_sets.append(val_data)
            valid_names.append("val")
            if self._early_stopping and self._early_stopping > 0:
                callbacks.append(lgb.early_stopping(self._early_stopping, verbose=False))

        callbacks.append(lgb.log_evaluation(period=0))

        self._model = lgb.train(
            self._params,
            train_data,
            num_boost_round=self._n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, test_X):
        # 预测时直接复用 Booster，对齐研究侧评分逻辑。
        if self._model is None:
            raise RuntimeError("model must be fit before predict")
        X = test_X.to_numpy(dtype=float)
        return self._model.predict(X)


class IndependentMultiHeadModel(object):
    def __init__(self, kind, head_configs, params=None):
        self.kind = kind
        self.head_configs = deepcopy(head_configs or {})
        self.params = deepcopy(params or {})
        self.models_ = {}

    def fit(self, train_X, train_targets, val_X=None, val_targets=None):
        # 多头模型本质上是每个目标各自训练一套独立基模型。
        if train_targets is None or len(train_targets) == 0:
            raise ValueError("train_targets must not be empty")
        for head_name, head_config in self.head_configs.items():
            target_column = head_config["target_column"]
            if target_column not in train_targets.columns:
                raise ValueError("missing target column for {}: {}".format(head_name, target_column))
            head_params = dict(self.params)
            head_params.update(head_config.get("params", {}))
            model = create_model(self.kind, params=head_params)
            fit_kwargs = {}
            if (
                supports_validation(model)
                and val_X is not None
                and val_targets is not None
                and target_column in val_targets.columns
            ):
                fit_kwargs["val_X"] = val_X
                fit_kwargs["val_y"] = val_targets[target_column]
            model.fit(train_X, train_targets[target_column], **fit_kwargs)
            self.models_[head_name] = {
                "model": model,
                "target_column": target_column,
            }
        return self

    def predict(self, test_X):
        # 预测时保持 head_name -> ndarray 的稳定映射，便于后续拼装输出列。
        if not self.models_:
            raise RuntimeError("model must be fit before predict")
        return {
            head_name: head_state["model"].predict(test_X)
            for head_name, head_state in self.models_.items()
        }


def supports_validation(model):
    return hasattr(model, "fit") and "val_X" in model.fit.__code__.co_varnames


def create_model(kind, params=None):
    if kind == "linear":
        return LinearBaselineModel()
    if kind == "tree":
        return TreeBaselineModel()
    if kind == "lgbm":
        return LightGBMModel(params=params)
    raise ValueError("unsupported model kind: {}".format(kind))


def create_multi_head_model(kind, head_configs, params=None):
    return IndependentMultiHeadModel(kind=kind, head_configs=head_configs, params=params)


def dump_model_bundle(model, *, model_kind: str, feature_columns: list[str]) -> dict:
    """将训练后的模型导出为可序列化 bundle。"""
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    if isinstance(model, IndependentMultiHeadModel):
        if not model.models_:
            raise RuntimeError("multi-head model must be fit before dump")
        state = {
            "base_model_kind": model.kind,
            "head_configs": deepcopy(model.head_configs),
            "heads": {},
        }
        for head_name, head_state in model.models_.items():
            state["heads"][head_name] = {
                "target_column": head_state["target_column"],
                "bundle": dump_model_bundle(
                    head_state["model"],
                    model_kind=model.kind,
                    feature_columns=feature_columns,
                ),
            }
        return {
            "model_kind": "multi_head",
            "feature_columns": list(feature_columns),
            "state": state,
        }
    if model_kind == "linear":
        if model.coef_ is None:
            raise RuntimeError("linear model must be fit before dump")
        state = {
            "coef": np.asarray(model.coef_, dtype=float).tolist(),
        }
    elif model_kind == "tree":
        if model.threshold_ is None:
            raise RuntimeError("tree model must be fit before dump")
        state = {
            "threshold": float(model.threshold_),
            "left_value": float(model.left_value_),
            "right_value": float(model.right_value_),
            "feature_idx": int(model.feature_idx_),
        }
    elif model_kind == "lgbm":
        if model._model is None:
            raise RuntimeError("lightgbm model must be fit before dump")
        state = {
            "model_str": model._model.model_to_string(),
            "params": dict(model._params),
            "n_estimators": int(model._n_estimators),
            "early_stopping_rounds": int(model._early_stopping),
        }
    else:
        raise ValueError("unsupported model kind: {}".format(model_kind))
    return {
        "model_kind": model_kind,
        "feature_columns": list(feature_columns),
        "state": state,
    }


def load_model_bundle(bundle: dict):
    """从序列化 bundle 恢复模型实例。"""
    if not isinstance(bundle, dict):
        raise ValueError("bundle must be a dict")
    model_kind = bundle.get("model_kind")
    state = bundle.get("state", {})
    if model_kind == "multi_head":
        model = IndependentMultiHeadModel(
            kind=state["base_model_kind"],
            head_configs=deepcopy(state.get("head_configs", {})),
            params=None,
        )
        model.models_ = {}
        for head_name, head_state in state.get("heads", {}).items():
            loaded_model = load_model_bundle(head_state["bundle"])
            model.models_[head_name] = {
                "model": loaded_model,
                "target_column": head_state["target_column"],
            }
        return model
    model = create_model(model_kind, params=state.get("params"))
    if model_kind == "linear":
        model.coef_ = np.asarray(state.get("coef", []), dtype=float)
    elif model_kind == "tree":
        model.threshold_ = float(state["threshold"])
        model.left_value_ = float(state["left_value"])
        model.right_value_ = float(state["right_value"])
        model.feature_idx_ = int(state["feature_idx"])
    elif model_kind == "lgbm":
        model._n_estimators = int(state.get("n_estimators", model._n_estimators))
        model._early_stopping = int(state.get("early_stopping_rounds", model._early_stopping))
        model._model = lgb.Booster(model_str=state["model_str"])
    else:
        raise ValueError("unsupported model kind: {}".format(model_kind))
    return model
