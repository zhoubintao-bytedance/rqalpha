# -*- coding: utf-8 -*-

import numpy as np
import lightgbm as lgb


class LinearBaselineModel(object):
    def __init__(self):
        self.coef_ = None

    def fit(self, train_X, train_y):
        X = self._design_matrix(train_X)
        y = np.asarray(train_y, dtype=float)
        self.coef_, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, test_X):
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
        if self._model is None:
            raise RuntimeError("model must be fit before predict")
        X = test_X.to_numpy(dtype=float)
        return self._model.predict(X)


def create_model(kind, params=None):
    if kind == "linear":
        return LinearBaselineModel()
    if kind == "tree":
        return TreeBaselineModel()
    if kind == "lgbm":
        return LightGBMModel(params=params)
    raise ValueError("unsupported model kind: {}".format(kind))
