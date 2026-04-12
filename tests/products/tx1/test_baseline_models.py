import numpy as np
import pandas as pd

from skyeye.products.tx1.baseline_models import (
    create_model,
    create_multi_head_model,
    dump_model_bundle,
    load_model_bundle,
)


def test_linear_baseline_fit_predict_minimal_case():
    train_X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 1.0, 1.0]})
    train_y = pd.Series([0.0, 1.0, 2.0])
    test_X = pd.DataFrame({"x1": [3.0, 4.0], "x2": [1.0, 1.0]})

    model = create_model("linear")
    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
    assert pred[1] > pred[0]


def test_tree_baseline_fit_predict_minimal_case():
    train_X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 1.0, 1.0, 1.0]})
    train_y = pd.Series([0.0, 0.0, 2.0, 2.0])
    test_X = pd.DataFrame({"x1": [0.5, 2.5], "x2": [1.0, 1.0]})

    model = create_model("tree")
    model.fit(train_X, train_y)
    pred = model.predict(test_X)

    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
    assert pred[0] != pred[1]


def test_multi_head_model_fits_independent_targets():
    train_X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 0.0, 1.0, 0.0]})
    train_targets = pd.DataFrame(
        {
            "target_return": [0.0, 1.0, 2.0, 3.0],
            "target_volatility": [3.0, 2.0, 1.0, 0.0],
            "target_max_drawdown": [0.2, 0.1, 0.3, 0.4],
        }
    )
    test_X = pd.DataFrame({"x1": [1.5, 2.5], "x2": [1.0, 0.0]})

    model = create_multi_head_model(
        "linear",
        {
            "return": {"target_column": "target_return"},
            "volatility": {"target_column": "target_volatility"},
            "max_drawdown": {"target_column": "target_max_drawdown"},
        },
    )
    model.fit(train_X, train_targets)
    pred = model.predict(test_X)

    assert set(pred) == {"return", "volatility", "max_drawdown"}
    assert pred["return"].shape == (2,)
    assert pred["volatility"].shape == (2,)
    assert pred["max_drawdown"].shape == (2,)
    assert np.isfinite(pred["return"]).all()
    assert np.isfinite(pred["volatility"]).all()
    assert np.isfinite(pred["max_drawdown"]).all()


def test_linear_model_bundle_round_trip_preserves_predictions():
    """验证线性模型序列化前后预测保持一致。"""
    train_X = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "x2": [1.0, 1.0, 1.0]})
    train_y = pd.Series([0.0, 1.0, 2.0])
    test_X = pd.DataFrame({"x1": [3.0, 4.0], "x2": [1.0, 1.0]})

    model = create_model("linear")
    model.fit(train_X, train_y)
    before = model.predict(test_X)
    bundle = dump_model_bundle(model, model_kind="linear", feature_columns=list(test_X.columns))
    restored = load_model_bundle(bundle)
    after = restored.predict(test_X)

    assert bundle["model_kind"] == "linear"
    assert bundle["feature_columns"] == ["x1", "x2"]
    assert np.allclose(before, after)


def test_tree_model_bundle_round_trip_preserves_predictions():
    """验证树模型序列化前后预测保持一致。"""
    train_X = pd.DataFrame({"x1": [0.0, 1.0, 2.0, 3.0], "x2": [1.0, 1.0, 1.0, 1.0]})
    train_y = pd.Series([0.0, 0.0, 2.0, 2.0])
    test_X = pd.DataFrame({"x1": [0.5, 2.5], "x2": [1.0, 1.0]})

    model = create_model("tree")
    model.fit(train_X, train_y)
    before = model.predict(test_X)
    bundle = dump_model_bundle(model, model_kind="tree", feature_columns=list(test_X.columns))
    restored = load_model_bundle(bundle)
    after = restored.predict(test_X)

    assert bundle["model_kind"] == "tree"
    assert np.allclose(before, after)
