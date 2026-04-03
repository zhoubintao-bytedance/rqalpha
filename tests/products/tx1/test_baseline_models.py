import numpy as np
import pandas as pd

from skyeye.products.tx1.baseline_models import create_model, create_multi_head_model


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
