import numpy as np
import pytest

from src.discrete_sindy import (
    fit_discrete_sindy,
    predict_discrete_sindy,
)


def test_discrete_sindy_recovers_linear_map():
    states = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    targets = 1.0 + 0.5 * states

    coefficients, feature_names = fit_discrete_sindy(
        states,
        targets,
        degree=2,
        threshold=0.05,
        max_iter=10,
    )

    predictions = predict_discrete_sindy(
        states,
        coefficients,
        degree=2,
    )

    assert coefficients.shape == (3, 1)
    assert feature_names == ["1", "x1", "x1^2"]

    assert np.isclose(coefficients[0, 0], 1.0)
    assert np.isclose(coefficients[1, 0], 0.5)
    assert np.isclose(coefficients[2, 0], 0.0)

    assert np.allclose(predictions, targets)


def test_discrete_sindy_rejects_mismatched_rows():
    states = np.zeros((5, 1))
    targets = np.zeros((4, 1))

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        fit_discrete_sindy(states, targets)