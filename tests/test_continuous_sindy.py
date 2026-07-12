import numpy as np
import pytest

from src.continuous_sindy import (
    finite_difference,
    fit_continuous_sindy,
    predict_continuous_sindy,
)


def test_finite_difference_linear_data():
    dt = 0.1

    states = np.array([
        [0.0],
        [0.1],
        [0.2],
        [0.3],
    ])

    current_states, derivatives = finite_difference(states, dt)

    expected_current_states = np.array([
        [0.0],
        [0.1],
        [0.2],
    ])

    expected_derivatives = np.array([
        [1.0],
        [1.0],
        [1.0],
    ])

    assert np.allclose(current_states, expected_current_states)
    assert np.allclose(derivatives, expected_derivatives)


def test_continuous_sindy_recovers_linear_dynamics():
    dt = 0.01

    # Exact solution of dx/dt = -x.
    time = np.arange(0.0, 2.0, dt)
    states = np.exp(-time).reshape(-1, 1)

    coefficients, feature_names, current_states, derivatives = (
        fit_continuous_sindy(
            states,
            dt=dt,
            degree=2,
            threshold=0.05,
            max_iter=10,
        )
    )

    predicted_derivatives = predict_continuous_sindy(
        current_states,
        coefficients,
        degree=2,
    )

    assert coefficients.shape == (3, 1)
    assert feature_names == ["1", "x1", "x1^2"]
    assert predicted_derivatives.shape == derivatives.shape

    # Expected discovered model: dx/dt ≈ -x.
    assert np.isclose(coefficients[0, 0], 0.0, atol=0.1)
    assert np.isclose(coefficients[1, 0], -1.0, atol=0.1)
    assert np.isclose(coefficients[2, 0], 0.0, atol=0.1)

    assert np.mean((predicted_derivatives - derivatives) ** 2) < 1e-4


def test_finite_difference_rejects_non_positive_dt():
    states = np.array([
        [0.0],
        [1.0],
    ])

    with pytest.raises(ValueError, match="dt must be positive"):
        finite_difference(states, dt=0.0)