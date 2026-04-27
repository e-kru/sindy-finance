import numpy as np

from src.continous_sindy import (
    finite_difference,
    fit_continuous_sindy,
    predict_derivative_continuous_sindy,
)


def test_finite_difference_linear_data():
    dt = 0.1

    X = np.array([
        [0.0],
        [0.1],
        [0.2],
        [0.3],
    ])

    X_current, dXdt = finite_difference(X, dt)

    print("Test: finite difference on linear data")
    print("X_current:")
    print(X_current)
    print("dXdt:")
    print(dXdt)
    print()

    expected_X_current = np.array([
        [0.0],
        [0.1],
        [0.2],
    ])

    expected_dXdt = np.array([
        [1.0],
        [1.0],
        [1.0],
    ])

    assert np.allclose(X_current, expected_X_current)
    assert np.allclose(dXdt, expected_dXdt)


def test_continuous_sindy_recovers_linear_dynamics():
    dt = 0.01

    # Data from exact solution of x_dot = -x:
    # x(t) = exp(-t)
    t = np.arange(0, 2, dt)
    X = np.exp(-t).reshape(-1, 1)

    Xi, names, X_current, dXdt = fit_continuous_sindy(
        X,
        dt=dt,
        degree=2,
        threshold=0.05,
        max_iter=10,
    )

    dXdt_hat = predict_derivative_continuous_sindy(
        X_current,
        Xi,
        degree=2,
    )

    print("Test: continuous-time SINDy recovers x_dot = -x")
    print("Feature names:", names)
    print("Xi:")
    print(Xi)
    print("First 5 true derivatives:")
    print(dXdt[:5])
    print("First 5 predicted derivatives:")
    print(dXdt_hat[:5])
    print()

    assert Xi.shape == (3, 1)
    assert names == ["1", "x1", "x1^2"]

    # Expected model: x_dot ≈ -x
    assert np.allclose(Xi[0, 0], 0.0, atol=0.1)
    assert np.allclose(Xi[1, 0], -1.0, atol=0.1)
    assert np.allclose(Xi[2, 0], 0.0, atol=0.1)


if __name__ == "__main__":
    test_finite_difference_linear_data()
    test_continuous_sindy_recovers_linear_dynamics()
    print("All continuous SINDy tests passed ✅")