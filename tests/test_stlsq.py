import numpy as np

from src.library import build_polynomial_library
from src.stlsq import stlsq


def test_stlsq_recovers_linear_model():
    # Data generated from y = 1 + 2x
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    Y = 1.0 + 2.0 * X

    # Library: [1, x1, x1^2]
    Theta, names = build_polynomial_library(X, degree=2)

    Xi = stlsq(Theta, Y, threshold=0.05, max_iter=10)

    print("Test: STLSQ recovers y = 1 + 2x")
    print("Feature names:", names)
    print("Theta shape:", Theta.shape)
    print("Y shape:", Y.shape)
    print("Xi:")
    print(Xi)
    print()

    assert Theta.shape == (5, 3)
    assert Y.shape == (5, 1)
    assert Xi.shape == (3, 1)

    # Expected model: y = 1 + 2x + 0x^2
    assert np.allclose(Xi[0, 0], 1.0)
    assert np.allclose(Xi[1, 0], 2.0)
    assert np.allclose(Xi[2, 0], 0.0)


def test_stlsq_removes_small_quadratic_term():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    # Mostly linear, tiny quadratic term
    Y = 1.0 + 2.0 * X + 0.001 * X**2

    Theta, names = build_polynomial_library(X, degree=2)

    Xi = stlsq(Theta, Y, threshold=0.05, max_iter=10)

    print("Test: STLSQ removes tiny quadratic term")
    print("Feature names:", names)
    print("Xi:")
    print(Xi)
    print()

    assert Xi.shape == (3, 1)

    # Quadratic coefficient should be thresholded to zero
    assert np.allclose(Xi[2, 0], 0.0)


if __name__ == "__main__":
    test_stlsq_recovers_linear_model()
    test_stlsq_removes_small_quadratic_term()
    print("All STLSQ tests passed ")