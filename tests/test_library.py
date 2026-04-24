import numpy as np
from src.library import build_polynomial_library


def test_one_feature_degree_2():
    X = np.array([[1.0], [2.0], [3.0]])
    Theta, names = build_polynomial_library(X, degree=2)

    print("Test 1: one feature, degree 2")
    print("Theta shape:", Theta.shape)
    print("Feature names:", names)
    print(Theta)
    print()

    # ASSERTS
    assert Theta.shape == (3, 3)
    assert names == ["1", "x1", "x1^2"]

    expected = np.array([
        [1, 1, 1],
        [1, 2, 4],
        [1, 3, 9]
    ])
    assert np.allclose(Theta, expected)


def test_two_features_degree_2():
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])
    Theta, names = build_polynomial_library(X, degree=2)

    print("Test 2: two features, degree 2")
    print("Theta shape:", Theta.shape)
    print("Feature names:", names)
    print(Theta)
    print()

    # ASSERTS
    assert Theta.shape == (3, 6)
    assert names == ["1", "x1", "x2", "x1^2", "x1x2", "x2^2"]

    expected = np.array([
        [1, 1, 2, 1, 2, 4],
        [1, 3, 4, 9, 12, 16],
        [1, 5, 6, 25, 30, 36]
    ])
    assert np.allclose(Theta, expected)


def test_three_features_degree_1():
    X = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    Theta, names = build_polynomial_library(X, degree=1)

    print("Test 3: three features, degree 1")
    print("Theta shape:", Theta.shape)
    print("Feature names:", names)
    print(Theta)
    print()

    # ASSERTS
    assert Theta.shape == (2, 4)
    assert names == ["1", "x1", "x2", "x3"]

    expected = np.array([
        [1, 1, 2, 3],
        [1, 4, 5, 6]
    ])
    assert np.allclose(Theta, expected)


if __name__ == "__main__":
    test_one_feature_degree_2()
    test_two_features_degree_2()
    test_three_features_degree_1()
    print("All tests passed ")