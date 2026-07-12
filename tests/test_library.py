import numpy as np
import pytest

from src.library import build_polynomial_library


def test_one_feature_degree_two():
    states = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    theta_matrix, feature_names = build_polynomial_library(
        states,
        degree=2,
    )

    expected = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0],
        [1.0, 3.0, 9.0],
    ])

    assert theta_matrix.shape == (3, 3)
    assert feature_names == ["1", "x1", "x1^2"]
    assert np.allclose(theta_matrix, expected)


def test_two_features_degree_two():
    states = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    theta_matrix, feature_names = build_polynomial_library(
        states,
        degree=2,
    )

    expected = np.array([
        [1.0, 1.0, 2.0, 1.0, 2.0, 4.0],
        [1.0, 3.0, 4.0, 9.0, 12.0, 16.0],
        [1.0, 5.0, 6.0, 25.0, 30.0, 36.0],
    ])

    assert theta_matrix.shape == (3, 6)
    assert feature_names == [
        "1",
        "x1",
        "x2",
        "x1^2",
        "x1x2",
        "x2^2",
    ]
    assert np.allclose(theta_matrix, expected)


def test_three_features_degree_one():
    states = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    theta_matrix, feature_names = build_polynomial_library(
        states,
        degree=1,
    )

    expected = np.array([
        [1.0, 1.0, 2.0, 3.0],
        [1.0, 4.0, 5.0, 6.0],
    ])

    assert theta_matrix.shape == (2, 4)
    assert feature_names == ["1", "x1", "x2", "x3"]
    assert np.allclose(theta_matrix, expected)


def test_library_rejects_unsupported_degree():
    states = np.ones((3, 1))

    with pytest.raises(ValueError, match="degree must be 1 or 2"):
        build_polynomial_library(states, degree=3)