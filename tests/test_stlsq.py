import numpy as np
import pytest

from src.library import build_polynomial_library
from src.stlsq import stlsq


def test_stlsq_recovers_linear_model():
    states = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    targets = 1.0 + 2.0 * states

    theta_matrix, _ = build_polynomial_library(
        states,
        degree=2,
    )

    coefficients = stlsq(
        theta_matrix,
        targets,
        threshold=0.05,
        max_iter=10,
    )

    assert coefficients.shape == (3, 1)

    assert np.isclose(coefficients[0, 0], 1.0)
    assert np.isclose(coefficients[1, 0], 2.0)
    assert np.isclose(coefficients[2, 0], 0.0)


def test_stlsq_removes_small_quadratic_term():
    states = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    targets = 1.0 + 2.0 * states + 0.001 * states**2

    theta_matrix, _ = build_polynomial_library(
        states,
        degree=2,
    )

    coefficients = stlsq(
        theta_matrix,
        targets,
        threshold=0.05,
        max_iter=10,
    )

    assert np.isclose(coefficients[2, 0], 0.0)


def test_stlsq_rejects_negative_threshold():
    theta_matrix = np.ones((5, 2))
    targets = np.ones((5, 1))

    with pytest.raises(
        ValueError,
        match="threshold must be non-negative",
    ):
        stlsq(
            theta_matrix,
            targets,
            threshold=-0.1,
        )