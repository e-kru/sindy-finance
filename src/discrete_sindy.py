import numpy as np

from src.library import build_polynomial_library
from src.stlsq import stlsq


def fit_discrete_sindy(
    states: np.ndarray,
    targets: np.ndarray,
    degree: int = 2,
    threshold: float = 0.05,
    max_iter: int = 10,
) -> tuple[np.ndarray, list[str]]:
    """
    Fit a discrete-time SINDy model.

    The model has the form

        targets ≈ Theta(states) @ coefficients.

    A typical use is

        X_t     -> states
        X_{t+1} -> targets.

    Parameters
    ----------
    states:
        Input states with shape (n_samples, n_features).
    targets:
        Target states with shape (n_samples, n_targets).
    degree:
        Maximum polynomial degree of the candidate library.
    threshold:
        STLSQ sparsity threshold.
    max_iter:
        Maximum number of STLSQ iterations.

    Returns
    -------
    coefficients:
        Sparse coefficient matrix with shape
        (n_library_terms, n_targets).
    feature_names:
        Names of the candidate-library terms.
    """
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy array")

    if not isinstance(targets, np.ndarray):
        raise TypeError("targets must be a numpy array")

    if states.ndim != 2:
        raise ValueError("states must be 2D")

    if targets.ndim != 2:
        raise ValueError("targets must be 2D")

    if states.shape[0] != targets.shape[0]:
        raise ValueError(
            "states and targets must have the same number of rows"
        )

    theta_matrix, feature_names = build_polynomial_library(
        states,
        degree=degree,
    )

    coefficients = stlsq(
        theta_matrix,
        targets,
        threshold=threshold,
        max_iter=max_iter,
    )

    return coefficients, feature_names


def predict_discrete_sindy(
    states: np.ndarray,
    coefficients: np.ndarray,
    degree: int = 2,
) -> np.ndarray:
    """
    Predict target states using a fitted discrete-time SINDy model.

    The prediction is

        predictions = Theta(states) @ coefficients.

    Parameters
    ----------
    states:
        Input states with shape (n_samples, n_features).
    coefficients:
        Learned SINDy coefficient matrix.
    degree:
        Polynomial degree used during fitting.

    Returns
    -------
    np.ndarray
        Predicted target states.
    """
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy array")

    if not isinstance(coefficients, np.ndarray):
        raise TypeError("coefficients must be a numpy array")

    if states.ndim != 2:
        raise ValueError("states must be 2D")

    if coefficients.ndim != 2:
        raise ValueError("coefficients must be 2D")

    theta_matrix, _ = build_polynomial_library(
        states,
        degree=degree,
    )

    if theta_matrix.shape[1] != coefficients.shape[0]:
        raise ValueError(
            "Library matrix and coefficient matrix have incompatible shapes"
        )

    return theta_matrix @ coefficients