import numpy as np

from src.library import build_polynomial_library
from src.stlsq import stlsq


def finite_difference(
    states: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate time derivatives using forward finite differences.

    For observations x_k at equally spaced times,

        dx/dt ≈ (x_{k+1} - x_k) / dt.

    Parameters
    ----------
    states:
        State observations with shape
        (n_time_steps, n_features).
    dt:
        Positive time-step size.

    Returns
    -------
    current_states:
        States x_k with shape
        (n_time_steps - 1, n_features).
    derivatives:
        Forward-difference derivative estimates with shape
        (n_time_steps - 1, n_features).
    """
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy array")

    if states.ndim != 2:
        raise ValueError("states must be 2D")

    if states.shape[0] < 2:
        raise ValueError("states must contain at least two time steps")

    if dt <= 0:
        raise ValueError("dt must be positive")

    current_states = states[:-1]
    next_states = states[1:]

    derivatives = (next_states - current_states) / dt

    return current_states, derivatives


def fit_continuous_sindy(
    states: np.ndarray,
    dt: float,
    degree: int = 2,
    threshold: float = 0.05,
    max_iter: int = 10,
) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """
    Fit a continuous-time SINDy model.

    The model has the form

        dX/dt ≈ Theta(X) @ coefficients.

    The derivatives are estimated using forward finite differences.

    Parameters
    ----------
    states:
        State observations with shape
        (n_time_steps, n_features).
    dt:
        Time-step size.
    degree:
        Maximum polynomial degree of the candidate library.
    threshold:
        STLSQ sparsity threshold.
    max_iter:
        Maximum number of STLSQ iterations.

    Returns
    -------
    coefficients:
        Sparse SINDy coefficient matrix.
    feature_names:
        Names of the candidate-library terms.
    current_states:
        States used to construct the library.
    derivatives:
        Estimated time derivatives.
    """
    current_states, derivatives = finite_difference(states, dt)

    theta_matrix, feature_names = build_polynomial_library(
        current_states,
        degree=degree,
    )

    coefficients = stlsq(
        theta_matrix,
        derivatives,
        threshold=threshold,
        max_iter=max_iter,
    )

    return coefficients, feature_names, current_states, derivatives


def predict_continuous_sindy(
    states: np.ndarray,
    coefficients: np.ndarray,
    degree: int = 2,
) -> np.ndarray:
    """
    Predict state derivatives using a fitted continuous-time SINDy model.

    The prediction is

        predicted_derivatives = Theta(states) @ coefficients.

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
        Predicted state derivatives.
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