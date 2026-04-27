import numpy as np

from src.library import build_polynomial_library
from src.stlsq import stlsq


def finite_difference(X, dt):
    """
    Estimate time derivatives using forward finite differences.

    Parameters
    ----------
    X : np.ndarray
        State data of shape (n_time_steps, n_features)
    dt : float
        Time step size

    Returns
    -------
    X_current : np.ndarray
        States x_k of shape (n_time_steps - 1, n_features)
    dXdt : np.ndarray
        Estimated derivatives of shape (n_time_steps - 1, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if dt <= 0:
        raise ValueError("dt must be positive")

    X_current = X[:-1]
    X_next = X[1:]

    dXdt = (X_next - X_current) / dt

    return X_current, dXdt


def fit_continuous_sindy(X, dt, degree=2, threshold=0.05, max_iter=10):
    """
    Fit a continuous-time SINDy model.

    Model:
        dXdt ≈ Theta(X) Xi
    """
    X_current, dXdt = finite_difference(X, dt)

    Theta, feature_names = build_polynomial_library(X_current, degree=degree)

    Xi = stlsq(
        Theta,
        dXdt,
        threshold=threshold,
        max_iter=max_iter,
    )

    return Xi, feature_names, X_current, dXdt


def predict_derivative_continuous_sindy(X, Xi, degree=2):
    """
    Predict derivatives using a fitted continuous-time SINDy model.

    Model:
        dXdt_hat = Theta(X) Xi
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if not isinstance(Xi, np.ndarray):
        raise TypeError("Xi must be a numpy array")

    Theta, _ = build_polynomial_library(X, degree=degree)

    if Theta.shape[1] != Xi.shape[0]:
        raise ValueError("Theta and Xi have incompatible shapes")

    return Theta @ Xi