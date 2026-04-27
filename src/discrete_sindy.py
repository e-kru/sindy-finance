import numpy as np

from src.library import build_polynomial_library
from src.stlsq import stlsq


def fit_discrete_sindy(X, Y, degree=2, threshold=0.05, max_iter=10):
    """
    Fit a discrete-time SINDy model.

    Model:
        Y ≈ Theta(X) Xi

    Parameters
    ----------
    X : np.ndarray
        Input states of shape (n_samples, n_features)
    Y : np.ndarray
        Target states of shape (n_samples, n_targets)
    degree : int
        Polynomial library degree
    threshold : float
        STLSQ threshold
    max_iter : int
        Maximum STLSQ iterations

    Returns
    -------
    Xi : np.ndarray
        Sparse coefficient matrix
    feature_names : list[str]
        Names of library features
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array")

    if X.ndim != 2:
        raise ValueError("X must be 2D")

    if Y.ndim != 2:
        raise ValueError("Y must be 2D")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows")

    Theta, feature_names = build_polynomial_library(X, degree=degree)
    Xi = stlsq(Theta, Y, threshold=threshold, max_iter=max_iter)

    return Xi, feature_names


def predict_discrete_sindy(X, Xi, degree=2):
    """
    Predict targets using a fitted discrete-time SINDy model.
    X new data, Xi learned coeficients


    Model:
        Y_hat = Theta(X) Xi
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if not isinstance(Xi, np.ndarray):
        raise TypeError("Xi must be a numpy array")

    Theta, _ = build_polynomial_library(X, degree=degree)

    if Theta.shape[1] != Xi.shape[0]:
        raise ValueError("Theta and Xi have incompatible shapes")

    return Theta @ Xi
