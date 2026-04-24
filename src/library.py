import numpy as np


def build_polynomial_library(X, degree=2, include_constant=True):
    """
    Build a polynomial feature library Theta(X).

    Parameters
    ----------
    X : np.ndarray
        Shape (n_samples, n_features)
    degree : int
        Maximum polynomial degree (currently supports 1 or 2)
    include_constant : bool
        Whether to include a constant column of ones

    Returns
    -------
    Theta : np.ndarray
        Library matrix of shape (n_samples, n_library_features)
    feature_names : list[str]
        Names of the columns in Theta
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy array")

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")

    if degree not in [1, 2]:
        raise ValueError("Currently only degree=1 or degree=2 is supported")

    n_samples, n_features = X.shape

    features = []
    feature_names = []

    if include_constant:
        features.append(np.ones((n_samples, 1)))
        feature_names.append("1")

    # linear terms
    features.append(X)
    for i in range(n_features):
        feature_names.append(f"x{i+1}")

    # quadratic terms
    if degree >= 2:
        for i in range(n_features):
            for j in range(i, n_features):
                new_feature = (X[:, i] * X[:, j]).reshape(-1, 1)
                features.append(new_feature)

                if i == j:
                    feature_names.append(f"x{i+1}^2")
                else:
                    feature_names.append(f"x{i+1}x{j+1}")

    Theta = np.hstack(features)
    return Theta, feature_names