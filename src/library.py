import numpy as np

def build_polynomial_library(X, degree=2):
    n_samples, n_features = X.shape

    features = [np.ones((n_samples, 1)), X]

    if degree >= 2:
        for i in range(n_features):
            for j in range(i, n_features):
                features.append((X[:, i] * X[:, j]).reshape(-1, 1))

    return np.hstack(features)