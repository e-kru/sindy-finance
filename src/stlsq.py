import numpy as np


def stlsq(Theta, Y, threshold=0.05, max_iter=10):
    """
    Sequential Thresholded Least Squares with simple convergence logic.
    """
    if not isinstance(Theta, np.ndarray):
        raise TypeError("Theta must be a numpy array")

    if not isinstance(Y, np.ndarray):
        raise TypeError("Y must be a numpy array")

    if Theta.ndim != 2:
        raise ValueError("Theta must be 2D")

    if Y.ndim != 2:
        raise ValueError("Y must be 2D")

    if Theta.shape[0] != Y.shape[0]:
        raise ValueError("Theta and Y must have the same number of rows")

    # initial least-squares fit
    Xi, *_ = np.linalg.lstsq(Theta, Y, rcond=None)

    previous_support = None

    for _ in range(max_iter):
        # 1) threshold small coefficients
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0

        # 2) current support = which coefficients are nonzero
        current_support = Xi != 0.0

        # 3) stop if support did not change
        if previous_support is not None and np.array_equal(current_support, previous_support):
            break

        previous_support = current_support.copy()

        # 4) refit each target only on active features
        for target_idx in range(Y.shape[1]):
            big_idx = current_support[:, target_idx]

            if np.sum(big_idx) == 0:
                continue

            Xi[big_idx, target_idx], *_ = np.linalg.lstsq(
                Theta[:, big_idx],
                Y[:, target_idx],
                rcond=None
            )

    return Xi