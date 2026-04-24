import numpy as np


def stlsq(Theta, Y, threshold=0.05, max_iter=10):
    """
    Sequential Thresholded Least Squares.

    Parameters
    ----------
    Theta : np.ndarray
        Feature library matrix of shape (n_samples, n_library_features)
    Y : np.ndarray
        Target matrix of shape (n_samples, n_targets)
    threshold : float
        Coefficients with absolute value below this are set to zero
    max_iter : int
        Number of thresholding/refitting iterations

    Returns
    -------
    Xi : np.ndarray
        Sparse coefficient matrix of shape (n_library_features, n_targets)
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

    # Initial least-squares fit
    Xi, *_ = np.linalg.lstsq(Theta, Y, rcond=None)

    # Iterative thresholding + refitting
    for _ in range(max_iter):
        small = np.abs(Xi) < threshold
        Xi[small] = 0.0

        for target_idx in range(Y.shape[1]):
            big_idx = ~small[:, target_idx]

            if np.sum(big_idx) == 0:
                continue

            Xi[big_idx, target_idx], *_ = np.linalg.lstsq(
                Theta[:, big_idx],
                Y[:, target_idx],
                rcond=None
            )

    return Xi