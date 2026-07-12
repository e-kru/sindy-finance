import numpy as np


def stlsq(
    theta_matrix: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.05,
    max_iter: int = 10,
) -> np.ndarray:
    """
    Fit a sparse linear model using Sequential Thresholded Least Squares.

    The method approximately solves

        targets ≈ theta_matrix @ coefficients

    by repeatedly:

    1. fitting least squares,
    2. setting small coefficients to zero,
    3. refitting only on the remaining active features.

    Parameters
    ----------
    theta_matrix:
        Candidate-library matrix with shape
        (n_samples, n_library_terms).
    targets:
        Regression targets with shape
        (n_samples, n_targets).
    threshold:
        Coefficients with absolute value below this value are removed.
    max_iter:
        Maximum number of threshold-refit iterations.

    Returns
    -------
    np.ndarray
        Sparse coefficient matrix with shape
        (n_library_terms, n_targets).
    """
    if not isinstance(theta_matrix, np.ndarray):
        raise TypeError("theta_matrix must be a numpy array")

    if not isinstance(targets, np.ndarray):
        raise TypeError("targets must be a numpy array")

    if theta_matrix.ndim != 2:
        raise ValueError("theta_matrix must be 2D")

    if targets.ndim != 2:
        raise ValueError("targets must be 2D")

    if theta_matrix.shape[0] != targets.shape[0]:
        raise ValueError(
            "theta_matrix and targets must have the same number of rows"
        )

    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    coefficients, *_ = np.linalg.lstsq(
        theta_matrix,
        targets,
        rcond=None,
    )

    previous_support = None

    for _ in range(max_iter):
        coefficients[np.abs(coefficients) < threshold] = 0.0

        current_support = coefficients != 0.0

        if (
            previous_support is not None
            and np.array_equal(current_support, previous_support)
        ):
            break

        previous_support = current_support.copy()

        for target_idx in range(targets.shape[1]):
            active_features = current_support[:, target_idx]

            if not np.any(active_features):
                continue

            fitted_coefficients, *_ = np.linalg.lstsq(
                theta_matrix[:, active_features],
                targets[:, target_idx],
                rcond=None,
            )

            coefficients[active_features, target_idx] = fitted_coefficients

    return coefficients