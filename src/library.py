import numpy as np


def build_polynomial_library(
    states: np.ndarray,
    degree: int = 2,
    include_constant: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Construct a polynomial candidate library for SINDy.

    For two state variables and degree 2, the resulting library is

        [1, x1, x2, x1^2, x1*x2, x2^2].

    Parameters
    ----------
    states:
        State observations with shape (n_samples, n_features).
    degree:
        Maximum polynomial degree. Currently supports degree 1 or 2.
    include_constant:
        Whether to include a constant column.

    Returns
    -------
    theta_matrix:
        Candidate-library matrix.
    feature_names:
        Human-readable names of the library terms.
    """
    if not isinstance(states, np.ndarray):
        raise TypeError("states must be a numpy array")

    if states.ndim != 2:
        raise ValueError(
            "states must be a 2D array with shape "
            "(n_samples, n_features)"
        )

    if states.shape[1] == 0:
        raise ValueError("states must contain at least one feature")

    if degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")

    n_samples, n_features = states.shape

    features = []
    feature_names = []

    if include_constant:
        features.append(np.ones((n_samples, 1)))
        feature_names.append("1")

    features.append(states)

    for feature_idx in range(n_features):
        feature_names.append(f"x{feature_idx + 1}")

    if degree == 2:
        for first_idx in range(n_features):
            for second_idx in range(first_idx, n_features):
                interaction = (
                    states[:, first_idx] * states[:, second_idx]
                ).reshape(-1, 1)

                features.append(interaction)

                if first_idx == second_idx:
                    feature_names.append(f"x{first_idx + 1}^2")
                else:
                    feature_names.append(
                        f"x{first_idx + 1}x{second_idx + 1}"
                    )

    theta_matrix = np.hstack(features)

    return theta_matrix, feature_names