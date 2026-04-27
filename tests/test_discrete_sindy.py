import numpy as np

from src.discrete_sindy import fit_discrete_sindy, predict_discrete_sindy


def test_discrete_sindy_linear_map():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ])

    Y = 1.0 + 0.5 * X

    Xi, names = fit_discrete_sindy(
        X,
        Y,
        degree=2,
        threshold=0.05,
        max_iter=10,
    )

    Y_hat = predict_discrete_sindy(X, Xi, degree=2)

    print("Test: discrete-time SINDy on linear map")
    print("Feature names:", names)
    print("Xi:")
    print(Xi)
    print("Y_hat:")
    print(Y_hat)
    print()

    assert Xi.shape == (3, 1)
    assert names == ["1", "x1", "x1^2"]

    assert np.allclose(Xi[0, 0], 1.0)
    assert np.allclose(Xi[1, 0], 0.5)
    assert np.allclose(Xi[2, 0], 0.0)

    assert np.allclose(Y_hat, Y)


if __name__ == "__main__":
    test_discrete_sindy_linear_map()
    print("All discrete SINDy tests passed")