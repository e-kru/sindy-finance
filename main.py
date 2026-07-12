from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.continuous_sindy import (
    fit_continuous_sindy,
    predict_continuous_sindy,
)
from src.discrete_sindy import (
    fit_discrete_sindy,
    predict_discrete_sindy,
)
from src.simulation import simulate_vasicek


FIGURE_DIR = Path("figures")


def format_equation(
    feature_names: list[str],
    coefficients: np.ndarray,
    target_name: str,
    tolerance: float = 1e-10,
) -> str:
    """
    Format a one-dimensional SINDy model as a readable equation.
    """
    terms = []

    for feature_name, coefficient in zip(
        feature_names,
        coefficients[:, 0],
    ):
        if abs(coefficient) <= tolerance:
            continue

        if feature_name == "1":
            term = f"{coefficient:.4f}"
        else:
            term = f"{coefficient:+.4f}*{feature_name}"

        terms.append(term)

    if not terms:
        return f"{target_name} = 0"

    equation = " ".join(terms)

    if equation.startswith("+"):
        equation = equation[1:].lstrip()

    return f"{target_name} = {equation}"


def run_vasicek_experiment() -> None:
    """
    Simulate a Vasicek process and fit discrete- and continuous-time
    SINDy models.
    """
    n_steps = 2000
    dt = 0.01
    kappa = 1.5
    long_run_mean = 0.03
    sigma = 0.02
    initial_rate = 0.01

    rates = simulate_vasicek(
        n_steps=n_steps,
        dt=dt,
        kappa=kappa,
        theta=long_run_mean,
        sigma=sigma,
        r0=initial_rate,
        seed=42,
    )

    full_states = rates.reshape(-1, 1)

    # Discrete-time model:
    # r_{k+1} ≈ Theta(r_k) @ coefficients
    discrete_states = rates[:-1].reshape(-1, 1)
    discrete_targets = rates[1:].reshape(-1, 1)

    discrete_coefficients, discrete_features = fit_discrete_sindy(
        discrete_states,
        discrete_targets,
        degree=1,
        threshold=1e-5,
        max_iter=10,
    )

    discrete_predictions = predict_discrete_sindy(
        discrete_states,
        discrete_coefficients,
        degree=1,
    )

    discrete_mse = np.mean(
        (discrete_targets - discrete_predictions) ** 2
    )

    # Continuous-time model:
    # dr/dt ≈ Theta(r) @ coefficients
    (
        continuous_coefficients,
        continuous_features,
        current_states,
        derivatives,
    ) = fit_continuous_sindy(
        full_states,
        dt=dt,
        degree=1,
        threshold=1e-5,
        max_iter=10,
    )

    derivative_predictions = predict_continuous_sindy(
        current_states,
        continuous_coefficients,
        degree=1,
    )

    continuous_mse = np.mean(
        (derivatives - derivative_predictions) ** 2
    )

    print_results(
        discrete_features=discrete_features,
        discrete_coefficients=discrete_coefficients,
        discrete_mse=discrete_mse,
        continuous_features=continuous_features,
        continuous_coefficients=continuous_coefficients,
        continuous_mse=continuous_mse,
        kappa=kappa,
        long_run_mean=long_run_mean,
        dt=dt,
    )

    save_figures(
        rates=rates,
        discrete_targets=discrete_targets,
        discrete_predictions=discrete_predictions,
        derivatives=derivatives,
        derivative_predictions=derivative_predictions,
        dt=dt,
        long_run_mean=long_run_mean,
    )


def print_results(
    discrete_features: list[str],
    discrete_coefficients: np.ndarray,
    discrete_mse: float,
    continuous_features: list[str],
    continuous_coefficients: np.ndarray,
    continuous_mse: float,
    kappa: float,
    long_run_mean: float,
    dt: float,
) -> None:
    """
    Print recovered and theoretical Vasicek coefficients.
    """
    discrete_equation = format_equation(
        discrete_features,
        discrete_coefficients,
        target_name="r_next",
    )

    continuous_equation = format_equation(
        continuous_features,
        continuous_coefficients,
        target_name="r_dot",
    )

    expected_discrete_constant = kappa * long_run_mean * dt
    expected_discrete_linear = 1.0 - kappa * dt

    expected_continuous_constant = kappa * long_run_mean
    expected_continuous_linear = -kappa

    print("\nDiscrete-time SINDy")
    print(discrete_equation)
    print(f"MSE: {discrete_mse:.8f}")

    print("\nContinuous-time SINDy")
    print(continuous_equation)
    print(f"MSE: {continuous_mse:.8f}")

    print("\nTheoretical Vasicek coefficients")
    print(
        "Discrete: "
        f"r_next ≈ {expected_discrete_constant:.6f} "
        f"+ {expected_discrete_linear:.6f}*r"
    )
    print(
        "Continuous: "
        f"r_dot ≈ {expected_continuous_constant:.6f} "
        f"{expected_continuous_linear:+.6f}*r"
    )


def save_figures(
    rates: np.ndarray,
    discrete_targets: np.ndarray,
    discrete_predictions: np.ndarray,
    derivatives: np.ndarray,
    derivative_predictions: np.ndarray,
    dt: float,
    long_run_mean: float,
) -> None:
    """
    Save the three main Vasicek experiment figures.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    n_plot = 300
    time = np.arange(len(rates)) * dt

    plt.figure(figsize=(10, 5))
    plt.plot(
        time[:n_plot],
        rates[:n_plot],
        label="Simulated short rate",
    )
    plt.axhline(
        long_run_mean,
        linestyle="--",
        label="Long-run mean",
    )
    plt.xlabel("Time")
    plt.ylabel("Short rate")
    plt.title("Simulated Vasicek short-rate path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "vasicek_path.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        discrete_targets[:n_plot],
        label="True next rate",
    )
    plt.plot(
        discrete_predictions[:n_plot],
        label="Discrete SINDy prediction",
    )
    plt.xlabel("Time step")
    plt.ylabel("Short rate")
    plt.title("Discrete-time SINDy on Vasicek data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "vasicek_discrete_prediction.png",
        dpi=200,
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(
        derivatives[:n_plot],
        label="Finite-difference derivative",
    )
    plt.plot(
        derivative_predictions[:n_plot],
        label="Continuous SINDy prediction",
    )
    plt.xlabel("Time step")
    plt.ylabel("Rate derivative")
    plt.title("Continuous-time SINDy derivative fit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "vasicek_continuous_derivative.png",
        dpi=200,
    )
    plt.close()

    print(f"\nSaved figures to {FIGURE_DIR}/")


if __name__ == "__main__":
    run_vasicek_experiment()