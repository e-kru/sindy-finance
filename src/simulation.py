import numpy as np


def simulate_vasicek(
    n_steps: int = 1000,
    dt: float = 0.01,
    kappa: float = 1.5,
    theta: float = 0.03,
    sigma: float = 0.02,
    r0: float = 0.01,
    seed: int | None = 0,
) -> np.ndarray:
    """
    Simulate the Vasicek short-rate model with Euler-Maruyama.

    The model is

        dr_t = kappa * (theta - r_t) dt + sigma dW_t,

    where
        kappa : mean-reversion speed,
        theta : long-run mean,
        sigma : volatility.

    Parameters
    ----------
    n_steps:
        Number of simulated observations.
    dt:
        Time-step size.
    kappa:
        Mean-reversion speed.
    theta:
        Long-run mean.
    sigma:
        Diffusion volatility.
    r0:
        Initial short rate.
    seed:
        Random seed. Use None for non-deterministic output.

    Returns
    -------
    np.ndarray
        Simulated short-rate path with shape (n_steps,).
    """
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")

    if dt <= 0:
        raise ValueError("dt must be positive")

    if kappa < 0:
        raise ValueError("kappa must be non-negative")

    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)

    rates = np.zeros(n_steps, dtype=float)
    rates[0] = r0

    for t in range(1, n_steps):
        previous_rate = rates[t - 1]

        drift = kappa * (theta - previous_rate) * dt
        diffusion = sigma * np.sqrt(dt) * rng.normal()

        rates[t] = previous_rate + drift + diffusion

    return rates