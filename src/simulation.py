import numpy as np

def simulate_ou(n_steps=1000, dt=0.01, theta=1.0, mu=0.0, sigma=0.2):
    x = np.zeros(n_steps)

    for t in range(1, n_steps):
        x[t] = x[t-1] + theta*(mu - x[t-1])*dt + sigma*np.sqrt(dt)*np.random.randn()

    return x


def simulate_vasicek(n_steps=1000, dt=0.01, kappa=1.5, theta=0.03, sigma=0.02, r0=0.01, seed=0):
    """
    Simulate the Vasicek short-rate model using Euler-Maruyama.

    Model:
        dr_t = kappa * (theta - r_t) dt + sigma dW_t
    """
    np.random.seed(seed)

    r = np.zeros(n_steps)
    r[0] = r0

    for t in range(1, n_steps):
        drift = kappa * (theta - r[t - 1]) * dt
        diffusion = sigma * np.sqrt(dt) * np.random.randn()
        r[t] = r[t - 1] + drift + diffusion

    return r