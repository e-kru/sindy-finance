import numpy as np

def simulate_ou(n_steps=1000, dt=0.01, theta=1.0, mu=0.0, sigma=0.2):
    x = np.zeros(n_steps)

    for t in range(1, n_steps):
        x[t] = x[t-1] + theta*(mu - x[t-1])*dt + sigma*np.sqrt(dt)*np.random.randn()

    return x