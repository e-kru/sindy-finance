import numpy as np
import matplotlib.pyplot as plt

from src.simulation import simulate_vasicek
from src.discrete_sindy import fit_discrete_sindy, predict_discrete_sindy
from src.continous_sindy import fit_continuous_sindy, predict_derivative_continuous_sindy


def format_equation(feature_names, Xi, target_name="y", tolerance=1e-10):
    """Format learned coefficients as a readable equation."""
    terms = []

    for name, coefficient in zip(feature_names, Xi[:, 0]):
        if abs(coefficient) <= tolerance:
            continue

        if name == "1":
            terms.append(f"{coefficient:.4f}")
        else:
            terms.append(f"{coefficient:.4f}*{name}")

    if not terms:
        return f"{target_name} = 0"

    return f"{target_name} = " + " + ".join(terms)


# Parameters for the Vasicek / OU short-rate process
n_steps = 2000
dt = 0.01
kappa = 1.5
theta = 0.03
sigma = 0.02
r0 = 0.01

# Simulate Vasicek short-rate data
r = simulate_vasicek(
    n_steps=n_steps,
    dt=dt,
    kappa=kappa,
    theta=theta,
    sigma=sigma,
    r0=r0,
    seed=42,
)

R_full = r.reshape(-1, 1)

# ------------------------------------------------------------
# 1) Discrete-time SINDy: r_{k+1} ≈ F(r_k)
# ------------------------------------------------------------
X_discrete = r[:-1].reshape(-1, 1)
Y_discrete = r[1:].reshape(-1, 1)

Xi_discrete, names_discrete = fit_discrete_sindy(
    X_discrete,
    Y_discrete,
    degree=1,
    threshold=0.00001,
    max_iter=10,
)

Y_discrete_hat = predict_discrete_sindy(
    X_discrete,
    Xi_discrete,
    degree=1,
)

discrete_mse = np.mean((Y_discrete - Y_discrete_hat) ** 2)
discrete_equation = format_equation(
    names_discrete,
    Xi_discrete,
    target_name="r_next",
)

# ------------------------------------------------------------
# 2) Continuous-time SINDy: r_dot ≈ f(r)
# ------------------------------------------------------------
Xi_continuous, names_continuous, R_current, dRdt = fit_continuous_sindy(
    R_full,
    dt=dt,
    degree=1,
    threshold=0.00001,
    max_iter=10,
)

dRdt_hat = predict_derivative_continuous_sindy(
    R_current,
    Xi_continuous,
    degree=1,
)

continuous_mse = np.mean((dRdt - dRdt_hat) ** 2)
continuous_equation = format_equation(
    names_continuous,
    Xi_continuous,
    target_name="r_dot",
)

expected_discrete_constant = kappa * theta * dt
expected_discrete_linear = 1 - kappa * dt
expected_continuous_constant = kappa * theta
expected_continuous_linear = -kappa

# ------------------------------------------------------------
# 3) Print comparison
# ------------------------------------------------------------
print("Discrete-time SINDy")
print("Feature names:", names_discrete)
print("Xi:")
print(Xi_discrete)
print("Recovered equation:")
print(discrete_equation)
print("MSE:", discrete_mse)
print()

print("Continuous-time SINDy")
print("Feature names:", names_continuous)
print("Xi:")
print(Xi_continuous)
print("Recovered equation:")
print(continuous_equation)
print("MSE:", continuous_mse)
print()

print("Theoretical comparison for Vasicek / OU short-rate model:")
print(f"Expected discrete form: r_next ≈ {expected_discrete_constant:.6f} + {expected_discrete_linear:.6f}*r")
print(f"Expected continuous form: r_dot ≈ {expected_continuous_constant:.6f} + {expected_continuous_linear:.6f}*r")

# ------------------------------------------------------------
# 4) Plot comparison
# ------------------------------------------------------------
time = np.arange(n_steps) * dt
n_plot = 300

plt.figure(figsize=(10, 5))
plt.plot(time[:n_plot], r[:n_plot], label="Simulated Vasicek short rate")
plt.axhline(theta, linestyle="--", label="Long-run mean theta")
plt.xlabel("Time")
plt.ylabel("Short rate")
plt.title("Simulated Vasicek / OU short-rate path")
plt.legend()
plt.tight_layout()
plt.savefig("figures/vasicek_path.png", dpi=200)

plt.figure(figsize=(10, 5))
plt.plot(Y_discrete[:n_plot], label="True r_next")
plt.plot(Y_discrete_hat[:n_plot], label="Discrete SINDy prediction")
plt.xlabel("Time step")
plt.ylabel("Short rate")
plt.title("Discrete-time SINDy prediction on Vasicek data")
plt.legend()
plt.tight_layout()
plt.savefig("figures/vasicek_discrete_prediction.png", dpi=200)

plt.figure(figsize=(10, 5))
plt.plot(dRdt[:n_plot], label="Finite-difference derivative")
plt.plot(dRdt_hat[:n_plot], label="Continuous SINDy derivative prediction")
plt.xlabel("Time step")
plt.ylabel("r_dot")
plt.title("Continuous-time SINDy derivative fit on Vasicek data")
plt.legend()
plt.tight_layout()
plt.savefig("figures/vasicek_continuous_derivative.png", dpi=200)

print("Saved plots:")
print("figures/vasicek_path.png")
print("figures/vasicek_discrete_prediction.png")
print("figures/vasicek_continuous_derivative.png")