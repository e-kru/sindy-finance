import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_ou
from src.discrete_sindy import fit_discrete_sindy, predict_discrete_sindy


# Formatting function for recovered equation
def format_equation(feature_names, Xi, target_name="x_next", tolerance=1e-10):
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

x = simulate_ou()

X = x[:-1].reshape(-1, 1)
Y = x[1:].reshape(-1, 1)

Xi, feature_names = fit_discrete_sindy(
    X,
    Y,
    degree=2,
    threshold=0.05,
    max_iter=10,
)

Y_hat = predict_discrete_sindy(X, Xi, degree=2)

mse = np.mean((Y - Y_hat) ** 2)

# Format recovered equation
equation = format_equation(feature_names, Xi)

print("Feature names:", feature_names)
print("Xi:")
print(Xi)

print("Recovered equation:")
print(equation)

print("First 5 predictions:")
print(Y_hat[:5])

print("First 5 true values:")
print(Y[:5])

print("Mean squared error:")
print(mse)

n_plot = 100

plt.figure(figsize=(10, 5))
plt.plot(Y[:n_plot], label="True values")
plt.plot(Y_hat[:n_plot], label="SINDy predictions")
plt.xlabel("Time step")
plt.ylabel("x")
plt.title("Discrete-time SINDy prediction on OU process")
plt.legend()
plt.tight_layout()
output_path = "figures/ou_prediction.png"
plt.savefig(output_path, dpi=200)
print(f"Saved plot to {output_path}")