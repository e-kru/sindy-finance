from src.simulation import simulate_ou
from src.library import build_polynomial_library
from src.stlsq import stlsq

x = simulate_ou()

X = x[:-1].reshape(-1, 1)
Y = x[1:].reshape(-1, 1)

Theta, feature_names = build_polynomial_library(X, degree=2)

Xi = stlsq(Theta, Y, threshold=0.05, max_iter=10)

print("Feature names:", feature_names)
print("Xi shape:", Xi.shape)
print("Recovered coefficients:")
print(Xi)