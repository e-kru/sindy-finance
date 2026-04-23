import numpy as np
from src.simulation import simulate_ou
from src.library import build_polynomial_library

# simulate data
x = simulate_ou()

# prepare data
X = x[:-1].reshape(-1, 1)
Y = x[1:].reshape(-1, 1)

# build library
Theta = build_polynomial_library(X)

print("Shapes:")
print("X:", X.shape)
print("Theta:", Theta.shape)