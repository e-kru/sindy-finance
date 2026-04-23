# SINDy Finance

Sparse identification of nonlinear dynamical systems (SINDy) applied to financial factor dynamics.

## Overview

This project implements SINDy from scratch to discover interpretable dynamics in financial data.

The goal is to recover low-dimensional governing equations from time series using sparse regression techniques.

## Motivation

Financial systems are often modeled using stochastic differential equations (e.g. Ornstein-Uhlenbeck, Vasicek, stochastic volatility models).

This project explores whether such dynamics can be recovered directly from data using sparse system identification.

## Method

We model the system as:

X_{t+1} ≈ Θ(X_t) Ξ

where:
- Θ(X) is a library of candidate nonlinear functions
- Ξ is a sparse coefficient matrix learned via regression

The algorithm:
1. Construct feature library (polynomial basis)
2. Fit regression model
3. Apply thresholding (STLSQ) to enforce sparsity
4. Recover interpretable dynamics

## Project Structure

- `src/` — core implementation (SINDy, simulation, utilities)
- `notebooks/` — experiments and analysis
- `data/` — datasets
- `figures/` — plots and visualizations

## Current Status

- OU process simulation implemented
- Polynomial feature library implemented
- Initial SINDy setup completed

## Next Steps

- Implement STLSQ (sparse regression)
- Evaluate model on synthetic systems
- Apply to financial factor data (e.g. yield curve)

## Technologies

- Python
- NumPy
- SciPy
- Matplotlib

## Author

GitHub: https://github.com/e-kru