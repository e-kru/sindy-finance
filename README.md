# Sparse Discovery of Dynamical Systems with SINDy

This project implements a small from-scratch version of **SINDy** — Sparse Identification of Nonlinear Dynamics — and applies it to synthetic dynamical systems and financial time-series examples.

The goal is not to build a black-box forecasting model. Instead, the project studies when sparse equation discovery works well, when it fails, and why real financial data are harder than clean physical systems.

SINDy is based on a simple idea: many dynamical systems can be represented by only a few active terms from a larger candidate library. Given observations of a system and its derivatives, SINDy uses sparse regression to identify the governing equations.

---

## Project structure

```text
sindy-finance/
├── src/
│   ├── library.py
│   ├── stlsq.py
│   ├── discrete_sindy.py
│   ├── continuous_sindy.py
│   └── simulation.py
│
├── tests/
│   ├── test_library.py
│   ├── test_stlsq.py
│   └── test_continuous_sindy.py
│
├── notebooks/
│   ├── 01_vasicek_sindy.ipynb
│   ├── 02_real_yield_sindy.ipynb
│   ├── 04_yield_curve_factors_sindy.ipynb
│   └── 05_lorenz_sindy.ipynb
│
├── figures/
├── main.py
└── README.md
```

---

## Methodology

For a continuous-time dynamical system, SINDy starts from

$$\dot{x}(t) = f(x(t)).$$

The observed states are collected in a matrix `X`, and the time derivatives are collected in `dXdt`.

A candidate library is constructed from possible nonlinear functions of the state:

```text
Theta(X) = [1, X, X^2, ...]
```

SINDy then solves the sparse regression problem

$$\dot{X} \approx \Theta(X)\Xi.$$

Here:

- `Theta(X)` is the candidate library matrix.
- `Xi` is the sparse coefficient matrix.
- Each column of `Xi` corresponds to one discovered equation.

The key assumption is that only a few candidate terms are actually needed.

---

## Polynomial library

The function `build_polynomial_library` constructs polynomial feature libraries.

For one variable and degree 2:

```text
[1, x, x^2]
```

For two variables and degree 2:

```text
[1, x1, x2, x1^2, x1*x2, x2^2]
```

For three variables and degree 2:

```text
[1, x1, x2, x3, x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2]
```

This is the candidate space from which SINDy selects sparse governing equations.

---

## Sequential Thresholded Least Squares

The optimizer `stlsq` implements **Sequential Thresholded Least Squares**.

The regression problem is

$$Y \approx \Theta(X)\Xi.$$

The algorithm is:

1. Fit an initial least-squares model.
2. Set coefficients with absolute value below a threshold to zero.
3. Refit least squares only on the remaining active terms.
4. Repeat until convergence or until the maximum number of iterations is reached.

The threshold controls sparsity:

```text
small threshold  -> many active terms
large threshold  -> fewer active terms
```

---

## Discrete-time SINDy

Discrete-time SINDy fits models of the form

$$X_{t+1} \approx F(X_t).$$

For example, in one dimension:

$$x_{t+1} \approx c_0 + c_1 x_t + c_2 x_t^2.$$

This is useful for one-step prediction experiments.

---

## Continuous-time SINDy

Continuous-time SINDy fits models of the form

$$\dot{X} \approx \Theta(X)\Xi.$$

For example, in one dimension:

$$\dot{x} \approx c_0 + c_1 x + c_2 x^2.$$

This is closer to the original SINDy formulation for discovering governing differential equations.

---

# Experiments

---

## 1. Synthetic Vasicek / Ornstein-Uhlenbeck dynamics

Notebook:

```text
notebooks/01_vasicek_sindy.ipynb
```

The Vasicek short-rate model is a finance version of an Ornstein-Uhlenbeck process:

$$dr_t = \kappa(\theta-r_t)dt + \sigma dW_t.$$

Using an Euler approximation:

$$r_{t+\Delta t} = r_t + \kappa(\theta-r_t)\Delta t + \sigma\sqrt{\Delta t}\epsilon_t.$$

Ignoring the noise term for the conditional mean gives

$$\mathbb{E}[r_{t+\Delta t}\mid r_t] = \kappa\theta\Delta t + (1-\kappa\Delta t)r_t.$$

So the discrete-time model has the form

$$r_{t+1} \approx c_0 + c_1 r_t.$$

**Takeaway:** SINDy recovers known sparse mean-reverting dynamics on synthetic finance data.

---

## 2. Real Treasury yield experiment: DGS10

Notebook:

```text
notebooks/02_real_yield_sindy.ipynb
```

This experiment applies discrete-time SINDy to the real U.S. 10-year Treasury yield series `DGS10` from FRED.

The model fits

$$y_{t+1} \approx c_0 + c_1 y_t.$$

The estimated coefficient `c1` is very close to 1, so the fitted model is close to

$$y_{t+1} \approx y_t.$$

This means that the daily 10-year Treasury yield behaves like a highly persistent near-random-walk process.

**Takeaway:** On daily DGS10 data, SINDy mostly recovers persistence rather than useful nonlinear forecasting structure.

---

## 3. Real yield-curve factor dynamics

Notebook:

```text
notebooks/04_yield_curve_factors_sindy.ipynb
```

This experiment builds a three-factor representation of the Treasury yield curve.

```text
level     = DGS10
slope     = DGS10 - DGS2
curvature = 2*DGS5 - DGS2 - DGS10
```

The state vector is

$$X_t = (L_t, S_t, C_t).$$

The model studies daily factor changes:

$$\Delta X_{t+1} = X_{t+1} - X_t.$$

The quadratic candidate library is

```text
[1, L, S, C, L^2, L*S, L*C, S^2, S*C, C^2]
```

The model is

$$\Delta X_{t+1} \approx \Theta(L_t,S_t,C_t)\Xi.$$

The zero-change baseline is

$$\widehat{\Delta X}_{t+1}=0.$$

### Full-sample result

The full-sample linear SINDy model identifies economically interpretable mean-reversion-like coefficients, especially for slope and curvature.

For example, the linear library produced negative self-effects in the slope and curvature equations:

```text
slope     -> Delta slope      : negative
curvature -> Delta curvature  : negative
```

This is consistent with weak mean-reversion-like behavior in yield-curve shape factors.

However, out-of-sample validation shows that the zero-change baseline is difficult to beat.

**Takeaway:** SINDy identifies plausible candidate structures, but daily yield-curve factor changes are too noisy for robust one-day forecasting in this setup.

---

## 4. Threshold sweep on yield-curve factors

The quadratic library contains 30 possible coefficients because there are 10 library terms and 3 target equations.

A threshold sweep shows how sparsity changes with the STLSQ threshold.

For small thresholds, the model remains dense. For a high threshold,

$$\lambda = 0.1,$$

the model becomes much more sparse and keeps only three active terms:

$$\Delta L_{t+1} = -0.417483 C_t^2,$$

$$\Delta S_{t+1} = 0.304801 C_t^2,$$

$$\Delta C_{t+1} = -0.179575 L_t C_t.$$

The most interpretable equation is

$$\Delta C_{t+1} = -0.179575 L_t C_t.$$

Since nominal rate levels are usually positive, this behaves like curvature mean reversion:

$$\Delta C_{t+1} \approx -k_t C_t, \quad k_t = 0.179575 L_t > 0.$$

Thus, positive curvature is pulled downward and negative curvature is pulled upward. This suggests a level-dependent curvature mean-reversion candidate structure.

However, the out-of-sample improvement over the zero-change baseline is very small.

**Takeaway:** Thresholding improves sparsity and interpretability, but it does not create robust predictive power.

---

## 5. Regime-specific yield-curve analysis

The project also tests a shorter monetary-policy regime starting on `2024-09-18`, the beginning of the recent Fed cutting-cycle period.

The motivation is that SINDy assumes relatively stable dynamics. A single equation over several decades of financial data may be too restrictive because rates are regime-dependent.

The regime-specific linear SINDy model produces stronger mean-reversion-like self-effects:

```text
level     -> Delta level      : negative
slope     -> Delta slope      : negative
curvature -> Delta curvature  : negative
```

Out-of-sample validation is mixed:

- The linear model improves the level forecast relative to the zero-change baseline.
- It performs worse for slope and curvature.
- The quadratic model performs worse across all three factors.

**Takeaway:** Regime selection can make discovered equations more interpretable, but it does not automatically create robust predictive power.

---

## 6. Lorenz system: positive control experiment

Notebook:

```text
notebooks/05_lorenz_sindy.ipynb
```

The Lorenz system is a classic nonlinear dynamical system:

$$\dot{x} = \sigma(y-x),$$

$$\dot{y} = x(\rho-z)-y,$$

$$\dot{z} = xy-\beta z.$$

Using the classical parameters

```text
sigma = 10
rho   = 28
beta  = 8/3
```

the system becomes

$$\dot{x} = -10x + 10y,$$

$$\dot{y} = 28x - y - xz,$$

$$\dot{z} = xy - \frac{8}{3}z.$$

SINDy is given the larger degree-2 polynomial library

```text
[1, x, y, z, x^2, x*y, x*z, y^2, y*z, z^2]
```

The true equations only require a few of these candidate terms:

```text
x_dot: x, y
y_dot: x, y, x*z
z_dot: x*y, z
```

SINDy recovers the equations almost exactly:

$$\dot{x} = -10x + 10y,$$

$$\dot{y} = 28x - y - xz,$$

$$\dot{z} = xy - 2.6667z.$$

The derivative MSE is essentially zero, and the coefficient error is at numerical precision.

**Takeaway:** When the true dynamics are sparse, stable, and contained in the candidate library, SINDy recovers the governing equations almost exactly.

---

# Main conclusion

This project shows both the strength and the limitation of SINDy.

## Where SINDy works well

SINDy performs very well when:

- the underlying system is governed by stable differential equations,
- the observed state variables form a reasonably closed system,
- the correct candidate terms are included in the library,
- the dynamics are sparse in that library,
- derivative estimates are clean.

This is clearly visible in the Lorenz experiment.

## Where SINDy struggles

SINDy is less effective as a direct forecasting tool for daily financial time series.

Real yield data are affected by:

- monetary policy,
- inflation expectations,
- growth expectations,
- term premia,
- risk sentiment,
- regime changes,
- macro news shocks.

Therefore, daily Treasury yield factors do not behave like a closed physical system in the observed variables alone.

**Final takeaway:** SINDy is useful as an interpretable model-discovery tool, but not as a strong one-day forecasting model for daily Treasury yields.

---

# How to run

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy requests certifi pytest
```

Run tests:

```bash
pytest
```

Run the main script:

```bash
python main.py
```

Open notebooks:

```bash
jupyter notebook
```

or open the notebooks directly in PyCharm.

---

# Key dependencies

```text
numpy
pandas
matplotlib
scipy
requests
certifi
pytest
```

---

# Status

This is an educational research project built from scratch for a seminar on Scientific Machine Learning and SINDy.

The implementation is intentionally simple and focuses on understanding the method rather than replacing mature packages such as PySINDy.

---

# References

- Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. Proceedings of the National Academy of Sciences.
- PySINDy documentation: Sparse Identification of Nonlinear Dynamics.
- Federal Reserve H.15: Treasury constant maturity yield methodology.
- Lorenz system: classical nonlinear system with parameters $\sigma=10$, $\rho=28$, $\beta=8/3$.
