# Sparse Identification of Dynamical Systems from Scratch

A compact, from-scratch implementation of **Sparse Identification of Nonlinear Dynamics (SINDy)**, with experiments on synthetic dynamical systems, chaotic dynamics, and U.S. Treasury yields.

The project focuses on a central question in scientific machine learning:

> Can we recover interpretable governing equations directly from data?

Rather than using SINDy as a black-box package, this repository implements the core pipeline in NumPy and studies both where sparse equation discovery works and where it breaks down.

<p align="center">
  <img src="figures/vasicek_path.png" alt="Simulated Vasicek short-rate path" width="800">
</p>

## What this project demonstrates

- Construction of polynomial candidate libraries from scratch
- Sequential Thresholded Least Squares (STLSQ)
- Discrete-time and continuous-time SINDy
- Numerical simulation and derivative estimation
- Unit testing with `pytest`
- Chronological train/test evaluation for time-series data
- Comparison against simple forecasting baselines
- Critical analysis of model assumptions and failure modes

## Core idea

For a continuous-time dynamical system,

```math
\dot X(t)=f(X(t)),
```

SINDy assumes that the dynamics can be represented by a sparse combination of candidate functions:

```math
\dot X \approx \Theta(X)\Xi.
```

Here:

- $\Theta(X)$ is a candidate-library matrix,
- $\Xi$ is a sparse coefficient matrix,
- each column of $\Xi$ defines one discovered state equation.

For example, a quadratic library for three variables contains

```math
\Theta(X)
=
\left[
1,\,
x,\,
y,\,
z,\,
x^2,\,
xy,\,
xz,\,
y^2,\,
yz,\,
z^2
\right].
```

The optimizer first fits least squares, removes small coefficients, and then refits only on the surviving terms.

## Experiments and key results

| Experiment | Main question | Result |
|---|---|---|
| **Lorenz system** | Can SINDy recover known nonlinear equations? | The correct 7 active coefficients are recovered from 30 candidates, with coefficient error near numerical precision. |
| **Vasicek / Ornstein–Uhlenbeck process** | Can SINDy identify mean reversion? | The discrete-time model closely recovers the theoretical Euler coefficients. The continuous-time model recovers the correct drift structure but is noisier because finite differences amplify diffusion noise. |
| **10-year Treasury yield** | Does SINDy improve one-day forecasts? | The learned coefficient is approximately $0.9997$, showing strong persistence. Linear SINDy performs almost identically to the persistence baseline, while the quadratic model is marginally worse. |
| **Yield-curve factors** | Can sparse equations identify dynamics in level, slope, and curvature? | Small out-of-sample improvements appear for level and slope, but not for curvature. Linear regime models are more interpretable than dense quadratic models. |

## 1. Lorenz system

The Lorenz equations are

```math
\dot x=-10x+10y,
```

```math
\dot y=28x-y-xz,
```

```math
\dot z=xy-\frac{8}{3}z.
```

SINDy is given the larger quadratic candidate library

```math
\left[
1,\,
x,\,
y,\,
z,\,
x^2,\,
xy,\,
xz,\,
y^2,\,
yz,\,
z^2
\right],
```

and correctly selects only the required terms.

Across the three equations, the library contains 30 possible coefficients. Only seven are active in the true Lorenz system.

This serves as a **positive control**: the system is closed, stationary, sparse, and fully represented by the candidate library.

## 2. Vasicek mean-reversion experiment

The Vasicek short-rate model is

```math
dr_t
=
\kappa(\theta-r_t)\,dt
+
\sigma\,dW_t.
```

Its deterministic drift is linear:

```math
\kappa(\theta-r_t)
=
\kappa\theta-\kappa r_t.
```

Therefore, the true drift is sparse in the library

```math
\Theta(r)=[1,r].
```

The discrete-time SINDy model learns the conditional one-step transition, while the continuous-time model fits finite-difference derivative estimates.

Using Euler–Maruyama discretization,

```math
r_{k+1}
=
r_k
+
\kappa(\theta-r_k)\Delta t
+
\sigma\sqrt{\Delta t}\,\varepsilon_k.
```

The conditional one-step relation is therefore

```math
\mathbb{E}[r_{k+1}\mid r_k]
=
\kappa\theta\Delta t
+
(1-\kappa\Delta t)r_k.
```

The discrete-time model closely recovers these theoretical coefficients.

<p align="center">
  <img src="figures/vasicek_discrete_prediction.png" alt="Discrete-time SINDy prediction on Vasicek data" width="800">
</p>

For the continuous-time model, finite differences produce

```math
\frac{r_{k+1}-r_k}{\Delta t}
=
\kappa(\theta-r_k)
+
\frac{\sigma}{\sqrt{\Delta t}}\varepsilon_k.
```

The stochastic term is therefore amplified by a factor proportional to

```math
\frac{1}{\sqrt{\Delta t}}.
```

This makes continuous-time coefficient recovery noisier than discrete-time estimation.

<p align="center">
  <img src="figures/vasicek_continuous_derivative.png" alt="Continuous-time SINDy derivative fit" width="800">
</p>

The experiment shows that SINDy recovers the correct mean-reverting drift structure, but not the full stochastic differential equation. The Brownian-motion term is not represented by the deterministic feature library.

## 3. Real 10-year Treasury-yield experiment

The first real-data experiment uses the U.S. 10-Year Treasury Constant Maturity Rate (`DGS10`) from FRED.

Let $y_t$ denote the yield observed on day $t$. We fit the discrete-time model

```math
y_{t+1}
\approx
c_0+c_1y_t.
```

The learned equation is approximately

```math
y_{t+1}
\approx
0.00001642
+
0.999735\,y_t.
```

The coefficient on today's yield is extremely close to one. This shows that daily 10-year Treasury yields are highly persistent and behave approximately like a random walk over a one-day horizon.

The implied fixed point is

```math
\bar y
=
\frac{c_0}{1-c_1}.
```

In this experiment, it is approximately $6.2\%$. However, because $c_1$ is extremely close to one, the fixed-point estimate is highly sensitive to small coefficient changes and should not be interpreted as a robust long-run equilibrium.

The models are evaluated against the persistence baseline

```math
\widehat y_{t+1}=y_t.
```

Out of sample:

- linear SINDy performs almost identically to persistence,
- quadratic SINDy performs marginally worse,
- the additional nonlinear term does not provide meaningful predictive value.

The dominant one-day structure is persistence.

## 4. Treasury yield-curve factors

The second real-data experiment constructs three observable yield-curve factors:

```math
L_t=y_t^{10},
```

```math
S_t=y_t^{10}-y_t^{2},
```

```math
C_t=2y_t^{5}-y_t^{2}-y_t^{10}.
```

These represent:

- **level**,
- **slope**,
- **curvature**.

The state vector is

```math
X_t
=
\begin{pmatrix}
L_t\\
S_t\\
C_t
\end{pmatrix}.
```

The model predicts daily factor changes:

```math
\Delta X_{t+1}
=
X_{t+1}-X_t.
```

SINDy fits

```math
\Delta X_{t+1}
\approx
\Theta(X_t)\Xi.
```

The models are compared against the zero-change baseline

```math
\widehat{\Delta X}_{t+1}=0.
```

The full-sample results are mixed:

- linear and quadratic SINDy slightly improve the test MSE for level,
- the largest improvement occurs for slope,
- both models underperform the baseline for curvature,
- the quadratic model performs particularly poorly for curvature.

The regime-specific linear model identifies negative self-effects for level, slope, and curvature, which is consistent with weak mean-reversion-like behavior.

However, the quadratic regime model produces dense equations with large interaction coefficients. These coefficients are difficult to interpret reliably and are more consistent with instability, multicollinearity, or overfitting than with robust economic structure.

## Main conclusions

1. **SINDy works well when the true system is sparse and contained in the candidate library.**  
   This is demonstrated by the near-exact recovery of the Lorenz equations.

2. **SINDy identifies the correct mean-reverting structure in the Vasicek experiment.**  
   The discrete-time coefficients closely match the theoretical Euler coefficients.

3. **Continuous-time identification is sensitive to stochastic noise.**  
   Finite differences amplify diffusion noise by a factor proportional to $1/\sqrt{\Delta t}$.

4. **Daily Treasury yields are dominated by persistence.**  
   Linear SINDy performs almost identically to a persistence baseline.

5. **Additional nonlinear terms do not automatically improve forecasting.**  
   Quadratic libraries can produce dense or unstable equations and may perform worse out of sample.

6. **Interpretability and predictive performance are different.**  
   Economically plausible coefficients do not necessarily imply robust forecasts.

7. **SINDy is more convincing here as a model-discovery tool than as a one-day forecasting model.**

The financial experiments should be understood as model-discovery exercises, not as evidence of a profitable trading strategy.

## Repository structure

```text
sindy-finance/
├── figures/
│   ├── vasicek_path.png
│   ├── vasicek_discrete_prediction.png
│   └── vasicek_continuous_derivative.png
├── notebooks/
│   ├── 01_vasicek_sindy.ipynb
│   ├── 02_real_yield_sindy.ipynb
│   ├── 03_yield_curve_factors_sindy.ipynb
│   └── 04_lorenz_sindy.ipynb
├── src/
│   ├── __init__.py
│   ├── continuous_sindy.py
│   ├── discrete_sindy.py
│   ├── library.py
│   ├── simulation.py
│   └── stlsq.py
├── tests/
│   ├── test_continuous_sindy.py
│   ├── test_discrete_sindy.py
│   ├── test_library.py
│   └── test_stlsq.py
├── main.py
├── pytest.ini
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/e-kru/sindy-finance.git
cd sindy-finance
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- NumPy
- pandas
- Matplotlib
- SciPy
- requests
- certifi
- pytest
- Jupyter

## Run the tests

```bash
pytest -v
```

The test suite covers:

- polynomial-library construction,
- sparse coefficient recovery,
- discrete-time prediction,
- continuous-time derivative estimation,
- input validation,
- error handling.

## Run the Vasicek example

```bash
python main.py
```

This script:

1. simulates a Vasicek short-rate process,
2. fits discrete-time SINDy,
3. fits continuous-time SINDy,
4. prints the recovered equations,
5. saves the main figures.

## Run the notebooks

```bash
jupyter notebook
```

The Treasury-yield notebooks download public FRED data at runtime and therefore require an internet connection.

## Main limitations

- The polynomial library currently supports degrees 1 and 2 only.
- STLSQ uses a fixed absolute threshold.
- Continuous-time SINDy relies on forward finite differences.
- The implementation does not include regularization or smoothing.
- The implementation does not estimate uncertainty.
- Hyperparameter selection is not automated.
- Financial factor dynamics are not a closed system.
- Real Treasury yields are affected by omitted macroeconomic variables and regime changes.

These limitations are intentional. The implementation is designed to make the mechanics of sparse equation discovery transparent rather than to replace mature libraries such as PySINDy.

## References

- Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems*. Proceedings of the National Academy of Sciences.
- Kaptanoglu, A. A. et al. (2022). *PySINDy: A comprehensive Python package for robust sparse system identification*. Journal of Open Source Software.
- Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis.