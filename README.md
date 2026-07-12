# Sparse Identification of Dynamical Systems from Scratch

A from-scratch implementation of the core **Sparse Identification of Nonlinear Dynamics (SINDy)** pipeline, with experiments on chaotic systems, stochastic mean reversion, and U.S. Treasury yields.

The project studies a central question in scientific machine learning:

> Can we recover interpretable governing equations directly from observed data?

The SINDy methodology and Sequential Thresholded Least Squares procedure follow Brunton, Proctor, and Kutz (2016) [1]. The core SINDy components in this repository were implemented specifically for this project using NumPy rather than relying on PySINDy.

<p align="center">
  <img src="figures/vasicek_path.png" alt="Simulated Vasicek short-rate path" width="800">
</p>

## Project overview

This repository implements:

- polynomial candidate-library construction,
- Sequential Thresholded Least Squares,
- discrete-time SINDy,
- continuous-time SINDy,
- numerical simulation,
- finite-difference derivative estimation,
- chronological train/test evaluation,
- comparisons against simple forecasting baselines,
- unit tests with `pytest`.

The experiments deliberately include both successful and unsuccessful applications:

1. **Lorenz system:** near-exact equation recovery under clean conditions.
2. **Vasicek process:** recovery of a stochastic mean-reverting drift.
3. **10-year Treasury yield:** persistence dominates one-day prediction.
4. **Yield-curve factors:** weak and non-uniform predictive structure.

## Attribution and scope

The mathematical ideas used in this project are based on established literature:

- the SINDy framework and STLSQ algorithm follow Brunton et al. (2016) [1],
- the Lorenz experiment uses the classical Lorenz system [2],
- the mean-reverting short-rate experiment uses the Vasicek model [3],
- the Treasury-yield data come from FRED [4].

All core SINDy components in `src/` were implemented specifically for this project using NumPy rather than relying on PySINDy.

The repository is intended to make the mechanics of sparse equation discovery transparent. It does not aim to replace mature libraries such as PySINDy [5].

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

- $\Theta(X)$ is the candidate-library matrix,
- $\Xi$ is the sparse coefficient matrix,
- each column of $\Xi$ defines one discovered state equation.

For three state variables and polynomial degree two, the library is

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

SINDy then solves a sparse regression problem:

1. fit least squares,
2. remove coefficients below a threshold,
3. refit using only the surviving terms,
4. repeat until the active set stabilizes.

## Experiments and key results

| Experiment | Main question | Result |
|---|---|---|
| **Lorenz system** | Can SINDy recover known nonlinear equations? | The correct 7 active coefficients are recovered from 30 candidates, with error near numerical precision. |
| **Vasicek process** | Can SINDy identify stochastic mean reversion? | The discrete-time model closely recovers the theoretical Euler coefficients. The continuous-time model recovers the correct drift structure but is noisier. |
| **10-year Treasury yield** | Can SINDy improve one-day forecasts? | The learned coefficient is approximately 0.9997. Linear SINDy performs almost identically to persistence, while the quadratic model is marginally worse. |
| **Yield-curve factors** | Can sparse models identify level, slope, and curvature dynamics? | Small improvements appear for level and slope, but not for curvature. Linear regime models are more interpretable than dense quadratic models. |

## 1. Lorenz system

Notebook: [`04_lorenz_sindy.ipynb`](notebooks/04_lorenz_sindy.ipynb)

The Lorenz system [2] is

```math
\dot x=-10x+10y,
```

```math
\dot y=28x-y-xz,
```

```math
\dot z=xy-\frac{8}{3}z.
```

SINDy receives the larger quadratic library

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
\right].
```

Across three equations, this creates 30 possible coefficients. Only seven are active in the true system.

Using exact derivatives, SINDy recovers:

- the correct seven active terms,
- the correct coefficient values,
- derivative error close to numerical precision.

This serves as a **positive control**. The system is:

- closed,
- governed by a fixed autonomous system,
- sparse,
- correctly represented by the candidate library,
- observed without derivative-estimation noise.

A small threshold sensitivity experiment also shows that the correct support remains stable across a broad threshold range. Once the threshold exceeds the magnitude of genuine coefficients, true terms are removed and the recovery error increases sharply.

## 2. Vasicek mean-reversion experiment

Notebook: [`01_vasicek_sindy.ipynb`](notebooks/01_vasicek_sindy.ipynb)

The Vasicek short-rate model [3] is

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

Therefore, the drift is sparse in the library

```math
\Theta(r)=[1,r].
```

### Discrete-time formulation

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

The conditional one-step relation is

```math
\mathbb{E}[r_{k+1}\mid r_k]
=
\kappa\theta\Delta t
+
(1-\kappa\Delta t)r_k.
```

The discrete-time SINDy model closely recovers these theoretical coefficients.

<p align="center">
  <img src="figures/vasicek_discrete_prediction.png" alt="Discrete-time SINDy prediction on Vasicek data" width="800">
</p>

### Continuous-time formulation

Finite-difference derivatives satisfy

```math
\frac{r_{k+1}-r_k}{\Delta t}
=
\kappa(\theta-r_k)
+
\frac{\sigma}{\sqrt{\Delta t}}\varepsilon_k.
```

The stochastic term is amplified by a factor proportional to

```math
\frac{1}{\sqrt{\Delta t}}.
```

As a result, the continuous-time model identifies the correct positive constant and negative linear drift structure, but its coefficients are noisier than in the discrete-time model.

<p align="center">
  <img src="figures/vasicek_continuous_derivative.png" alt="Continuous-time SINDy derivative fit" width="800">
</p>

The experiment also illustrates an important limitation:

> Deterministic SINDy identifies the drift structure, not the Brownian-motion term of the full stochastic differential equation.

## 3. Real 10-year Treasury yield

Notebook: [`02_real_yield_sindy.ipynb`](notebooks/02_real_yield_sindy.ipynb)

This experiment uses the U.S. 10-Year Treasury Constant Maturity Rate (`DGS10`) from FRED [4].

Let $y_t$ denote the yield observed on day $t$. The linear model is

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

The coefficient on the current yield is extremely close to one. This indicates that daily 10-year Treasury yields are highly persistent and behave approximately like a random walk over a one-day horizon.

The implied fixed point is

```math
\bar y
=
\frac{c_0}{1-c_1}.
```

It is approximately 6.2% in this sample. However, because $c_1$ is extremely close to one, the fixed-point estimate is highly sensitive to small coefficient changes and should not be interpreted as a robust long-run equilibrium.

The models are compared against the persistence baseline

```math
\widehat y_{t+1}=y_t.
```

Out of sample:

- linear SINDy performs almost identically to persistence,
- quadratic SINDy performs marginally worse,
- the nonlinear term does not provide meaningful predictive value.

The dominant one-day structure is persistence.

## 4. Treasury yield-curve factors

Notebook: [`03_yield_curve_factors_sindy.ipynb`](notebooks/03_yield_curve_factors_sindy.ipynb)

The experiment constructs three transparent factor proxies from the 2-, 5-, and 10-year Treasury yields:

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

They are simple observable proxies rather than estimated Nelson–Siegel factors or principal components.

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

The models are evaluated against the zero-change baseline

```math
\widehat{\Delta X}_{t+1}=0.
```

### Full-sample results

- Linear and quadratic SINDy slightly improve the test MSE for level.
- The largest improvement appears for slope.
- Both models underperform the baseline for curvature.
- The quadratic model performs particularly poorly for curvature.

### Regime-specific results

The linear regime model identifies negative self-effects for:

- level,
- slope,
- curvature.

These coefficients are consistent with weak mean-reversion-like behavior.

The quadratic regime model produces dense equations with large interaction coefficients. Given the shorter sample and correlated polynomial features, these coefficients are treated as unstable rather than as robust economic relationships.

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

7. **SINDy is more convincing here as a model-discovery tool than as a one-day financial forecasting model.**

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

The script:

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

## Limitations

- The polynomial library currently supports degrees 1 and 2 only.
- STLSQ uses a fixed absolute threshold.
- Continuous-time SINDy relies on forward finite differences.
- The implementation does not include regularization or smoothing.
- The implementation does not estimate uncertainty.
- Hyperparameter selection is not automated.
- Financial factor dynamics are not a closed system.
- Real Treasury yields are affected by omitted macroeconomic variables and regime changes.

These limitations are intentional. The implementation is designed to make sparse equation discovery transparent rather than to replace mature libraries.

## References

### Methodology

[1] Brunton, S. L., Proctor, J. L., and Kutz, J. N. (2016).  
*Discovering governing equations from data by sparse identification of nonlinear dynamical systems.*  
Proceedings of the National Academy of Sciences, 113(15), 3932–3937.

### Dynamical systems and financial models

[2] Lorenz, E. N. (1963).  
*Deterministic nonperiodic flow.*  
Journal of the Atmospheric Sciences, 20(2), 130–141.

[3] Vasicek, O. (1977).  
*An equilibrium characterization of the term structure.*  
Journal of Financial Economics, 5(2), 177–188.

### Data

[4] Federal Reserve Economic Data (FRED), Federal Reserve Bank of St. Louis.  
Series used: `DGS2`, `DGS5`, and `DGS10`.

### Reference implementation

[5] Kaptanoglu, A. A. et al. (2022).  
*PySINDy: A comprehensive Python package for robust sparse system identification.*  
Journal of Open Source Software, 7(69), 3994.