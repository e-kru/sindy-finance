# Sparse Discovery of Dynamical Systems with SINDy

This project implements a small from-scratch version of **SINDy** — Sparse Identification of Nonlinear Dynamics — and applies it to synthetic dynamical systems and financial time-series examples.

The goal is not to build a black-box forecasting model. Instead, the project studies when sparse equation discovery works well, when it fails, and why real financial data are harder than clean physical systems.

SINDy is based on the idea that many dynamical systems can be represented by only a few active terms from a larger candidate library. Given state observations and their time derivatives, SINDy solves a sparse regression problem to identify the governing equations. The original SINDy framework was introduced by Brunton, Proctor, and Kutz, and PySINDy provides a widely used Python implementation of the method.

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

## 1. SINDy methodology

The central SINDy assumption is that the dynamics of a system can be written as a sparse combination of candidate functions.

For a continuous-time dynamical system,

$$
\dot{x}(t) = f(x(t)),
$$

we collect state observations in a matrix

$$
X =
\begin{pmatrix}
x_1^\top \\
x_2^\top \\
\vdots \\
x_n^\top
\end{pmatrix},
$$

and derivative observations in

$$
\dot{X} =
\begin{pmatrix}
\dot{x}_1^\top \\
\dot{x}_2^\top \\
\vdots \\
\dot{x}_n^\top
\end{pmatrix}.
$$

A candidate library is then constructed:

$$
\Theta(X)
=
\begin{pmatrix}
1 & X & X^2 & \cdots
\end{pmatrix}.
$$

SINDy solves the sparse regression problem

$$
\dot{X} \approx \Theta(X)\Xi,
$$

where the coefficient matrix $\Xi$ should be sparse.

Each column of $\Xi$ represents one discovered governing equation.

---

## 2. Polynomial library

The function `build_polynomial_library` constructs polynomial candidate libraries.

For one variable and degree 2:

$$
\Theta(x) = [1, x, x^2].
$$

For two variables and degree 2:

$$
\Theta(x_1,x_2)
=
[1, x_1, x_2, x_1^2, x_1x_2, x_2^2].
$$

For three variables and degree 2:

$$
\Theta(x_1,x_2,x_3)
=
[1, x_1, x_2, x_3, x_1^2, x_1x_2, x_1x_3, x_2^2, x_2x_3, x_3^2].
$$

---

## 3. Sequential Thresholded Least Squares

The optimizer `stlsq` implements **Sequential Thresholded Least Squares**.

The regression problem is

$$
Y \approx \Theta(X)\Xi.
$$

The algorithm is:

1. Fit an initial least-squares model.
2. Set coefficients with absolute value below a threshold to zero.
3. Refit least squares only on the remaining active terms.
4. Repeat until convergence or until the maximum number of iterations is reached.

Mathematically, the initial least-squares step solves

$$
\Xi^{(0)}
=
\arg\min_{\Xi}
\left\|
Y - \Theta(X)\Xi
\right\|_2^2.
$$

Then small coefficients are removed:

$$
\xi_{ij} = 0
\quad
\text{if}
\quad
|\xi_{ij}| < \lambda.
$$

Here, $\lambda$ is the threshold controlling sparsity.

---

## 4. Discrete-time SINDy

Discrete-time SINDy fits models of the form

$$
X_{t+1} \approx F(X_t).
$$

In matrix form:

$$
Y \approx \Theta(X)\Xi,
$$

where

$$
X =
\begin{pmatrix}
X_0 \\
X_1 \\
\vdots \\
X_{T-1}
\end{pmatrix},
\quad
Y =
\begin{pmatrix}
X_1 \\
X_2 \\
\vdots \\
X_T
\end{pmatrix}.
$$

For example, in one dimension:

$$
x_{t+1}
\approx
c_0 + c_1x_t + c_2x_t^2.
$$

---

## 5. Continuous-time SINDy

Continuous-time SINDy fits models of the form

$$
\dot{X} \approx \Theta(X)\Xi.
$$

For example, in one dimension:

$$
\dot{x}
\approx
c_0 + c_1x + c_2x^2.
$$

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

$$
dr_t = \kappa(\theta-r_t)dt + \sigma dW_t.
$$

Using an Euler approximation,

$$
r_{t+\Delta t}
=
r_t + \kappa(\theta-r_t)\Delta t
+ \sigma\sqrt{\Delta t}\epsilon_t.
$$

Ignoring the stochastic noise term for the conditional mean gives

$$
\mathbb{E}[r_{t+\Delta t}\mid r_t]
=
\kappa\theta\Delta t
+
(1-\kappa\Delta t)r_t.
$$

So the discrete-time model has the form

$$
r_{t+1}
\approx
c_0 + c_1r_t.
$$

The experiment verifies that SINDy can recover this simple mean-reverting structure when the data-generating process is known and matches the candidate library.

Main takeaway:

$$
\text{SINDy recovers known sparse mean-reverting dynamics on synthetic finance data.}
$$

---

## 2. Real Treasury yield experiment: DGS10

Notebook:

```text
notebooks/02_real_yield_sindy.ipynb
```

This experiment applies discrete-time SINDy to real U.S. Treasury yield data from FRED.

The first real-data test uses

$$
\text{DGS10}
=
\text{10-Year Treasury Constant Maturity Rate}.
$$

The model fits

$$
y_{t+1}
\approx
c_0 + c_1y_t.
$$

The estimated coefficient $c_1$ is very close to 1, so the model is close to

$$
y_{t+1} \approx y_t.
$$

This means that the daily 10-year Treasury yield behaves like a highly persistent near-random-walk process.

Main takeaway:

$$
\text{On daily DGS10 data, SINDy mostly recovers persistence rather than useful nonlinear forecasting structure.}
$$

---

## 3. Real yield-curve factor dynamics

Notebook:

```text
notebooks/04_yield_curve_factors_sindy.ipynb
```

This experiment builds a three-factor representation of the Treasury yield curve:

$$
X_t =
\begin{pmatrix}
L_t \\
S_t \\
C_t
\end{pmatrix},
$$

where

$$
L_t = y_t^{10Y},
$$

$$
S_t = y_t^{10Y} - y_t^{2Y},
$$

and

$$
C_t = 2y_t^{5Y} - y_t^{2Y} - y_t^{10Y}.
$$

Here:

- $L_t$ is a level proxy.
- $S_t$ is a slope proxy.
- $C_t$ is a curvature proxy.

The model studies daily factor changes:

$$
\Delta X_{t+1}
=
X_{t+1} - X_t.
$$

The candidate library includes both linear and quadratic terms:

$$
\Theta(L,S,C)
=
[1,L,S,C,L^2,LS,LC,S^2,SC,C^2].
$$

The model is

$$
\Delta X_{t+1}
\approx
\Theta(L_t,S_t,C_t)\Xi.
$$

The zero-change baseline is

$$
\widehat{\Delta X}_{t+1}=0.
$$

### Main full-sample result

The full-sample linear SINDy model identifies economically interpretable mean-reversion-like coefficients, especially for slope and curvature.

For example, the linear library produced equations of the form

$$
\Delta S_{t+1}
\approx
a_S
+
b_{SL}L_t
+
b_{SS}S_t
+
b_{SC}C_t,
$$

with

$$
b_{SS}<0,
$$

and

$$
\Delta C_{t+1}
\approx
a_C
+
b_{CL}L_t
+
b_{CS}S_t
+
b_{CC}C_t,
$$

with

$$
b_{CC}<0.
$$

This is consistent with weak mean-reversion-like behavior in slope and curvature.

However, out-of-sample validation shows that the zero-change baseline is difficult to beat.

Main takeaway:

$$
\text{SINDy identifies plausible candidate structures, but daily yield-curve factor changes are too noisy for robust one-day forecasting in this setup.}
$$

---

## 4. Threshold sweep on yield-curve factors

The quadratic library contains 30 possible coefficients because there are 10 library terms and 3 target equations.

A threshold sweep shows how sparsity changes with the STLSQ threshold.

For small thresholds, the model remains dense.

For a high threshold,

$$
\lambda = 0.1,
$$

the model becomes much more sparse and keeps only three active terms:

$$
\Delta L_{t+1}
=
-0.417483 C_t^2,
$$

$$
\Delta S_{t+1}
=
0.304801 C_t^2,
$$

$$
\Delta C_{t+1}
=
-0.179575 L_t C_t.
$$

The most interpretable equation is

$$
\Delta C_{t+1}
=
-0.179575 L_t C_t.
$$

Since nominal rate levels are usually positive, this behaves like

$$
\Delta C_{t+1}
\approx
-k_t C_t,
\quad
k_t = 0.179575 L_t > 0.
$$

Thus, positive curvature is pulled downward and negative curvature is pulled upward. This suggests a level-dependent curvature mean-reversion candidate structure.

However, the out-of-sample improvement over the zero-change baseline is very small.

Main takeaway:

$$
\text{Thresholding improves sparsity and interpretability, but it does not create robust predictive power.}
$$

---

## 5. Regime-specific yield-curve analysis

The project also tests a shorter monetary-policy regime starting on

$$
2024\text{-}09\text{-}18,
$$

the beginning of the recent Fed cutting-cycle period.

The motivation is that SINDy assumes relatively stable dynamics. A single equation over several decades of financial data may be too restrictive because rates are regime-dependent.

The regime-specific linear SINDy model produces stronger mean-reversion-like self-effects:

$$
L_t \rightarrow \Delta L_{t+1}: \text{negative},
$$

$$
S_t \rightarrow \Delta S_{t+1}: \text{negative},
$$

$$
C_t \rightarrow \Delta C_{t+1}: \text{negative}.
$$

Out-of-sample validation is mixed:

- The linear model improves the level forecast relative to the zero-change baseline.
- It performs worse for slope and curvature.
- The quadratic model performs worse across all three factors.

Main takeaway:

$$
\text{Regime selection can make discovered equations more interpretable, but it does not automatically create robust predictive power.}
$$

---

## 6. Lorenz system: positive control experiment

Notebook:

```text
notebooks/05_lorenz_sindy.ipynb
```

The Lorenz system is a classic nonlinear dynamical system:

$$
\dot{x} = \sigma(y-x),
$$

$$
\dot{y} = x(\rho-z)-y,
$$

$$
\dot{z} = xy-\beta z.
$$

Using the classical parameters

$$
\sigma=10,
\quad
\rho=28,
\quad
\beta=\frac{8}{3},
$$

the system becomes

$$
\dot{x} = -10x + 10y,
$$

$$
\dot{y} = 28x - y - xz,
$$

$$
\dot{z} = xy - \frac{8}{3}z.
$$

SINDy is given the larger degree-2 polynomial library

$$
\Theta(x,y,z)
=
[1,x,y,z,x^2,xy,xz,y^2,yz,z^2].
$$

The true equations only require a few of these candidate terms:

$$
\dot{x}: x,y,
$$

$$
\dot{y}: x,y,xz,
$$

$$
\dot{z}: xy,z.
$$

SINDy recovers the equations almost exactly:

$$
\dot{x} = -10x + 10y,
$$

$$
\dot{y} = 28x - y - xz,
$$

$$
\dot{z} = xy - 2.6667z.
$$

The derivative MSE is essentially zero, and the coefficient error is at numerical precision.

Main takeaway:

$$
\text{When the true dynamics are sparse, stable, and contained in the candidate library, SINDy recovers the governing equations almost exactly.}
$$

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

The real-data experiments suggest:

$$
\text{SINDy is useful as an interpretable model-discovery tool, but not as a strong one-day forecasting model for daily Treasury yields.}
$$

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
