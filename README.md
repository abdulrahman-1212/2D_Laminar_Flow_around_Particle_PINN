# Physics-Informed Neural Network (PINN) for Laminar Flow Around a Circular Particle

Implementation of a **Physics-Informed Neural Network** to solve the steady-state 2D incompressible **Navier–Stokes equations** for laminar flow around a circular particle at **Re = 20**.

Based on the paper:

> Applying Physics-Informed Neural Networks to Solve Navier–Stokes Equations for Laminar Flow around a Particle  
> *Mathematical and Computational Applications* 2023, 28(5), 102  
> https://doi.org/10.3390/mca28050102

## Project Overview

This project trains a PINN to predict the velocity field (u, v) and pressure (p) in a 2D channel containing a circular obstacle. The model learns **purely from the governing equations and boundary conditions** — no labeled CFD simulation data is used.

### Key features

- Steady, incompressible Navier–Stokes equations
- Domain: rectangular channel x ∈ [−2, 6], y ∈ [−2, 2], circular particle of radius r = 0.5 centered at (0,0)
- Parabolic inlet velocity profile
- No-slip (u = v = 0) on top/bottom walls and cylinder surface
- Soft zero-pressure condition at outlet
- Fully physics-informed training (PDE residuals + boundary losses)
- Simple MLP architecture with tanh activation
- Adam optimizer (with optional L-BFGS fine-tuning)

## Governing Equations

### Steady incompressible Navier–Stokes

$$
\nabla \cdot \mathbf{u} = 0
$$

$$
\rho \, (\mathbf{u} \cdot \nabla) \mathbf{u} = -\nabla p + \mu \nabla^2 \mathbf{u}
$$

or in component form:

$$
\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0
$$

$$
\rho \left( u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} \right) = -\frac{\partial p}{\partial x} + \mu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)
$$

$$
\rho \left( u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} \right) = -\frac{\partial p}{\partial y} + \mu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)
$$

### Parabolic inlet velocity profile (x = −2)

$$
u(y) = 4 U_\mathrm{max} \frac{(y - y_\mathrm{min})(y_\mathrm{max} - y)}{H^2}, \quad v(y) = 0
$$

where H = y_max − y_min.

## Loss Function

The total loss is a weighted sum of:

$$
\mathcal{L} = \mathcal{L}_\mathrm{PDE} + \lambda_1 \mathcal{L}_\mathrm{inlet} + \lambda_2 \mathcal{L}_\mathrm{outlet} + \lambda_3 \mathcal{L}_\mathrm{wall}
$$

- $\mathcal{L}_\mathrm{PDE}$: mean squared residual of continuity + momentum equations at collocation points
- $\mathcal{L}_\mathrm{inlet}$: MSE between predicted and target parabolic velocity at inlet
- $\mathcal{L}_\mathrm{outlet}$: MSE of pressure at outlet (soft constraint p ≈ 0)
- $\mathcal{L}_\mathrm{wall}$: MSE of velocity on walls and cylinder surface (u² + v²)

## Results

### Velocity and pressure fields (Re = 20)

Predicted velocity magnitude, streamlines, pressure field and speed contours:

![Flow field results](results.png)



## How to Run

1. Open `particle_flow_pinn.ipynb` in **Google Colab** or **Kaggle**
2. Select GPU runtime (T4 or better recommended)
3. Run all cells sequentially
4. Training typically takes 10–40 minutes (depending on epochs and hardware)
5. Final plots appear automatically at the end

## Dependencies

```text
torch
numpy
matplotlib
tqdm          # optional progress bar
