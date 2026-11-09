# Physics-Informed Neural Networks (PINNs)

### Solving Differential Equations with Deep Learning

## Overview

This project investigates the use of **Physics-Informed Neural Networks (PINNs)** to solve ordinary and partial differential equations, using PyTorch to implement and train neural networks constrained by governing physical equations.

Rather than learning solutions exclusively from labelled data, PINNs incorporate the underlying physics directly into their training objectives. Automatic differentiation is used to calculate the derivatives required by the governing equations, while residual-based loss functions encourage the network to satisfy the physical constraints.

I developed this project to understand how PINNs work from first principles, explore their application to increasingly complex physical systems, and evaluate when they offer advantages over established numerical methods.

## Problems investigated

The project progresses through four problems of increasing complexity:

**1. One-dimensional ordinary differential equation (ODE)**

A simple introductory problem used to establish the fundamental PINN workflow: defining a neural approximation, calculating derivatives through automatic differentiation and minimising the residual of the governing equation.

**2. Heat equation**

A time-dependent diffusion problem in which the network learns a continuous space-time temperature field. The predicted solution is compared against a conventional finite-difference reference implemented using SciPy.

**3. Wave equation**

An oscillatory PDE that highlights the importance of neural network architecture and physical constraints. Sinusoidal activation functions and hard boundary conditions are investigated to improve the network's ability to represent wave-like solutions.

**4. Pressure Poisson equation**

An elliptic PDE arising in incompressible fluid dynamics. The network approximates a steady pressure field by minimising the governing equation's residual, providing a mesh-free approach to solving the problem.

## Methodology

The implementations explore several central components of PINN development:

- **Automatic differentiation:** Using PyTorch's autograd functionality to calculate spatial and temporal derivatives directly from the neural network.
- **Physics-informed loss functions:** Incorporating PDE or ODE residuals alongside initial and boundary condition constraints.
- **Network architecture:** Investigating how activation functions and boundary-condition enforcement affect convergence and solution accuracy.
- **Numerical validation:** Comparing neural predictions against conventional numerical solutions where reference solutions are available.
- **Visualisation and error analysis:** Examining predicted solution fields and their deviations from reference results.

The project involved building the PyTorch implementations from the underlying mathematical principles and adapting existing open-source libraries where necessary to accommodate the requirements of individual problems.

## Results and observations

The project demonstrates that PINNs are best understood as a complementary computational method rather than a direct replacement for established numerical solvers.

Several observations emerged:

- **Continuous solutions:** PINNs provide differentiable approximations across the entire spatial and temporal domain, rather than solutions restricted to discrete mesh points.
- **Physics-based learning:** Governing equations can be incorporated directly into the loss function, reducing dependence on large labelled datasets.
- **Architectural sensitivity:** The choice of activation function and enforcement of boundary conditions can have a substantial effect on performance, particularly for oscillatory problems such as the wave equation.
- **Computational trade-offs:** For relatively simple, low-dimensional problems, conventional numerical methods are generally faster and more reliable.
- **Potential flexibility:** The mesh-free formulation offers opportunities for problems involving more complicated geometries, physical constraints or repeated evaluations.

## Technologies

- **PyTorch** – neural network implementation, optimisation and automatic differentiation
- **NumPy** – numerical operations
- **SciPy** – conventional numerical reference solutions
- **Matplotlib** – solution visualisation and error analysis

## Conclusion

This project provided practical experience developing machine-learning models for scientific computing, from implementing automatic differentiation and physics-based objectives to training networks on increasingly complex differential equations.

More importantly, it demonstrated the importance of validating machine-learning approaches against established methods and understanding the limitations of a model rather than relying solely on its ability to fit a solution.

The work reflects my interest in combining physical understanding with machine learning to develop computational methods for challenging scientific and engineering problems.
