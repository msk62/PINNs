# Pressure_Possion.py

import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from IPython.display import HTML
from mpl_toolkits.mplot3d import Axes3D

!pip install "pina-mathlab"
!pip install torch torchvision
!pip install pytorch-lightning

from pina import Condition, Trainer
from pina.problem import SpatialProblem
from pina.operator import laplacian
from pina.model import FeedForward
from pina.model.block import PeriodicBoundaryEmbedding
from pina.solver import PINN
from pina.domain import CartesianDomain
from pina.equation import Equation
from pina.callback import MetricTracker
#importing everything from PINA tutorial 9

torch.manual_seed(42)


# PDE residual
# ----------------------------------------------------------
def pressure_poisson_equation(input_, output_):
    """
    Residual for periodic pressure poisson equation ∇²p = f(x,y)
    with exact solution p(x,y) = sin(2pix) cos(2piy).
    """
    x = input_.extract("x")
    y = input_.extract("y")

    # network pressure prediction.
    p = output_.extract("p")

    # PINA computes the laplacian using automatic differentiation.
    p_lap = laplacian(
        output_,
        input_,
        components=["p"],
        d=["x", "y"]
    )

    forcing = -8.0 * torch.pi**2 * torch.sin(2.0 * torch.pi * x) * torch.cos(2.0 * torch.pi * y)

    return p_lap - forcing


# PINA problem definition
# ----------------------------------------------------------
class PeriodicPressurePoisson(SpatialProblem):
    """
    pressure field is periodic on [0,1]x[0,1]
    """
    output_variables = ["p"]

    spatial_domain = CartesianDomain({
        "x": [0, 1],
        "y": [0, 1],
    })

    conditions = {
        "pde": Condition(
            domain=spatial_domain,
            equation=Equation(pressure_poisson_equation)
        )
    }

    def solution(self, pts):
        """Analytical manufactured solution."""
        x = pts.extract("x")
        y = pts.extract("y")
        return torch.sin(2.0 * torch.pi * x) * torch.cos(2.0 * torch.pi * y)


# Build problem and discretise collocation points
# ----------------------------------------------------------
problem = PeriodicPressurePoisson()

# Collocation points for PDE residual
problem.discretise_domain(
    n=2500,
    mode="random",
    domains=["pde"]
)


# Periodic PINN model
# ----------------------------------------------------------
''' PeriodicBoundaryEmbedding maps each coordinate into periodic features
for two spatial dimensions embedded input has dimension 2+2x2 = 6'''
model = torch.nn.Sequential(
    PeriodicBoundaryEmbedding(
        input_dimension=2,
        periods= 1.0
    ),
    FeedForward(
        input_dimensions=6,
        output_dimensions=1,
        layers=[64, 64, 64],
        func=torch.nn.Tanh
    )
)


# Solver and trainer
# ----------------------------------------------------------
solver = PINN(
    problem=problem,
    model=model
)

trainer = Trainer(
    solver=solver,
    max_epochs=2000,    #adjust this to be lower if running to check the code works, but this is ideal epoch no. for full training
    accelerator="cpu",     #this can be adjusted to gpu?
    enable_model_summary=False,
    callbacks=[MetricTracker()]
)

trainer.train()


# evaluate on a regular grid
# ----------------------------------------------------------
N_GRID = 100

x = torch.linspace(0, 1, N_GRID)
y = torch.linspace(0, 1, N_GRID)

X, Y = torch.meshgrid(x, y, indexing="ij")

x_flat = X.reshape(-1, 1)
y_flat = Y.reshape(-1, 1)

# Create PINA style labelled points by sampling Cartesian domain
pts = problem.spatial_domain.sample(N_GRID, mode="grid")

with torch.no_grad():
    p_pred_flat = solver(pts).extract("p").tensor.detach()
    p_true_flat = problem.solution(pts).detach()

# Reshape to 2D fields
p_pred = p_pred_flat.reshape(N_GRID, N_GRID)
p_true = p_true_flat.reshape(N_GRID, N_GRID)

# pressure is defined up to an arbitrary constant so align the mean
p_pred = p_pred - torch.mean(p_pred)
p_true = p_true - torch.mean(p_true)

abs_error = torch.abs(p_pred - p_true)

X_np = X.numpy()
Y_np = Y.numpy()
p_true_np = p_true.numpy()
p_pred_np = p_pred.numpy()
abs_error_np = abs_error.numpy()

mse = torch.mean((p_pred - p_true)**2).item()
max_error = torch.max(abs_error).item()

print(f"Mean squared error: {mse:.3e}")
print(f"Maximum absolute error: {max_error:.3e}")


# Training convergence
# ----------------------------------------------------------
trainer_metrics = trainer.callbacks[0].metrics

plt.figure(figsize=(8, 4))
plt.plot(trainer_metrics["train_loss"], color="#20808D", linewidth=2)
plt.yscale("log")
plt.title("Periodic pressure-Poisson PINN convergence")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# Colormesh plots analytic, PINN, absolute error
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

pcm0 = axes[0].pcolormesh(
    X_np,
    Y_np,
    p_true_np,
    shading="auto",
    cmap="viridis"
)
axes[0].set_title("Analytical pressure")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
fig.colorbar(pcm0, ax=axes[0], label="p")

pcm1 = axes[1].pcolormesh(
    X_np,
    Y_np,
    p_pred_np,
    shading="auto",
    cmap="viridis"
)
axes[1].set_title("PINN pressure")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
fig.colorbar(pcm1, ax=axes[1], label="p")

pcm2 = axes[2].pcolormesh(
    X_np,
    Y_np,
    abs_error_np,
    shading="auto",
    cmap="inferno"
)
axes[2].set_title("|Error|")
axes[2].set_xlabel("x")
axes[2].set_ylabel("y")
fig.colorbar(pcm2, ax=axes[2], label="|p_PINN - p_exact|")

plt.show()


# 3D surface
# ----------------------------------------------------------
fig = plt.figure(figsize=(16, 5), constrained_layout=True)

ax1 = fig.add_subplot(1, 3, 1, projection="3d")
ax1.plot_surface(X_np, Y_np, p_true_np, cmap="viridis", edgecolor="none", alpha=0.95)
ax1.contour(X_np, Y_np, p_true_np, zdir="z", offset=np.min(p_true_np), cmap="viridis")
ax1.set_title("Analytical pressure surface")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("p")

ax2 = fig.add_subplot(1, 3, 2, projection="3d")
ax2.plot_surface(X_np, Y_np, p_pred_np, cmap="viridis", edgecolor="none", alpha=0.95)
ax2.contour(X_np, Y_np, p_pred_np, zdir="z", offset=np.min(p_pred_np), cmap="viridis")
ax2.set_title("PINN pressure surface")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("p")

ax3 = fig.add_subplot(1, 3, 3, projection="3d")
ax3.plot_surface(X_np, Y_np, abs_error_np, cmap="inferno", edgecolor="none", alpha=0.95)
ax3.contour(X_np, Y_np, abs_error_np, zdir="z", offset=0, cmap="inferno")
ax3.set_title("Absolute error surface")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.set_zlabel("|error|")

plt.show()


# Animation of horizontal slices through pressure field scans through y-slices
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)

ax_line, ax_map = axes

line_exact, = ax_line.plot(
    x.numpy(),
    p_true_np[:, 0],
    color="blue",
    linewidth=2,
    label="analytical"
)

line_pinn, = ax_line.plot(
    x.numpy(),
    p_pred_np[:, 0],
    color="orange",
    linestyle="--",
    linewidth=2,
    label="PINN"
)

ax_line.set_xlim(0, 1)
ax_line.set_ylim(
    1.1 * min(p_true_np.min(), p_pred_np.min()),
    1.1 * max(p_true_np.max(), p_pred_np.max())
)
ax_line.set_xlabel("x")
ax_line.set_ylabel("p(x,y)")
ax_line.grid(True, alpha=0.3)
ax_line.legend()

pcm = ax_map.pcolormesh(
    X_np,
    Y_np,
    abs_error_np,
    shading="auto",
    cmap="inferno"
)
slice_marker = ax_map.axhline(y=0.0, color="white", linewidth=2)
ax_map.set_title("|Error| with current y-slice")
ax_map.set_xlabel("x")
ax_map.set_ylabel("y")
fig.colorbar(pcm, ax=ax_map, label="|error|")

title = fig.suptitle("Pressure slice comparison, y = 0.00")


def update(frame):
    y_val = y[frame].item()

    line_exact.set_ydata(p_true_np[:, frame])
    line_pinn.set_ydata(p_pred_np[:, frame])

    slice_marker.set_ydata([y_val, y_val])
    title.set_text(f"Pressure slice comparison, y = {y_val:.2f}")

    return line_exact, line_pinn, slice_marker, title


ani = animation.FuncAnimation(
    fig,
    update,
    frames=N_GRID,
    interval=60,
    blit=False
)

plt.show()
HTML(ani.to_jshtml())
