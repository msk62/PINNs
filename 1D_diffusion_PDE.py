# 1D_diffusion_PDE.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import solve_ivp
from IPython.display import display, HTML
import matplotlib.animation as animation

torch.manual_seed(42)

alpha = 0.1

class HeatPINN(nn.Module):
    """PINN mapping coordinates (x,t) to the scalar field u(x,t)"""

    def __init__(self, hidden_neurons=64, hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_neurons), nn.Tanh()]

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_neurons, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.network(inputs)


def diffusion_loss(model, x_col, t_col, x_bc0, x_bc1, t_bc, x_ic, t_ic):
    """Compute PDE BC AND IC losses"""
    u = model(x_col, t_col)

    du_dx = torch.autograd.grad(
        u, x_col, torch.ones_like(u), create_graph=True
    )[0]
    du_dt = torch.autograd.grad(
        u, t_col, torch.ones_like(u), create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        du_dx, x_col, torch.ones_like(du_dx), create_graph=True
    )[0]

    loss_pde = torch.mean((du_dt - alpha * d2u_dx2)**2)
    loss_bc = torch.mean(model(x_bc0, t_bc)**2) + torch.mean(model(x_bc1, t_bc)**2)

    u_ic_pred = model(x_ic, t_ic)
    u_ic_true = torch.sin(torch.pi * x_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic_true)**2)

    return loss_pde + loss_bc + loss_ic


# Train PINN
# ----------------------------------------------------------
model = HeatPINN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 3000

N_col = 2000
x_col = torch.rand(N_col, 1, requires_grad=True)
t_col = torch.rand(N_col, 1, requires_grad=True)

N_bc = 400
x_bc0 = torch.zeros(N_bc, 1)
x_bc1 = torch.ones(N_bc, 1)
t_bc = torch.rand(N_bc, 1)

N_ic = 400
x_ic = torch.rand(N_ic, 1)
t_ic = torch.zeros(N_ic, 1)

N_grid = 100
x_grid = np.linspace(0, 1, N_grid)
u0 = np.sin(np.pi * x_grid)

loss_history = []
print("Training diffusion PINN...")
start_time = time.time()

snap_epochs = {500, 1000, 2000, 3000}
snapshots = {}

for epoch in range(1, epochs + 1):
    optimizer.zero_grad()

    loss = diffusion_loss(
        model,
        x_col, t_col,
        x_bc0, x_bc1, t_bc,
        x_ic, t_ic
    )

    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    if epoch == 1 or epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs} -> Loss {loss.item():.6e}")

    if epoch in snap_epochs:
        with torch.no_grad():
            x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
            preds = []
            for t_val in np.linspace(0, 1, 100):
                t_tensor = torch.full_like(x_tensor, float(t_val))
                preds.append(model(x_tensor, t_tensor).cpu().numpy().flatten())
        snapshots[epoch] = np.array(preds)

pinn_train_time = time.time() - start_time
print(f"PINN training time: {pinn_train_time:.2f} s")


# SciPy ref, finite diff in space + RK45 in time
# ----------------------------------------------------------
def diffusion_system(t, u, alpha, N):
    dx = 1 / (N - 1)
    dudt = np.zeros_like(u)
    dudt[1:-1] = alpha * (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return dudt

start_scipy = time.time()
t_eval = np.linspace(0, 1, 100)
sol = solve_ivp(
    diffusion_system,
    (0, 1),
    u0,
    args=(alpha, N_grid),
    t_eval=t_eval,
    method="RK45"
)
scipy_time = time.time() - start_scipy
print(f"SciPy solver time: {scipy_time:.4f} s")
print(f"Time ratio (PINN/SciPy): {pinn_train_time / scipy_time:.1f}x")


# Evaluate PINN on the same grid as SciPy
# ----------------------------------------------------------
pinn_solution = []
with torch.no_grad():
    x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)
    for t_val in t_eval:
        t_tensor = torch.full_like(x_tensor, float(t_val))
        pred = model(x_tensor, t_tensor).cpu().numpy().flatten()
        pinn_solution.append(pred)

pinn_solution = np.array(pinn_solution)
scipy_solution = sol.y.T

error = pinn_solution - scipy_solution
abs_error = np.abs(error)
time_mse = np.mean(error**2, axis=1)

print(f"Mean field MSE: {np.mean(error**2):.3e}")
print(f"Maximum absolute error: {np.max(abs_error):.3e}")

# ----------------------------------------------------------
# Animated PINN vs SciPy evolution (IPython display)

fig, ax = plt.subplots(figsize=(8, 5))

line1, = ax.plot(
    x_grid,
    scipy_solution[0],
    linewidth=2,
    label="SciPy RK45"
)

line2, = ax.plot(
    x_grid,
    pinn_solution[0],
    linestyle="--",
    linewidth=2,
    label="PINN"
)

ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.set_ylim(
    min(scipy_solution.min(), pinn_solution.min()),
    max(scipy_solution.max(), pinn_solution.max())
)
ax.grid(True, alpha=0.3)
ax.legend()


def update(frame):
    line1.set_ydata(scipy_solution[frame])
    line2.set_ydata(pinn_solution[frame])
    ax.set_title(f"t = {t_eval[frame]:.2f}")
    return line1, line2


ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(t_eval),
    interval=50,
    blit=False
)

plt.tight_layout()

display(HTML(ani.to_jshtml()))

# Error heatmaps
# ----------------------------------------------------------
X, T = np.meshgrid(x_grid, t_eval)   # NOTE: matches solution shape (t, x)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

#SciPy heatmap
pcm0 = axes[0].pcolormesh(
    X, T, scipy_solution,
    shading='auto', cmap='viridis'
)
axes[0].set_title("SciPy reference")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(pcm0, ax=axes[0], label="u")

#PINN heatmap
pcm1 = axes[1].pcolormesh(
    X, T, pinn_solution,
    shading='auto', cmap='viridis'
)
axes[1].set_title("PINN solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(pcm1, ax=axes[1], label="u")

# abs error heatmap
pcm2 = axes[2].pcolormesh(
    X, T, abs_error,
    shading='auto', cmap='inferno'
)
axes[2].set_title("|Error|")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(pcm2, ax=axes[2], label="|u_PINN - u_SciPy|")

plt.tight_layout()
plt.show()


# Training convergence
# ----------------------------------------------------------
plt.figure(figsize=(11, 4))
plt.plot(np.arange(1, epochs + 1), loss_history, color="#20808D")
plt.yscale("log")
plt.title("Heat PINN training convergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3D surface comparison scipy vs PINN
# ----------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure(figsize=(14, 5))

#sciPy surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(
    X, T, scipy_solution,
    cmap='viridis',
    edgecolor='none'
)
ax1.set_title("SciPy solution")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("u(x,t)")

#PINN surface
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(
    X, T, pinn_solution,
    cmap='viridis',
    edgecolor='none'
)
ax2.set_title("PINN solution")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("u(x,t)")

plt.tight_layout()
plt.show()

#slightly different notagtion for sets of axes just for differentiating
