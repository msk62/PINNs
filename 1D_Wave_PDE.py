import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from mpl_toolkits.mplot3d import Axes3D 
from IPython.display import HTML

torch.manual_seed(42)

# Physical setup
# ------------------------------------------------------------------------
c = 1.0 # wave speed
t_final = 4.0/c  #full period of fundamental mode


class Sine(nn.Module):
    """sinusoidal activation for oscillatory wave solution"""
    def forward(self, x):
        return torch.sin(x)


class WavePINN(nn.Module):
    """
    raw network N(z,tau) is wrapped in a hard constrained trial solution:
    u(x,t) = u0(x) + tau^2 * x(2-x) * N(x(2-x), tau) where tau = t / t_final.

    this HARD enforces (no soft loss penalties):
        u(x,0)= sin(pi x/2)
        u_t(x,0)= 0
        u(0,t)= 0
        u_x(1,t)= 0
    """

    def __init__(self, hidden_neurons=64, hidden_layers=4):
        super().__init__()

        layers = [nn.Linear(2, hidden_neurons), Sine()]

        for i in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(Sine())

        layers.append(nn.Linear(hidden_neurons, 1))
        self.network = nn.Sequential(*layers)

        self.initialise_weights()

    def initialise_weights(self):
        """Xavier initialisation gives a stable starting point for LBFGS"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, t):
        """
        evaluate hard constrained solution u(x,t)

        Parameters: x - torch.Tensor
                        Spatial coordinate, shape (N, 1).
                    t - torch.Tensor
                        Physical time coordinate, shape (N, 1).

        Returns torch.Tensor
            Predicted displacement u(x,t), shape (N, 1).
        """
        tau = t / t_final

        # initial displacement satisfying mixed BCs
        u0 = torch.sin(torch.pi * x / 2.0)

        # spatial factor used to enforce u(0,t)=0 and u_x(1,t)=0
        boundary_factor = x * (2.0 - x)

        # monotone coordinate transform, derivative also vanishes at x=1
        z = boundary_factor

        raw_output = self.network(torch.cat([z, tau], dim=1))

        # constrained trial solution
        return u0 + tau**2 * boundary_factor * raw_output


def exact_wave_solution(x, t, wave_speed=1.0):
    """
    analytical solution for the mixed BC wave
    """
    return np.sin(np.pi * x / 2.0) * np.cos(wave_speed * np.pi * t / 2.0)


def wave_residual_loss(model, x_col, t_col):
    """
    PDE residual loss
    BC/IC are hard constraints so the loss only contains the wave eq residual
    """
    u = model(x_col, t_col)

    du_dx = torch.autograd.grad(
        u,
        x_col,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    du_dt = torch.autograd.grad(
        u,
        t_col,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    d2u_dx2 = torch.autograd.grad(
        du_dx,
        x_col,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]

    d2u_dt2 = torch.autograd.grad(
        du_dt,
        t_col,
        grad_outputs=torch.ones_like(du_dt),
        create_graph=True
    )[0]

    residual = d2u_dt2 - c**2 * d2u_dx2
    return torch.mean(residual**2)


# Training data
# ------------------------------------------------------------------------
N_col = 3000

# sobol points (quasirandom) give more even coverage than random points
sobol = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=42)
points = sobol.draw(N_col)

x_col = points[:, 0:1].clone().detach().requires_grad_(True)
t_col = (t_final * points[:, 1:2]).clone().detach().requires_grad_(True)


# Train with L-BFGS
# ------------------------------------------------------------------------
model = WavePINN(hidden_neurons=64, hidden_layers=4)

optimizer = optim.LBFGS(
    model.parameters(),
    lr=1.0,
    max_iter=700,
    max_eval=900,
    history_size=100,
    tolerance_grad=1e-9,
    tolerance_change=1e-11,
    line_search_fn="strong_wolfe"
)

loss_history = []

print("Training wave PINN with L-BFGS...")
start_time = time.time()

def closure():
    optimizer.zero_grad()

    loss = wave_residual_loss(model, x_col, t_col)
    loss.backward()

    # records closure evaluations, including line search trial steps
    loss_history.append(loss.item())

    if len(loss_history) == 1 or len(loss_history) % 50 == 0:
        print(f"Closure {len(loss_history):4d} -> PDE loss: {loss.item():.3e}")

    return loss

optimizer.step(closure)

wave_train_time = time.time() - start_time
print(f"Wave PINN training time: {wave_train_time:.2f} s")


# Hard constraint diagnostics
# ------------------------------------------------------------------------
x_check = torch.linspace(0, 1, 200).view(-1, 1).requires_grad_(True)
t_zero = torch.zeros_like(x_check, requires_grad=True)

u_initial_pred = model(x_check, t_zero)
u_initial_true = torch.sin(torch.pi * x_check / 2.0)

du_initial_dt = torch.autograd.grad(
    u_initial_pred,
    t_zero,
    grad_outputs=torch.ones_like(u_initial_pred),
    create_graph=True
)[0]

t_check = torch.linspace(0, t_final, 200).view(-1, 1)

x_left = torch.zeros_like(t_check, requires_grad=True)
u_left = model(x_left, t_check)

x_right = torch.ones_like(t_check, requires_grad=True)
u_right = model(x_right, t_check)

du_dx_right = torch.autograd.grad(
    u_right,
    x_right,
    grad_outputs=torch.ones_like(u_right),
    create_graph=True
)[0]

 #realistically these will all show up as 0 to 3dp - PINN is working well
print(f"Max initial displacement error: {torch.max(torch.abs(u_initial_pred - u_initial_true)).item():.3e}")
print(f"Max initial velocity error: {torch.max(torch.abs(du_initial_dt)).item():.3e}")
print(f"Max |u(0,t)|:{torch.max(torch.abs(u_left)).item():.3e}")
print(f"Max |u_x(1,t)|: {torch.max(torch.abs(du_dx_right)).item():.3e}")


# Evaluate PINN against analytical solution
# ------------------------------------------------------------------------
N_grid = 140
x_grid = np.linspace(0, 1, N_grid)
t_eval = np.linspace(0, t_final, N_grid)

X, T = np.meshgrid(x_grid, t_eval)

analytical_solution = exact_wave_solution(X, T, wave_speed=c)

pinn_solution = []

model.eval()
with torch.no_grad():
    x_tensor = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)

    for t_val in t_eval:
        t_tensor = torch.full_like(x_tensor, float(t_val))
        pred = model(x_tensor, t_tensor).cpu().numpy().flatten()
        pinn_solution.append(pred)

pinn_solution = np.array(pinn_solution)

error = pinn_solution - analytical_solution
abs_error = np.abs(error)
time_mse = np.mean(error**2, axis=1)

print(f"Mean field MSE:{np.mean(error**2):.3e}")
print(f"Max absolute error:{np.max(abs_error):.3e}")


# Animated comparison analytical vs PINN ipython
# ----------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

line1, = ax.plot(
    x_grid,
    analytical_solution[0],
    linewidth=2,
    label="Analytical solution"
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
    min(analytical_solution.min(), pinn_solution.min()),
    max(analytical_solution.max(), pinn_solution.max())
)
ax.grid(True, alpha=0.3)
ax.legend()


def update(frame):
    line1.set_ydata(analytical_solution[frame])
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


# Error heatmaps w/ pcolormesh
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

pcm0 = axes[0].pcolormesh(
    X,
    T,
    analytical_solution,
    shading="auto",
    cmap="viridis"
)
axes[0].set_title("Analytical solution")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig.colorbar(pcm0, ax=axes[0], label="u")

pcm1 = axes[1].pcolormesh(
    X,
    T,
    pinn_solution,
    shading="auto",
    cmap="viridis"
)
axes[1].set_title("PINN solution")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig.colorbar(pcm1, ax=axes[1], label="u")

pcm2 = axes[2].pcolormesh(
    X,
    T,
    abs_error,
    shading="auto",
    cmap="inferno"
)
axes[2].set_title("|Error|")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig.colorbar(pcm2, ax=axes[2], label="|u_PINN - u_exact|")

plt.tight_layout()
plt.show()


# 3D surface analytical wave vs PINN
# -------------------------------------------------------------------
fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
surf1 = ax1.plot_surface(
    X,
    T,
    analytical_solution,
    cmap="viridis",
    edgecolor="none"
)
ax1.set_title("Analytical wave solution")
ax1.set_xlabel("x")
ax1.set_ylabel("t")
ax1.set_zlabel("u(x,t)")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
surf2 = ax2.plot_surface(
    X,
    T,
    pinn_solution,
    cmap="viridis",
    edgecolor="none"
)
ax2.set_title("PINN solution")
ax2.set_xlabel("x")
ax2.set_ylabel("t")
ax2.set_zlabel("u(x,t)")

plt.tight_layout()
plt.show()


# LBFGS convergence plot
# ------------------------------------------------------------------------
plt.figure(figsize=(9, 5))

plt.plot(loss_history, color="blue", linewidth=2, label="PDE residual loss")

plt.yscale("log")
plt.title("Wave PINN convergence with LBFGS")
plt.xlabel("LBFGS closure eval")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
