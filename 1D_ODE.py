# Tanh PINN architecture for a nonoscillatory 1D ODE
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class DecayPINN(nn.Module):
    def __init__(self, hidden_neurons=32, hidden_layers=3, activation=nn.Tanh):
        """
        PINN architecture mapping time t to the scalar solution u(t).
        The default tanh activation is smooth and non-oscillatory.
        """
        super().__init__()
        layers = []

        # Input layer: one time coordinate -> hidden representation
        layers.append(nn.Linear(1, hidden_neurons))
        layers.append(activation())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(activation())

        # Output layer: hidden representation -> scalar u(t)
        layers.append(nn.Linear(hidden_neurons, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        """Evaluate the network at time coordinate(s) t."""
        return self.network(t)

# Physical parameter for du/dt + lambda*u = 0
decay_rate = 1.0

def compute_physics_loss(model, t_collocation):
    """Compute the ODE residual loss using automatic differentiation."""
    u = model(t_collocation)

    du_dt = torch.autograd.grad(
        u,
        t_collocation,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    residual = du_dt + decay_rate * u
    return torch.mean(residual**2)

def decay_system(t, y, decay_rate):
    """First-order exponential decay ODE: du/dt = -lambda*u."""
    return [-decay_rate * y[0]]

if __name__ == "__main__":
    torch.manual_seed(42)
    model = DecayPINN()
    t_test = torch.tensor([[0.5]], requires_grad=True)
    u_test = model(t_test)

    print(f"Input t shape: {t_test.shape}, value: {t_test.item()}")
    print(f"Output u shape: {u_test.shape}, value: {u_test.item():.4f}")
    print("Forward pass successful")
    
    pinn_model = DecayPINN(hidden_neurons=32, hidden_layers=3)
    # Adam optimiser minimising the PINN parameters
    optimizer = optim.Adam(pinn_model.parameters(), lr=1e-3)
    # Collocation points on t in [0, 5]
    t_physics = torch.linspace(0, 5, 300).view(-1, 1).requires_grad_(True)
    # Initial condition u(0)=1
    t_ic = torch.tensor([[0.0]])
    u_ic_target = torch.tensor([[1.0]])

    epochs = 2500
    start_time = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        loss_ic = torch.mean((pinn_model(t_ic) - u_ic_target)**2)
        loss_ode = compute_physics_loss(pinn_model, t_physics)
        loss = loss_ic + loss_ode

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}/{epochs} -> "
                f"Total Loss: {loss.item():.6e}, "
                f"IC: {loss_ic.item():.2e}, ODE: {loss_ode.item():.2e}"
            )

    pinn_training_time = time.time() - start_time
    print(f"Training completed in {pinn_training_time:.2f} seconds.")

    t_span = (0, 5)
    t_eval = np.linspace(0, 5, 300)
    y0 = [1.0]

    start_time_scipy = time.time()
    sol = solve_ivp(
        decay_system,
        t_span,
        y0,
        args=(decay_rate,),
        t_eval=t_eval,
        method="RK45"
    )
    scipy_execution_time = time.time() - start_time_scipy

    print("SciPy solver successful")
    print(f"Execution time: {scipy_execution_time:.4f} seconds")

    
    with torch.no_grad():
        t_tensor = torch.tensor(t_eval, dtype=torch.float32).view(-1, 1)
        u_pinn_pred = pinn_model(t_tensor).numpy().flatten()

    u_scipy = sol.y[0]
    u_exact = np.exp(-decay_rate * t_eval)

    mse_vs_scipy = np.mean((u_pinn_pred - u_scipy)**2)
    mse_vs_exact = np.mean((u_pinn_pred - u_exact)**2)

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval, u_exact, color="blue", linewidth=2, label="Analytical solution")
    plt.plot(t_eval, u_scipy, color="red", linestyle="-", label="SciPy RK45")
    plt.plot(t_eval, u_pinn_pred, color="yellow", linestyle="--", label="PINN prediction")
    plt.title("Non-oscillatory 1D ODE: PINN vs SciPy")
    plt.xlabel("Time, t")
    plt.ylabel("Solution, u(t)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("Comparing time and error")
    print(f"MSE vs SciPy: {mse_vs_scipy:.2e}")
    print(f"MSE vs analytical solution: {mse_vs_exact:.2e}")
    print(f"PINN training time: {pinn_training_time:.4f} seconds")
    print(f"SciPy execution time: {scipy_execution_time:.4f} seconds")
    print(f"Time ratio (PINN/SciPy): {pinn_training_time / scipy_execution_time:.1f}x")



