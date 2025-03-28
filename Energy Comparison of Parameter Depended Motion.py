import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

'''
Nondimensionalized forced damped pendulum equation:
  d²θ/dφ² + β dθ/dφ + sin(θ) = ε cos(Ω φ)
  θ: angular displacement
  ξ: angular velocity (dθ/dφ)
  β: dimensionless damping coefficient
  ε: dimensionless forcing amplitude
  Ω: dimensionless forcing frequency

state = [θ, ξ] (for phase space)
'''

def forced_pendulum(phi, state, beta, epsilon, Omega):
    theta, xi = state
    dtheta_dphi = xi
    dxi_dphi    = -beta * xi - np.sin(theta) + epsilon * np.cos(Omega * phi)
    return [dtheta_dphi, dxi_dphi]

def compute_energies(theta_array, xi_array):
    kinetic = 0.5 * (xi_array**2)
    potential = 1.0 - np.cos(theta_array)
    total = kinetic + potential
    return kinetic, potential, total


param_sets = [
    {"beta": 0.1, "epsilon": 2.0, "Omega": 1.1, "label": "Periodic"},
    {"beta": 0.2, "epsilon": 1.2, "Omega": 1.1,   "label": "Chaotic"}
]

# Time span
phi_span = (0, 100)
phi_eval_count = 1000

# Initial conditions
theta0 = 0.5
xi0 = 0.0

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
fig.suptitle("Energy Comparisons: Periodic vs. Chaotic")

for i, params in enumerate(param_sets):
    beta    = params["beta"]
    epsilon = params["epsilon"]
    Omega   = params["Omega"]
    label   = params["label"]
    
    phi_eval = np.linspace(phi_span[0], phi_span[1], phi_eval_count)
    sol = solve_ivp(
        forced_pendulum,
        phi_span,
        [theta0, xi0],
        args=(beta, epsilon, Omega),
        t_eval=phi_eval,
        method='RK45'
    )
    
    phi_vals   = sol.t
    theta_vals = sol.y[0]
    xi_vals    = sol.y[1]
    
    kin, pot, tot = compute_energies(theta_vals, xi_vals)
    
    ax = axes[i]
    ax.plot(phi_vals, kin,  label="Kinetic",  linewidth=1.2)
    ax.plot(phi_vals, pot,  label="Potential",linewidth=1.2)
    ax.plot(phi_vals, tot,  label="Total",    linewidth=1.2)
    ax.set_title(f"{label}\nβ={beta}, ε={epsilon}, Ω={Omega}")
    ax.set_xlabel("Dimensionless Time (φ)")
    if i == 0:
        ax.set_ylabel("Energy")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()
