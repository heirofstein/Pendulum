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



param_sets = [
    # (β, ε, Ω)
    (1.0, 0.5,  1.1,  "Small ε, large β → small oscillations"),
    (0.2, 1.2,  1.1,  "Mid. ε, small β → steady-state oscillations"),
    (0.1, 2.0,  1.1,  "Large ε, small β → chaotic oscillation")
]

# Time span
phi_span = (0, 100)  
num_points = 1000 

# Initial conditions
theta0 = 0.2
xi0    = 0.0

# Plot #########################################################
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey='row')
fig.suptitle("Forced Damped Pendulum")

for i, (beta, epsilon, Omega, lbl) in enumerate(param_sets):
    phi_eval = np.linspace(phi_span[0], phi_span[1], num_points)
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
    
    axes[0, i].plot(phi_vals, theta_vals, 'b-', linewidth=1.0)
    axes[0, i].set_title(f"{lbl}\nβ={beta}, ε={epsilon}, Ω={Omega}")
    axes[0, i].set_xlabel("Dimensionless Time (φ)")
    if i == 0:
        axes[0, i].set_ylabel("θ")
    axes[0, i].grid(True)
    
    axes[1, i].plot(theta_vals, xi_vals, color='red', linewidth=0.8)
    axes[1, i].set_xlabel("θ")
    if i == 0:
        axes[1, i].set_ylabel("ξ")
    axes[1, i].grid(True)

plt.tight_layout()
plt.show()
