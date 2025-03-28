import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# System definition
def forced_pendulum(phi, state, beta, epsilon, omega):
    theta, xi = state
    dtheta_dphi = xi
    dxi_dphi = -beta*xi - np.sin(theta) + epsilon*np.cos(omega*phi)
    return [dtheta_dphi, dxi_dphi]

# Poincaré section generator
def poincare_section(beta, epsilon, omega, theta0=0.2, xi0=0.0, num_periods=2000, steps_per_period=1000):
    T = 2.0 * np.pi / omega
    phi_max = num_periods * T
    phi_eval = np.linspace(0, phi_max, steps_per_period * num_periods)

    sol = solve_ivp(
        forced_pendulum,
        (0, phi_max),
        [theta0, xi0],
        args=(beta, epsilon, omega),
        t_eval=phi_eval,
        method='RK45'
    )

    sample_times = np.arange(1, num_periods + 1) * T
    theta_vals = np.interp(sample_times, sol.t, sol.y[0])
    xi_vals = np.interp(sample_times, sol.t, sol.y[1])

    return theta_vals, xi_vals, sol

# Parameters for animation
beta, epsilon, omega = 0.25, 1.7 , 2/3  # Chaotic regime
theta_p, xi_p, sol = poincare_section(beta, epsilon, omega, num_periods=300)

# --- 1. Poincaré section plot ---
plt.figure(figsize=(5, 4))
plt.scatter(theta_p, xi_p, s=1, c='k')
plt.title("Poincaré Section")
plt.xlabel("θ")
plt.ylabel("ξ")
plt.grid(True)
plt.tight_layout()

# --- 2. Phase space plot (θ vs ξ) ---
plt.figure(figsize=(6, 4))
plt.plot(sol.y[0], sol.y[1], lw=0.5)
plt.title("Phase Space Trajectory (θ vs ξ)")
plt.xlabel("θ")
plt.ylabel("ξ")
plt.grid(True)
plt.tight_layout()


# --- 3. Pendulum Animation ---
# Normalize angle to [-π, π]
theta_anim = (sol.y[0] + np.pi) % (2 * np.pi) - np.pi
x = np.sin(theta_anim)
y = -np.cos(theta_anim)

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.grid(True)

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    time_text.set_text('')
    return line, time_text

def update(i):
    line.set_data([0, x[i]], [0, y[i]])
    time_text.set_text(f'Time = {sol.t[i]:.2f}')
    return line, time_text

ani = FuncAnimation(fig, update, frames=range(0, len(sol.t), 10),
                    init_func=init, blit=True, interval=10)

plt.title("Forced Damped Pendulum Animation")
plt.show()
