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
'''
def forced_pendulum(phi, y, beta, epsilon, Omega):
    theta, xi = y  # xi = dθ/dφ
    dydphi = [
        xi,
        -beta * xi - np.sin(theta) + epsilon * np.cos(Omega * phi)
    ]
    return dydphi


# Parameters
beta = 0.25       # dimensionless damping
epsilon = 3.0     # dimensionless forcing amplitude
Omega = 0.5       # dimensionless forcing frequency

# Initial conditions [θ(0), ξ(0)]
y0 = [0.5, 0.0]  

# Time span
phi_span = (0, 100)
phi_eval = np.linspace(*phi_span, 10000)

####################
sol = solve_ivp(
    forced_pendulum,
    phi_span,
    y0,
    args=(beta, epsilon, Omega),
    t_eval=phi_eval,
    method='RK45'
)
####################



# θ vs. Time φ Plot

plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], color='teal', label='θ')
plt.xlabel('φ (Dimensionless Time)')
plt.ylabel('θ')
plt.title('Forced Damped Pendulum')
plt.grid(True)
plt.legend()


# Phase-space (ξ vs. θ) Plot

plt.figure(figsize=(8, 8))
plt.plot(sol.y[0], sol.y[1], color='crimson', linewidth=0.5)
plt.xlabel('θ')
plt.ylabel('ξ = dθ/dφ (Angular Velocity)')
plt.title('Phase-Space Trajectory of Forced Damped Pendulum')
plt.grid(True)


# Energy vs. Time Plot

theta_vals = sol.y[0]
xi_vals    = sol.y[1]

kinetic    = 0.5 * (xi_vals**2)
potential  = 1.0 - np.cos(theta_vals)
total      = kinetic + potential

plt.figure(figsize=(12, 6))
plt.plot(sol.t, kinetic,   'r--',  label='Kinetic Energy')
plt.plot(sol.t, potential, 'g--',  label='Potential Energy')
plt.plot(sol.t, total,     'b-',   label='Total Energy')

plt.xlabel('φ (Dimensionless Time)')
plt.ylabel('Energy')
plt.title('Energy vs. Time in Forced Damped Pendulum')
plt.grid(True)
plt.legend()
plt.show()



