import numpy as np
import matplotlib.pyplot as plt


'''
Nondimensionalized forced damped pendulum equation:
  d²θ/dφ² + β dθ/dφ + sin(θ) = ε cos(Ω φ)
  θ: angular displacement
  ξ: angular velocity (dθ/dφ)
  β: dimensionless damping coefficient
  ε: dimensionless forcing amplitude
  Ω: dimensionless forcing frequency
'''

# Parameters
beta = 0.25       # Damping coefficient
epsilon = 3.0     # Forcing amplitude
Omega = 0.5       # Forcing frequency
# xi = dθ/dφ


# Initial conditions: [theta, xi]
y0 = [0.5, 0.0]   
phi_start = 0.0
phi_end = 100.0
N = 10000                     # Number of time steps
h = (phi_end - phi_start) / N  # Step size


phi_vals = np.linspace(phi_start, phi_end, N)
theta_vals = np.zeros(N)
xi_vals = np.zeros(N)

# Initial conditions
theta_vals[0] = y0[0]
xi_vals[0] = y0[1]


def d(phi, theta, xi):
    dthetadphi = xi
    dxidphi = -beta * xi - np.sin(theta) + epsilon * np.cos(Omega * phi)
    return dthetadphi, dxidphi

# RK4
for i in range(N - 1):
    phi = phi_vals[i]
    theta = theta_vals[i]
    xi = xi_vals[i]

    # k1
    k1_theta, k1_xi = d(phi, theta, xi)

    # k2
    k2_theta, k2_xi = d(phi + 0.5*h, theta + 0.5*h*k1_theta, xi + 0.5*h*k1_xi)

    # k3
    k3_theta, k3_xi = d(phi + 0.5*h, theta + 0.5*h*k2_theta, xi + 0.5*h*k2_xi)

    # k4
    k4_theta, k4_xi = d(phi + h, theta + h*k3_theta, xi + h*k3_xi)



    theta_vals[i+1] = theta + (h / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    xi_vals[i+1] = xi + (h / 6) * (k1_xi + 2*k2_xi + 2*k3_xi + k4_xi)

# θ vs. Time φ Plot
plt.figure(figsize=(12, 6))
plt.plot(phi_vals, theta_vals, color='teal', label='θ')
plt.xlabel('φ (Dimensionless Time)')
plt.ylabel('θ')
plt.title('Forced Damped Pendulum (RK4 Method)')
plt.grid(True)
plt.legend()

# Phase-space (ξ vs. θ) Plot
plt.figure(figsize=(8, 8))
plt.plot(theta_vals, xi_vals, color='crimson', linewidth=0.5)
plt.xlabel('θ')
plt.ylabel('ξ = dθ/dφ')
plt.title('Phase-Space Trajectory (RK4 Method)')
plt.grid(True)

# Energy vs. Time Plot
kinetic = 0.5 * (xi_vals**2)
potential = 1.0 - np.cos(theta_vals)
total = kinetic + potential

plt.figure(figsize=(12, 6))
plt.plot(phi_vals, kinetic,   'r--', label='Kinetic Energy')
plt.plot(phi_vals, potential, 'g--', label='Potential Energy')
plt.plot(phi_vals, total,     'b-',  label='Total Energy')
plt.xlabel('φ (Dimensionless Time)')
plt.ylabel('Energy')
plt.title('Energy vs. Time (RK4 Method)')
plt.grid(True)
plt.legend()
plt.show()
