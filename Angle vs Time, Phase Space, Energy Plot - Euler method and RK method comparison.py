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


def d(phi, theta, xi):
    dtheta = xi
    dxi = -beta * xi - np.sin(theta) + epsilon * np.cos(Omega * phi)
    return dtheta, dxi




theta_euler = np.zeros(N)
xi_euler = np.zeros(N)
theta_rk4 = np.zeros(N)
xi_rk4 = np.zeros(N)



# Initial conditions
theta_euler[0], xi_euler[0] = y0
theta_rk4[0], xi_rk4[0] = y0

### Euler Method
for i in range(N - 1):
    phi = phi_vals[i]
    theta = theta_euler[i]
    xi = xi_euler[i]
    
    dtheta, dxi = d(phi, theta, xi)
    
    theta_euler[i+1] = theta + h * dtheta
    xi_euler[i+1] = xi + h * dxi

### RK4 Method
for i in range(N - 1):
    phi = phi_vals[i]
    theta = theta_rk4[i]
    xi = xi_rk4[i]
    
    k1_theta, k1_xi = d(phi, theta, xi)
    k2_theta, k2_xi = d(phi + 0.5*h, theta + 0.5*h*k1_theta, xi + 0.5*h*k1_xi)
    k3_theta, k3_xi = d(phi + 0.5*h, theta + 0.5*h*k2_theta, xi + 0.5*h*k2_xi)
    k4_theta, k4_xi = d(phi + h, theta + h*k3_theta, xi + h*k3_xi)
    
    theta_rk4[i+1] = theta + (h / 6) * (k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
    xi_rk4[i+1] = xi + (h / 6) * (k1_xi + 2*k2_xi + 2*k3_xi + k4_xi)

### Plots

# θ vs Time
plt.figure(figsize=(12, 6))
plt.plot(phi_vals, theta_euler, label='Euler θ', linestyle='-')
plt.plot(phi_vals, theta_rk4, label='RK4 θ', linestyle='-')
plt.xlabel('φ')
plt.ylabel('θ')
plt.title('Angle vs Time: Euler vs RK4')
plt.legend()
plt.grid(True)

# Phase Space
plt.figure(figsize=(8, 8))
plt.plot(theta_euler, xi_euler, label='Euler', linestyle='--')
plt.plot(theta_rk4, xi_rk4, label='RK4', linestyle='-')
plt.xlabel('θ')
plt.ylabel('ξ')
plt.title('Phase Space: Euler vs RK4')
plt.legend()
plt.grid(True)

#Comparison
kinetic_euler = 0.5 * xi_euler**2
potential_euler = 1.0 - np.cos(theta_euler)
total_euler = kinetic_euler + potential_euler

kinetic_rk4 = 0.5 * xi_rk4**2
potential_rk4 = 1.0 - np.cos(theta_rk4)
total_rk4 = kinetic_rk4 + potential_rk4

plt.figure(figsize=(12, 6))
plt.plot(phi_vals, total_euler, label='Euler Total Energy', linestyle='-')
plt.plot(phi_vals, total_rk4, label='RK4 Total Energy', linestyle='-')
plt.xlabel('φ')
plt.ylabel('Total Energy')
plt.title('Total Energy vs Time: Euler vs RK4')
plt.legend()
plt.grid(True)

plt.show()
