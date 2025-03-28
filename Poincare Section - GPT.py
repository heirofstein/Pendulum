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

def pendulum(phi, y, beta, epsilon, omega):
    theta, xi = y
    return [xi, -beta*xi - np.sin(theta) + epsilon*np.cos(omega*phi)]

beta = 0.3      
epsilon = 1.2
omega = 0.3     

y0 = [0.2, 0]

T = 2 * np.pi / omega

num_periods = 200
phi_span = (0, num_periods * T)
phi_eval = np.linspace(*phi_span, 10000*num_periods)

sol = solve_ivp(pendulum, phi_span, y0, args=(beta, epsilon,omega), t_eval=phi_eval, method='RK45')

poincare_phi = np.arange(0, num_periods) * T
poincare_points = np.interp(poincare_phi, sol.t, sol.y[0]), np.interp(poincare_phi, sol.t, sol.y[1])

plt.figure(figsize=(8, 8))
plt.scatter(poincare_points[0], poincare_points[1], s=10, color='magenta', marker='*')
plt.xlabel('Angular Displacement (θ)')
plt.ylabel('Angular Velocity (ξ = dθ/dφ)')
plt.title('Poincaré Section of Forced Damped Pendulum')
plt.grid(True)
plt.show()
