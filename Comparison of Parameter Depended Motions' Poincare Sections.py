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

def forced_pendulum(phi, state, beta, epsilon, omega):
    theta, xi = state
    dtheta_dphi = xi
    dxi_dphi    = -beta*xi - np.sin(theta) + epsilon*np.cos(omega*phi)
    return [dtheta_dphi, dxi_dphi]


def poincare_section(beta, epsilon, omega,
                     theta0=0.2, 
                     xi0=0.0,
                     num_periods=2000,
                     steps_per_period=1000):
    """
    Returns arrays (theta_points, xi_points) for the Poincaré section.
    """
    T = 2.0 * np.pi / omega  
    phi_max = num_periods * T
 
    phi_eval = np.linspace(0, phi_max, steps_per_period*num_periods)

    sol = solve_ivp(
        forced_pendulum,
        (0, phi_max),
        [theta0, xi0],
        args=(beta, epsilon, omega),
        t_eval=phi_eval,
        method='RK45'
    )

    sample_times = np.arange(1, num_periods+1) * T

    theta_vals = np.interp(sample_times, sol.t, sol.y[0])
    xi_vals    = np.interp(sample_times, sol.t, sol.y[1])

    return theta_vals, xi_vals

param_sets = [

    (0.2, 1.06, 2/3,  'Set A: near periodic'),
    (0.25, 1.2,  2/3,  'Set B: moderate forcing'),
    (0.25, 1.5,  2/3,  'Set C: chaotic regime')
]

plt.figure(figsize=(14, 4.5))

for i, (beta, epsilon, omega, label) in enumerate(param_sets, start=1):

    theta_p, xi_p = poincare_section(beta, epsilon, omega)

    plt.subplot(1, 3, i)
    plt.scatter(theta_p, xi_p, s=2, c='k')
    plt.title(label)
    plt.xlabel('θ')
    plt.ylabel('ξ')
    plt.grid(True)

plt.tight_layout()
plt.show()

