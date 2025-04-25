# ABOUT POINCARE SECTION PLOTS #

# "Comparison of Parameter Depended Motions' Poincare Sections.py "

(1) The code starts with importing required libraries for numerically solving differential equation for the forced damped pendulum:
    
 "  import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp "

(2) The equation for nondimensionalized forced damped pendulum is defined:

"   def forced_pendulum(phi, state, beta, epsilon, omega):
    theta, xi = state
    dtheta_dphi = xi
    dxi_dphi = -beta* xi - np.sin(theta) + epsilon* np.cos(omega*phi)
    return [dtheta_dphi, dxi_dphi] "


where:
    theta: angular displacement
    xi: angular frequency
    beta: damping factor
    epsion: forcing amplitude
    omega: driving frequency

NOTE: The system is turned into a first-order differential equation by writing dtheta/dphi = xi and dxi/dphi = -beta* xi - sin(theta) + epsilon* cos(omega*phi).

(3) The poincare section function is defined:

"   def poincare_section(beta, epsilon, omega, theta0=0.2, xi0=0.0, num_periods=2000, steps_per_period=1000):   "

where:
    T = 2*pi / omega: Periof of the driving force.
    phi_eval = np.linspace(...): To obtain spaced time points.
    sample_times = np.arange(1, num_periods+1) * T: Takes snapshots at times where the time is a multiple of the driving force.

This function solves the ODE I have written in (2) and samples the "state" (theta, xi) once per driving force period to create a Poincare section.


(4) Integration and Sampling Part

"    sol = solve_ivp(forced_pendulum, (0, phi_max), [theta0, xi0], args=(beta, epsilon, omega), t_eval=phi_eval, method='RK45') "

This part is where the numerical integration happens from phi = 0 to phi = phi' using Runge-Kutta method of order 5(4).

"   theta_vals = np.interp(sample_times, sol.t, sol.y[ 0 ])
    xi_vals = np.interp(sample_times, sol.t, sol.y[ 1 ])  "

Uses np.interp to the solution obtained from integration to get theta and xi values at each full periof of the driving force.

(5) Setting the Parameters

"   param_sets = [
    (0.2, 1.06, 2/3,  'Set A: near periodic'),
    (0.25, 1.2,  2/3,  'Set B: moderate forcing'),
    (0.25, 1.5,  2/3,  'Set C: chaotic regime')
]

Three different parameter sets are used. Low epsilon represents regular or periodic motion and the higher the epsilon value is, higher the chaotic behavior starts.

(6) Plotting Poincare Sections

" for i, (beta, epsilon, omega, label) in enumerate(param_sets, start=1):
    theta_p, xi_p = poincare_section(beta, epsilon, omega)
    plt.subplot(1, 3, i)
    plt.scatter(theta_p, xi_p, s=2, c='k') "

Each subplot is scaterring the plot of theta vs xi points taken once per period.

The pattern of the plots should reveal whether the motion is periodic or chaotic depending on the points: Closed loops in pahse space reveals periodic motion and scattered cloud reveals chaotic motion.