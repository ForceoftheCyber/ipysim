"""
widgets.py

Interactive visualization module for maglev simulation using Jupyter widgets.

Provides slider controls for PD gains (Kp, Kd) and plots:
- x and z position over time
- Phase plot of x vs theta

Users can optionally pass in custom simulation parameters and initial state.
"""

from typing import Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

from ipysim.core import simulate_maglev, maglev_measurements
from ipysim.params import params as default_params, state0 as default_state0


def interactive_simulation(
    *,
    params: Optional[Dict[str, float]] = None,
    state0: Optional[List[float]] = None,
    T: float = 1.0,
    dt: float = 0.001,
    Kp_default: float = 600.0,
    Kd_default: float = 30.0,
) -> None:
    """
    Launch interactive slider-based simulation with customizable parameters.

    Args:
        params: Optional dictionary of simulation parameters.
        state0: Optional list of initial state [x, z, theta, dx, dz, dtheta].
        T: Total simulation time (s).
        dt: Time step for integration.
        Kp_default: Default proportional gain for PD control.
        Kd_default: Default derivative gain for PD control.
    """
    params = params or default_params
    state0 = state0 or default_state0

    def simulate_and_plot(Kp: float, Kd: float) -> None:
        t, sol = simulate_maglev(Kp, Kd, T, dt, state0, params)

        u_array = np.array([
            -Kp * maglev_measurements(s, params["m"], params["mu0"])[0]
            -Kd * maglev_measurements(s, params["m"], params["mu0"])[1]
            for s in sol
        ])

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(t, sol[:, 1], label='z (height)')
        plt.plot(t, sol[:, 0], label='x (horizontal)')
        plt.xlabel('Time [s]')
        plt.ylabel('Position [m]')
        plt.title('Position of levitating magnet')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(sol[:, 0], sol[:, 2])
        plt.xlabel('x')
        plt.ylabel('theta')
        plt.title('Phase plot: x vs theta')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    interact(
        simulate_and_plot,
        Kp=FloatSlider(value=Kp_default, min=0, max=1000, step=10.0, description='Kp'),
        Kd=FloatSlider(value=Kd_default, min=0, max=200, step=5.0, description='Kd')
    )
