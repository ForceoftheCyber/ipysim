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
from ipywidgets import interact, FloatSlider, Button, Output, VBox
from IPython.display import display
import warnings
from scipy.integrate import ODEintWarning
# import solara
from ipysim.core import simulate_maglev, maglev_measurements
from ipysim.params import params as default_params, state0 as default_state0

# Globals for external use
t = None
sol = None
Kp = None
Kd = None
last_valid_Kp = None
last_valid_Kd = None

def interactive_simulation(
    params: Optional[Dict[str, float]] = None,
    state0: Optional[List[float]] = None,
    T: float = 1.0,
    dt: float = 0.001,
    Kp_default: float = 600.0,
    Kd_default: float = 30.0,
) -> None:
    """
    Create an interactive simulation for the maglev system using Jupyter widgets.

    This function allows users to:
    - Adjust the proportional (`Kp`) and derivative (`Kd`) gains using sliders.
    - Visualize the system's behavior over time.

    Args:
        params (Optional[Dict[str, float]]): Simulation parameters (e.g., mass, magnetic properties).
        state0 (Optional[List[float]]): Initial state of the system [x, z, theta, dx, dz, dtheta].
        T (float): Total simulation time in seconds.
        dt (float): Time step for the simulation.
        Kp_default (float): Default proportional gain for the PD controller.
        Kd_default (float): Default derivative gain for the PD controller.

    Returns:
        None
    """
    # Suppress ODEintWarning
    warnings.filterwarnings("ignore", category=ODEintWarning)

    global Kp, Kd, last_valid_Kp, last_valid_Kd
    params = params or default_params
    state0 = state0 or default_state0

    out = Output()
    print_button = Button(description="Output")

    def simulate_and_plot(Kp: float, Kd: float) -> None:
        """
        Simulate the maglev system and plot the results.

        Args:
            Kp (float): Proportional gain for the PD controller.
            Kd (float): Derivative gain for the PD controller.

        Returns:
            None
        """
        global t, sol, last_valid_Kp, last_valid_Kd
        try:
            t, sol = simulate_maglev(Kp, Kd, T, dt, state0, params)

            # Validate simulation results
            if t is None or sol is None or np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
                raise ValueError("Simulation produced invalid results.")

            # Update last valid values
            last_valid_Kp = Kp
            last_valid_Kd = Kd

            with out:
                out.clear_output(wait=True)
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

        except Exception as e:
            # Roll back to last valid values
            with out:
                out.clear_output(wait=True)
                print(f"Error: {e}. Rolling back to last valid parameters (Kp={last_valid_Kp}, Kd={last_valid_Kd}).")
            Kp_slider.value = last_valid_Kp
            Kd_slider.value = last_valid_Kd

    def print_arrays(_):
        """
        Print the simulation time and solution arrays.

        Args:
            _ : Unused argument (required for button callback).

        Returns:
            None
        """
        with out:
            out.clear_output()
            if t is not None and sol is not None:
                print(f"Time: (len={len(t)}): {t[:5]} ...")
                print(f"Solution: (shape={sol.shape}):\n{sol[:5]} ...")
            else:
                print("Simulation not yet run.")

    print_button.on_click(print_arrays)
    Kp_slider = FloatSlider(value=Kp_default, min=0, max=1000, step=10.0, description='Kp')
    Kd_slider = FloatSlider(value=Kd_default, min=0, max=200, step=5.0, description='Kd')
    # Display the interactive sliders and button/output separately
    interact(
        simulate_and_plot,
        Kp=FloatSlider(value=Kp_default, min=0, max=1000, step=10.0, description='Kp'),
        Kd=FloatSlider(value=Kd_default, min=0, max=200, step=5.0, description='Kd')
    )

    display(VBox([print_button, out]))

# @solara.component
# def MaglevControl(
#     params: Optional[Dict[str, float]] = None,
#     state0: Optional[List[float]] = None,
#     T: float = 1.0,
#     dt: float = 0.001,
#     Kp_default: float = 600.0,
#     Kd_default: float = 30.0,
# ):
#     Kp = solara.use_reactive(Kp_default)
#     Kd = solara.use_reactive(Kd_default)

#     def simulate_and_plot(Kp_val: float, Kd_val: float) -> None:
#         simulate_maglev(Kp_val, Kd_val, T, dt, state0 or default_state0, params or default_params)

#     solara.SliderFloat("Kp", value=Kp, min=0, max=1000, step=10.0)
#     solara.SliderFloat("Kd", value=Kd, min=0, max=200, step=5.0)

#     # simulate_and_plot(Kp.value, Kd.value)