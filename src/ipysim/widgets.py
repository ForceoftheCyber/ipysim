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
from ipysim.evaluation import SimulationEvaluator

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
    target_kp: float = 600.0,
    target_kd: float = 30.0,
) -> None:
    """
    Create an interactive simulation for the maglev system using Jupyter widgets.

    This function allows users to:
    - Adjust the proportional (`Kp`) and derivative (`Kd`) gains using sliders.
    - Visualize the system's behavior over time.
    - Evaluate if the student-selected `Kp` and `Kd` match the target values.

    Args:
        params (Optional[Dict[str, float]]): Simulation parameters (e.g., mass, magnetic properties).
        state0 (Optional[List[float]]): Initial state of the system [x, z, theta, dx, dz, dtheta].
        T (float): Total simulation time in seconds.
        dt (float): Time step for the simulation.
        Kp_default (float): Default proportional gain for the PD controller.
        Kd_default (float): Default derivative gain for the PD controller.
        target_kp (float): Target proportional gain for evaluation.
        target_kd (float): Target derivative gain for evaluation.

    Returns:
        None
    """
    # Suppress ODEintWarning
    warnings.filterwarnings("ignore", category=ODEintWarning)

    global t, sol
    params = params or default_params
    state0 = state0 or default_state0

    out = Output()
    result_output = Output()

    def simulate_and_plot(Kp: float, Kd: float) -> None:
        """
        Simulate the maglev system and plot the results.

        Args:
            Kp (float): Proportional gain for the PD controller.
            Kd (float): Derivative gain for the PD controller.

        Returns:
            None
        """
        global t, sol
        try:
            t, sol = simulate_maglev(Kp, Kd, T, dt, state0, params)

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
            with out:
                out.clear_output(wait=True)
                print(f"Error: {e}")

    def evaluate_parameters(_) -> None:
        """
        Evaluate if the current Kp and Kd match the target values.

        Args:
            _ : Unused argument (required for button callback).

        Returns:
            None
        """
        with result_output:
            result_output.clear_output(wait=True)
            if Kp_slider.value == target_kp and Kd_slider.value == target_kd:
                #print(f"Correct! Kp={Kp_slider.value}, Kd={Kd_slider.value} match the target values.")
                print("Correct!")
            else:
                # print(f"Incorrect. Kp={Kp_slider.value}, Kd={Kd_slider.value} do not match the target values.")
                print("Incorrect!")

    Kp_slider = FloatSlider(value=Kp_default, min=0, max=1000, step=10.0, description='Kp')
    Kd_slider = FloatSlider(value=Kd_default, min=0, max=200, step=5.0, description='Kd')
    evaluate_button = Button(description="Evaluate")
    evaluate_button.on_click(evaluate_parameters)

    interact(
        simulate_and_plot,
        Kp=Kp_slider,
        Kd=Kd_slider
    )

    display(VBox([evaluate_button, result_output, out]))