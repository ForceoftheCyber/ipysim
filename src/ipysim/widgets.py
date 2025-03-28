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
import solara

from ipysim.core import simulate_maglev, maglev_measurements
from ipysim.params import params as default_params, state0 as default_state0

# Globals for external use
t = None
sol = None

def interactive_simulation(
    params: Optional[Dict[str, float]] = None,
    state0: Optional[List[float]] = None,
    T: float = 1.0,
    dt: float = 0.001,
    Kp_default: float = 600.0,
    Kd_default: float = 30.0,
) -> None:
    params = params or default_params
    state0 = state0 or default_state0

    out = Output()
    print_button = Button(description="Output")

    def simulate_and_plot(Kp: float, Kd: float) -> None:
        global t, sol
        t, sol = simulate_maglev(Kp, Kd, T, dt, state0, params)

        plt.close('all')  # Close existing figures
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

    def print_arrays(_):
        with out:
            out.clear_output()
            if t is not None and sol is not None:
                print(f"Time: (len={len(t)}): {t[:5]} ...")
                print(f"Solution: (shape={sol.shape}):\n{sol[:5]} ...")
            else:
                print("Simulation not yet run.")

    print_button.on_click(print_arrays)

    # Display the interactive sliders and button/output separately
    interact(
        simulate_and_plot,
        Kp=FloatSlider(value=Kp_default, min=0, max=1000, step=10.0, description='Kp'),
        Kd=FloatSlider(value=Kd_default, min=0, max=200, step=5.0, description='Kd')
    )

    display(VBox([print_button, out]))

@solara.component
def MaglevControl(
    params: Optional[Dict[str, float]] = None,
    state0: Optional[List[float]] = None,
    T: float = 1.0,
    dt: float = 0.001,
    Kp_default: float = 600.0,
    Kd_default: float = 30.0,
):
    Kp = solara.use_reactive(Kp_default)
    Kd = solara.use_reactive(Kd_default)

    def simulate_and_plot(Kp_val: float, Kd_val: float) -> None:
        simulate_maglev(Kp_val, Kd_val, T, dt, state0 or default_state0, params or default_params)

    solara.SliderFloat("Kp", value=Kp, min=0, max=1000, step=10.0)
    solara.SliderFloat("Kd", value=Kd, min=0, max=200, step=5.0)

    # simulate_and_plot(Kp.value, Kd.value)