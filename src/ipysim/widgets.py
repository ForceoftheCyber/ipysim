"""
widgets.py

Interactive visualization module for maglev simulation using Jupyter widgets.

Provides slider controls for PD gains (Kp, Kd) and plots:
- x and z position over time
- Phase plot of x vs theta

Users can optionally pass in custom simulation parameters and initial state.
"""

from typing import Callable, Optional, Dict, List
import numpy as np
import matplotlib.pyplot as plt
import warnings
from ipywidgets import interact, FloatSlider, Button, Output, VBox, HTML
from IPython.display import display
from ipysim.core import simulate_maglev
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
    Kp_min: float = 20.0,  # Increased minimum Kp to prevent computational errors
    Kd_min: float = 10.0,  # Increased minimum Kd to prevent computational errors
    evaluation_function: Callable[[np.ndarray, np.ndarray], bool] | None = None,
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
        Kp_min (float): Minimum allowed value for Kp to prevent computational errors.
        Kd_min (float): Minimum allowed value for Kd to prevent computational errors.

    Returns:
        None
    """

    global t, sol, Kp, Kd, last_valid_Kp, last_valid_Kd
    params = params or default_params
    state0 = state0 or default_state0
    
    # Initialize last valid values with defaults
    last_valid_Kp = max(Kp_default, Kp_min)
    last_valid_Kd = max(Kd_default, Kd_min)

    out = Output()
    result_output = Output()
    status_message = HTML(value="")

    def validate_parameters(Kp: float, Kd: float) -> bool:
        """Validate controller parameters to prevent computation errors."""
        if Kp < Kp_min:
            status_message.value = f"<span style='color:red'>Warning: Kp must be at least {Kp_min}</span>"
            return False
        if Kd < Kd_min:
            status_message.value = f"<span style='color:red'>Warning: Kd must be at least {Kd_min}</span>"
            return False
        status_message.value = ""
        return True

    def is_valid_solution(solution: np.ndarray) -> bool:
        """Check if the solution is valid and doesn't contain extreme values."""
        if solution is None or solution.size == 0:
            return False
            
        # Check for NaN or Inf values
        if np.isnan(solution).any() or np.isinf(solution).any():
            return False
            
        # Check for extreme values that might cause overflow
        max_abs_value = np.max(np.abs(solution))
        if max_abs_value > 1e10:  # If any value is extremely large
            return False
            
        return True

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
        
        # Validate parameters before simulation
        if not validate_parameters(Kp, Kd):
            # Use the last valid values instead
            Kp_slider.value = last_valid_Kp
            Kd_slider.value = last_valid_Kd
            return
        
        try:
            # Store current values before simulation attempt
            attempted_Kp = Kp
            attempted_Kd = Kd
            
            t, sol = simulate_maglev(Kp, Kd, T, dt, state0, params)
            
            # Validate the solution
            if not is_valid_solution(sol):
                with out:
                    out.clear_output(wait=True)
                    print(f"Simulation with Kp={attempted_Kp}, Kd={attempted_Kd} produced unstable results.")
                    print(f"Rolling back to last valid values: Kp={last_valid_Kp}, Kd={last_valid_Kd}")
                
                # Roll back to last valid values
                Kp_slider.value = last_valid_Kp
                Kd_slider.value = last_valid_Kd
                
                # Re-run simulation with last valid parameters
                t, sol = simulate_maglev(last_valid_Kp, last_valid_Kd, T, dt, state0, params)
                
                # Make sure even this solution is valid
                if not is_valid_solution(sol):
                    with out:
                        out.clear_output(wait=True)
                        print("Even the last valid settings produced unstable results.")
                        print("Please try with different Kp and Kd values.")
                    return
            else:
                # Store successful values only if the simulation was valid
                last_valid_Kp = Kp
                last_valid_Kd = Kd

            with out:
                out.clear_output(wait=True)
                
                # Additional safety check before plotting
                try:
                    plt.figure(figsize=(12, 5))
                    
                    # First subplot
                    plt.subplot(1, 2, 1)
                    plt.plot(t, sol[:, 1], label='z (height)')
                    plt.plot(t, sol[:, 0], label='x (horizontal)')
                    plt.xlabel('Time [s]')
                    plt.ylabel('Position [m]')
                    plt.title('Position of levitating magnet')
                    plt.legend()
                    plt.grid(True)
                    
                    # Second subplot
                    plt.subplot(1, 2, 2)
                    plt.plot(sol[:, 0], sol[:, 2])
                    plt.xlabel('x')
                    plt.ylabel('theta')
                    plt.title('Phase plot: x vs theta')
                    plt.grid(True)
                    
                    plt.tight_layout()
                    plt.show()
                except (ValueError, OverflowError) as e:
                    # Catch specific errors during plotting
                    plt.close('all')  # Close any partially created figures
                    print(f"Error during plotting: {str(e)}")
                    print(f"The simulation may have produced extreme values that cannot be displayed.")
                    print(f"Try different parameters with higher Kp and Kd values.")

        except Exception as e:
            with out:
                out.clear_output(wait=True)
                # Roll back to last valid values
                Kp_slider.value = last_valid_Kp
                Kd_slider.value = last_valid_Kd
                
                # Display error message with rollback information
                print(f"Error: {e}")
                print(f"Rolling back to last valid values: Kp={last_valid_Kp}, Kd={last_valid_Kd}")
                
                # Use last valid values to show a working plot
                if t is not None and sol is not None:
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

            # Button that calls this function will not be shown if evaluation_function is None
            assert evaluation_function  
            
            global sol, t
            if sol is None or t is None:
                print("Simulation has not been run, change the parameters.")
                return

            if evaluation_function(sol, t):
                print("Correct!")
            else:
                print("Incorrect!")

    Kp_slider = FloatSlider(value=max(Kp_default, Kp_min), min=Kp_min, max=1000, step=10.0, description='Kp')
    Kd_slider = FloatSlider(value=max(Kd_default, Kd_min), min=Kd_min, max=200, step=5.0, description='Kd')
    evaluate_button = Button(description="Evaluate")
    evaluate_button.on_click(evaluate_parameters)

    interact(
        simulate_and_plot,
        Kp=Kp_slider,
        Kd=Kd_slider
    )

    output_widgets = [status_message, out]
    if evaluation_function is not None:
        # Adds widgets for evalution
        output_widgets += [evaluate_button, result_output]
    display(VBox(output_widgets))