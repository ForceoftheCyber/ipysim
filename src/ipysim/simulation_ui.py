# ipysim/simulation_ui.py

import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Button, Output, VBox, HBox, Checkbox, Label, HTML
from IPython.display import display, Javascript, HTML as IPythonHTML
import traceback
import numpy as np
from .utils import redirect_stderr_to_console

def interactive_simulation(simulate_fn, plot_fn=None, animation_fn=None, evaluation_function=None,
                           params=None, state0=None, T=10.0, dt=0.01, sliders_config=None):
    """
    Create an interactive simulation interface.
    
    simulate_fn: function(params, state0, T, dt, **slider_values) -> (t, sol)
    plot_fn: optional plotting function: plot_fn(t, sol)
    animation_fn: optional animation function: animation_fn(t, sol)
    evaluation_function: optional function for student evaluation: evaluation_function(t, sol) -> bool
    params: simulation parameters (dictionary)
    state0: initial state (list or array)
    T: total simulation time
    dt: simulation time step
    sliders_config: dict specifying additional slider configuration
    """
    params = params or {}
    state0 = state0 or []
    
    # Default slider configuration, for example gains.
    if sliders_config is None:
        sliders_config = {
            "Kp": {"default": 100.0, "min": 0.0, "max": 1000.0, "step": 10.0, "description": "Kp"},
            "Kd": {"default": 10.0, "min": 0.0, "max": 200.0, "step": 5.0, "description": "Kd"},
        }
    
    sliders = {name: FloatSlider(value=config["default"], min=config["min"],
                                 max=config["max"], step=config["step"],
                                 description=config["description"])
               for name, config in sliders_config.items()}
    
    # Store the last valid values for all sliders for rollback
    last_valid_values = {name: sliders[name].value for name in sliders}
    
    # Store the last valid simulation results
    last_valid_results = {'t': None, 'sol': None}
    
    # Store if we've had at least one successful simulation
    had_successful_sim = [False]
    
    out = Output()
    anim_out = Output()
    eval_out = Output()
    error_out = Output()
    
    # Create an error container that can be hidden/shown
    error_header = HTML("<b>Status:</b>")
    error_container = VBox([error_header, error_out])
    # Initially hide the error container since we have no errors yet
    error_container.layout.display = 'none'
        
    def is_valid_solution(solution):
        """Check if the solution contains valid numerical values."""
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
    
    def validate_parameters(**slider_values):
        """
        Validate controller parameters to prevent computation errors.
        Override this with simulation-specific validation if needed.
        """
        # Basic validation - ensure all values are in their acceptable ranges
        for name, value in slider_values.items():
            slider = sliders.get(name)
            if slider and (value < slider.min or value > slider.max):
                return False
        return True
    
    # Function to show or hide the error container
    def show_error(show=True):
        error_container.layout.display = 'flex' if show else 'none'
    
    # Function to run simulation using the current slider values.
    def run_simulation(**slider_values):
        with out:
            out.clear_output(wait=True)
            
            # Clear any previous error messages
            with error_out:
                error_out.clear_output(wait=True)
                
            # Validate parameters before running simulation
            if not validate_parameters(**slider_values):
                # Display error and roll back to last valid values
                with error_out:
                    print("Invalid parameter values. Rolling back to previous valid settings.")
                show_error(True)
                
                # Restore sliders to last valid values without triggering callbacks
                with redirect_stderr_to_console():
                    for name, value in last_valid_values.items():
                        if name in sliders:
                            sliders[name].value = value
                return
            
            try:
                # Run the simulation with current parameters
                t, sol = None, None
                with redirect_stderr_to_console():
                    t, sol = simulate_fn(params, state0, T, dt, **slider_values)
                
                # Validate the solution
                if not is_valid_solution(sol):
                    with error_out:
                        print("Simulation produced unstable results.")
                        print("Rolling back to last valid settings.")
                    show_error(True)
                    
                    if had_successful_sim[0]:
                        # Restore sliders to last valid values
                        with redirect_stderr_to_console():
                            for name, value in last_valid_values.items():
                                if name in sliders:
                                    sliders[name].value = value
                                    
                        # Re-run with last valid parameters if we have them
                        t, sol = last_valid_results['t'], last_valid_results['sol']
                        
                        # Make sure even this solution is valid
                        if not is_valid_solution(sol):
                            with error_out:
                                print("Even the last valid settings produced unstable results.")
                                print("Please try with different parameter values.")
                            return
                    else:
                        with error_out:
                            print("No previous valid settings found. Please try with different values.")
                        return
                else:
                    # Store these as the last valid values and results
                    for name, value in slider_values.items():
                        if name in last_valid_values:
                            last_valid_values[name] = value
                    
                    last_valid_results['t'] = t
                    last_valid_results['sol'] = sol
                    had_successful_sim[0] = True
                    
                    # Hide the error container since we have a successful simulation
                    show_error(False)
                
                # Plot the results
                if plot_fn:
                    try:
                        with redirect_stderr_to_console():
                            with redirect_stderr_to_console():  # Double wrapping to ensure capture
                                # Close any existing figures first
                                plt.close('all')
                                # Create new figure and plot
                                #fig = plt.figure(figsize=(8, 4))
                                plot_fn(t, sol)
                                # Show plot and immediately capture any output
                                plt.show()
                    except Exception as plot_error:
                        with error_out:
                            print("Error plotting simulation results.")
                            print("Please try different parameter values.")
                        show_error(True)
                
                # Save results so that animation and evaluation buttons can use them.
                run_simulation.t = t
                run_simulation.sol = sol
                
            except Exception as e:
                with error_out:
                    # Only show user-friendly error message, not the technical details
                    print("Simulation failed with current parameter values.")
                    
                    # Provide a more user-friendly error message if possible
                    if "overflow" in str(e).lower() or "underflow" in str(e).lower():
                        print("The simulation produced extreme values.")
                        print("Try using different controller parameters.")
                    elif "singular" in str(e).lower():
                        print("The simulation encountered a mathematical singularity.")
                        print("Try adjusting your initial conditions or parameters.")
                    
                    if had_successful_sim[0]:
                        print("Rolling back to last valid settings.")
                
                # Send the actual error details to the console
                with redirect_stderr_to_console():
                    error_details = traceback.format_exc()
                    # Uncomment for debugging
                    # print(f"Simulation error: {str(e)}\n{error_details}")
                
                # Show the error container
                show_error(True)
                
                # Roll back to last valid values if we have them
                if had_successful_sim[0]:
                    with redirect_stderr_to_console():
                        for name, value in last_valid_values.items():
                            if name in sliders:
                                sliders[name].value = value
    
    # Animation trigger.
    def run_animation(_):
        with anim_out:
            anim_out.clear_output(wait=True)
            if not hasattr(run_simulation, "t") or not hasattr(run_simulation, "sol"):
                print("No simulation results available. Adjust parameters first.")
                show_error(True)
            else:
                try:
                    # Add a loading message while animation is being created
                    display(HTML("<p>Creating animation, please wait...</p>"))
                    
                    # Create and display the animation
                    html_anim = None
                    with redirect_stderr_to_console():
                        # Close any existing figures first to avoid warnings
                        plt.close('all')
                        html_anim = animation_fn(run_simulation.t, run_simulation.sol)
                    
                    # Clear the loading message and show animation
                    anim_out.clear_output(wait=True)
                    display(html_anim)
                    with redirect_stderr_to_console():
                        display(Javascript("void(0);"))
                except Exception as e:
                    anim_out.clear_output(wait=True)
                    print("Failed to create animation with current results.")
                    # Send full error details to console
                    with redirect_stderr_to_console():
                        error_details = traceback.format_exc()
    
    # Evaluation trigger, if desired.
    def evaluate(_):
        with eval_out:
            eval_out.clear_output(wait=True)
            if not hasattr(run_simulation, "t") or not hasattr(run_simulation, "sol"):
                print("No simulation results available. Adjust parameters first.")
                show_error(True)
            else:
                try:
                    result = False
                    with redirect_stderr_to_console():
                        result = evaluation_function(run_simulation.t, run_simulation.sol)
                    if result:
                        print("Evaluation: Correct!")
                    else:
                        print("Evaluation: Incorrect!")
                except Exception as e:
                    print(f"Evaluation error: {str(e)}")
                    traceback.print_exc()
                    show_error(True)
    
    # Helper function to get all current slider values
    def get_slider_values():
        return {name: slider.value for name, slider in sliders.items()}
    
    # Add observe callbacks to each slider for real-time updates
    def on_slider_change(change):
        run_simulation(**get_slider_values())
    
    # Register callbacks for all sliders
    for name, slider in sliders.items():
        slider.observe(on_slider_change, names='value')
    
    # Create buttons (no Run Simulation button as requested)
    buttons_list = []

    # Only add animation button if animation_fn is provided
    if animation_fn:
        anim_button = Button(description="Run Animation")
        anim_button.on_click(run_animation)
        buttons_list.append(anim_button)
    
    # Add evaluation button if provided
    if evaluation_function:
        eval_button = Button(description="Evaluate")
        eval_button.on_click(evaluate)
        buttons_list.append(eval_button)
    
    # Create UI layout with error output (now hidden by default)
    controls = HBox([VBox(list(sliders.values()))])
    ui_elements = [controls]
    
    # Add buttons row if any buttons exist
    if buttons_list:
        ui_elements.append(HBox(buttons_list))
    
    # Add the error container (initially hidden)
    ui_elements.append(error_container)
    
    # Always include output area
    ui_elements.append(out)
    
    # Add animation output if needed
    if animation_fn:
        ui_elements.append(anim_out)
    
    # Add evaluation output if needed
    if evaluation_function:
        ui_elements.append(eval_out)
    
    ui = VBox(ui_elements)
    
    # Initial run to show something when UI first appears
    run_simulation(**get_slider_values())
    
    display(ui)
