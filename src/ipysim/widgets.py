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
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, Ellipse
import warnings
import contextlib
import io
import sys
import os
import base64
from ipywidgets import interact, FloatSlider, Button, Output, VBox, HTML, HBox
from IPython.display import display, Javascript, HTML as IPythonHTML
from ipysim.core import simulate_maglev
from ipysim.params import params as default_params, state0 as default_state0

# Globals for external use
t = None
sol = None
Kp = None
Kd = None
last_valid_Kp = None
last_valid_Kd = None

# Context manager to redirect stderr to browser console
@contextlib.contextmanager
def redirect_stderr_to_console():
    """
    Context manager to redirect stderr output to the browser's JavaScript console.
    
    This captures ipywidgets warnings and errors and displays them in the browser console
    instead of in the notebook output.
    """
    # Create a StringIO object to capture stderr
    stderr_capture = io.StringIO()
    
    # Save the original stderr
    old_stderr = sys.stderr
    
    try:
        # Redirect stderr to our capture object
        sys.stderr = stderr_capture
        yield  # Execute the code block inside the with statement
    finally:
        # Get the captured content
        stderr_content = stderr_capture.getvalue()
        
        # Restore the original stderr
        sys.stderr = old_stderr
        
        # If there's content, display it in the browser console
        if stderr_content:
            # Escape quotes and newlines for JavaScript
            stderr_content = stderr_content.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
            # Send to browser console wrapped in a try-catch to prevent errors
            js_code = f"""
            try {{
                console.error('[IPySim error message]: ' + '{stderr_content}');
            }} catch(e) {{
                console.error('Logging failed')
            }}
            """
            try:
                display(Javascript(js_code))
            except:
                pass

def interactive_simulation(
    params: Optional[Dict[str, float]] = None,
    state0: Optional[List[float]] = None,
    T: float = 1.0,
    dt: float = 0.001,
    Kp_default: float = 600.0,
    Kd_default: float = 30.0,
    Kp_min: float = 20.0,  
    Kd_min: float = 10.0,  
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
    animation_out = Output()
    result_output = Output()

    def validate_parameters(Kp: float, Kd: float) -> bool:
        """Validate controller parameters to prevent computation errors."""
        if Kp < Kp_min:
            return False
        if Kd < Kd_min:
            return False
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

    def create_maglev_animation(t, sol):
        """
        Create a base64 frame animation using JavaScript and HTML.
        Returns:
            IPythonHTML: HTML+JS snippet with a styled animation player.
        """

        try:
            # Prepare subsampled trajectory
            target_frames = min(100, len(t))
            frame_skip = max(1, len(t) // target_frames)
            t_sub = t[::frame_skip]
            x = sol[::frame_skip, 0]
            z = sol[::frame_skip, 1]
            theta = sol[::frame_skip, 2]

            # Generate PNG base64 frames
            base64_frames = []
            for i in range(len(t_sub)):
                fig, ax = plt.subplots(figsize=(4, 3))
                ax.set_xlim(-0.05, 0.05)
                ax.set_ylim(0, 0.1)
                ax.set_aspect('equal')
                ax.set_title(f"t = {t_sub[i]:.2f}s")

                # Draw base and magnet
                ax.add_patch(Rectangle((-0.06, 0), 0.12, 0.01, fc='#3a3a3a'))
                for pos in [-0.03, -0.01, 0.01, 0.03]:
                    ax.add_patch(Circle((pos, 0.01), 0.005, fc='#e63946'))
                ax.add_patch(Ellipse((x[i], z[i]), 0.024, 0.006, angle=np.degrees(theta[i]), fc='#1d71b8'))

                # Save to base64
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode("utf-8")
                base64_frames.append(b64)
                plt.close(fig)

            # Generate HTML and JS animation
            js_script = f"""
            <div id="maglev-animation" style="background:#f8f9fa;padding:1em;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1);text-align:center;">
                <h3 style="font-family:Arial,sans-serif;color:#1d3557;">Magnetic Levitation Animation</h3>
                <img id="maglev-frame" style="width:100%;max-width:400px;border:1px solid #ccc;border-radius:6px;" />
            </div>
            <script>
            const frames = {base64_frames};
            let i = 0;
            const img = document.getElementById("maglev-frame");
            setInterval(() => {{
                img.src = "data:image/png;base64," + frames[i];
                i = (i + 1) % frames.length;
            }}, 1000 / 30);  // 30 FPS
            </script>
            """
            return IPythonHTML(js_script)

        except Exception as e:
            return IPythonHTML(f"<p style='color:red;'>Failed to render animation: {e}</p>")


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
            with redirect_stderr_to_console():
                Kp_slider.value = last_valid_Kp
                Kd_slider.value = last_valid_Kd
            return
        
        try:
            # Store current values before simulation attempt
            attempted_Kp = Kp
            attempted_Kd = Kd
            
            with redirect_stderr_to_console():
                t, sol = simulate_maglev(Kp, Kd, T, dt, state0, params)
            
            # Validate the solution
            if not is_valid_solution(sol):
                with out:
                    out.clear_output(wait=True)
                    print(f"Simulation with Kp={attempted_Kp}, Kd={attempted_Kd} produced unstable results.")
                    print(f"Rolling back to last valid values: Kp={last_valid_Kp}, Kd={last_valid_Kd}")
                
                # Roll back to last valid values
                with redirect_stderr_to_console():
                    Kp_slider.value = last_valid_Kp
                    Kd_slider.value = last_valid_Kd
                
                # Re-run simulation with last valid parameters
                with redirect_stderr_to_console():
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

            # Clear animation output first so it doesn't show before the plots
            with animation_out:
                animation_out.clear_output(wait=True)
                
            with out:
                out.clear_output(wait=True)
                
                # Additional safety check before plotting
                try:
                    with redirect_stderr_to_console():
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
                    with redirect_stderr_to_console():
                        plt.close('all')  # Close any partially created figures
                    print(f"Error during plotting: {str(e)}")
                    print(f"The simulation may have produced extreme values that cannot be displayed.")
                    print(f"Try different parameters with higher Kp and Kd values.")
            
            # Now create and display animation after the plots
            with animation_out:
                try:
                    # Add a loading message while animation is being created
                    display(HTML("<p>Creating animation, please wait...</p>"))
                    
                    # Create and display the animation
                    animation_html = create_maglev_animation(t, sol)
                    
                    # Clear the loading message and show animation
                    animation_out.clear_output(wait=True)
                    display(animation_html)
                except Exception as e:
                    animation_out.clear_output(wait=True)
                    print(f"Error creating animation: {str(e)}")

        except Exception as e:
            with out:
                out.clear_output(wait=True)
                # Roll back to last valid values
                with redirect_stderr_to_console():
                    Kp_slider.value = last_valid_Kp
                    Kd_slider.value = last_valid_Kd
                
                # Display error message with rollback information
                print(f"Error: {e}")
                print(f"Rolling back to last valid values: Kp={last_valid_Kp}, Kd={last_valid_Kd}")
                
                # Use last valid values to show a working plot
                if t is not None and sol is not None:
                    with redirect_stderr_to_console():
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

            with redirect_stderr_to_console():
                if evaluation_function(sol, t):
                    print("Correct!")
                else:
                    print("Incorrect!")

    Kp_slider = FloatSlider(value=max(Kp_default, Kp_min), min=Kp_min, max=1000, step=10.0, description='Kp')
    Kd_slider = FloatSlider(value=max(Kd_default, Kd_min), min=Kd_min, max=200, step=5.0, description='Kd')
    evaluate_button = Button(description="Evaluate")
    evaluate_button.on_click(evaluate_parameters)

    with redirect_stderr_to_console():
        interact(
            simulate_and_plot,
            Kp=Kp_slider,
            Kd=Kd_slider
        )

    output_widgets = [out, animation_out]
    if evaluation_function is not None:
        # Adds widgets for evalution
        output_widgets += [evaluate_button, result_output]
    with redirect_stderr_to_console():
        display(VBox(output_widgets))