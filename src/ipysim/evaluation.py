from typing import Callable, Dict, Any
import numpy as np
from ipysim.core import simulate_maglev

class SimulationEvaluator:
    def __init__(self):
        """
        Initialize the evaluator with an empty set of evaluation functions.
        """
        self.evaluation_functions = {}

    def add_evaluation_function(self, name: str, func: Callable[[Dict[str, Any]], Any]) -> None:
        """
        Add a new evaluation function.

        Args:
            name (str): Name of the evaluation function.
            func (Callable): Function that takes a dictionary of inputs and returns a result.
        """
        self.evaluation_functions[name] = func

    def evaluate(self, name: str, **kwargs) -> Any:
        """
        Evaluate a specific function with the provided arguments.

        Args:
            name (str): Name of the evaluation function to use.
            kwargs: Arguments to pass to the evaluation function.

        Returns:
            Any: Result of the evaluation.
        """
        if name not in self.evaluation_functions:
            raise ValueError(f"Evaluation function '{name}' not found.")
        return self.evaluation_functions[name](kwargs)

# Example evaluation function
def time_to_stabilize_evaluation(params: Dict[str, Any]) -> float:
    """
    Evaluate the time it takes for the simulation to stabilize.

    Args:
        params (Dict[str, Any]): Dictionary containing 'Kp', 'Kd', 'state0', 'params', 'T', and 'dt'.

    Returns:
        float: Time to stabilize in seconds.
    """
    Kp = params["Kp"]
    Kd = params["Kd"]
    state0 = params["state0"]
    sim_params = params["params"]
    T = params["T"]
    dt = params["dt"]

    t, sol = simulate_maglev(Kp, Kd, T, dt, state0, sim_params)
    z = sol[:, 1]  # Extract the z (height) position
    threshold = 1e-3  # Stabilization threshold
    for i in range(len(z) - 1):
        if abs(z[i + 1] - z[i]) < threshold:
            return t[i]
    return T  # Return total time if not stabilized

# Add the example function to the evaluator
evaluator = SimulationEvaluator()
evaluator.add_evaluation_function("time_to_stabilize", time_to_stabilize_evaluation)
