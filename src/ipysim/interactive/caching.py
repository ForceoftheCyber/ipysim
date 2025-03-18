import numpy as np
import itertools
from ipysim.core.integrators import simulate

def quantize(min, max, step):
    """
    Generates a range of discrete values for a parameter.

    Parameters:
    - min (float): Minimum value.
    - max (float): Maximum value.
    - step (float): Step size.

    Returns:
    - np.array: Array of discrete parameter values.
    """
    return np.arange(min, max + step, step)

def precompute(dynamics, initial_state, t_span, params_ranges):
    """
    Precomputes simulation results for discrete parameter values.

    Parameters:
    - dynamics (function): System dynamics function.
    - initial_state (list): Initial state of the system.
    - t_span (np.array): Time steps for the simulation.
    - params_ranges (dict): Dictionary of parameter name → list of values.

    Returns:
    - dict: Precomputed solutions mapped to parameter values.
    """
    param_names = list(params_ranges.keys())
    param_values = itertools.product(*params_ranges.values())
    cache = {}

    for values in param_values:
        params = dict(zip(param_names, values))
        solution = simulate(dynamics, initial_state, t_span, params)
        cache[tuple(values)] = solution

    return cache
