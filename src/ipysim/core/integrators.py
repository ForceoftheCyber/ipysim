from scipy.integrate import odeint

def simulate(dynamics_func, initial_state, t_span, params):
    """
    Simulates a dynamical system using numerical integration.

    Parameters:
    - dynamics_func (function): Function defining system dynamics.
    - initial_state (list): Initial state of the system.
    - t_span (np.array): Array of time steps for the simulation.
    - params (object): Parameters required by the dynamics function.

    Returns:
    - np.array: Simulated state trajectories over time.
    """
    return odeint(dynamics_func, initial_state, t_span, args=(params,))
