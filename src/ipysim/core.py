# ipysim/core.py

import numpy as np
from scipy.integrate import odeint

def simulate_closed_loop(dynamics_fn, controller_fn, state0, params, T, dt):
    """
    Simulate a closed-loop system.
    
    dynamics_fn: function(state, t, u, params) -> state_dot
    controller_fn: function(state, t, params) -> u
    state0: initial state vector (list or numpy array)
    params: dictionary of simulation parameters
    T: total simulation time (seconds)
    dt: simulation time step (seconds)
    
    Returns:
        t: time vector (numpy array)
        sol: state trajectory (numpy array with shape [len(t), len(state0)])
    """
    t = np.arange(0, T, dt)

    def closed_loop(state, time):
        u = controller_fn(state, time, params)
        return dynamics_fn(state, time, u, params)
        
    sol = odeint(closed_loop, state0, t)
    return t, sol


def simulate_open_loop(dynamics_fn, input_fn, state0, params, T, dt):
    """
    Simulate an open-loop system.
    
    dynamics_fn: function(state, t, u, params) -> state_dot
    input_fn: function(t) -> u
    state0: initial state vector
    params: simulation parameters
    T: total simulation time
    dt: simulation time step
    
    Returns:
        t: time vector
        sol: state trajectory
    """
    t = np.arange(0, T, dt)

    def open_loop(state, time):
        u = input_fn(time)
        return dynamics_fn(state, time, u, params)
    
    sol = odeint(open_loop, state0, t)
    return t, sol
