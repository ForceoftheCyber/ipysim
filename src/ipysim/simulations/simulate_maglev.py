# ipysim/simulations/simulate_maglev.py

import numpy as np
from scipy.integrate import odeint
from ipysim.core import simulate_closed_loop

# --- Simulation-specific utility functions ---
def cross2D(a, b):
    return a[0]*b[1] - a[1]*b[0]

def field(state, m, mu0):
    x, z, theta = state[0], state[1], state[2]
    r = np.array([x, z])
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.zeros(2)
    m_vec = m * np.array([-np.sin(theta), np.cos(theta)])
    B = mu0 / (4*np.pi*r_norm**3) * (3*np.dot(m_vec, r)/r_norm**2 * r - m_vec)
    return B

def maglev_measurements(state, m, mu0, eps=1e-6):
    y = field(state, m, mu0)[0]
    grad = np.zeros(3)
    for i in range(3):
        state_plus = state.copy()
        state_minus = state.copy()
        state_plus[i] += eps
        state_minus[i] -= eps
        y_plus = field(state_plus, m, mu0)[0]
        y_minus = field(state_minus, m, mu0)[0]
        grad[i] = (y_plus - y_minus) / (2 * eps)
    state_dot = np.array(state[3:6])
    y_dot = np.dot(grad, state_dot)
    return y, y_dot

def force(m_i, m, r, mu0):
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.zeros_like(r)
    term1 = np.dot(m_i, r) * m
    term2 = np.dot(m, r) * m_i
    term3 = np.dot(m_i, m) * r
    term4 = 5 * np.dot(m_i, r) * np.dot(m, r) / r_norm**2 * r
    return (3*mu0/(4*np.pi*r_norm**5))*(term1 + term2 + term3 - term4)

def torque(m_i, m, r, mu0):
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return 0.0
    r_hat = r / r_norm
    return (mu0 / (4*np.pi*r_norm**3)) * cross2D(m, 3*np.dot(m_i, r_hat)*r_hat - m_i)

# --- Dynamics and Controller ---
def maglev_state_dynamics(state, t, u, params):
    x, z, theta, dx, dz, dtheta = state
    M = params["M"]
    m_val = params["m"]
    l = params["l"]
    g = params["g"]
    m_support = params["m_support"]
    k = params["k"]
    J = params["J"]
    mu0 = params["mu0"]
    
    r1 = np.array([l/2, 0])
    r2 = np.array([-l/2, 0])
    m1 = np.array([0.0, m_support + k*u])
    m2 = np.array([0.0, m_support - k*u])
    m_lev = m_val * np.array([-np.sin(theta), np.cos(theta)])
    r = np.array([x, z])
    
    F1 = force(m1, m_lev, r - r1, mu0)
    F2 = force(m2, m_lev, r - r2, mu0)
    F_total = F1 + F2 + M * np.array([0.0, -g])
    ddx, ddz = F_total / M
    ddz += -5*dz  # damping term
    torque_total = torque(m1, m_lev, r - r1, mu0) + torque(m2, m_lev, r - r2, mu0)
    ddtheta = torque_total / J
    return [dx, dz, dtheta, ddx, ddz, ddtheta]

def controller(state, t, params, Kp, Kd):
    """
    A PD controller for the maglev system.
    """
    y, y_dot = maglev_measurements(state, params["m"], params["mu0"])
    return -Kp*y - Kd*y_dot

def simulate(params, state0, T, dt, Kp, Kd):
    """
    Entry point for simulating the maglev system.
    Wraps the core.simulate_closed_loop.
    """
    def dynamics(state, t, u, params):
        return maglev_state_dynamics(state, t, u, params)
    
    def control_fn(state, t, params):
        return controller(state, t, params, Kp, Kd)
    
    return simulate_closed_loop(dynamics, control_fn, state0, params, T, dt)

# --- Visualization functions for this simulation ---
def plot_maglev(t, sol):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(t, sol[:, 0], label='x')
    plt.plot(t, sol[:, 1], label='z')
    plt.xlabel("Time [s]")
    plt.ylabel("Position")
    plt.title("Positions")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(sol[:, 0], sol[:, 2], label='theta')
    plt.xlabel("x")
    plt.ylabel("theta")
    plt.title("Phase plot: x vs theta")
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_frame_maglev(ax, sol, i):
    """
    Draw a simple representation of the maglev system:
    plot the magnet as a point.
    """
    ax.set_title("Maglev Animation")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    # Simple visualization: plot a point
    ax.plot(sol[i, 0], sol[i, 1], 'o', markersize=10)
    ax.set_xlim(-0.1, 0.1)
    ax.set_ylim(0, 0.1)

def animate_maglev(t, sol):
    """
    Create an animation for the maglev simulation using the generic
    create_animation function provided by the framework.
    """
    from ipysim.plotting import create_animation
    return create_animation(t, sol, draw_frame_maglev, interval=50)
