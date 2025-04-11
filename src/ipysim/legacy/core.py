"""
core.py

Core dynamics, control, and simulation logic for the magnetic levitation system.

Implements:
- Magnetic field computation
- Magnetic force and torque between dipoles
- Plant dynamics and PD control
- Numerical integration of the full closed-loop system
"""

import numpy as np
from scipy.integrate import odeint
from typing import List, Tuple, Dict


def cross2D(a: np.ndarray, b: np.ndarray) -> float:
    """2D scalar cross product."""
    return float(a[0] * b[1] - a[1] * b[0])


def field(state: List[float], m: float, mu0: float) -> np.ndarray:
    """Calculate magnetic field vector from levitating dipole."""
    x, z, theta = state[0], state[1], state[2]
    r = np.array([x, z])
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.zeros(2)
    m_vec = m * np.array([-np.sin(theta), np.cos(theta)])
    B = mu0 / (4 * np.pi * r_norm**3) * (
        3 * np.dot(m_vec, r) / r_norm**2 * r - m_vec
    )
    return B


def maglev_measurements(state: List[float], m: float, mu0: float, eps: float = 1e-6) -> Tuple[float, float]:
    """Compute magnetic x-field and its derivative using finite differences."""
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

    state_dot = np.array([state[3], state[4], state[5]])
    y_dot = float(np.dot(grad, state_dot))
    return float(y), y_dot


def force(m_i: np.ndarray, m: np.ndarray, r: np.ndarray, mu0: float) -> np.ndarray:
    """Calculate magnetic force on levitating dipole."""
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return np.zeros_like(r)

    term1 = np.dot(m_i, r) * m
    term2 = np.dot(m, r) * m_i
    term3 = np.dot(m_i, m) * r
    term4 = 5 * np.dot(m_i, r) * np.dot(m, r) / r_norm**2 * r

    return (3 * mu0 / (4 * np.pi * r_norm**5)) * (term1 + term2 + term3 - term4)


def torque(m_i: np.ndarray, m: np.ndarray, r: np.ndarray, mu0: float) -> float:
    """Calculate magnetic torque on levitating dipole."""
    r_norm = np.linalg.norm(r)
    if r_norm == 0:
        return 0.0
    r_hat = r / r_norm
    return float(mu0 / (4 * np.pi * r_norm**3) * cross2D(m, 3 * np.dot(m_i, r_hat) * r_hat - m_i))


def maglev_state_dynamics(state: List[float], t: float, u: float, params: Dict[str, float]) -> List[float]:
    """Compute time-derivative of the system state with input u."""
    x, z, theta, dx, dz, dtheta = state
    M = params["M"]
    m_val = params["m"]
    l = params["l"]
    g = params["g"]
    m_sup = params["m_support"]
    k = params["k"]
    J = params["J"]
    mu0 = params["mu0"]

    r1 = np.array([l / 2, 0])
    r2 = np.array([-l / 2, 0])
    m1 = np.array([0.0, m_sup + k * u])
    m2 = np.array([0.0, m_sup - k * u])
    m_lev = m_val * np.array([-np.sin(theta), np.cos(theta)])
    r = np.array([x, z])

    F1 = force(m1, m_lev, r - r1, mu0)
    F2 = force(m2, m_lev, r - r2, mu0)
    F_total = F1 + F2 + M * np.array([0.0, -g])
    ddx, ddz = F_total / M
    ddz += -5 * dz  # Damping

    torque_total = torque(m1, m_lev, r - r1, mu0) + torque(m2, m_lev, r - r2, mu0)
    ddtheta = torque_total / J

    return [dx, dz, dtheta, ddx, ddz, ddtheta]


def closed_loop_dynamics(state: List[float], t: float, params: Dict[str, float], Kp: float, Kd: float) -> List[float]:
    """Simulate the plant under a PD control loop."""
    y, y_dot = maglev_measurements(state, params["m"], params["mu0"])
    u = -Kp * y - Kd * y_dot
    return maglev_state_dynamics(state, t, u, params)


def simulate_maglev(Kp: float, Kd: float, T: float, dt: float, state0: List[float], params: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run full simulation of the levitating magnet.

    Returns:
        t: Time vector
        sol: State trajectory over time
    """
    t = np.arange(0, T, dt)
    sol = odeint(closed_loop_dynamics, state0, t, args=(params, Kp, Kd))
    return t, sol
