# ipysim/simulations/simulate_flip_magnet.py

import numpy as np
import matplotlib.pyplot as plt

def compute_force_z(z, params):
    """
    Compute net force in the z direction for a given height `z` and system parameters.
    """
    M = params["M"]
    m = params["m"]
    l = params["l"]
    g = params["g"]
    m_support = params["m_support"]
    mu0 = params["mu0"]

    r1 = np.array([l / 2, 0])
    r2 = np.array([-l / 2, 0])
    r = np.array([0, z])
    m_vec = m * np.array([0.0, 1.0])
    m1 = np.array([0.0, m_support])
    m2 = np.array([0.0, m_support])

    def magnetic_force(m_i, m, r_vec):
        r_norm = np.linalg.norm(r_vec)
        if r_norm == 0:
            return np.zeros_like(r_vec)
        term1 = np.dot(m_i, r_vec) * m
        term2 = np.dot(m, r_vec) * m_i
        term3 = np.dot(m_i, m) * r_vec
        term4 = 5 * np.dot(m_i, r_vec) * np.dot(m, r_vec) / r_norm**2 * r_vec
        return (3 * mu0 / (4 * np.pi * r_norm**5)) * (term1 + term2 + term3 - term4)

    F1 = magnetic_force(m1, m_vec, r - r1)
    F2 = magnetic_force(m2, m_vec, r - r2)
    F_total = F1 + F2 - M * np.array([0.0, g])
    return F_total[1]

def flip_magnet_simulation(params, state0=None, T=1.0, dt=0.01, **controls):
    """
    Simulate the vertical magnetic force acting on a levitating magnet.
    
    Args:
        params: dictionary of physical parameters
        state0: unused
        T: unused
        dt: unused
        controls: includes M, l, and flip
    
    Returns:
        z: array of height values
        fz: array of corresponding force values
    """
    M = controls.get("M", 0.075)
    l = controls.get("l", 0.046)
    flip = controls.get("flip", False)

    params = dict(params)
    params["M"] = M
    params["l"] = l
    params["m"] = 9.375 if not flip else -9.375

    z = np.linspace(0.001, 0.1, 200)
    fz = np.array([compute_force_z(zi, params) for zi in z])
    return z, fz

def plot_flip_magnet(z, fz):
    """
    Plot the force vs. height curve and highlight equilibrium points.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(z, fz, label='Force')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    first_dot = True
    for i in range(len(z) - 1):
        if (fz[i] < 0 and fz[i+1] >= 0) or (fz[i] > 0 and fz[i+1] <= 0):
            z_zero = z[i] - fz[i] * (z[i+1] - z[i]) / (fz[i+1] - fz[i])
            label = 'Equilibrium (f=0)' if first_dot else None
            plt.plot(z_zero, 0, 'ro', label=label)
            first_dot = False

    plt.xlabel("z (m)")
    plt.ylabel("Force in z-direction (N)")
    plt.title("Force on Magnet vs. Height")
    plt.grid(True)
    plt.legend()
    plt.show()
