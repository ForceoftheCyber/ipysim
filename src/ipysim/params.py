# ipysim/params.py

import numpy as np

def default_params():
    return {
        "M": 0.075,          # mass of levitating magnet (kg)
        "m": 9.375,          # magnetic moment magnitude (A·m²)
        "l": 0.046,          # magnet spacing (m)
        "g": 9.81,           # gravity (m/s²)
        "m_support": 0.6250, # baseline support dipole moment
        "k": 0.0377,         # current-to-moment scaling factor
        "J": 0.12e-4,        # moment of inertia (kg·m²)
        "mu0": 4 * np.pi * 1e-7  # vacuum permeability (H/m)
    }

def default_state0():
    return [
        0.004,           # x position
        0.0243 + 0.02,   # z position (elevated)
        -0.2,            # initial angle
        0.0, 0.0, 0.0    # initial velocities
    ]
