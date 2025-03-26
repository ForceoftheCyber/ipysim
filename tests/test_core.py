import pytest
import numpy as np
from ipysim.core import (
    force,
    torque,
    field,
    maglev_measurements,
    maglev_state_dynamics,
    simulate_maglev,
)

default_params = {
    "M": 0.075,
    "m": 9.375,
    "l": 0.046,
    "g": 9.81,
    "m_support": 0.6250,
    "k": 0.0377,
    "J": 1.2e-5,
    "mu0": 4 * np.pi * 1e-7,
}


def test_force_output_shape_and_type():
    m1 = np.array([0.0, 1.0])
    m2 = np.array([1.0, 0.0])
    r = np.array([0.01, 0.01])
    f = force(m1, m2, r, default_params["mu0"])
    assert f.shape == (2,)
    assert np.all(np.isfinite(f))


def test_torque_output_type():
    m1 = np.array([0.0, 1.0])
    m2 = np.array([1.0, 0.0])
    r = np.array([0.01, 0.01])
    t = torque(m1, m2, r, default_params["mu0"])
    assert isinstance(t, float)
    assert np.isfinite(t)


def test_field_vector_output():
    s = [0.01, 0.01, 0.0]
    B = field(s, default_params["m"], default_params["mu0"])
    assert isinstance(B, np.ndarray)
    assert B.shape == (2,)
    assert np.all(np.isfinite(B))


def test_maglev_measurements_output():
    s = [0.01, 0.01, 0.0, 0.0, 0.0, 0.0]
    y, y_dot = maglev_measurements(s, default_params["m"], default_params["mu0"])
    assert isinstance(y, float)
    assert isinstance(y_dot, float)
    assert np.isfinite(y)
    assert np.isfinite(y_dot)


def test_maglev_state_dynamics_structure():
    s = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
    out = maglev_state_dynamics(s, 0.0, 0.0, default_params)
    assert isinstance(out, list)
    assert len(out) == 6
    assert all(isinstance(x, float) for x in out)


def test_simulation_shape_and_stability():
    state0 = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
    t, sol = simulate_maglev(
        Kp=600,
        Kd=30,
        T=0.1,
        dt=0.001,
        state0=state0,
        params=default_params
    )
    assert isinstance(t, np.ndarray)
    assert isinstance(sol, np.ndarray)
    assert sol.ndim == 2 and sol.shape[1] == 6
    assert t.shape[0] == sol.shape[0]
    assert np.all(np.isfinite(sol))


def test_simulation_response_to_zero_gains():
    state0 = [0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
    t, sol = simulate_maglev(
        Kp=0,
        Kd=0,
        T=0.1,
        dt=0.001,
        state0=state0,
        params=default_params
    )
    assert np.all(np.isfinite(sol))
    assert not np.allclose(sol[:, 1], 0.1)  # some dynamics should happen
