import numpy as np

def pendulum_dynamics(state, t, params):
    """
    Computes the state derivatives for an inverted pendulum on a cart with PD control.

    Parameters:
    - state (list): [x, x_dot, theta, theta_dot] representing the cart position, velocity, pendulum angle, and angular velocity.
    - t (float): Time step (required for odeint but not used directly).
    - params (PendulumParams): Object containing physical and control parameters.

    Returns:
    - np.array: The time derivative of the state, [x_dot, x_ddot, theta_dot, theta_ddot].
    """
    x, x_dot, theta, theta_dot = state
    Kp, Kd = params.Kp, params.Kd
    M, m, l, g = params.M, params.m, params.l, params.g

    # Compute control force using PD controller
    F = Kp * theta + Kd * theta_dot

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    denom = M + m * sin_theta**2

    # Compute accelerations
    x_ddot = (F + m * sin_theta * (l * theta_dot**2 + g * cos_theta)) / denom
    theta_ddot = (-F * cos_theta - m * l * theta_dot**2 * cos_theta * sin_theta - (M + m) * g * sin_theta) / (l * denom)

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])
