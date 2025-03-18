from dataclasses import dataclass

@dataclass
class PendulumParams:
    """
    Dataclass containing physical and control parameters for the inverted pendulum simulation.

    Attributes:
    - M (float): Mass of the cart (kg).
    - m (float): Mass of the pendulum (kg).
    - l (float): Distance to pendulum center of mass (m).
    - g (float): Gravitational acceleration (m/s²).
    - Kp (float): Proportional gain for PD controller.
    - Kd (float): Derivative gain for PD controller.
    """
    M: float = 1.0
    m: float = 0.1
    l: float = 0.5
    g: float = 9.81
    Kp: float = 100.0
    Kd: float = 20.0
