import matplotlib.pyplot as plt

def plot_simulation(t, solution, labels):
    """
    Plots the simulation results over time and adds a phase plot (θ vs. θ̇).

    Parameters:
    - t (np.array): Time steps.
    - solution (np.array): Simulation output (state trajectories).
    - labels (list): List of state variable names.

    Returns:
    - None: Displays the plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time Series Plot
    ax1 = axes[0]
    for idx, label in enumerate(labels):
        ax1.plot(t, solution[:, idx], label=label)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("State Variables")
    ax1.legend()
    ax1.grid()
    ax1.set_title("Time Series")

    # Phase Plot (θ vs. θ̇)
    ax2 = axes[1]
    theta = solution[:, 2]  # Pendulum angle θ
    theta_dot = solution[:, 3]  # Angular velocity θ̇
    ax2.plot(theta, theta_dot, 'm-', lw=2)
    ax2.set_xlabel("Pendulum Angle θ [rad]")
    ax2.set_ylabel("Angular Velocity θ̇ [rad/s]")
    ax2.set_title("Phase Plot (θ vs. θ̇)")
    ax2.grid()

    plt.tight_layout()
    plt.show()
