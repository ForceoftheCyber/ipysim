# ipysim/plotting.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_time_series(t, sol, indices=None, labels=None, title="State Evolution"):
    """
    Generic time-series plot.
    
    t: time array
    sol: state trajectory (2D array)
    indices: which state indices to plot (default: [0, 1])
    labels: list of labels corresponding to the indices
    """
    if indices is None:
        indices = [0, 1]
    if labels is None:
        labels = [f"State {i}" for i in indices]
    
    plt.title(title)
    plt.xlabel("Time [s]")
    plt.ylabel("States")
    for idx, label in zip(indices, labels):
        plt.plot(t, sol[:, idx], label=label)
    plt.legend()
    plt.grid(True)

def create_animation(t, sol, draw_frame_fn, interval=50):
    """
    Create a generic animation using matplotlib's FuncAnimation.
    
    draw_frame_fn: a function with signature draw_frame_fn(ax, sol, i)
                   that draws a single frame (given axis, simulation data, frame index).
    """
    fig, ax = plt.subplots()
    
    def init():
        ax.clear()
        draw_frame_fn(ax, sol, 0)
        return []
    
    def update(i):
        ax.clear()
        draw_frame_fn(ax, sol, i)
        return []
    
    ani = FuncAnimation(fig, update, frames=len(t), init_func=init, interval=interval)
    plt.close(fig)
    return HTML(ani.to_jshtml())
