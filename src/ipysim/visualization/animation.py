import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def animate_system(t, sol, draw_func, xlim=None, ylim=None, interval=30):
    """
    Generic animation function for any simulation.

    Parameters:
    - t: Time array
    - sol: State solution array (each row is a state at a timestep)
    - draw_func: Function to draw the system, takes (ax, state) as input
    - xlim: Tuple (xmin, xmax) for the animation plot
    - ylim: Tuple (ymin, ymax) for the animation plot
    - interval: Delay between frames in milliseconds

    Returns:
    - HTML animation object.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    # Store objects to update during animation
    draw_objects = draw_func(ax, sol[0])  # Initialize with the first state
    
    def init():
        return draw_objects  # Initial frame setup
    
    def update(frame):
        state = sol[frame]  # Get the state at this time step
        return draw_func(ax, state, update=True, objects=draw_objects)
    
    ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init,
                                  blit=True, interval=interval)
    
    plt.close()
    return HTML(ani.to_jshtml())
