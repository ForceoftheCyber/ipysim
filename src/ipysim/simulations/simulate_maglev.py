# ipysim/simulations/simulate_maglev.py

import numpy as np
from scipy.integrate import odeint
from ipysim.core import simulate_closed_loop
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Ellipse
from matplotlib import transforms
from IPython.display import HTML, Javascript

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
    return create_animation(t, sol, draw_frame_maglev, interval=16.67)
def create_maglev_animation(t, sol, state0):
    """
    Create a animation of the maglev system.
    
    This function uses Matplotlib's FuncAnimation to animate
    an elaborate visualization including the base, two solenoids with windings,
    and a magnet represented as a cylinder with rounded edges.
    
    Args:
        t (array-like): The time vector.
        sol (ndarray): The simulation solution as a 2D array (columns: x, z, theta, ...).
        state0 (list or array): The initial state; used here to set the vertical axis.
        
    Returns:
        IPython.display.HTML: An HTML object containing the animation.
    """
    try:
        # Downsample simulation data to avoid browser overload.
        max_frames = 1000
        if len(t) > max_frames:
            frame_step = len(t) // max_frames
            t_anim = t[::frame_step]
            sol_anim = sol[::frame_step]
        else:
            t_anim = t
            sol_anim = sol

        # Extract state variables.
        x = sol_anim[:, 0]
        z = sol_anim[:, 1]
        theta = sol_anim[:, 2]

        # Create figure and axis.
        fig = plt.figure(figsize=(8, 6), dpi=80)
        ax_anim = fig.add_subplot(111)

        # Set up the base and solenoid parameters.
        base_height = 0.01
        solenoid_height = 0.012
        y_offset = base_height + 0.001 + solenoid_height

        ax_anim.set_xlim(-0.06, 0.06)
        initial_z = state0[1]
        margin_pct = 0.4
        ax_anim.set_ylim(-y_offset, initial_z * (1 + margin_pct))
        ax_anim.set_aspect('equal')
        ax_anim.set_title('Maglev Animation')
        ax_anim.grid(False)

        # Draw the static base.
        base = Rectangle((-0.06, -y_offset), 0.12, base_height, fc='#3a3a3a')
        ax_anim.add_patch(base)

        # Draw two solenoids.
        solenoid_positions = [-0.01, 0.01]
        solenoid_width = 0.009
        copper_color = '#b87333'
        dark_copper = '#8c5e28'

        for pos in solenoid_positions:
            # Draw solenoid base (bottom cap).
            solenoid_base = Rectangle(
                (pos - solenoid_width / 2 - 0.001, -y_offset + base_height), 
                solenoid_width + 0.002, 0.001, 
                fc=dark_copper, ec='black', lw=0.5
            )
            ax_anim.add_patch(solenoid_base)
            
            # Draw the main solenoid body.
            solenoid_body = Rectangle(
                (pos - solenoid_width / 2, -y_offset + base_height + 0.001), 
                solenoid_width, solenoid_height, 
                fc=copper_color, ec='black', lw=0.5
            )
            ax_anim.add_patch(solenoid_body)
            
            # Draw the solenoid top cap.
            solenoid_top = Rectangle(
                (pos - solenoid_width / 2 - 0.001, -0.001), 
                solenoid_width + 0.002, 0.001, 
                fc=dark_copper, ec='black', lw=0.5
            )
            ax_anim.add_patch(solenoid_top)
            
            # Draw coil windings.
            num_windings = 6
            winding_spacing = solenoid_height / (num_windings + 1)
            for i in range(1, num_windings + 1):
                y_pos = -y_offset + base_height + 0.001 + i * winding_spacing
                ax_anim.plot(
                    [pos - solenoid_width/2, pos + solenoid_width/2],
                    [y_pos, y_pos],
                    lw=0.5, color='black', alpha=0.7
                )
                
            # Draw the core indicator.
            core = plt.Line2D(
                [pos, pos],
                [-y_offset + base_height + 0.001, -0.001],
                lw=1, color=dark_copper, alpha=0.8
            )
            ax_anim.add_line(core)

        # Draw the magnet as a cylinder with rounded edges.
        w_body = 0.032
        h_body = 0.006

        # Cylinder body.
        cylinder_body = Rectangle(
            (x[0] - w_body / 2, z[0] - h_body / 2),
            w_body, h_body, fc='#808080', ec='black'
        )
        ax_anim.add_patch(cylinder_body)

        # Top ellipse for the rounded top.
        top_width = w_body
        top_height = 0.005
        offset_x_top = - (h_body / 2) * np.sin(np.radians(theta[0]))
        offset_y_top = (h_body / 2) * np.cos(np.radians(theta[0]))
        cylinder_top = Ellipse(
            (x[0] + offset_x_top, z[0] + offset_y_top),
            top_width, top_height, fc='#707070', ec='black'
        )
        ax_anim.add_patch(cylinder_top)

        # Bottom ellipse for the rounded bottom.
        bottom_width = w_body
        bottom_height = 0.005
        offset_x_bottom = (h_body / 2) * np.sin(np.radians(theta[0]))
        offset_y_bottom = - (h_body / 2) * np.cos(np.radians(theta[0]))
        cylinder_bottom = Ellipse(
            (x[0] + offset_x_bottom, z[0] + offset_y_bottom),
            bottom_width, bottom_height, fc='#707070', ec='black'
        )
        ax_anim.add_patch(cylinder_bottom)

        # Timer text.
        timer_text = ax_anim.text(0.02, 0.95, '', transform=ax_anim.transAxes, fontsize=12, color='black')

        # Animation initialization function.
        def init():
            current_x, current_z = x[0], z[0]
            current_theta = theta[0]
            trans = transforms.Affine2D().rotate_around(current_x, current_z, current_theta) + ax_anim.transData
            cylinder_body.set_xy((current_x - w_body/2, current_z - h_body/2))
            cylinder_body.set_transform(trans)
            offset_x_top = - (h_body/2) * np.sin(current_theta)
            offset_y_top = (h_body/2) * np.cos(current_theta)
            cylinder_top.center = (current_x + offset_x_top, current_z + offset_y_top)
            cylinder_top.angle = np.degrees(current_theta)
            offset_x_bottom = (h_body/2) * np.sin(current_theta)
            offset_y_bottom = - (h_body/2) * np.cos(current_theta)
            cylinder_bottom.center = (current_x + offset_x_bottom, current_z + offset_y_bottom)
            cylinder_bottom.angle = np.degrees(current_theta)
            timer_text.set_text("Time: 0.00 s")
            return [cylinder_body, cylinder_top, cylinder_bottom, timer_text]

        # Animation update function.
        def update(i):
            current_x, current_z = x[i], z[i]
            current_theta = theta[i]
            trans = transforms.Affine2D().rotate_around(current_x, current_z, current_theta) + ax_anim.transData
            cylinder_body.set_xy((current_x - w_body/2, current_z - h_body/2))
            cylinder_body.set_transform(trans)
            offset_x_top = - (h_body/2) * np.sin(current_theta)
            offset_y_top = (h_body/2) * np.cos(current_theta)
            cylinder_top.center = (current_x + offset_x_top, current_z + offset_y_top)
            cylinder_top.angle = np.degrees(current_theta)
            offset_x_bottom = (h_body/2) * np.sin(current_theta)
            offset_y_bottom = - (h_body/2) * np.cos(current_theta)
            cylinder_bottom.center = (current_x + offset_x_bottom, current_z + offset_y_bottom)
            cylinder_bottom.angle = np.degrees(current_theta)
            timer_text.set_text("Time: {:.2f} s".format(t_anim[i]))
            return [cylinder_body, cylinder_top, cylinder_bottom, timer_text]

        # Create the animation.
        ani = animation.FuncAnimation(
            fig, update, frames=len(t_anim), init_func=init, blit=True, interval=50
        )
        plt.tight_layout()
        plt.close(fig)  # Prevent duplicate figures in notebooks.

        # Convert the animation to embeddable HTML.
        html_anim = ani.to_jshtml(default_mode='once')
        return HTML(html_anim)
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return HTML(
            "<div style='color:red; border:1px solid #ffaaaa; padding:10px; "
            "background-color:#ffeeee; border-radius:5px;'>"
            "<h3>Animation Error</h3>"
            "<p>Failed to render animation: {}</p>"
            "<details><summary>Error Details</summary><pre>{}</pre></details></div>"
            .format(str(e), error_details)
        )