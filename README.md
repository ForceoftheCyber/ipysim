# ipysim

**Interactive Simulations in Jupyter Notebooks**

---

## Overview

**ipysim** is a modular Python package for building, visualizing, and interacting with dynamical system simulations directly inside Jupyter notebooks.  

---

## Package Structure

| File/Module                      | Purpose |
|----------------------------------|---------|
| `core.py`                        | Defines the general-purpose ODE solver and simulation engine. |
| `params.py`                      | Provides default parameters and initial states for simulations. |
| `plotting.py`                    | Contains utility functions for plotting simulation results. |
| `simulation_ui.py`               | Provides an interactive UI for running simulations with live controls. |
| `utils.py`                       | Miscellaneous helper functions (e.g., for array handling). |
| `simulations/simulate_maglev.py` | Simulation code for a magnetic levitation (maglev) system. |
| `simulations/simulate_flip_magnet.py` | Simulation code for a magnet flip system (alternative physics model). |
| `simulations/simulate_maglev_with_noise.py` | Maglev simulation with sensor noise modeling. |

---

## Installation

```bash
pip install ipysim
```
**Requirements**:
- `numpy`
- `matplotlib`
- `ipywidgets`
- `scipy`

---

## Quickstart Example
*See also [examples/maglev_dynamical_system_simulation.ipynb](examples/maglev_dynamical_system_simulation.ipynb) for example usage.*

Here is a minimal working example to get started with **ipysim**:

```python
from ipysim.simulation_ui import interactive_simulation
from ipysim.simulations.simulate_maglev import simulate, plot_maglev, create_maglev_animation
from ipysim.params import default_params, default_state0

# Optional: Create animation function
detailed_animation_fn = lambda t, sol: create_maglev_animation(t, sol, default_state0())

# Optional: Define evaluation check function
def evaluation_check(sol, t):
    # Insert custom logic to evaluate simulation quality
    pass

# Slider configuration for tunable parameters
sliders_config = {
    "Kp": {"default": 300.0, "min": 0.0, "max": 1000.0, "step": 10.0, "description": "Kp"},
    "Kd": {"default": 30.0, "min": 0.0, "max": 200.0, "step": 5.0, "description": "Kd"},
}

# Launch the interactive simulation
interactive_simulation(
    simulate_fn=simulate,
    plot_fn=plot_maglev,
    animation_fn=detailed_animation_fn,
    params=default_params(),
    state0=default_state0(),
    T=5.0,
    evaluation_function=evaluation_check,
    sliders_config=sliders_config
)
```

This example sets up:
- A simulation running for 5 seconds (`T=5.0`)
- Live plots and optional animations
- Slider controls for `Kp` and `Kd` parameters in the PD controller
- Evaluation function which can be defined and must return a boolean

---

## Available Predefined Simulations

| Simulation | Description |
|------------|-------------|
| `simulate_maglev.py` | Simulates vertical magnetic levitation using a PD controller. |
| `simulate_flip_magnet.py` | Simulates the flipping dynamics of a suspended magnet. |
| `simulate_maglev_with_noise.py` | Adds sensor noise to the maglev simulation for robustness testing. |

Depending on the simulation, they will export:
- A `simulate()` function (the core dynamics)
- A `plot_*()` function
- An optional `create_*_animation()` function
---
## Function Hierarchy
![image](/docs/ipysim_functional_hierarchy.png)

## Sequence Diagram of Maglev Example Instance
![image](/docs/ipysim_sequence_diagram.png)

---

## License

MIT License

---
