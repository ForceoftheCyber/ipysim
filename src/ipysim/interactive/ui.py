import ipywidgets as widgets
from IPython.display import display

def create_button(label, callback):
    """
    Creates an interactive button.

    Parameters:
    - label (str): The text displayed on the button.
    - callback (function): Function executed when the button is clicked.

    Returns:
    - widgets.Button: The created button.
    """
    button = widgets.Button(description=label)
    button.on_click(callback)
    return button

def create_dropdown(options, default, label, callback):
    """
    Creates an interactive dropdown.

    Parameters:
    - options (list): List of selectable options.
    - default (str): Default selected value.
    - label (str): Label for the dropdown.
    - callback (function): Function executed when the value changes.

    Returns:
    - widgets.Dropdown: The created dropdown.
    """
    dropdown = widgets.Dropdown(
        options=options,
        value=default,
        description=label,
    )
    
    dropdown.observe(lambda change: callback(change['new']), names='value')
    return dropdown

def setup_ui(simulation_func, slider_dict):
    """
    Sets up the UI for running the simulation.

    Parameters:
    - simulation_func (function): Function to run the simulation.
    - slider_dict (dict): Dictionary containing sliders.

    Returns:
    - widgets.VBox: UI container.
    """
    run_button = create_button("Run Simulation", lambda _: simulation_func())
    reset_button = create_button("Reset", lambda _: [s.reset() for s in slider_dict.values()])

    ui = widgets.VBox([run_button, reset_button] + list(slider_dict.values()))
    display(ui)
