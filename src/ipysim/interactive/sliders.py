import ipywidgets as widgets

def create_slider(name, min, max, step, default):
    """
    Creates an interactive slider for parameter selection.

    Parameters:
    - name (str): Label for the slider.
    - min (float): Minimum slider value.
    - max (float): Maximum slider value.
    - step (float): Step size for the slider.
    - default (float): Default value of the slider.

    Returns:
    - widgets.FloatSlider: The generated interactive slider.
    """
    return widgets.FloatSlider(
        value=default, min=min, max=max, step=step, description=name
    )

def generate_sliders(slider_specs):
    """
    Generates multiple sliders based on parameter specifications.

    Parameters:
    - slider_specs (list of dicts): List of parameter settings (name, min, max, step, default).

    Returns:
    - dict: A dictionary containing sliders mapped to their names.
    """
    return {spec["name"]: create_slider(**spec) for spec in slider_specs}
