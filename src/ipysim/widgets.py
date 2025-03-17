"""
widgets.py

Provides convenient factory functions to quickly generate common interactive
widgets (e.g., sliders, dropdown menus) for use with simulations. Simplifies
widget creation by abstracting common widget setup patterns.

Functions:
    Slider(name: str, min: float, max: float, step: float, default: float)
        Creates a floating-point slider widget with the specified range and step.

    IntSlider(name: str, min: int, max: int, step: int, default: int)
        Creates an integer slider widget with the specified range and step.

    Dropdown(name: str, options: list, default: Any)
        Creates a dropdown widget with predefined selectable options.
"""

import ipywidgets as widgets

def Slider(name, min, max, step, default):
    return widgets.FloatSlider(
        description=name,
        min=min,
        max=max,
        step=step,
        value=default,
        continuous_update=False,
        style={'description_width': 'initial'}
    )

def IntSlider(name, min, max, step, default):
    return widgets.IntSlider(
        description=name,
        min=min,
        max=max,
        step=step,
        value=default,
        continuous_update=False,
        style={'description_width': 'initial'}
    )

def Dropdown(name, options, default):
    return widgets.Dropdown(
        description=name,
        options=options,
        value=default,
        style={'description_width': 'initial'}
    )
