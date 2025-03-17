"""
core.py

Defines the Simulator class, which provides a generic, interactive interface
for running user-defined simulations with dynamic parameters controlled by
widgets such as sliders. Abstracts away interaction logic from simulation logic.

Classes:
    Simulator(simulation_fn: callable)
        Manages interactive execution of simulations, automatically updating
        output in response to changes in widget parameters.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output

class Simulator:
    def __init__(self, simulation_fn):
        self.simulation_fn = simulation_fn
        self.widgets = []
        self.output = widgets.Output()

    def add_widget(self, widget):
        self.widgets.append(widget)

    def _on_change(self, change):
        params = {widget.description: widget.value for widget in self.widgets}
        with self.output:
            clear_output(wait=True)
            result = self.simulation_fn(**params)
            display(result)

    def display(self):
        for widget in self.widgets:
            widget.observe(self._on_change, names='value')

        ui = widgets.VBox(self.widgets + [self.output])
        self._on_change(None)  # initial run
        display(ui)
