"""
simulation.py

Defines an abstract base class for simulations. Users can inherit from this
class to ensure a consistent simulation interface across various implementations.

Classes:
    Simulation(ABC)
        Abstract base class specifying a standardized interface for user-defined
        simulations. Subclasses must implement the 'run' method.
"""

from abc import ABC, abstractmethod

class Simulation(ABC):
    @abstractmethod
    def run(self, **params):
        pass
