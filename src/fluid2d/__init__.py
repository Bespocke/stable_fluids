"""Small 2D incompressible fluid solver based on Stable Fluids and Numba."""

from .state import Fluid
from .solver.simulation import step

__all__ = ["Fluid", "step"]
