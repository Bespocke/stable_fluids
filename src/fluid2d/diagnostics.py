from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def mass_start(density: np.ndarray, solid_mask: np.ndarray, dx: float) -> float:
    """Return the initial scalar mass-like integral over fluid cells."""
    n = density.shape[0]
    cell_area = dx * dx
    total = 0.0
    for i in range(n):
        for j in range(n):
            if solid_mask[i, j]:
                continue
            total += density[i, j] * cell_area
    return total


@njit(cache=True)
def mass_total(density: np.ndarray, solid_mask: np.ndarray, dx: float) -> float:
    """Return the scalar integral over fluid cells."""
    return mass_start(density, solid_mask, dx)


def relative_error(value: float, reference: float) -> float:
    """Relative error as a dimensionless float; returns 0 for reference=0 and value=0."""
    if reference == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return (value - reference) / reference
