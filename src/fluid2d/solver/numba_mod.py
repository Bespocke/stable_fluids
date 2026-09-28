from __future__ import annotations

from numba import njit, prange
import numpy as np


@njit(cache=True, parallel=True)
def advect_numba(source: np.ndarray, destination: np.ndarray, vx: np.ndarray, vy: np.ndarray, dx: float, dt: float):

    N = source.shape[0]
    scale = dt/dx

    for i in prange(N):
        for j in range(N):

            x = j - vx[i,j]*scale
            y = i - vy[i,j]*scale

            if x>=0 and x<N-1 and y>=0 and y<N-1:

                x0=int(x)
                y0=int(y)

                a=x-x0
                b=y-y0

                destination[i,j]=(
                    source[y0,x0]*(1-a)*(1-b)
                    +source[y0,x0+1]*a*(1-b)
                    +source[y0+1,x0]*(1-a)*b
                    +source[y0+1,x0+1]*a*b
                )

            else:
                destination[i,j]=0


@njit(cache=True)
def open_boundary_numba(field: np.ndarray):

    N = field.shape[0]

    for i in range(N):
        field[0,i] = field[1,i]
        field[N-1,i] = field[N-2,i]

        field[i,0] = field[i,1]
        field[i,N-1] = field[i,N-2]


@njit(cache=True, parallel=True)
def jacobi_numba(table: np.ndarray, table_tmp: np.ndarray, b: np.ndarray, alpha: float, beta: float):

    N = table.shape[0]
    invbeta = 1.0/beta

    for i in prange(1,N-1):
        for j in range(1,N-1):

            table_tmp[i,j] = (
                table[i,j+1]
                + table[i,j-1]
                + table[i+1,j]
                + table[i-1,j]
                + alpha*b[i,j]
            ) * invbeta

    open_boundary_numba(table_tmp)


@njit(cache=True)
def poisson_solver_numba(pressure: np.ndarray, pressure_tmp: np.ndarray, b: np.ndarray, iterations: int, dx: float):
    pressure.fill(0)
    pressure_tmp.fill(0)

    alpha = -dx**2
    beta = 4
    
    for _ in range(iterations):
        jacobi_numba(pressure, pressure_tmp, b, alpha, beta)
        pressure, pressure_tmp = pressure_tmp, pressure

    return pressure, pressure_tmp


@njit(cache=True)
def diffuse_numba(field: np.ndarray, field_tmp: np.ndarray, rhs: np.ndarray, diffusion_rate: float, dt: float, dx: float, iterations: int):
    if diffusion_rate <= 0:
        return field, field_tmp

    alpha = dx**2 / (diffusion_rate * dt)
    beta = 4 + alpha

    rhs[:] = field

    for _ in range(iterations):
        jacobi_numba(field, field_tmp, rhs, alpha, beta)

        field, field_tmp = field_tmp, field

    return field, field_tmp


# from boundaries.open_bounds.open_boundary_operators import div, grad
from ..boundaries.operators import divergence, gradx, grady

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fluid_numba_buff import Fluid

def project_numba(fluid: "Fluid", iterations: int):
    
    divergence(fluid.vx, fluid.vy, fluid.div_tmp, fluid.dx)
    
    fluid.pressure, fluid.pressure_tmp = poisson_solver_numba(fluid.pressure, fluid.pressure_tmp, fluid.div_tmp, iterations, fluid.dx)

    gradx(fluid.pressure, fluid.gradx_tmp, fluid.dx)
    grady(fluid.pressure, fluid.grady_tmp, fluid.dx)

    fluid.vx -= fluid.gradx_tmp
    fluid.vy -= fluid.grady_tmp