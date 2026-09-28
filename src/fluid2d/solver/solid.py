from __future__ import annotations

from numba import njit, prange
import numpy as np

from .numba_mod import open_boundary_numba

@njit(cache=True)
def get_speed(v: np.ndarray, solid_mask: np.ndarray, i_wanted: int, j_wanted: int, i_asking: int, j_asking: int):
    if solid_mask[i_wanted,j_wanted]:
        return -v[i_asking,j_asking]
    return v[i_wanted,j_wanted]


@njit(cache=True, parallel=True)
def advect_speed_numba_solid(source: np.ndarray, destination: np.ndarray, vx: np.ndarray, vy: np.ndarray, dx: float, dt: float, solid_mask: np.ndarray):

    N = source.shape[0]
    scale = dt/dx

    for i in prange(N):
        for j in range(N):
            if solid_mask[i,j]:
                destination[i,j]=source[i,j]
                continue  

            x = j - vx[i,j]*scale
            y = i - vy[i,j]*scale

            if x>=0 and x<N-1 and y>=0 and y<N-1:

                x0=int(x)
                y0=int(y)

                a=x-x0
                b=y-y0

                destination[i,j]=(
                    get_speed(source, solid_mask, y0, x0, i, j)*(1-a)*(1-b)
                    +get_speed(source, solid_mask, y0, x0+1, i, j)*a*(1-b)
                    +get_speed(source, solid_mask, y0+1, x0, i, j)*(1-a)*b
                    +get_speed(source, solid_mask, y0+1, x0+1, i, j)*a*b
                )

            else:
                destination[i,j]=0


@njit(cache=True, parallel=True)
def jacobi_smago_speed_solid(table: np.ndarray, table_tmp: np.ndarray, b: np.ndarray, alpha: np.ndarray, nu: np.ndarray, solid_mask: np.ndarray):

    N = table.shape[0]

    for i in prange(1, N-1):
        for j in range(1, N-1):
            if solid_mask[i,j]:
                table_tmp[i,j]=0
                continue    

            nuE = 0.5*(nu[i,j] + nu[i,j+1])
            nuW = 0.5*(nu[i,j] + nu[i,j-1])
            nuN = 0.5*(nu[i,j] + nu[i-1,j])
            nuS = 0.5*(nu[i,j] + nu[i+1,j])

            a = alpha[i,j]

            inv = 1.0/(a + nuE + nuW + nuN + nuS)

            table_tmp[i,j] = (
                a*b[i,j]
                + nuE*get_speed(table, solid_mask, i, j+1, i, j)
                + nuW*get_speed(table, solid_mask, i, j-1, i, j)
                + nuN*get_speed(table, solid_mask, i-1, j, i, j)
                + nuS*get_speed(table, solid_mask, i+1, j, i, j)
            ) * inv

    open_boundary_numba(table_tmp)

from ..physics.smagorinsky import nueff, compute_alpha

@njit(cache=True)
def diffuse_smago_speed_solid(field: np.ndarray, field_tmp: np.ndarray, rhs: np.ndarray, diffusion_rate: float, nu: np.ndarray, normS: np.ndarray, alpha: np.ndarray, dt: float, dx: float, iterations: int, solid_mask: np.ndarray):
    if diffusion_rate <= 0:
        return field, field_tmp

    nueff(nu, normS, diffusion_rate, dx, Cs=0.17)

    compute_alpha(nu, alpha, dx, dt)

    rhs[:] = field

    for _ in range(iterations):
        jacobi_smago_speed_solid(field, field_tmp, rhs, alpha, nu, solid_mask)

        field, field_tmp = field_tmp, field

    return field, field_tmp


# @njit(cache=True, parallel=True)
# def jacobi_smago_poisson_solid(pressure: np.ndarray, pressure_tmp: np.ndarray, b: np.ndarray, dx: float, solid_mask: np.ndarray):

#     N = pressure.shape[0]
#     dx2 = dx**2

#     for i in prange(1, N-1):
#         for j in range(1, N-1):
#             if solid_mask[i, j]:
#                 pressure_tmp[i, j] = pressure[i, j]
#                 continue

#             east  = pressure[i, j] if solid_mask[i, j+1] else pressure[i, j+1]
#             west  = pressure[i, j] if solid_mask[i, j-1] else pressure[i, j-1]
#             north = pressure[i, j] if solid_mask[i-1, j] else pressure[i-1, j]
#             south = pressure[i, j] if solid_mask[i+1, j] else pressure[i+1, j]

#             pressure_tmp[i,j] = (
#                 east + west + north + south
#                 - dx2 * b[i,j]
#             ) * 0.25

#     open_boundary_numba(pressure_tmp)

@njit(cache=True, parallel=True)
def jacobi_smago_poisson_solid(pressure: np.ndarray, pressure_tmp: np.ndarray, b: np.ndarray, dx: float, solid_mask: np.ndarray):
    N = pressure.shape[0]
    dx2 = dx * dx

    for i in prange(1, N - 1):
        for j in range(1, N - 1):

            if solid_mask[i, j]:
                pressure_tmp[i, j] = pressure[i, j]
                continue

            p = 0.0
            n = 0

            if not solid_mask[i, j + 1]:
                p += pressure[i, j + 1]
                n += 1

            if not solid_mask[i, j - 1]:
                p += pressure[i, j - 1]
                n += 1

            if not solid_mask[i - 1, j]:
                p += pressure[i - 1, j]
                n += 1

            if not solid_mask[i + 1, j]:
                p += pressure[i + 1, j]
                n += 1

            if n > 0:
                pressure_tmp[i, j] = (
                    p - dx2 * b[i, j]
                ) / n
            else:
                pressure_tmp[i, j] = pressure[i, j]

    open_boundary_numba(pressure_tmp)


@njit(cache=True)
def poisson_solver_numba_solid(pressure: np.ndarray, pressure_tmp: np.ndarray, b: np.ndarray, iterations: int, dx: float, solid_mask: np.ndarray):
    # pressure*=0.5
    # pressure_tmp[:] = pressure

    pressure.fill(0)
    pressure_tmp.fill(0)

    for _ in range(iterations):
        jacobi_smago_poisson_solid(pressure, pressure_tmp, b, dx, solid_mask)
        pressure, pressure_tmp = pressure_tmp, pressure
        # print("pressure")
        # print(pressure)
        # print("pressure_tmp")
        # print(pressure_tmp)

    return pressure, pressure_tmp

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fluid_numba_buff import Fluid


@njit(cache=True, parallel=True)
def annex(vx: np.ndarray, vy: np.ndarray, gradx_tmp: np.ndarray, grady_tmp: np.ndarray, solid_mask: np.ndarray, N: int):
    for i in prange(1, N - 1):
        for j in range(1, N - 1):

            if solid_mask[i,j]:
                vx[i,j] = 0.0
                vy[i,j] = 0.0
                continue

            vx[i,j] -= gradx_tmp[i,j]
            vy[i,j] -= grady_tmp[i,j]


from ..boundaries.operators import divergence_solid, gradx_solid, grady_solid

def project_numba_solid(fluid: Fluid, iterations: int):
    
    divergence_solid(fluid.vx, fluid.vy, fluid.div_tmp, fluid.dx, fluid.solid_mask)
    
    fluid.pressure, fluid.pressure_tmp = poisson_solver_numba_solid(fluid.pressure, fluid.pressure_tmp, fluid.div_tmp, iterations, fluid.dx, fluid.solid_mask)

    gradx_solid(fluid.pressure, fluid.gradx_tmp, fluid.solid_mask, fluid.dx)
    grady_solid(fluid.pressure, fluid.grady_tmp, fluid.solid_mask, fluid.dx)

    annex(fluid.vx, fluid.vy, fluid.gradx_tmp, fluid.grady_tmp, fluid.solid_mask, fluid.N)

    # fluid.vx -= fluid.gradx_tmp
    # fluid.vy -= fluid.grady_tmp

    # null_fill(fluid.vx, fluid.vy, fluid.solid_mask)


# pour les autres grandeures que la vitesse

@njit(cache=True, parallel=True)
def advect_numba_solid(source: np.ndarray, destination: np.ndarray, vx: np.ndarray, vy: np.ndarray, dx: float, dt: float, solid_mask: np.ndarray):

    N = source.shape[0]
    scale = dt/dx

    for i in prange(N):
        for j in range(N):
            if solid_mask[i,j]:
                destination[i,j]=source[i,j]
                continue 

            x = j - vx[i,j]*scale
            y = i - vy[i,j]*scale

            if x>=0 and x<N-1 and y>=0 and y<N-1:

                x0=int(x)
                y0=int(y)

                a=x-x0
                b=y-y0

                value = 0.0
                weight = 0.0

                w = (1-a)*(1-b)
                if not solid_mask[y0,x0]:
                    value += w*source[y0,x0]
                else:
                    value += w * source[i, j]
                weight += w

                w = a*(1-b)
                if not solid_mask[y0,x0+1]:
                    value += w*source[y0,x0+1]
                else:
                    value += w * source[i, j]
                weight += w

                w = (1-a)*b
                if not solid_mask[y0+1,x0]:
                    value += w*source[y0+1,x0]
                else:
                    value += w * source[i, j]
                weight += w

                w = a*b
                if not solid_mask[y0+1,x0+1]:
                    value += w*source[y0+1,x0+1]
                else:
                    value += w * source[i, j]
                weight += w

                if weight > 0:
                    destination[i,j] = value
                else:
                    destination[i,j] = source[i,j]


@njit(cache=True, parallel=True)
def jacobi_smago_solid(table: np.ndarray, table_tmp: np.ndarray, b: np.ndarray, alpha: np.ndarray, nu: np.ndarray, solid_mask: np.ndarray):

    N = table.shape[0]

    for i in prange(1, N-1):
        for j in range(1, N-1):
            if solid_mask[i,j]:
                table_tmp[i,j]=table[i,j]
                continue

            a = alpha[i,j]

            if solid_mask[i,j+1]:
                east = table[i,j]
                nuE = nu[i,j]
            else:
                east = table[i,j+1]
                nuE = 0.5*(nu[i,j] + nu[i,j+1])

            if solid_mask[i,j-1]:
                west = table[i,j]
                nuW = nu[i,j]
            else:
                west = table[i,j-1]
                nuW = 0.5*(nu[i,j] + nu[i,j-1])

            if solid_mask[i-1,j]:
                north = table[i,j]
                nuN = nu[i,j]
            else:
                north = table[i-1,j]
                nuN = 0.5*(nu[i,j] + nu[i-1,j])

            if solid_mask[i+1,j]:
                south = table[i,j]
                nuS = nu[i,j]
            else:
                south = table[i+1,j]
                nuS = 0.5*(nu[i,j] + nu[i+1,j])

            inv = 1.0/(a + nuE + nuW + nuN + nuS)

            table_tmp[i,j] = (
                a*b[i,j]
                + nuE*east
                + nuW*west
                + nuN*north
                + nuS*south
            ) * inv

    open_boundary_numba(table_tmp)

@njit(cache=True)
def diffuse_smago_solid(field: np.ndarray, field_tmp: np.ndarray, rhs: np.ndarray, diffusion_rate: float, nu: np.ndarray, normS: np.ndarray, alpha: np.ndarray, dt: float, dx: float, iterations: int, solid_mask: np.ndarray):
    if diffusion_rate <= 0:
        return field, field_tmp

    nueff(nu, normS, diffusion_rate, dx, Cs=0.8)

    compute_alpha(nu, alpha, dx, dt)

    rhs[:] = field

    for _ in range(iterations):
        jacobi_smago_solid(field, field_tmp, rhs, alpha, nu, solid_mask)

        field, field_tmp = field_tmp, field

    return field, field_tmp


