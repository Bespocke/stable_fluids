import numpy as np
from ..boundaries.operators import normalize, gradx_solid, grady_solid
from numba import njit, prange


@njit(cache=True, parallel=True)
def abs_field(table: np.ndarray, result: np.ndarray):
    N = table.shape[0]

    for i in prange(N):
        for j in range(N):
            result[i,j] = abs(table[i,j])

@njit(cache=True, parallel=True)
def vorticity_confinment_numba(vx: np.ndarray, vy: np.ndarray, curl: np.ndarray, curl_abs: np.ndarray, gradx_tmp: np.ndarray, grady_tmp: np.ndarray, norm_tmp: np.ndarray, Nx: np.ndarray, Ny: np.ndarray, dx: float, dt: float, solid_mask: np.ndarray, confinment_intensity: float):
    N = vx.shape[0]

    abs_field(curl, curl_abs)

    gradx_solid(curl_abs, gradx_tmp, solid_mask, dx)
    grady_solid(curl_abs, grady_tmp, solid_mask, dx)

    normalize(gradx_tmp, grady_tmp, norm_tmp, Nx, Ny)

    for i in prange(N):
        for j in range(N):
            if solid_mask[i, j]:
                continue

            force = confinment_intensity * curl[i,j] * dt

            vx[i,j] += force * Ny[i,j]
            vy[i,j] -= force * Nx[i,j]


# from boundaries.open_bounds.open_boundary_operators import normalize, grad

# def vorticity_confinment(fluid, confinment_intensity):

#     curl_abs = np.abs(fluid.curl)

#     Nx, Ny = grad(
#         curl_abs,
#         fluid.dx
#     )

#     Nx, Ny = normalize(
#         Nx,
#         Ny
#     )

#     fluid.vx += confinment_intensity * Ny * fluid.curl * fluid.dt
#     fluid.vy -= confinment_intensity * Nx * fluid.curl * fluid.dt