from numba import njit, prange
import numpy as np

@njit(cache=True, parallel=True)
def add_forces(forces: np.ndarray, vx: np.ndarray, vy: np.ndarray, dt: float, solid_mask: np.ndarray, table: np.ndarray):
    N = vx.shape[0]
    for i in prange(1, N-1):
        for j in range(1, N-1):
            if solid_mask[i,j] or solid_mask[i+1,j] or solid_mask[i-1,j] or solid_mask[i,j+1] or solid_mask[i,j-1] or table[i,j] == 0.0 :
                continue
            else:
                vx[i,j] += forces[0]*dt
                vy[i,j] += forces[1]*dt

# @njit(cache=True, parallel=True)
# def add_forces(forces: np.ndarray, vx: np.ndarray, vy: np.ndarray, dt: float, solid_mask: np.ndarray):
#     N = vx.shape[0]
#     for i in prange(N):
#         for j in range(N):
#             if not solid_mask[i,j]:
#                 vx[i,j] += forces[0]*dt
#                 vy[i,j] += forces[1]*dt