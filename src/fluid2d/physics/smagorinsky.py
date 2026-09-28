from numba import njit, prange
import numpy as np

from ..boundaries.operators import gradx_solid, grady_solid, gradx, grady

@njit(cache=True)
def open_boundary_numba(field: np.ndarray):

    N = field.shape[0]

    for i in range(N):
        field[0,i] = field[1,i]
        field[N-1,i] = field[N-2,i]

        field[i,0] = field[i,1]
        field[i,N-1] = field[i,N-2]


@njit(cache=True) # inutile si opti
def strain(vx: np.ndarray, vy: np.ndarray, gradxy_tmp: np.ndarray, gradyx_tmp: np.ndarray, gradxx_tmp: np.ndarray, gradyy_tmp: np.ndarray, S: np.ndarray, d: float):

    N = vx.shape[0]

    gradx(vy, gradyx_tmp, d)
    grady(vx, gradxy_tmp, d)
    gradx(vx, gradxx_tmp, d)
    grady(vy, gradyy_tmp, d)

    for i in range(N):
        for j in range(N):
            S[i,j,0] = gradxx_tmp[i,j] # 0,0
            S[i,j,1] = gradyy_tmp[i,j] # 1,1
            S[i,j,2] = 0.5*(gradxy_tmp[i,j]+gradyx_tmp[i,j]) # 1,0 et 0,1 mais égaux

@njit(cache=True, parallel=True)
def norm_strain(vx: np.ndarray, vy: np.ndarray, gradxy_tmp: np.ndarray, gradyx_tmp: np.ndarray, gradxx_tmp: np.ndarray, gradyy_tmp: np.ndarray, normS: np.ndarray, d: float, solid_mask: np.ndarray):

    N = normS.shape[0]
    Sxy = 0.0

    gradx_solid(vy, gradyx_tmp, solid_mask, d)
    grady_solid(vx, gradxy_tmp, solid_mask, d)
    gradx_solid(vx, gradxx_tmp, solid_mask, d)
    grady_solid(vy, gradyy_tmp, solid_mask, d)

    for i in prange(N):
        for j in range(N):
            Sxy = 0.5*(gradxy_tmp[i,j]+gradyx_tmp[i,j]) # 1,0 et 0,1 mais égaux
            normS[i,j] = np.sqrt(
                2*(gradxx_tmp[i,j]**2 + gradyy_tmp[i,j]**2 + 2*(Sxy**2))
            )

@njit(cache=True, parallel=True)
def nueff(nu: np.ndarray, normS: np.ndarray, diffusion_rate: float, dx: float, Cs: float):

    N = normS.shape[0]
    Coeff = (Cs*dx)**2

    for i in prange(N):
        for j in range(N):
            nu[i,j] = Coeff*normS[i,j] + diffusion_rate

@njit(cache=True, parallel=True)
def compute_alpha(nu: np.ndarray, alpha: np.ndarray, dx: float, dt: float):

    N = nu.shape[0]
    dxcarre = dx**2

    for i in prange(N):
        for j in range(N):
            alpha[i,j] = dxcarre / (nu[i,j]*dt)

@njit(cache=True, parallel=True)
def jacobi_smago(table: np.ndarray, table_tmp: np.ndarray, b: np.ndarray, alpha: np.ndarray, nu: np.ndarray):

    N = table.shape[0]

    for i in prange(1, N-1):
        for j in range(1, N-1):

            nuE = 0.5*(nu[i,j] + nu[i,j+1])
            nuW = 0.5*(nu[i,j] + nu[i,j-1])
            nuN = 0.5*(nu[i,j] + nu[i-1,j])
            nuS = 0.5*(nu[i,j] + nu[i+1,j])

            a = alpha[i,j]

            inv = 1.0/(a + nuE + nuW + nuN + nuS)

            table_tmp[i,j] = (
                a*b[i,j]
                + nuE*table[i,j+1]
                + nuW*table[i,j-1]
                + nuN*table[i-1,j]
                + nuS*table[i+1,j]
            ) * inv

    open_boundary_numba(table_tmp)


@njit(cache=True)
def diffuse_smago(field: np.ndarray, field_tmp: np.ndarray, rhs: np.ndarray, diffusion_rate: float, nu: np.ndarray, normS: np.ndarray, alpha: np.ndarray, dt: float, dx: float, iterations: int):
    if diffusion_rate <= 0:
        return field, field_tmp

    nueff(nu, normS, diffusion_rate, dx, Cs=0.8)

    compute_alpha(nu, alpha, dx, dt)

    rhs[:] = field

    for _ in range(iterations):
        jacobi_smago(field, field_tmp, rhs, alpha, nu)

        field, field_tmp = field_tmp, field

    return field, field_tmp




# @njit(cache=True, parallel=True)
# def jacobi_smago2(table, table_tmp, b, vx, vy, gradxy_tmp, gradyx_tmp, gradxx_tmp, gradyy_tmp, nu, dx, dt, diffusion_rate, Cs):
#     N = table.shape[0]
#     dxcarre = dx**2
#     Coeff = (Cs*dx)**2
#     Sxy = 0.0
#     normS = 0.0 
#     gradx(vy, gradyx_tmp, dx)
#     grady(vx, gradxy_tmp, dx)
#     gradx(vx, gradxx_tmp, dx)
#     grady(vy, gradyy_tmp, dx)
#     for i in prange(N):
#             for j in range(N):
#                 Sxy = 0.5*(gradxy_tmp[i,j]+gradyx_tmp[i,j]) # 1,0 et 0,1 mais égaux
#                 normS = np.sqrt(
#                     2*(gradxx_tmp[i,j]**2 + gradyy_tmp[i,j]**2 + 2*(Sxy**2))
#                 )
#                 nu[i,j] = Coeff*normS + diffusion_rate
#     for i in prange(1, N-1):
#         for j in range(1, N-1):
#             nuE = 0.5*(nu[i,j] + nu[i,j+1])
#             nuW = 0.5*(nu[i,j] + nu[i,j-1])
#             nuN = 0.5*(nu[i,j] + nu[i-1,j])
#             nuS = 0.5*(nu[i,j] + nu[i+1,j])
#             a = dxcarre / (nu[i,j]*dt)
#             inv = 1.0/(a + nuE + nuW + nuN + nuS)
#             table_tmp[i,j] = (
#                 a*b[i,j]
#                 + nuE*table[i,j+1]
#                 + nuW*table[i,j-1]
#                 + nuN*table[i-1,j]
#                 + nuS*table[i+1,j]
#             ) * inv
#     open_boundary_numba(table_tmp)
# @njit(cache=True) # moins rapide
# def diffuse_smago2(field: np.ndarray, field_tmp: np.ndarray, rhs,vx, vy, gradxy_tmp, gradyx_tmp, gradxx_tmp, gradyy_tmp, diffusion_rate, nu, dt, dx, iterations):
#     if diffusion_rate <= 0:
#         return field, field_tmp
#     rhs[:] = field
#     for _ in range(iterations):
#         jacobi_smago2(field, field_tmp, rhs, vx, vy, gradxy_tmp, gradyx_tmp, gradxx_tmp, gradyy_tmp, nu, dx, dt, diffusion_rate, Cs=0.2)
#         field, field_tmp = field_tmp, field
#     return field, field_tmp




@njit(cache=True, parallel=True)
def compute_nu(vx: np.ndarray, vy: np.ndarray, d: float, nu: np.ndarray, diffusion_rate: float, Cs: float):

    N = vx.shape[0]

    inv2d = 1.0 / (2.0*d)
    coeff = (Cs*d)**2

    for i in prange(1, N-1):
        for j in range(1, N-1):

            dvxdx = (vx[i,j+1] - vx[i,j-1]) * inv2d
            dvxdy = (vx[i+1,j] - vx[i-1,j]) * inv2d

            dvydx = (vy[i,j+1] - vy[i,j-1]) * inv2d
            dvydy = (vy[i+1,j] - vy[i-1,j]) * inv2d

            Sxy = 0.5*(dvxdy + dvydx)

            normS = np.sqrt(
                2.0*(
                    dvxdx*dvxdx
                    + dvydy*dvydy
                    + 2.0*Sxy*Sxy
                )
            )

            nu[i,j] = coeff*normS + diffusion_rate

    open_boundary_numba(nu)

@njit(cache=True, parallel=True)
def compute_coeffs(nu: np.ndarray, nuE: np.ndarray, nuW: np.ndarray, nuN: np.ndarray, nuS: np.ndarray, a: np.ndarray, inv: np.ndarray, dx: float, dt: float):

    N = nu.shape[0]
    dxcarre = dx*dx

    for i in prange(1, N-1):
        for j in range(1, N-1):
            nuij = nu[i,j]
            fnuE = 0.5*(nuij + nu[i,j+1])
            fnuW = 0.5*(nuij + nu[i,j-1])
            fnuN = 0.5*(nuij + nu[i-1,j])
            fnuS = 0.5*(nuij + nu[i+1,j])

            nuE[i,j] = fnuE
            nuW[i,j] = fnuW
            nuN[i,j] = fnuN
            nuS[i,j] = fnuS

            a[i,j] = dxcarre / (nuij*dt)
            
            inv[i,j] = 1.0/(a[i,j] + fnuE + fnuW + fnuN + fnuS)

@njit(cache=True, parallel=True)
def jacobi_smago3(table: np.ndarray, table_tmp: np.ndarray, b: np.ndarray, nuE: np.ndarray, nuW: np.ndarray, nuN: np.ndarray, nuS: np.ndarray, a: np.ndarray, inv: np.ndarray):

    N = table.shape[0]

    for i in prange(1, N-1):
        for j in range(1, N-1):

            fnuE = nuE[i,j]
            fnuW = nuW[i,j]
            fnuN = nuN[i,j]
            fnuS = nuS[i,j]

            table_tmp[i,j] = (
                a[i,j]*b[i,j]
                + fnuE*table[i,j+1]
                + fnuW*table[i,j-1]
                + fnuN*table[i-1,j]
                + fnuS*table[i+1,j]
            ) * inv[i,j]

    open_boundary_numba(table_tmp)


@njit(cache=True)
def diffuse_smago3(field: np.ndarray, field_tmp: np.ndarray, rhs: np.ndarray, vx: np.ndarray, vy: np.ndarray, diffusion_rate: float, nu: np.ndarray, nuE: np.ndarray, nuW: np.ndarray, nuN: np.ndarray, nuS: np.ndarray, a: np.ndarray, inv: np.ndarray, dt: float, dx: float, iterations: int):
    if diffusion_rate <= 0:
        return field, field_tmp

    compute_nu(vx, vy, dx, nu, diffusion_rate, Cs=0.2)
    compute_coeffs(nu, nuE, nuW, nuN, nuS, a, inv, dx, dt)

    rhs[:] = field

    for _ in range(iterations):
        jacobi_smago3(field, field_tmp, rhs, nuE, nuW, nuN, nuS, a, inv)

        field, field_tmp = field_tmp, field

    return field, field_tmp